from django.shortcuts import render
from rest_framework import viewsets, permissions, serializers
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import action
from django.contrib.auth import get_user_model

from .models import VideoJob
from .serializers import VideoJobSerializer
from .tasks import process_video_job

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone
from django.core.files.storage import default_storage
from uuid import uuid4
import os
import tempfile
import logging
import json

# Import the food waste estimation function
from apps.video_analytics.analytics.food_waste_estimation import analyze_food_image, analyze_multiple_food_images
from apps.video_analytics.convert import convert_to_web_image

logger = logging.getLogger(__name__)

class VideoJobViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing VideoJob objects.
    Handles creation, listing, and retrieval of video analytics jobs for authenticated users.
    """
    queryset = VideoJob.objects.all()
    serializer_class = VideoJobSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        """
        Handles creation of a new VideoJob instance.
        - Validates emergency_lines for emergency_count jobs.
        - Associates the job with the current user.
        - Triggers asynchronous processing via Celery.
        """
        data = self.request.data
        job_type = data.get("job_type", "emergency_count")

        if job_type == "emergency_count":
            # Validate emergency_lines for emergency_count jobs
            emergency_lines = data.get("emergency_lines")
            if not emergency_lines:
                raise serializers.ValidationError("'emergency_lines' is required for emergency_count jobs.")
            # Accept both JSON string and list
            import json
            if isinstance(emergency_lines, str):
                try:
                    emergency_lines = json.loads(emergency_lines)
                except Exception:
                    raise serializers.ValidationError("'emergency_lines' must be a valid JSON list of line dicts.")
            if not isinstance(emergency_lines, list) or not all(isinstance(line, dict) for line in emergency_lines):
                raise serializers.ValidationError("'emergency_lines' must be a list of dicts.")
            # Validate each line dict
            for idx, line in enumerate(emergency_lines):
                for key in ["start_x", "start_y", "end_x", "end_y"]:
                    if key not in line:
                        raise serializers.ValidationError(f"Line {idx+1} missing '{key}' in emergency_lines.")
                    try:
                        val = float(line[key])
                    except (ValueError, TypeError):
                        raise serializers.ValidationError(f"Line {idx+1} '{key}' must be a valid number.")
                    if (key.endswith('x') and not (0 <= val <= 1920)) or (key.endswith('y') and not (0 <= val <= 1080)):
                        raise serializers.ValidationError(f"Line {idx+1} '{key}' out of bounds.")
            # Validate inDirection
            if "inDirection" not in line:
                raise serializers.ValidationError(f"Line {idx+1} missing 'inDirection' in emergency_lines.")
            if line["inDirection"] not in ["UP", "DOWN", "LR", "RL"]:
                raise serializers.ValidationError(f"Line {idx+1} 'inDirection' must be one of: 'UP', 'DOWN', 'LR', 'RL'.")
            data._mutable = True
            data["emergency_lines"] = emergency_lines

        User = get_user_model()
        user = self.request.user if self.request.user.is_authenticated else User.objects.first()
        instance = serializer.save(user=user, status='pending')
        # Trigger asynchronous processing of the job
        process_video_job.delay(instance.id)

    def get_queryset(self):
        """
        Returns jobs belonging to the authenticated user, ordered by creation date.
        """
        user = self.request.user
        if user.is_authenticated:
            return VideoJob.objects.filter(user=user).order_by('-created_at')
        return VideoJob.objects.none()

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user_view(request):
    """
    API endpoint to retrieve the current authenticated user's ID and username.
    """
    user = request.user
    return Response({
        "id": user.id,
        "username": user.username,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def food_waste_estimation_view(request):
    """
    API endpoint for food waste estimation from uploaded images.
    Accepts one or more images, processes them, and returns the results.
    """
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    image_files = request.FILES.getlist('images')
    if not image_files and 'image' in request.FILES:
        image_files = [request.FILES['image']]

    if not image_files:
        return Response({'error': 'No image(s) file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    temp_paths = []
    saved_files = []
    try:
        for image_file in image_files:
            # Save each uploaded image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                for chunk in image_file.chunks():
                    tmp.write(chunk)
                temp_paths.append(tmp.name)
            # Save each image file for database storage
            image_file.seek(0)
            image_content = ContentFile(image_file.read(), name=f"food_{uuid4()}.jpg")
            saved_files.append(image_content)

        # Create a VideoJob for the food waste estimation
        # Store the first image as input_video (for backward compatibility)
        # Store all images in results for frontend access
        job = VideoJob.objects.create(
            user=user,
            status='pending',
            input_video=saved_files[0],
            job_type='food_waste_estimation',
            created_at=timezone.now(),
            updated_at=timezone.now(),
        )
        # Analyze the images
        result_data = analyze_multiple_food_images(temp_paths)

        # Add input images to results for frontend access
        if isinstance(result_data, list):
            # Multiple images - save each image to storage and get the correct path
            input_image_paths = []
            for i, image_content in enumerate(saved_files):
                saved_path = default_storage.save(f"uploads/{image_content.name}", image_content)
                input_image_paths.append(saved_path)
            for i, result in enumerate(result_data):
                if i < len(input_image_paths):
                    result['input_image'] = input_image_paths[i]
        else:
            # Single image - use the path from the job's input_video field
            result_data['input_image'] = job.input_video.name

        job.results = result_data
        job.status = 'done'
        job.save()
    finally:
        # Clean up temporary files
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
        'results': job.results,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def pothole_detection_image_view(request):
    """
    API endpoint for pothole detection from an uploaded image.
    Processes the image and returns the detection results and output path.
    """
    from apps.video_analytics.analytics.pothole_detection import run_pothole_image_detection

    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    image_file = request.FILES.get('image')
    if not image_file:
        return Response({'error': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    temp_path = f"/tmp/pothole_{uuid4()}.jpg"
    with open(temp_path, 'wb') as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    output_path = temp_path.replace('.jpg', '_out.jpg')
    result_data = run_pothole_image_detection(temp_path, output_path)

    # Convert to web-friendly WebP image
    web_output_path = output_path.replace('.jpg', '_web.webp')
    if convert_to_web_image(output_path, web_output_path, format="WEBP", quality=80):
        final_output_path = web_output_path
    else:
        final_output_path = output_path

    # Check if output file exists before trying to open
    if not os.path.exists(final_output_path):
        return Response({'error': 'Detection failed, output file not found.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Convert to web-accessible path
    with open(final_output_path, 'rb') as out_f:
        saved_name = f"results/pothole_{uuid4()}.webp" if final_output_path.endswith('.webp') else f"results/pothole_{uuid4()}.jpg"
        saved_path = default_storage.save(saved_name, ContentFile(out_f.read()))
        output_url = default_storage.url(saved_path)
        result_data['output_path'] = output_url

    # Create a VideoJob for the pothole detection
    job = VideoJob.objects.create(
        user=user,
        status='done',
        input_video=ContentFile(image_file.read(), name=f"pothole_{uuid4()}.jpg"),
        job_type='pothole_detection',
        results=result_data,
        created_at=timezone.now(),
        updated_at=timezone.now(),
    )

    # Clean up temporary files
    os.remove(temp_path)
    os.remove(output_path)
    if os.path.exists(web_output_path):
        os.remove(web_output_path)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
        'results': job.results,
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def car_count_view(request):
    """
    API endpoint for car counting from an uploaded video.
    Triggers asynchronous processing and returns the job info.
    """
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    video_file = request.FILES.get('video')
    if not video_file:
        return Response({'error': 'No video file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    video_file.seek(0)
    video_content = ContentFile(video_file.read(), name=f"carcount_{uuid4()}.mp4")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=video_content,
        job_type='car_count',
        created_at=timezone.now(),
        updated_at=timezone.now(),
    )

    process_video_job.delay(job.id)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def parking_analysis_view(request):
    """
    API endpoint for parking analysis from an uploaded video.
    Logs request data for debugging, triggers async processing, and returns job info.
    """
    # Debug: Log all request data and files for backend logging
    logger.info("=== PARKING ANALYSIS DEBUG ===")
    logger.info("Request data keys: %s", list(request.data.keys()))
    logger.info("Request FILES keys: %s", list(request.FILES.keys()))
    logger.debug("Request data (raw): %s", dict(request.data))
    logger.debug("Request FILES (raw): %s", {k: v.name for k, v in request.FILES.items()})
    logger.info("job_type: %s", request.data.get("job_type"))
    logger.info("analysis_type: %s", request.data.get("analysis_type"))

    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    video_file = request.FILES.get('video')
    if not video_file:
        return Response({'error': 'No video file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    video_file.seek(0)
    video_content = ContentFile(video_file.read(), name=f"carcount_{uuid4()}.mp4")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=video_content,
        job_type='parking_analysis',
        created_at=timezone.now(),
        updated_at=timezone.now(),
    )

    process_video_job.delay(job.id)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def wildlife_detection_image_view(request):
    """
    API endpoint for wildlife detection from an uploaded image.
    Triggers asynchronous processing and returns the job info.
    """
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    image_file = request.FILES.get('image')
    if not image_file:
        return Response({'error': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    image_file.seek(0)
    image_content = ContentFile(image_file.read(), name=f"wildlife_{uuid4()}.jpg")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=image_content,
        job_type='wildlife_detection',
        created_at=timezone.now(),
        updated_at=timezone.now(),
    )

    process_video_job.delay(job.id)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def wildlife_detection_video_view(request):
    """
    API endpoint for wildlife detection from an uploaded video.
    Triggers asynchronous processing and returns the job info.
    """
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    video_file = request.FILES.get('video')
    if not video_file:
        return Response({'error': 'No video file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    video_file.seek(0)
    video_content = ContentFile(video_file.read(), name=f"wildlife_{uuid4()}.mp4")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=video_content,
        job_type='wildlife_detection',
        created_at=timezone.now(),
        updated_at=timezone.now(),
    )

    process_video_job.delay(job.id)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def emergency_count_view(request):
    """
    API endpoint for emergency counting from an uploaded video.
    Validates emergency_lines and triggers async processing.
    Also handles YouTube line configuration updates.
    """
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    # Check if this is a YouTube configuration update
    is_youtube_config = request.data.get('is_youtube_config') == 'true'
    job_id = request.data.get('job_id')
    
    if is_youtube_config and job_id:
        # Handle YouTube line configuration
        try:
            job = VideoJob.objects.get(id=job_id, user=user)
        except VideoJob.DoesNotExist:
            return Response({'error': 'Job not found or access denied.'}, status=status.HTTP_404_NOT_FOUND)
        
        if job.status != 'pending_config':
            return Response({'error': 'Job is not in pending configuration state.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Get and validate emergency_lines
        emergency_lines = request.data.get("emergency_lines")
        if isinstance(emergency_lines, str):
            try:
                emergency_lines = json.loads(emergency_lines)
            except Exception:
                return Response({'error': "'emergency_lines' must be valid JSON."}, status=status.HTTP_400_BAD_REQUEST)
        
        if not isinstance(emergency_lines, list) or len(emergency_lines) != 2:
            return Response({'error': "'emergency_lines' must be a list of exactly 2 line configurations."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Update job with line configuration and resume processing
        job.emergency_lines = emergency_lines
        job.video_width = int(request.data.get('video_width', 1920))
        job.video_height = int(request.data.get('video_height', 1080))
        job.status = 'processing'
        job.results = {
            **job.results,
            'emergency_lines': emergency_lines,
            'configuration_completed': True
        }
        job.save()
        
        # Resume emergency count processing with the saved YouTube video
        from .tasks import resume_youtube_emergency_count
        resume_youtube_emergency_count.delay(job.id)
        
        return Response({
            'message': 'Line configuration saved. Processing resumed.',
            'job_id': job.id,
            'status': job.status
        })
    
    # Original emergency count logic for uploaded videos
    video_file = request.FILES.get('video')
    if not video_file:
        return Response({'error': 'No video file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    # Debug: Log all request data
    logger.info("=== EMERGENCY COUNT DEBUG ===")
    logger.info("Request data keys: %s", list(request.data.keys()))
    logger.info("Request FILES keys: %s", list(request.FILES.keys()))

    # Validate emergency_lines
    emergency_lines = request.data.get("emergency_lines")
    logger.info("Emergency lines type: %s", type(emergency_lines))
    logger.info("Emergency lines value: %s", emergency_lines)

    if not emergency_lines:
        return Response({'error': "'emergency_lines' is required for emergency_count jobs."}, status=status.HTTP_400_BAD_REQUEST)
    import json
    if isinstance(emergency_lines, str):
        try:
            emergency_lines = json.loads(emergency_lines)
            logger.info("Parsed emergency_lines from JSON: %s", emergency_lines)
        except Exception as e:
            logger.error("JSON parsing error: %s", e)
            return Response({'error': "'emergency_lines' must be a valid JSON list of line dicts."}, status=status.HTTP_400_BAD_REQUEST)
    if not isinstance(emergency_lines, list) or not all(isinstance(line, dict) for line in emergency_lines):
        logger.error("Emergency lines validation failed - not a list of dicts")
        return Response({'error': "'emergency_lines' must be a list of dicts."}, status=status.HTTP_400_BAD_REQUEST)

    logger.info("Validating each line...")
    for idx, line in enumerate(emergency_lines):
        logger.info(f"Line {idx+1}: %s", line)
        for key in ["start_x", "start_y", "end_x", "end_y"]:
            if key not in line:
                logger.error(f"Missing key: {key}")
                return Response({'error': f"Line {idx+1} missing '{key}' in emergency_lines."}, status=status.HTTP_400_BAD_REQUEST)
            try:
                val = float(line[key])
            except (ValueError, TypeError):
                logger.error(f"Invalid number for {key}: {line[key]}")
                return Response({'error': f"Line {idx+1} '{key}' must be a valid number."}, status=status.HTTP_400_BAD_REQUEST)
            if (key.endswith('x') and not (0 <= val <= 1920)) or (key.endswith('y') and not (0 <= val <= 1080)):
                logger.error(f"Value out of bounds for {key}: {val}")
                return Response({'error': f"Line {idx+1} '{key}' out of bounds."}, status=status.HTTP_400_BAD_REQUEST)
        # Validate inDirection
        if "inDirection" not in line:
            logger.error(f"Missing inDirection in line {idx+1}")
            return Response({'error': f"Line {idx+1} missing 'inDirection' in emergency_lines."}, status=status.HTTP_400_BAD_REQUEST)
        if line["inDirection"] not in ["UP", "DOWN", "LR", "RL"]:
            logger.error(f"Invalid inDirection in line {idx+1}: {line['inDirection']}")
            return Response({'error': f"Line {idx+1} 'inDirection' must be one of: 'UP', 'DOWN', 'LR', 'RL'."}, status=status.HTTP_400_BAD_REQUEST)

    logger.info("All validation passed!")

    video_file.seek(0)
    video_content = ContentFile(video_file.read(), name=f"emergency_{uuid4()}.mp4")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=video_content,
        job_type='emergency_count',
        video_width=int(request.data['video_width']) if 'video_width' in request.data else None,
        video_height=int(request.data['video_height']) if 'video_height' in request.data else None,
        created_at=timezone.now(),
        updated_at=timezone.now(),
        emergency_lines=emergency_lines,
        results={"emergency_lines": emergency_lines},
    )
    # DEBUG: Log the emergency_lines and job id after creation
    logger.warning(f"DEBUG: Created job {job.id} with emergency_lines: {job.emergency_lines}")

    process_video_job.delay(job.id)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def room_readiness_view(request):
    """
    API endpoint for room readiness analysis from an uploaded image or video.
    Triggers asynchronous processing and returns the job info.
    """
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    # Accept both image and video uploads
    image_file = request.FILES.get('image')
    video_file = request.FILES.get('video')
    if not image_file and not video_file:
        return Response({'error': 'No image or video file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    if image_file:
        image_file.seek(0)
        input_content = ContentFile(image_file.read(), name=f"roomreadiness_{uuid4()}.jpg")
    else:
        video_file.seek(0)
        input_content = ContentFile(video_file.read(), name=f"roomreadiness_{uuid4()}.mp4")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=input_content,
        job_type='room_readiness',
        created_at=timezone.now(),
        updated_at=timezone.now(),
    )

    process_video_job.delay(job.id)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def lobby_detection_view(request):
    """
    API endpoint for lobby/crowd detection from an uploaded video.
    Triggers asynchronous processing and returns the job info.
    """
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    video_file = request.FILES.get('video')
    if not video_file:
        return Response({'error': 'No video file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    # Validate lobby_zones
    lobby_zones = request.data.get("lobby_zones")
    if not lobby_zones:
        return Response({'error': "'lobby_zones' is required for lobby detection jobs."}, status=status.HTTP_400_BAD_REQUEST)
    import json
    if isinstance(lobby_zones, str):
        try:
            lobby_zones = json.loads(lobby_zones)
        except Exception:
            return Response({'error': "'lobby_zones' must be a valid JSON list of zone dicts."}, status=status.HTTP_400_BAD_REQUEST)
    if not isinstance(lobby_zones, list) or not all(isinstance(zone, dict) for zone in lobby_zones):
        return Response({'error': "'lobby_zones' must be a list of dicts."}, status=status.HTTP_400_BAD_REQUEST)
    for idx, zone in enumerate(lobby_zones):
        if 'points' not in zone or 'threshold' not in zone:
            return Response({'error': f"Zone {idx+1} missing 'points' or 'threshold'."}, status=status.HTTP_400_BAD_REQUEST)
        if not isinstance(zone['points'], list) or not all(isinstance(pt, list) and len(pt) == 2 for pt in zone['points']):
            return Response({'error': f"Zone {idx+1} 'points' must be a list of [x, y] pairs."}, status=status.HTTP_400_BAD_REQUEST)
        if not isinstance(zone['threshold'], int):
            return Response({'error': f"Zone {idx+1} 'threshold' must be an integer."}, status=status.HTTP_400_BAD_REQUEST)

    video_file.seek(0)
    video_content = ContentFile(video_file.read(), name=f"lobby_{uuid4()}.mp4")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=video_content,
        job_type='lobby_detection',
        lobby_zones=lobby_zones,
        video_width=int(request.data['video_width']) if 'video_width' in request.data else None,
        video_height=int(request.data['video_height']) if 'video_height' in request.data else None,
        created_at=timezone.now(),
        updated_at=timezone.now(),
    )

    process_video_job.delay(job.id)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
    })

class AnalyzeYouTubeView(APIView):
    """API endpoint for analyzing a YouTube video with a selected job type."""
    permission_classes = [AllowAny]
    def post(self, request):
        job_type = request.data.get('job_type')
        youtube_url = request.data.get('youtube_url')
        if not job_type or not youtube_url:
            return Response({'success': False, 'message': 'job_type and youtube_url are required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Create a VideoJob record for tracking
        User = get_user_model()
        user = request.user if request.user.is_authenticated else User.objects.first()
        
        # Create a dummy file entry for YouTube URL
        from django.core.files.base import ContentFile
        dummy_content = ContentFile(f"YouTube URL: {youtube_url}".encode(), name=f"youtube_{uuid4()}.txt")
        
        job = VideoJob.objects.create(
            user=user,
            status='pending',
            input_video=dummy_content,
            job_type=job_type,
            created_at=timezone.now(),
            updated_at=timezone.now(),
            results={'youtube_url': youtube_url, 'source': 'youtube'}
        )
        
        # Start the analysis task with the job ID
        from .tasks import analyze_youtube_video_task
        analyze_youtube_video_task.delay(job.id, job_type, youtube_url)
        
        return Response({
            'job_id': job.id,
            'status': job.status,
            'created_at': job.created_at,
            'job_type': job.job_type,
        })
