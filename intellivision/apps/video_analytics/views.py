from django.shortcuts import render
from rest_framework import viewsets, permissions, serializers
from rest_framework.parsers import MultiPartParser, FormParser
from django.contrib.auth import get_user_model

from .models import VideoJob
from .serializers import VideoJobSerializer
from .tasks import process_video_job

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone
from django.core.files.storage import default_storage
from uuid import uuid4
import os
import tempfile

# Import the food waste estimation function
from apps.video_analytics.analytics.food_waste_estimation import analyze_food_image, analyze_multiple_food_images

class VideoJobViewSet(viewsets.ModelViewSet):
    queryset = VideoJob.objects.all()
    serializer_class = VideoJobSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        data = self.request.data
        job_type = data.get("job_type", "emergency_count")

        if job_type == "emergency_count":
            # Validate line coordinates for emergency_count jobs
            line_coords = [
                'line1_start_x', 'line1_start_y', 'line1_end_x', 'line1_end_y',
                'line2_start_x', 'line2_start_y', 'line2_end_x', 'line2_end_y'
            ]
            for coord in line_coords:
                if coord not in data:
                    raise serializers.ValidationError(f"Line coordinate '{coord}' is required for emergency_count jobs.")
                try:
                    float(data[coord])
                except (ValueError, TypeError):
                    raise serializers.ValidationError(f"Line coordinate '{coord}' must be a valid number.")

            # Validate that line coordinates are within reasonable bounds (0-1920 for x, 0-1080 for y)
            for coord in line_coords:
                value = float(data[coord])
                if 'x' in coord and not (0 <= value <= 1920):
                    raise serializers.ValidationError(f"Line coordinate '{coord}' must be between 0 and 1920.")
                if 'y' in coord and not (0 <= value <= 1080):
                    raise serializers.ValidationError(f"Line coordinate '{coord}' must be between 0 and 1080.")

        User = get_user_model()
        user = self.request.user if self.request.user.is_authenticated else User.objects.first()
        instance = serializer.save(user=user, status='pending')
        process_video_job.delay(instance.id)

    def get_queryset(self):
        user = self.request.user
        if user.is_authenticated:
            return VideoJob.objects.filter(user=user).order_by('-created_at')
        return VideoJob.objects.none()

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user_view(request):
    user = request.user
    return Response({
        "id": user.id,
        "username": user.username,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def food_waste_estimation_view(request):
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
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                for chunk in image_file.chunks():
                    tmp.write(chunk)
                temp_paths.append(tmp.name)
            if len(saved_files) == 0:
                image_file.seek(0)
                image_content = ContentFile(image_file.read(), name=f"food_{uuid4()}.jpg")
                saved_files.append(image_content)

        job = VideoJob.objects.create(
            user=user,
            status='pending',
            input_video=saved_files[0],
            job_type='food_waste_estimation',
            created_at=timezone.now(),
            updated_at=timezone.now(),
        )
        result_data = analyze_multiple_food_images(temp_paths)
        job.results = result_data
        job.status = 'done'
        job.save()
    finally:
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

    # Convert to web-accessible path
    with open(output_path, 'rb') as out_f:
        saved_name = f"results/pothole_{uuid4()}.jpg"
        saved_path = default_storage.save(saved_name, ContentFile(out_f.read()))
        output_url = default_storage.url(saved_path)
        result_data['output_path'] = output_url

    job = VideoJob.objects.create(
        user=user,
        status='done',
        input_video=ContentFile(image_file.read(), name=f"pothole_{uuid4()}.jpg"),
        job_type='pothole_detection',
        results=result_data,
        created_at=timezone.now(),
        updated_at=timezone.now(),
    )

    os.remove(temp_path)
    os.remove(output_path)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
        'results': job.results,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def pothole_detection_video_view(request):
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    video_file = request.FILES.get('video')
    if not video_file:
        return Response({'error': 'No video file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    video_file.seek(0)
    video_content = ContentFile(video_file.read(), name=f"pothole_{uuid4()}.mp4")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=video_content,
        job_type='pothole_detection',
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
def car_count_view(request):
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
def pest_monitoring_image_view(request):
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    image_file = request.FILES.get('image')
    if not image_file:
        return Response({'error': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    image_file.seek(0)
    image_content = ContentFile(image_file.read(), name=f"pest_{uuid4()}.jpg")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=image_content,
        job_type='pest_monitoring',
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
def pest_monitoring_video_view(request):
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    video_file = request.FILES.get('video')
    if not video_file:
        return Response({'error': 'No video file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    video_file.seek(0)
    video_content = ContentFile(video_file.read(), name=f"pest_{uuid4()}.mp4")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=video_content,
        job_type='pest_monitoring',
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
@permission_classes([AllowAny])
def anpr_callback_view(request):
    job_id = request.data.get('job_id')
    status = request.data.get('status')
    summary = request.data.get('summary')
    history = request.data.get('history')
    download_urls = request.data.get('download_urls', {})

    from .models import VideoJob
    try:
        job = VideoJob.objects.get(id=job_id)
        job.status = 'done' if status == 'completed' else 'failed'
        job.results = {
            'summary': summary,
            'history': history,
            **download_urls
        }
        job.save()
        return Response({'message': 'Callback received and job updated.'})
    except VideoJob.DoesNotExist:
        return Response({'error': 'Job not found.'}, status=404)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def emergency_count_view(request):
    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    video_file = request.FILES.get('video')
    if not video_file:
        return Response({'error': 'No video file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    # Validate line coordinates for emergency_count jobs
    line_coords = [
        'line1_start_x', 'line1_start_y', 'line1_end_x', 'line1_end_y',
        'line2_start_x', 'line2_start_y', 'line2_end_x', 'line2_end_y'
    ]
    for coord in line_coords:
        if coord not in request.data:
            return Response({'error': f"Line coordinate '{coord}' is required for emergency_count jobs."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            float(request.data[coord])
        except (ValueError, TypeError):
            return Response({'error': f"Line coordinate '{coord}' must be a valid number."}, status=status.HTTP_400_BAD_REQUEST)

        # Validate that line coordinates are within reasonable bounds
        value = float(request.data[coord])
        if 'x' in coord and not (0 <= value <= 1920):
            return Response({'error': f"Line coordinate '{coord}' must be between 0 and 1920."}, status=status.HTTP_400_BAD_REQUEST)
        if 'y' in coord and not (0 <= value <= 1080):
            return Response({'error': f"Line coordinate '{coord}' must be between 0 and 1080."}, status=status.HTTP_400_BAD_REQUEST)

    video_file.seek(0)
    video_content = ContentFile(video_file.read(), name=f"emergency_{uuid4()}.mp4")

    job = VideoJob.objects.create(
        user=user,
        status='pending',
        input_video=video_content,
        job_type='emergency_count',
        line1_start_x=float(request.data['line1_start_x']),
        line1_start_y=float(request.data['line1_start_y']),
        line1_end_x=float(request.data['line1_end_x']),
        line1_end_y=float(request.data['line1_end_y']),
        line2_start_x=float(request.data['line2_start_x']),
        line2_start_y=float(request.data['line2_start_y']),
        line2_end_x=float(request.data['line2_end_x']),
        line2_end_y=float(request.data['line2_end_y']),
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
