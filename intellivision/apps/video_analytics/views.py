# /apps/video_analytics/views.py

import json
import logging
import os
import tempfile
from uuid import uuid4

import cv2
import yt_dlp

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import VideoJob
from .serializers import VideoJobSerializer
from .tasks import process_video_job, ensure_api_media_url

# Import specific analytics functions for synchronous (non-Celery) tasks
from .analytics.food_waste_estimation import analyze_food_image
from .analytics.pothole_detection import run_pothole_image_detection

logger = logging.getLogger(__name__)


# =================================
# Helper Function for Async Job Creation (Now handles YouTube URLs)
# =================================

def _create_and_dispatch_job(request, job_type: str, extra_data: dict = None) -> Response:
    """
    Helper function to create a VideoJob instance, save the file (from upload or URL),
    and dispatch the processing task to Celery for asynchronous execution.
    """
    user = request.user
    file_obj = request.FILES.get('video') or request.FILES.get('image')
    youtube_url = request.data.get('youtube_url')

    if not file_obj and not youtube_url:
        return Response(
            {'error': "You must provide either a file to upload or a YouTube URL."},
            status=status.HTTP_400_BAD_REQUEST
        )
    if file_obj and youtube_url:
        return Response(
            {'error': "Please provide either a file or a YouTube URL, not both."},
            status=status.HTTP_400_BAD_REQUEST
        )

    job_data = {
        'user': user,
        'status': 'pending',
        'job_type': job_type,
    }
    if extra_data:
        job_data.update(extra_data)

    input_file_content = None
    temp_video_path = None

    try:
        if youtube_url:
            logger.info(f"Downloading video from YouTube URL using yt-dlp: {youtube_url}")

            # Create a temporary file path for yt-dlp to download to
            temp_video_path = tempfile.mktemp(suffix=".mp4")

            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/bestvideo[ext=mp4][vcodec!=av01]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!=av01]/best',
                'outtmpl': temp_video_path,
                'quiet': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            # Create a ContentFile from the downloaded video
            with open(temp_video_path, 'rb') as f:
                file_name = f"yt_{job_type}_{uuid4().hex}.mp4"
                input_file_content = ContentFile(f.read(), name=file_name)

            job_data['youtube_url'] = youtube_url

        else:  # A file was uploaded
            ext = os.path.splitext(file_obj.name)[1] or '.tmp'
            file_name = f"{job_type}_{uuid4().hex}{ext}"
            input_file_content = ContentFile(file_obj.read(), name=file_name)

        job_data['input_video'] = input_file_content

        job = VideoJob.objects.create(**job_data)
        process_video_job.delay(job.id)
        logger.info(f"Dispatched async job {job.id} ({job_type}) for user {user.username}")

        # Serialize the job object to include all its details in the response
        serializer = VideoJobSerializer(job)
        return Response(serializer.data, status=status.HTTP_202_ACCEPTED)

    except Exception as e:
        logger.error(f"Error during job creation: {e}", exc_info=True)
        # Check for specific yt-dlp download error
        if isinstance(e, yt_dlp.utils.DownloadError):
            return Response({'error': f"Failed to download video from URL. Please check the link. Error: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)
        return Response({'error': f"An unexpected error occurred: {str(e)}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # Clean up the temporary downloaded file
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)


# =================================
# NEW VIEW for YouTube Frame Extraction
# =================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_youtube_frame_view(request):
    """
    Accepts a YouTube URL, downloads the video using yt-dlp, extracts the first frame,
    saves it, and returns its public URL. Used for interactive canvas setup.
    """
    youtube_url = request.data.get('youtube_url')
    if not youtube_url:
        return Response({'error': 'youtube_url is required.'}, status=status.HTTP_400_BAD_REQUEST)

    temp_video_path = None
    temp_frame_path = None
    try:
        logger.info(f"Fetching frame for YouTube URL using yt-dlp: {youtube_url}")

        temp_video_path = tempfile.mktemp(suffix=".mp4")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/bestvideo[ext=mp4][vcodec!=av01]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!=av01]/best',
            'outtmpl': temp_video_path,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # Use OpenCV to capture the first frame
        cap = cv2.VideoCapture(temp_video_path)
        success, frame = cap.read()
        cap.release()

        if not success:
            return Response({'error': 'Could not extract a frame from the video.'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Save the frame to a temporary image file
        temp_frame_path = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(temp_frame_path, frame)

        # Save the frame to Django's default storage to get a permanent URL
        with open(temp_frame_path, 'rb') as f:
            frame_filename = f"thumbnails/yt_thumb_{uuid4().hex}.jpg"
            saved_path = default_storage.save(frame_filename, ContentFile(f.read()))
            
            # Use Django's storage URL generation for proper path handling
            relative_url = default_storage.url(saved_path)
            
            # Handle URL construction more robustly
            if relative_url.startswith('http'):
                # Storage already returned a full URL
                frame_url = relative_url
            else:
                # Ensure we have a clean relative path
                if relative_url.startswith('/'):
                    relative_url = relative_url[1:]  # Remove only the first slash
                
                # Build the full URL with proper protocol and domain
                frame_url = f"https://intellivision.aionos.co/{relative_url}"
            
            # Ensure the URL is properly formatted - safety net for malformed URLs
            if 'https//' in frame_url:
                frame_url = frame_url.replace('https//', 'https://')

        return Response({'frame_url': frame_url}, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error fetching YouTube frame: {e}", exc_info=True)
        if isinstance(e, yt_dlp.utils.DownloadError):
            return Response({'error': f"Failed to download video from URL. Please check the link. Error: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)
        return Response({'error': f'An error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # Clean up temporary files
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        if temp_frame_path and os.path.exists(temp_frame_path):
            os.unlink(temp_frame_path)


# =================================
# Asynchronous API Views (Updated to use the new helper)
# =================================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def people_count_view(request):
    """API endpoint for people counting from an uploaded video or YouTube URL."""
    return _create_and_dispatch_job(request, 'people_count')


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def car_count_view(request):
    """API endpoint for car counting from an uploaded video or YouTube URL."""
    return _create_and_dispatch_job(request, 'car_count')


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def parking_analysis_view(request):
    """API endpoint for parking analysis from an uploaded video or YouTube URL."""
    return _create_and_dispatch_job(request, 'parking_analysis')


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def wildlife_detection_video_view(request):
    """API endpoint for wildlife detection from an uploaded video or YouTube URL."""
    return _create_and_dispatch_job(request, 'wildlife_detection')


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def wildlife_detection_image_view(request):
    """API endpoint for wildlife detection from an uploaded image."""
    return _create_and_dispatch_job(request, 'wildlife_detection')


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def room_readiness_view(request):
    """API endpoint for room readiness analysis from an uploaded image or video."""
    return _create_and_dispatch_job(request, 'room_readiness')


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def emergency_count_view(request):
    """API endpoint for emergency counting, requiring line coordinates."""
    emergency_lines_str = request.data.get("emergency_lines")
    if not emergency_lines_str:
        return Response({'error': "'emergency_lines' is a required field."}, status=status.HTTP_400_BAD_REQUEST)
    try:
        emergency_lines = json.loads(emergency_lines_str)
    except json.JSONDecodeError:
        return Response({'error': "Invalid JSON format in 'emergency_lines'."}, status=status.HTTP_400_BAD_REQUEST)

    extra_data = {
        'emergency_lines': emergency_lines,
        'video_width': request.data.get('video_width'),
        'video_height': request.data.get('video_height'),
    }
    return _create_and_dispatch_job(request, 'emergency_count', extra_data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def lobby_detection_view(request):
    """API endpoint for lobby/crowd detection, requiring zone definitions."""
    lobby_zones_str = request.data.get("lobby_zones")
    if not lobby_zones_str:
        return Response({'error': "'lobby_zones' is a required field."}, status=status.HTTP_400_BAD_REQUEST)
    try:
        lobby_zones = json.loads(lobby_zones_str)
    except json.JSONDecodeError:
        return Response({'error': "Invalid JSON format in 'lobby_zones'."}, status=status.HTTP_400_BAD_REQUEST)

    extra_data = {
        'lobby_zones': lobby_zones,
        'video_width': request.data.get('video_width'),
        'video_height': request.data.get('video_height'),
    }
    return _create_and_dispatch_job(request, 'lobby_detection', extra_data)


# =================================
# Synchronous API Views (No changes needed)
# =================================
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def food_waste_estimation_view(request):
    """
    API endpoint for food waste estimation from a single uploaded image.
    This is a synchronous operation that calls the Azure OpenAI API.
    """
    image_file = request.FILES.get('image')
    if not image_file:
        return Response({'error': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    # Use a temporary file to get a path for the analytics function
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_in:
        for chunk in image_file.chunks():
            tmp_in.write(chunk)
        input_path = tmp_in.name

    try:
        # The analytics function handles the API call and returns a result dictionary
        result = analyze_food_image(input_path)

        if result.get('status') == 'failed':
            error_msg = result.get('error', 'Unknown error from analytics function')
            logger.error(f"Food waste estimation failed: {error_msg}")
            return Response({'error': error_msg}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Return the 'data' part of the result directly to the client
        return Response(result.get('data', {}), status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error in food_waste_estimation_view: {e}", exc_info=True)
        return Response({'error': 'An unexpected server error occurred.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # Ensure the temporary file is always cleaned up
        if os.path.exists(input_path):
            os.unlink(input_path)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def pothole_detection_image_view(request):
    """
    API endpoint for pothole detection from a single uploaded image.
    This is a synchronous operation.
    """
    image_file = request.FILES.get('image')
    if not image_file:
        return Response({'error': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    # Create temporary files for input and output
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_in:
        for chunk in image_file.chunks():
            tmp_in.write(chunk)
        input_path = tmp_in.name

    output_path = tempfile.mktemp(suffix=".jpg")

    try:
        # Run the synchronous analytics function
        result = run_pothole_image_detection(input_path, output_path)

        # Check if the analytics function returned an error
        if 'error' in result:
            logger.error(f"Pothole detection analytics failed: {result['error']}")
            return Response({'error': result['error']}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # The analytics function saves the annotated image to output_path.
        # Now, we save it to Django's storage to get a web-accessible URL.
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                # Create a unique name for the stored file
                output_filename = f"outputs/pothole_{uuid4().hex}.jpg"
                saved_path = default_storage.save(output_filename, ContentFile(f.read()))
                output_url = default_storage.url(saved_path)
                # Add the final URL to the result dictionary
                result['output_url'] = output_url
        else:
            # This case should ideally not happen if the analytics function is successful
            logger.warning(f"Pothole detection ran but output file not found at {output_path}")
            result['output_url'] = None

        return Response(result, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error in pothole_detection_image_view: {e}", exc_info=True)
        return Response({'error': 'An unexpected server error occurred.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # Ensure temporary files are always cleaned up
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


# =================================
# Standard ViewSet and User Info
# =================================

class VideoJobViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing VideoJob objects (listing, retrieving).
    Provides a standard RESTful interface for the VideoJob model.
    """
    queryset = VideoJob.objects.all()
    serializer_class = VideoJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """
        This view should return a list of all the jobs
        for the currently authenticated user.
        """
        return VideoJob.objects.filter(user=self.request.user).order_by('-created_at')


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user_view(request):
    """
    API endpoint to retrieve the current authenticated user's info.
    """
    user = request.user
    return Response({
        "id": user.id,
        "username": user.username,
        "email": user.email
    })
