# /apps/video_analytics/views.py

"""
=====================================
Imports
=====================================
Import necessary modules for Django REST Framework views, authentication, and permissions.
"""

import json
import logging
import os
import tempfile
import time
from uuid import uuid4
from typing import Dict, Any

import cv2
import yt_dlp
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from rest_framework import permissions, status, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
import mimetypes
from django.utils import timezone

from .models import VideoJob
from .serializers import VideoJobSerializer
from .tasks import process_video_job, ensure_api_media_url

logger = logging.getLogger(__name__)

# Valid file extensions and size limit
VALID_EXTENSIONS = {'.mp4', '.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

"""
=====================================
Helper Function
=====================================
Defines utility function for creating and dispatching jobs.
All error responses use standardized format (Issue #10).
"""

def _create_and_dispatch_job(request, job_type: str, extra_data: Dict[str, Any] = None) -> Response:
    """
    Create a VideoJob, save file (upload or YouTube), and dispatch to Celery.

    Args:
        request: HTTP request object
        job_type: Type of analytics job
        extra_data: Additional job parameters

    Returns:
        Response with standardized format:
        {
            'status': 'pending' | 'failed',
            'job_type': str,
            'output_image': None,
            'output_video': None,
            'data': {'job_id': int, ...},
            'meta': {'timestamp': str, 'request_time': float},
            'error': dict | None
        }
    """
    start_time = time.time()
    user = request.user
    file_obj = request.FILES.get('video') or request.FILES.get('image')
    youtube_url = request.data.get('youtube_url')

    if not file_obj and not youtube_url:
        return Response({
            'status': 'failed',
            'job_type': job_type,
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': 'Provide either a file or YouTube URL', 'code': 'MISSING_INPUT'}
        }, status=status.HTTP_400_BAD_REQUEST)

    if file_obj and youtube_url:
        return Response({
            'status': 'failed',
            'job_type': job_type,
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': 'Provide either a file or YouTube URL, not both', 'code': 'INVALID_INPUT_COMBINATION'}
        }, status=status.HTTP_400_BAD_REQUEST)

    job_data = {'user': user, 'status': 'pending', 'job_type': job_type}
    if extra_data:
        job_data.update(extra_data)

    input_file_content = None
    temp_video_path = None

    try:
        if youtube_url:
            logger.info(f"🎬 Downloading YouTube: {youtube_url} for {job_type} (user: {user.username})")
            temp_video_path = tempfile.mktemp(suffix=".mp4")
            ydl_opts = {
                'format': 'bestvideo[ext=mp4][vcodec=h264]+bestaudio[ext=m4a]/best[ext=mp4][vcodec=h264]/best[ext=mp4]',
                'outtmpl': temp_video_path,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            if not os.path.exists(temp_video_path):
                return Response({
                    'status': 'failed',
                    'job_type': job_type,
                    'output_image': None,
                    'output_video': None,
                    'data': {},
                    'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                    'error': {'message': 'Failed to download YouTube video', 'code': 'DOWNLOAD_FAILED'}
                }, status=status.HTTP_400_BAD_REQUEST)

            # Validate file
            ext = '.mp4'
            size = os.path.getsize(temp_video_path)
            if size > MAX_FILE_SIZE:
                return Response({
                    'status': 'failed',
                    'job_type': job_type,
                    'output_image': None,
                    'output_video': None,
                    'data': {},
                    'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                    'error': {'message': f"File size {size / (1024*1024):.2f}MB exceeds 500MB limit", 'code': 'FILE_TOO_LARGE'}
                }, status=status.HTTP_400_BAD_REQUEST)

            with open(temp_video_path, 'rb') as f:
                file_name = f"yt_{job_type}_{uuid4().hex}.mp4"
                input_file_content = ContentFile(f.read(), name=file_name)
            job_data['youtube_url'] = youtube_url
            logger.info(f"✅ Downloaded YouTube video: {file_name} ({size / (1024*1024):.2f}MB)")

        else:
            ext = os.path.splitext(file_obj.name)[1].lower()
            if ext not in VALID_EXTENSIONS:
                return Response({
                    'status': 'failed',
                    'job_type': job_type,
                    'output_image': None,
                    'output_video': None,
                    'data': {},
                    'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                    'error': {'message': f"Invalid file type: {ext}. Allowed: {', '.join(VALID_EXTENSIONS)}", 'code': 'INVALID_FILE_TYPE'}
                }, status=status.HTTP_400_BAD_REQUEST)
            if file_obj.size > MAX_FILE_SIZE:
                return Response({
                    'status': 'failed',
                    'job_type': job_type,
                    'output_image': None,
                    'output_video': None,
                    'data': {},
                    'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                    'error': {'message': f"File size {file_obj.size / (1024*1024):.2f}MB exceeds 500MB limit", 'code': 'FILE_TOO_LARGE'}
                }, status=status.HTTP_400_BAD_REQUEST)

            file_name = f"{job_type}_{uuid4().hex}{ext}"
            input_file_content = ContentFile(file_obj.read(), name=file_name)
            logger.info(f"📁 Uploaded: {file_name} ({file_obj.size / (1024*1024):.2f}MB)")

        job_data['input_video'] = input_file_content
        job = VideoJob.objects.create(**job_data)
        task = process_video_job.delay(job.id)
        job.task_id = task.id
        job.save()
        logger.info(f"🚀 Dispatched job {job.id} ({job_type}) with task ID {task.id}")

        serializer = VideoJobSerializer(job)
        return Response({
            'status': 'pending',
            'job_type': job_type,
            'output_image': None,
            'output_video': None,
            'data': serializer.data,
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': None
        }, status=status.HTTP_202_ACCEPTED)

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"YouTube download failed for {job_type}: {e}", exc_info=True)
        return Response({
            'status': 'failed',
            'job_type': job_type,
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': f"Failed to download YouTube video: {str(e)}", 'code': 'DOWNLOAD_ERROR'}
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Job creation failed for {job_type}: {e}", exc_info=True)
        return Response({
            'status': 'failed',
            'job_type': job_type,
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': f"Unexpected error: {str(e)}", 'code': 'SERVER_ERROR'}
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

"""
=====================================
YouTube Frame Extraction
=====================================
Handles extraction of the first frame from a YouTube video URL.
All error responses use standardized format (Issue #10).
"""

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_youtube_frame_view(request):
    """
    Extract first frame from a YouTube URL and return its URL.

    Args:
        request: HTTP request with youtube_url

    Returns:
        Response with standardized format:
        {
            'status': 'completed' | 'failed',
            'job_type': 'youtube_frame_extraction',
            'output_image': str | None,
            'output_video': None,
            'data': {'frame_url': str},
            'meta': {'timestamp': str, 'request_time': float},
            'error': dict | None
        }
    """
    start_time = time.time()
    youtube_url = request.data.get('youtube_url')
    if not youtube_url:
        return Response({
            'status': 'failed',
            'job_type': 'youtube_frame_extraction',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': 'youtube_url is required', 'code': 'MISSING_URL'}
        }, status=status.HTTP_400_BAD_REQUEST)

    temp_video_path = None
    temp_frame_path = None
    try:
        logger.info(f"🎬 Fetching frame for YouTube URL: {youtube_url}")
        temp_video_path = tempfile.mktemp(suffix=".mp4")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][vcodec=h264]+bestaudio[ext=m4a]/best[ext=mp4][vcodec=h264]/best[ext=mp4]',
            'outtmpl': temp_video_path,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        if not os.path.exists(temp_video_path):
            return Response({
                'status': 'failed',
                'job_type': 'youtube_frame_extraction',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {'message': 'Failed to download video', 'code': 'DOWNLOAD_FAILED'}
            }, status=status.HTTP_400_BAD_REQUEST)

        size = os.path.getsize(temp_video_path)
        if size > MAX_FILE_SIZE:
            return Response({
                'status': 'failed',
                'job_type': 'youtube_frame_extraction',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {'message': f"File size {size / (1024*1024):.2f}MB exceeds 500MB limit", 'code': 'FILE_TOO_LARGE'}
            }, status=status.HTTP_400_BAD_REQUEST)

        cap = cv2.VideoCapture(temp_video_path)
        success, frame = cap.read()
        cap.release()
        if not success:
            return Response({
                'status': 'failed',
                'job_type': 'youtube_frame_extraction',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {'message': 'Could not extract frame', 'code': 'FRAME_EXTRACT_FAILED'}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        temp_frame_path = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(temp_frame_path, frame)

        with open(temp_frame_path, 'rb') as f:
            frame_filename = f"thumbnails/yt_thumb_{uuid4().hex}.jpg"
            saved_path = default_storage.save(frame_filename, ContentFile(f.read()))
            frame_url = ensure_api_media_url(default_storage.url(saved_path))

        logger.info(f"✅ Extracted frame: {frame_url}")
        return Response({
            'status': 'completed',
            'job_type': 'youtube_frame_extraction',
            'output_image': frame_url,
            'output_video': None,
            'data': {'frame_url': frame_url},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': None
        }, status=status.HTTP_200_OK)

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"YouTube frame extraction failed: {e}", exc_info=True)
        return Response({
            'status': 'failed',
            'job_type': 'youtube_frame_extraction',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': f"Failed to download video: {str(e)}", 'code': 'DOWNLOAD_ERROR'}
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"YouTube frame extraction error: {e}", exc_info=True)
        return Response({
            'status': 'failed',
            'job_type': 'youtube_frame_extraction',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': f"Unexpected error: {str(e)}", 'code': 'SERVER_ERROR'}
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        for path in [temp_video_path, temp_frame_path]:
            if path and os.path.exists(path):
                os.unlink(path)

"""
=====================================
Asynchronous API Views
=====================================
API views for submitting various analytics jobs, all requiring authentication.
All error responses use standardized format (Issue #10).
"""

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def people_count_view(request):
    """Submit people counting job."""
    return _create_and_dispatch_job(request, 'people-count')

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def car_count_view(request):
    """Submit car counting job."""
    return _create_and_dispatch_job(request, 'car-count')

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def parking_analysis_view(request):
    """Submit parking analysis job."""
    return _create_and_dispatch_job(request, 'parking-analysis')

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def wildlife_detection_video_view(request):
    """Submit wildlife detection video job."""
    return _create_and_dispatch_job(request, 'wildlife-detection')

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def wildlife_detection_image_view(request):
    """Submit wildlife detection image job."""
    return _create_and_dispatch_job(request, 'wildlife-detection')

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def room_readiness_view(request):
    """Submit room readiness analysis job."""
    return _create_and_dispatch_job(request, 'room-readiness')

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def emergency_count_view(request):
    """Submit emergency counting job with line coordinates."""
    start_time = time.time()
    emergency_lines_str = request.data.get("emergency_lines")
    if not emergency_lines_str:
        return Response({
            'status': 'failed',
            'job_type': 'emergency-count',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': "'emergency_lines' is required", 'code': 'MISSING_EMERGENCY_LINES'}
        }, status=status.HTTP_400_BAD_REQUEST)
    try:
        emergency_lines = json.loads(emergency_lines_str)
    except json.JSONDecodeError:
        return Response({
            'status': 'failed',
            'job_type': 'emergency-count',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': "Invalid JSON in 'emergency_lines'", 'code': 'INVALID_JSON'}
        }, status=status.HTTP_400_BAD_REQUEST)

    extra_data = {
        'emergency_lines': emergency_lines,
        'video_width': request.data.get('video_width'),
        'video_height': request.data.get('video_height'),
    }
    return _create_and_dispatch_job(request, 'emergency-count', extra_data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def lobby_detection_view(request):
    """Submit lobby/crowd detection job with zone definitions."""
    start_time = time.time()
    lobby_zones_str = request.data.get("lobby_zones")
    if not lobby_zones_str:
        return Response({
            'status': 'failed',
            'job_type': 'lobby-detection',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': "'lobby_zones' is required", 'code': 'MISSING_LOBBY_ZONES'}
        }, status=status.HTTP_400_BAD_REQUEST)
    try:
        lobby_zones = json.loads(lobby_zones_str)
    except json.JSONDecodeError:
        return Response({
            'status': 'failed',
            'job_type': 'lobby-detection',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': "Invalid JSON in 'lobby_zones'", 'code': 'INVALID_JSON'}
        }, status=status.HTTP_400_BAD_REQUEST)

    extra_data = {
        'lobby_zones': lobby_zones,
        'video_width': request.data.get('video_width'),
        'video_height': request.data.get('video_height'),
    }
    return _create_and_dispatch_job(request, 'lobby-detection', extra_data)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def food_waste_estimation_view(request):
    """
    Submit food waste estimation job for an image.

    Args:
        request: HTTP request with image file

    Returns:
        Response with standardized format:
        {
            'status': 'pending' | 'failed',
            'job_type': 'food-waste-estimation',
            'output_image': None,
            'output_video': None,
            'data': {'job_id': int, ...},
            'meta': {'timestamp': str, 'request_time': float},
            'error': dict | None
        }
    """
    start_time = time.time()
    image_file = request.FILES.get('image')
    if not image_file:
        return Response({
            'status': 'failed',
            'job_type': 'food-waste-estimation',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': 'No image file provided', 'code': 'MISSING_IMAGE'}
        }, status=status.HTTP_400_BAD_REQUEST)

    ext = os.path.splitext(image_file.name)[1].lower()
    if ext not in {'.jpg', '.jpeg', '.png'}:
        return Response({
            'status': 'failed',
            'job_type': 'food-waste-estimation',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': f"Invalid file type: {ext}. Allowed: .jpg, .jpeg, .png", 'code': 'INVALID_FILE_TYPE'}
        }, status=status.HTTP_400_BAD_REQUEST)
    if image_file.size > MAX_FILE_SIZE:
        return Response({
            'status': 'failed',
            'job_type': 'food-waste-estimation',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': f"File size {image_file.size / (1024*1024):.2f}MB exceeds 500MB limit", 'code': 'FILE_TOO_LARGE'}
        }, status=status.HTTP_400_BAD_REQUEST)

    return _create_and_dispatch_job(request, 'food-waste-estimation')

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def pothole_detection_image_view(request):
    """
    Submit pothole detection job for an image.

    Args:
        request: HTTP request with image file

    Returns:
        Response with standardized format:
        {
            'status': 'pending' | 'failed',
            'job_type': 'pothole-detection',
            'output_image': None,
            'output_video': None,
            'data': {'job_id': int, ...},
            'meta': {'timestamp': str, 'request_time': float},
            'error': dict | None
        }
    """
    start_time = time.time()
    image_file = request.FILES.get('image')
    if not image_file:
        return Response({
            'status': 'failed',
            'job_type': 'pothole-detection',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': 'No image file provided', 'code': 'MISSING_IMAGE'}
        }, status=status.HTTP_400_BAD_REQUEST)

    ext = os.path.splitext(image_file.name)[1].lower()
    if ext not in {'.jpg', '.jpeg', '.png'}:
        return Response({
            'status': 'failed',
            'job_type': 'pothole-detection',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': f"Invalid file type: {ext}. Allowed: .jpg, .jpeg, .png", 'code': 'INVALID_FILE_TYPE'}
        }, status=status.HTTP_400_BAD_REQUEST)
    if image_file.size > MAX_FILE_SIZE:
        return Response({
            'status': 'failed',
            'job_type': 'pothole-detection',
            'output_image': None,
            'output_video': None,
            'data': {},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': {'message': f"File size {image_file.size / (1024*1024):.2f}MB exceeds 500MB limit", 'code': 'FILE_TOO_LARGE'}
        }, status=status.HTTP_400_BAD_REQUEST)

    return _create_and_dispatch_job(request, 'pothole-detection')

"""
=====================================
Standard ViewSet and User Info
=====================================
ViewSet for managing VideoJob objects and retrieving user information.
Added error handling for DoesNotExist to use standardized error format (Issue #10).
"""

class VideoJobViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing VideoJob objects with task termination and cleanup.
    """
    queryset = VideoJob.objects.all()
    serializer_class = VideoJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Return jobs for the authenticated user."""
        return VideoJob.objects.filter(user=self.request.user).order_by('-created_at')

    def destroy(self, request, *args, **kwargs):
        """
        Delete a job, terminate its Celery task, and clean up resources.

        Args:
            request: HTTP request object

        Returns:
            Response with standardized format
        """
        start_time = time.time()
        try:
            job = self.get_object()
        except VideoJob.DoesNotExist:
            return Response({
                'status': 'failed',
                'job_type': 'job_deletion',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {'message': 'Job not found', 'code': 'JOB_NOT_FOUND'}
            }, status=status.HTTP_404_NOT_FOUND)

        logger.info(f"Deleting job {job.id} (status: {job.status}, task_id: {job.task_id})")

        if job.task_id and job.status in ['pending', 'processing']:
            try:
                from celery import current_app
                current_app.control.revoke(job.task_id, terminate=True, signal='SIGKILL')
                logger.info(f"Revoked task {job.task_id} for job {job.id}")
            except Exception as e:
                logger.warning(f"Failed to revoke task {job.task_id}: {e}")

        temp_files = [
            f"/tmp/output_{job.id}.mp4",
            f"/tmp/output_{job.id}.jpg",
            f"/tmp/output_{job.id}.png",
            f"/tmp/yt_thumb_{job.id}.jpg"
        ]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    logger.info(f"Cleaned up: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete {temp_file}: {e}")

        response = super().destroy(request, *args, **kwargs)
        logger.info(f"Deleted job {job.id}")
        return Response({
            'status': 'completed',
            'job_type': 'job_deletion',
            'output_image': None,
            'output_video': None,
            'data': {'job_id': job.id},
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'error': None
        }, status=status.HTTP_204_NO_CONTENT)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user_view(request):
    """
    Retrieve current user's info.

    Args:
        request: HTTP request object

    Returns:
        Response with standardized format:
        {
            'status': 'completed',
            'job_type': 'user_info',
            'output_image': None,
            'output_video': None,
            'data': {'id': int, 'username': str, 'email': str},
            'meta': {'timestamp': str, 'request_time': float},
            'error': None
        }
    """
    start_time = time.time()
    user = request.user
    return Response({
        'status': 'completed',
        'job_type': 'user_info',
        'output_image': None,
        'output_video': None,
        'data': {
            'id': user.id,
            'username': user.username,
            'email': user.email
        },
        'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
        'error': None
    }, status=status.HTTP_200_OK)