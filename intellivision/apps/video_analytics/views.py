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
from io import BytesIO

import cv2
import yt_dlp
import requests
from PIL import Image
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from rest_framework import permissions, status, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
import mimetypes
from django.utils import timezone
from rest_framework.exceptions import PermissionDenied
from django.db import transaction

from .models import VideoJob
from .serializers import VideoJobSerializer
from .tasks import process_video_job, ensure_api_media_url, extract_youtube_frame
from .utils import create_standardized_response, create_error_response, validate_file_upload_for_job_type
from .rate_limiting import check_resource_availability, acquire_job_slot, release_job_slot

logger = logging.getLogger(__name__)

# Valid file extensions and size limit (now handled by job-type-specific validation)
VALID_EXTENSIONS = {'.mp4', '.jpg', '.jpeg', '.png'}  # Legacy - job-specific validation in utils.py
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

"""
=====================================
Memory Management Utilities
=====================================
Utilities for monitoring memory usage and preventing OOM crashes.
"""

def check_memory_usage():
    """
    Check current memory usage to prevent OOM crashes.

    Returns:
        Dict with memory info and whether it's safe to proceed
    """
    import psutil
    import gc

    try:
        # Force garbage collection to get accurate memory reading
        gc.collect()

        # Get current process memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        # Get system memory info
        system_memory = psutil.virtual_memory()

        # Calculate if it's safe to proceed
        # Threshold: Don't start new operations if using > 70% system memory
        # or > 85% of configured memory limit (24GB for this system)
        MAX_SYSTEM_MEMORY_GB = 24.0  # From settings.py
        current_memory_gb = memory_info.rss / (1024**3)
        memory_safe = (
            system_memory.percent < 70.0 and  # System memory usage < 70%
            current_memory_gb < (MAX_SYSTEM_MEMORY_GB * 0.85)  # Process memory < 85% of limit
        )

        return {
            'memory_safe': memory_safe,
            'current_memory_gb': current_memory_gb,
            'memory_percent': memory_percent,
            'system_memory_percent': system_memory.percent,
            'available_memory_gb': system_memory.available / (1024**3)
        }
    except Exception as e:
        logger.warning(f"Memory check failed: {e}")
        # If memory check fails, assume it's safe but log the issue
        return {
            'memory_safe': True,
            'current_memory_gb': 0,
            'memory_percent': 0,
            'system_memory_percent': 0,
            'available_memory_gb': 0,
            'error': str(e)
        }

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



    # Check resource availability
    if not check_resource_availability():
        return create_error_response(
            "System is currently at capacity. Please try again later.",
            'RESOURCE_UNAVAILABLE',
            job_type,
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    file_obj = request.FILES.get('video') or request.FILES.get('image')
    youtube_url = request.data.get('youtube_url')

    if not file_obj and not youtube_url:
        return create_error_response(
            'Provide either a file or YouTube URL',
            'MISSING_INPUT',
            job_type
        )

    if file_obj and youtube_url:
        return create_error_response(
            'Provide either a file or YouTube URL, not both',
            'INVALID_INPUT_COMBINATION',
            job_type
        )

    job_data = {'user': user, 'status': 'pending', 'job_type': job_type}
    if extra_data:
        job_data.update(extra_data)

    input_file_content = None
    temp_video_path = None

    try:
        if youtube_url:
            logger.info(f"🎬 Validating YouTube URL: {youtube_url} for {job_type} (user: {user.username})")

            # ======================================
            # OPTIMIZATION: Just validate URL, don't download in web worker
            # ======================================
            try:
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    # Only extract info to validate URL - NO DOWNLOAD
                    info = ydl.extract_info(youtube_url, download=False)
                    if not info:
                        return create_error_response(
                            'Invalid YouTube URL or video not available',
                            'INVALID_YOUTUBE_URL',
                            job_type
                        )

                    # Check estimated file size before dispatching to Celery
                    filesize = info.get('filesize') or info.get('filesize_approx', 0)
                    if filesize and filesize > MAX_FILE_SIZE:
                        return create_error_response(
                            f'Video size {filesize / (1024*1024):.1f}MB exceeds {MAX_FILE_SIZE / (1024*1024):.0f}MB limit',
                            'FILE_TOO_LARGE',
                            job_type
                        )

                    logger.info(f"✅ YouTube URL validated: {info.get('title', 'Unknown')} ({filesize / (1024*1024):.1f}MB estimated)")

            except yt_dlp.utils.DownloadError as e:
                logger.error(f"YouTube URL validation failed: {e}")
                return create_error_response(
                    f'Failed to validate YouTube URL: {str(e)}',
                    'YOUTUBE_VALIDATION_ERROR',
                    job_type
                )
            except Exception as e:
                logger.error(f"Unexpected error during YouTube URL validation: {e}")
                return create_error_response(
                    f'Unexpected error during YouTube URL validation: {str(e)}',
                    'YOUTUBE_VALIDATION_ERROR',
                    job_type
                )

            # Set job data for YouTube processing in Celery
            job_data['youtube_url'] = youtube_url

            # No file content needed - Celery will download
            input_file_content = None
            ext = '.mp4'  # Assume MP4 for YouTube videos

        else:
            # Use the new job-type-specific validation utility
            is_valid, error_msg = validate_file_upload_for_job_type(file_obj, job_type, MAX_FILE_SIZE)
            if not is_valid:
                # Determine error code based on error message content
                if 'Invalid file type' in error_msg:
                    error_code = 'INVALID_FILE_TYPE'
                elif 'File size' in error_msg and 'exceeds' in error_msg:
                    error_code = 'FILE_TOO_LARGE'
                elif 'No file provided' in error_msg:
                    error_code = 'MISSING_FILE'
                else:
                    error_code = 'VALIDATION_ERROR'

                return create_error_response(
                    error_msg,
                    error_code,
                    job_type
                )

            # Extract file extension from uploaded file
            original_name = file_obj.name
            ext = os.path.splitext(original_name)[1] if original_name else ''

            # Set appropriate default extension based on job type
            if not ext:
                if job_type in ['pothole-detection', 'food-waste-estimation']:
                    ext = '.jpg'  # Default to image for image-only jobs
                else:
                    ext = '.mp4'  # Default to video for video jobs

            file_name = f"{job_type}_{uuid4().hex}{ext}"
            input_file_content = ContentFile(file_obj.read(), name=file_name)
            logger.info(f"📁 Uploaded: {file_name} ({file_obj.size / (1024*1024):.2f}MB)")

        # Acquire job slot
        if not acquire_job_slot():
            return create_error_response(
                "System is currently at capacity. Please try again later.",
                'RESOURCE_UNAVAILABLE',
                job_type,
                status.HTTP_503_SERVICE_UNAVAILABLE
            )

        try:
            with transaction.atomic():
                # For YouTube URLs, don't set input_video - Celery will handle download
                if input_file_content:
                    job_data['input_video'] = input_file_content

                job = VideoJob.objects.create(**job_data)
                task = process_video_job.delay(job.id)
                job.task_id = task.id
                job.save()

                if youtube_url:
                    logger.info(f"🚀 Dispatched YouTube job {job.id} ({job_type}) with task ID {task.id} - video will be downloaded by Celery")
                else:
                    logger.info(f"🚀 Dispatched file job {job.id} ({job_type}) with task ID {task.id}")
        except Exception as e:
            release_job_slot()  # Release slot on error
            logger.error(f"Job creation failed: {e}")
            # Transaction will automatically rollback
            raise e

        serializer = VideoJobSerializer(job)
        return create_standardized_response(
            status_type='pending',
            job_type=job_type,
            data=serializer.data
        )

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"YouTube download failed for {job_type}: {e}", exc_info=True)
        # Custom error for authentication-required videos
        if 'requires authentication' in str(e).lower() or 'provide a valid cookies.txt' in str(e).lower():
            return create_error_response(
                "This YouTube video requires authentication. Please provide a valid cookies.txt file exported from your browser (see documentation).",
                'YOUTUBE_AUTH_REQUIRED',
                job_type
            )
        return create_error_response(
            f"Failed to download YouTube video: {str(e)}",
            'DOWNLOAD_ERROR',
            job_type
        )
    except Exception as e:
        logger.error(f"Job creation failed for {job_type}: {e}", exc_info=True)
        return create_error_response(
            f"Unexpected error: {str(e)}",
            'SERVER_ERROR',
            job_type,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        # Clean up temporary files
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
                logger.info(f"Cleaned up temporary video file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary video file {temp_video_path}: {e}")

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
    Create a Celery task to extract first frame from a YouTube URL.

    This returns immediately with a job ID, and the frame extraction
    happens asynchronously in a Celery worker to prevent OOM crashes.

    Args:
        request: HTTP request with youtube_url

    Returns:
        Response with job ID for polling:
        {
            'status': 'pending',
            'job_type': 'youtube_frame_extraction',
            'data': {'job_id': int},
            'meta': {'timestamp': str, 'request_time': float},
            'polling_url': str
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

    try:
        # ======================================
        # MEMORY SAFETY CHECK
        # ======================================
        memory_status = check_memory_usage()
        logger.info(f"💾 Memory status: {memory_status['current_memory_gb']:.2f}GB used, "
                   f"{memory_status['system_memory_percent']:.1f}% system memory")

        # Check resource availability
        if not check_resource_availability():
            return Response({
                'status': 'failed',
                'job_type': 'youtube_frame_extraction',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {
                    'message': 'System is currently at capacity. Please try again later.',
                    'code': 'RESOURCE_UNAVAILABLE'
                }
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        logger.info(f"🎬 Validating YouTube URL for frame extraction: {youtube_url}")

        # ======================================
        # VALIDATE URL (but don't download)
        # ======================================
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                # Only extract info to validate URL - NO DOWNLOAD
                info = ydl.extract_info(youtube_url, download=False)
                if not info:
                    return Response({
                        'status': 'failed',
                        'job_type': 'youtube_frame_extraction',
                        'output_image': None,
                        'output_video': None,
                        'data': {},
                        'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                        'error': {'message': 'Invalid YouTube URL or video not available', 'code': 'INVALID_YOUTUBE_URL'}
                    }, status=status.HTTP_400_BAD_REQUEST)

                logger.info(f"✅ YouTube URL validated: {info.get('title', 'Unknown')}")

        except yt_dlp.utils.DownloadError as e:
            logger.error(f"YouTube URL validation failed: {e}")
            return Response({
                'status': 'failed',
                'job_type': 'youtube_frame_extraction',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {'message': f"Failed to validate YouTube URL: {str(e)}", 'code': 'YOUTUBE_VALIDATION_ERROR'}
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Unexpected error during YouTube URL validation: {e}")
            return Response({
                'status': 'failed',
                'job_type': 'youtube_frame_extraction',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {'message': f"Unexpected error during YouTube URL validation: {str(e)}", 'code': 'YOUTUBE_VALIDATION_ERROR'}
            }, status=status.HTTP_400_BAD_REQUEST)

        # ======================================
        # CREATE CELERY JOB FOR ASYNC PROCESSING
        # ======================================
        if not acquire_job_slot():
            return Response({
                'status': 'failed',
                'job_type': 'youtube_frame_extraction',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {
                    'message': 'System is currently at capacity. Please try again later.',
                    'code': 'RESOURCE_UNAVAILABLE'
                }
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        try:
            with transaction.atomic():
                # Create job for YouTube frame extraction
                job_data = {
                    'user': request.user,
                    'status': 'pending',
                    'job_type': 'youtube_frame_extraction',
                    'youtube_url': youtube_url
                    # No input_video needed - Celery will download
                }

                job = VideoJob.objects.create(**job_data)
                task = extract_youtube_frame.delay(job.id)  # New Celery task
                job.task_id = task.id
                job.save()

                logger.info(f"🚀 Dispatched YouTube frame extraction job {job.id} with task ID {task.id}")

        except Exception as e:
            release_job_slot()  # Release slot on error
            logger.error(f"YouTube frame job creation failed: {e}")
            return Response({
                'status': 'failed',
                'job_type': 'youtube_frame_extraction',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {'message': f"Failed to create job: {str(e)}", 'code': 'JOB_CREATION_ERROR'}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Return job info for polling
        return Response({
            'status': 'pending',
            'job_type': 'youtube_frame_extraction',
            'output_image': None,
            'output_video': None,
            'data': {
                'job_id': job.id,
                'task_id': task.id,
                'message': 'Frame extraction started. Use polling_url to check status.'
            },
            'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
            'polling_url': f'/api/jobs/{job.id}/'
        }, status=status.HTTP_202_ACCEPTED)

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

        # Validate emergency lines structure
        if not isinstance(emergency_lines, list) or len(emergency_lines) != 2:
            return Response({
                'status': 'failed',
                'job_type': 'emergency-count',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {'message': "emergency_lines must be an array with exactly 2 lines", 'code': 'INVALID_EMERGENCY_LINES'}
            }, status=status.HTTP_400_BAD_REQUEST)

        # Validate each line has required fields
        for i, line in enumerate(emergency_lines):
            required_fields = ['start_x', 'start_y', 'end_x', 'end_y']
            if not all(field in line for field in required_fields):
                return Response({
                    'status': 'failed',
                    'job_type': 'emergency-count',
                    'output_image': None,
                    'output_video': None,
                    'data': {},
                    'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                    'error': {'message': f"Line {i+1} missing required fields: {required_fields}", 'code': 'INVALID_EMERGENCY_LINES'}
                }, status=status.HTTP_400_BAD_REQUEST)
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

        # Validate lobby zones structure
        if not isinstance(lobby_zones, list) or len(lobby_zones) == 0:
            return Response({
                'status': 'failed',
                'job_type': 'lobby-detection',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                'error': {'message': "lobby_zones must be a non-empty array", 'code': 'INVALID_LOBBY_ZONES'}
            }, status=status.HTTP_400_BAD_REQUEST)

        # Validate each zone has required fields
        for i, zone in enumerate(lobby_zones):
            if not isinstance(zone, dict):
                return Response({
                    'status': 'failed',
                    'job_type': 'lobby-detection',
                    'output_image': None,
                    'output_video': None,
                    'data': {},
                    'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                    'error': {'message': f"Zone {i+1} must be an object", 'code': 'INVALID_LOBBY_ZONES'}
                }, status=status.HTTP_400_BAD_REQUEST)

            required_fields = ['name', 'points']
            if not all(field in zone for field in required_fields):
                return Response({
                    'status': 'failed',
                    'job_type': 'lobby-detection',
                    'output_image': None,
                    'output_video': None,
                    'data': {},
                    'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                    'error': {'message': f"Zone {i+1} missing required fields: {required_fields}", 'code': 'INVALID_LOBBY_ZONES'}
                }, status=status.HTTP_400_BAD_REQUEST)

            if not isinstance(zone['points'], list) or len(zone['points']) < 3:
                return Response({
                    'status': 'failed',
                    'job_type': 'lobby-detection',
                    'output_image': None,
                    'output_video': None,
                    'data': {},
                    'meta': {'timestamp': timezone.now().isoformat(), 'request_time': time.time() - start_time},
                    'error': {'message': f"Zone {i+1} points must be an array with at least 3 points", 'code': 'INVALID_LOBBY_ZONES'}
                }, status=status.HTTP_400_BAD_REQUEST)
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

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def pothole_detection_video_view(request):
    """
    Submit pothole detection job for a video.

    Args:
        request: HTTP request with video file

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
    return _create_and_dispatch_job(request, 'pothole-detection')

"""
=====================================
Standard ViewSet and User Info
=====================================
ViewSet for managing VideoJob objects and retrieving user information.
Enhanced with better permissions, validation, and resource cleanup.
"""

class VideoJobViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing VideoJob objects with enhanced permissions and cleanup.
    """
    queryset = VideoJob.objects.all()
    serializer_class = VideoJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Return jobs for the authenticated user only."""
        return VideoJob.objects.filter(user=self.request.user).order_by('-created_at')

    def get_object(self):
        """Get object and ensure user ownership."""
        obj = super().get_object()

        # Additional permission check to ensure user owns the job
        if obj.user != self.request.user:
            raise PermissionDenied("You don't have permission to access this job.")

        return obj

    def destroy(self, request, *args, **kwargs):
        """
        Delete a job with enhanced validation, cleanup, and standardized responses.

        - Validates user ownership
        - Terminates running Celery tasks
        - Cleans up associated files
        - Releases resource slots
        - Returns standardized error format
        """
        start_time = time.time()
        job = None

        try:
            job = self.get_object()
            logger.info(f"🗑️ User {request.user.id} requesting deletion of job {job.id} (status: {job.status})")

            # Additional validation for active jobs
            if job.status == 'processing':
                logger.warning(f"Attempting to delete processing job {job.id}")
                # Allow deletion but warn user

            # Store job info for cleanup
            job_id = job.id
            job_type = job.job_type
            task_id = job.task_id
            input_file_path = job.input_video.path if job.input_video else None
            output_file_paths = []

            # Collect output file paths for cleanup
            if job.output_video:
                try:
                    output_file_paths.append(job.output_video.path)
                except (ValueError, AttributeError):
                    pass

            if job.output_image:
                try:
                    output_file_paths.append(job.output_image.path)
                except (ValueError, AttributeError):
                    pass

            # Terminate Celery task if running
            if task_id:
                try:
                    from celery import current_app
                    current_app.control.revoke(task_id, terminate=True)
                    logger.info(f"🛑 Terminated Celery task {task_id} for job {job_id}")
                except Exception as e:
                    logger.warning(f"Failed to revoke Celery task {task_id}: {e}")

            # Delete the job record
            job.delete()
            logger.info(f"✅ Successfully deleted job {job_id}")

            # Release job slot if applicable
            try:
                from .rate_limiting import release_job_slot
                release_job_slot()
                logger.info(f"Released job slot after deleting job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to release job slot: {e}")

            # Clean up files (in background to avoid blocking response)
            def cleanup_files():
                cleaned_files = []
                for file_path in [input_file_path] + output_file_paths:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            cleaned_files.append(file_path)
                        except OSError as e:
                            logger.warning(f"Failed to delete file {file_path}: {e}")

                if cleaned_files:
                    logger.info(f"🧹 Cleaned up {len(cleaned_files)} files for job {job_id}")

            # Run cleanup in background
            import threading
            cleanup_thread = threading.Thread(target=cleanup_files)
            cleanup_thread.daemon = True
            cleanup_thread.start()

            # Return standardized success response
            return Response({
                'status': 'success',
                'job_type': 'job_deletion',
                'data': {
                    'job_id': job_id,
                    'message': f'Job {job_id} deleted successfully'
                },
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'request_time': time.time() - start_time
                },
                'error': None
            }, status=status.HTTP_200_OK)

        except PermissionDenied as e:
            logger.warning(f"Permission denied for job deletion: {e}")
            return Response({
                'status': 'failed',
                'job_type': 'job_deletion',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'request_time': time.time() - start_time
                },
                'error': {
                    'message': 'You don\'t have permission to delete this job',
                    'code': 'PERMISSION_DENIED'
                }
            }, status=status.HTTP_403_FORBIDDEN)

        except VideoJob.DoesNotExist:
            logger.warning(f"Job not found for deletion by user {request.user.id}")
            return Response({
                'status': 'failed',
                'job_type': 'job_deletion',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'request_time': time.time() - start_time
                },
                'error': {
                    'message': 'Job not found or already deleted',
                    'code': 'JOB_NOT_FOUND'
                }
            }, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            logger.error(f"Unexpected error during job deletion: {e}", exc_info=True)
            return Response({
                'status': 'failed',
                'job_type': 'job_deletion',
                'output_image': None,
                'output_video': None,
                'data': {},
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'request_time': time.time() - start_time
                },
                'error': {
                    'message': 'Internal server error during job deletion',
                    'code': 'INTERNAL_ERROR'
                }
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def list(self, request, *args, **kwargs):
        """Enhanced list endpoint with better error handling."""
        try:
            queryset = self.get_queryset()
            serializer = self.get_serializer(queryset, many=True)

            logger.info(f"📋 User {request.user.id} retrieved {len(serializer.data)} jobs")
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error retrieving jobs for user {request.user.id}: {e}", exc_info=True)
            return Response({
                'error': {
                    'message': 'Failed to retrieve jobs',
                    'code': 'JOBS_RETRIEVAL_ERROR'
                }
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def retrieve(self, request, *args, **kwargs):
        """Enhanced retrieve endpoint with ownership validation."""
        try:
            instance = self.get_object()
            serializer = self.get_serializer(instance)

            logger.info(f"👁️ User {request.user.id} viewed job {instance.id}")
            return Response(serializer.data)

        except PermissionDenied as e:
            return Response({
                'error': {
                    'message': 'You don\'t have permission to view this job',
                    'code': 'PERMISSION_DENIED'
                }
            }, status=status.HTTP_403_FORBIDDEN)

        except VideoJob.DoesNotExist:
            return Response({
                'error': {
                    'message': 'Job not found',
                    'code': 'JOB_NOT_FOUND'
                }
            }, status=status.HTTP_404_NOT_FOUND)

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
