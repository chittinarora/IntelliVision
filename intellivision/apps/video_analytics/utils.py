# /apps/video_analytics/utils.py

"""
=====================================
Utility Functions
=====================================
Common utilities for video analytics app.
"""

import time
import torch
import logging
from typing import Dict, Any
from django.utils import timezone
from rest_framework.response import Response
from rest_framework import status
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def get_optimal_device():
    """
    Get optimal device for ML models based on available hardware.
    Optimized for Tesla P100 GPU with fallback to CPU.
    """
    if torch.cuda.is_available():
        # Check for Tesla P100 specifically
        gpu_name = torch.cuda.get_device_name(0).lower()
        if 'tesla p100' in gpu_name or 'p100' in gpu_name:
            logger.info(f"✅ Tesla P100 GPU detected: {torch.cuda.get_device_name(0)}")
            return "cuda:0"
        else:
            logger.info(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            return "cuda:0"
    elif torch.backends.mps.is_available():
        logger.info("✅ MPS (Apple Silicon) detected")
        return "mps"
    else:
        logger.warning("⚠️ No GPU detected, using CPU")
        return "cpu"

def load_yolo_model(model_path: str, device: str = None):
    """
    Load YOLO model with optimal device selection.

    Args:
        model_path: Path to YOLO model file
        device: Device to load model on (auto-detected if None)

    Returns:
        Loaded YOLO model
    """
    if device is None:
        device = get_optimal_device()

    try:
        model = YOLO(model_path)
        if device == "cuda:0" or device == "cuda":
            model.to("cuda")
            # Optimize for Tesla P100
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                # Set memory fraction to prevent OOM
                torch.cuda.empty_cache()
        logger.info(f"✅ YOLO model loaded on {device}: {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model {model_path}: {e}")
        raise


def create_standardized_response(
    status_type: str,
    job_type: str,
    data: Dict[str, Any] = None,
    error: Dict[str, Any] = None,
    output_image: str = None,
    output_video: str = None,
    http_status: int = None
) -> Response:
    """
    Create a standardized API response format.

    Args:
        status_type: 'pending', 'completed', 'failed'
        job_type: Type of analytics job
        data: Response data dictionary
        error: Error information dictionary
        output_image: URL to output image
        output_video: URL to output video
        http_status: HTTP status code (auto-determined if None)

    Returns:
        Response with standardized format
    """
    start_time = getattr(create_standardized_response, '_start_time', time.time())

    response_data = {
        'status': status_type,
        'job_type': job_type,
        'output_image': output_image,
        'output_video': output_video,
        'data': data or {},
        'meta': {
            'timestamp': timezone.now().isoformat(),
            'request_time': time.time() - start_time
        },
        'error': error
    }

    # Auto-determine HTTP status if not provided
    if http_status is None:
        if status_type == 'failed':
            http_status = status.HTTP_400_BAD_REQUEST
        elif status_type == 'pending':
            http_status = status.HTTP_202_ACCEPTED
        else:
            http_status = status.HTTP_200_OK

    return Response(response_data, status=http_status)


def create_error_response(
    message: str,
    code: str,
    job_type: str,
    http_status: int = status.HTTP_400_BAD_REQUEST
) -> Response:
    """
    Create a standardized error response.

    Args:
        message: Error message
        code: Error code
        job_type: Type of job that failed
        http_status: HTTP status code

    Returns:
        Response with standardized error format
    """
    return create_standardized_response(
        status_type='failed',
        job_type=job_type,
        error={'message': message, 'code': code},
        http_status=http_status
    )


def validate_file_upload(file_obj, max_size: int = 500 * 1024 * 1024, valid_extensions: set = None) -> tuple[bool, str]:
    """
    Validate uploaded file.

    Args:
        file_obj: Uploaded file object
        max_size: Maximum file size in bytes
        valid_extensions: Set of valid file extensions

    Returns:
        Tuple of (is_valid, error_message)
    """
    if valid_extensions is None:
        valid_extensions = {'.mp4', '.jpg', '.jpeg', '.png'}

    if not file_obj:
        return False, "No file provided"

    if file_obj.size > max_size:
        return False, f"File size {file_obj.size / (1024*1024):.2f}MB exceeds {max_size / (1024*1024)}MB limit"

    import os
    ext = os.path.splitext(file_obj.name)[1].lower()
    if ext not in valid_extensions:
        return False, f"Invalid file type: {ext}. Allowed: {', '.join(valid_extensions)}"

    return True, ""


# Define job-type-specific file requirements
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.mpg', '.mpeg', '.3gp'}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

JOB_TYPE_FILE_REQUIREMENTS = {
    'pothole-detection': {
        'allowed_extensions': IMAGE_EXTENSIONS,
        'description': 'images (JPG, JPEG, PNG)'
    },
    'food-waste-estimation': {
        'allowed_extensions': IMAGE_EXTENSIONS,
        'description': 'images (JPG, JPEG, PNG)'
    },
    'room-readiness': {
        'allowed_extensions': ALL_EXTENSIONS,
        'description': 'videos (MP4, AVI, MOV, MKV, WebM, etc.) or images (JPG, JPEG, PNG)'
    },
    'wildlife-detection': {
        'allowed_extensions': ALL_EXTENSIONS,
        'description': 'videos (MP4, AVI, MOV, MKV, WebM, etc.) or images (JPG, JPEG, PNG)'
    },
    'people_count': {
        'allowed_extensions': ALL_EXTENSIONS,
        'description': 'videos (MP4, AVI, MOV, MKV, WebM, etc.) or images (JPG, JPEG, PNG)'
    },
    'emergency_count': {
        'allowed_extensions': VIDEO_EXTENSIONS,
        'description': 'videos (MP4, AVI, MOV, MKV, WebM, etc.)'
    },
    'car-count': {
        'allowed_extensions': ALL_EXTENSIONS,
        'description': 'videos (MP4, AVI, MOV, MKV, WebM, etc.) or images (JPG, JPEG, PNG)'
    },
    'parking-analysis': {
        'allowed_extensions': VIDEO_EXTENSIONS,
        'description': 'videos (MP4, AVI, MOV, MKV, WebM, etc.)'
    },
    'pest_monitoring': {
        'allowed_extensions': ALL_EXTENSIONS,
        'description': 'videos (MP4, AVI, MOV, MKV, WebM, etc.) or images (JPG, JPEG, PNG)'
    },
    'lobby_detection': {
        'allowed_extensions': VIDEO_EXTENSIONS,
        'description': 'videos (MP4, AVI, MOV, MKV, WebM, etc.)'
    },
    # Default for other job types (video-only)
    'default': {
        'allowed_extensions': VIDEO_EXTENSIONS,
        'description': 'videos (MP4, AVI, MOV, MKV, WebM, etc.)'
    }
}

def validate_file_upload_for_job_type(file_obj, job_type: str, max_size: int = 500 * 1024 * 1024) -> tuple[bool, str]:
    """
    Validate uploaded file based on job type requirements.

    Args:
        file_obj: Uploaded file object
        job_type: Type of analytics job
        max_size: Maximum file size in bytes

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_obj:
        return False, "No file provided"

    if file_obj.size > max_size:
        return False, f"File size {file_obj.size / (1024*1024):.2f}MB exceeds {max_size / (1024*1024)}MB limit"

    import os
    ext = os.path.splitext(file_obj.name)[1].lower()

    # Get requirements for this job type, fallback to default
    requirements = JOB_TYPE_FILE_REQUIREMENTS.get(job_type, JOB_TYPE_FILE_REQUIREMENTS['default'])
    allowed_extensions = requirements['allowed_extensions']
    description = requirements['description']

    if ext not in allowed_extensions:
        return False, f"Invalid file type for {job_type}. Only {description} are supported."

    return True, ""
