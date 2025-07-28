# /apps/video_analytics/utils.py

"""
=====================================
Utility Functions
=====================================
Common utilities for video analytics app.
"""

import time
from typing import Dict, Any
from django.utils import timezone
from rest_framework.response import Response
from rest_framework import status


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
