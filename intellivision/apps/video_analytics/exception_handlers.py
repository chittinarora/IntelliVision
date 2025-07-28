"""
=====================================
Exception Handling Utilities
=====================================
Provides standardized exception handling for video analytics operations.
Replaces broad exception handling with specific error types and proper cleanup.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable
from functools import wraps
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response

logger = logging.getLogger(__name__)

# =====================================
# Exception Types
# =====================================

class VideoAnalyticsError(Exception):
    """Base exception for video analytics errors."""
    def __init__(self, message: str, error_code: str = "ANALYTICS_ERROR", details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class FileValidationError(VideoAnalyticsError):
    """Raised when file validation fails."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "FILE_VALIDATION_ERROR", details)

class GPUError(VideoAnalyticsError):
    """Raised when GPU operations fail."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "GPU_ERROR", details)

class ModelLoadingError(VideoAnalyticsError):
    """Raised when ML model loading fails."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "MODEL_LOADING_ERROR", details)

class ProcessingError(VideoAnalyticsError):
    """Raised when video/image processing fails."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "PROCESSING_ERROR", details)

class ResourceExhaustionError(VideoAnalyticsError):
    """Raised when system resources are exhausted."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "RESOURCE_EXHAUSTION_ERROR", details)

# =====================================
# Exception Handler Decorator
# =====================================

def handle_analytics_exceptions(func: Callable) -> Callable:
    """
    Decorator to handle analytics exceptions with proper cleanup.

    Args:
        func: Function to wrap with exception handling

    Returns:
        Wrapped function with exception handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except VideoAnalyticsError as e:
            # Log specific analytics error
            logger.error(f"Analytics error in {func.__name__}: {e.error_code} - {e.message}",
                        extra={'error_code': e.error_code, 'details': e.details})

            # Cleanup GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU memory cleaned after error")
            except Exception as cleanup_error:
                logger.warning(f"GPU cleanup failed: {cleanup_error}")

            # Return standardized error response
            return create_error_response(
                message=e.message,
                error_code=e.error_code,
                job_type=kwargs.get('job_type', 'unknown'),
                status_code=status.HTTP_400_BAD_REQUEST,
                details=e.details
            )
        except (ValueError, TypeError) as e:
            # Handle validation errors
            logger.error(f"Validation error in {func.__name__}: {str(e)}")
            return create_error_response(
                message=f"Invalid input: {str(e)}",
                error_code="VALIDATION_ERROR",
                job_type=kwargs.get('job_type', 'unknown'),
                status_code=status.HTTP_400_BAD_REQUEST
            )
        except (OSError, IOError) as e:
            # Handle file system errors
            logger.error(f"File system error in {func.__name__}: {str(e)}")
            return create_error_response(
                message=f"File operation failed: {str(e)}",
                error_code="FILE_SYSTEM_ERROR",
                job_type=kwargs.get('job_type', 'unknown'),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except MemoryError as e:
            # Handle memory errors
            logger.error(f"Memory error in {func.__name__}: {str(e)}")
            return create_error_response(
                message="System memory exhausted. Please try with a smaller file or contact support.",
                error_code="MEMORY_ERROR",
                job_type=kwargs.get('job_type', 'unknown'),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)

            # Cleanup GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            return create_error_response(
                message="An unexpected error occurred. Please try again or contact support.",
                error_code="UNEXPECTED_ERROR",
                job_type=kwargs.get('job_type', 'unknown'),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    return wrapper

# =====================================
# Response Utilities
# =====================================

def create_error_response(
    message: str,
    error_code: str,
    job_type: str,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[Dict[str, Any]] = None
) -> Response:
    """
    Create a standardized error response.

    Args:
        message: Error message
        error_code: Error code for frontend handling
        job_type: Type of job that failed
        status_code: HTTP status code
        details: Additional error details

    Returns:
        Standardized error response
    """
    error_data = {
        'message': message,
        'code': error_code,
        'details': details or {}
    }

    response_data = {
        'status': 'failed',
        'job_type': job_type,
        'output_image': None,
        'output_video': None,
        'data': {},
        'meta': {
            'timestamp': None,  # Will be set by caller
            'request_time': 0,  # Will be set by caller
        },
        'error': error_data
    }

    return Response(response_data, status=status_code)

def create_success_response(
    data: Dict[str, Any],
    job_type: str,
    output_image: Optional[str] = None,
    output_video: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None
) -> Response:
    """
    Create a standardized success response.

    Args:
        data: Response data
        job_type: Type of job
        output_image: Path to output image
        output_video: Path to output video
        meta: Additional metadata

    Returns:
        Standardized success response
    """
    response_data = {
        'status': 'completed',
        'job_type': job_type,
        'output_image': output_image,
        'output_video': output_video,
        'data': data,
        'meta': meta or {},
        'error': None
    }

    return Response(response_data, status=status.HTTP_200_OK)

# =====================================
# GPU Utilities
# =====================================

def check_gpu_memory(required_gb: float = 2.0) -> bool:
    """
    Check if GPU has enough memory for processing.

    Args:
        required_gb: Required GPU memory in GB

    Returns:
        True if enough memory is available
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return True  # CPU processing is always available

        gpu_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        available_memory_gb = total_memory_gb - gpu_memory_gb

        if available_memory_gb < required_gb:
            raise ResourceExhaustionError(
                f"Insufficient GPU memory. Available: {available_memory_gb:.1f}GB, Required: {required_gb}GB",
                {
                    'available_gb': round(available_memory_gb, 1),
                    'required_gb': required_gb,
                    'total_gb': round(total_memory_gb, 1),
                    'used_gb': round(gpu_memory_gb, 1)
                }
            )

        return True
    except Exception as e:
        if isinstance(e, ResourceExhaustionError):
            raise
        logger.warning(f"GPU memory check failed: {e}")
        return True  # Assume OK if check fails

def cleanup_gpu_memory():
    """Clean up GPU memory after processing."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleaned")
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")

# =====================================
# Validation Utilities
# =====================================

def validate_file_path(file_path: str) -> None:
    """
    Validate file path and existence.

    Args:
        file_path: Path to validate

    Raises:
        FileValidationError: If file is invalid
    """
    import os
    from django.core.files.storage import default_storage

    if not file_path:
        raise FileValidationError("File path is required")

    if not default_storage.exists(file_path):
        raise FileValidationError(f"File not found: {file_path}")

    if not os.path.exists(file_path):
        raise FileValidationError(f"File does not exist on disk: {file_path}")

def validate_file_size(file_path: str, max_size_mb: int = 500) -> None:
    """
    Validate file size.

    Args:
        file_path: Path to file
        max_size_mb: Maximum size in MB

    Raises:
        FileValidationError: If file is too large
    """
    from django.core.files.storage import default_storage

    size_bytes = default_storage.size(file_path)
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        raise FileValidationError(
            f"File size {size_mb:.1f}MB exceeds limit of {max_size_mb}MB",
            {
                'file_size_mb': round(size_mb, 1),
                'max_size_mb': max_size_mb
            }
        )
