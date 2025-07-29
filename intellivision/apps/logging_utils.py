"""
Enhanced Logging Utilities for IntelliVision Backend

Provides structured logging with security, performance, and API tracking
capabilities. Includes formatters, decorators, and specialized loggers.
"""

import logging
import time
import functools
from typing import Dict, Any, Optional, Callable
from django.http import HttpRequest
from django.contrib.auth.models import User
import json


def get_client_ip(request: HttpRequest) -> str:
    """Extract client IP from request, handling proxies."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR', 'unknown')
    return ip


def get_user_info(request: HttpRequest) -> Dict[str, Any]:
    """Extract user information from request."""
    if hasattr(request, 'user') and request.user.is_authenticated:
        return {
            'user_id': request.user.id,
            'username': request.user.username,
            'is_staff': request.user.is_staff,
            'is_superuser': request.user.is_superuser
        }
    return {'user_id': None, 'username': 'anonymous'}


class SecurityLogger:
    """Specialized logger for security events."""

    def __init__(self):
        self.logger = logging.getLogger('security_logger')

    def log_login_attempt(self, request: HttpRequest, username: str, success: bool, method: str = 'password'):
        """Log authentication attempts."""
        extra = {
            'client_ip': get_client_ip(request),
            'user_id': username,
            'action': f'login_attempt_{method}',
            'success': success,
            'user_agent': request.META.get('HTTP_USER_AGENT', 'unknown')
        }

        if success:
            self.logger.info(f"Successful login for user {username}", extra=extra)
        else:
            self.logger.warning(f"Failed login attempt for user {username}", extra=extra)

    def log_face_auth(self, request: HttpRequest, username: str, success: bool, confidence: float = None):
        """Log face authentication events."""
        extra = {
            'client_ip': get_client_ip(request),
            'user_id': username,
            'action': 'face_authentication',
            'success': success,
            'confidence': confidence
        }

        message = f"Face authentication {'succeeded' if success else 'failed'} for {username}"
        if confidence:
            message += f" (confidence: {confidence:.2f})"

        if success:
            self.logger.info(message, extra=extra)
        else:
            self.logger.warning(message, extra=extra)

    def log_permission_denied(self, request: HttpRequest, resource: str, action: str):
        """Log permission denied events."""
        user_info = get_user_info(request)
        extra = {
            'client_ip': get_client_ip(request),
            'user_id': user_info['user_id'],
            'action': f'permission_denied_{action}',
            'resource': resource
        }

        self.logger.warning(
            f"Permission denied for user {user_info['username']} on {resource}",
            extra=extra
        )

    def log_suspicious_activity(self, request: HttpRequest, activity_type: str, details: Dict[str, Any]):
        """Log suspicious activities."""
        user_info = get_user_info(request)
        extra = {
            'client_ip': get_client_ip(request),
            'user_id': user_info['user_id'],
            'action': f'suspicious_{activity_type}',
            'details': json.dumps(details)
        }

        self.logger.error(f"Suspicious activity detected: {activity_type}", extra=extra)


class PerformanceLogger:
    """Specialized logger for performance monitoring."""

    def __init__(self):
        self.logger = logging.getLogger('performance_logger')

    def log_operation(self, operation_name: str, duration_ms: float,
                     metadata: Dict[str, Any] = None, threshold_ms: float = 1000):
        """Log operation performance."""
        extra = {
            'operation': operation_name,
            'duration': round(duration_ms, 2),
            'metadata': json.dumps(metadata or {})
        }

        level = logging.WARNING if duration_ms > threshold_ms else logging.INFO
        message = f"Operation {operation_name} completed in {duration_ms:.2f}ms"

        self.logger.log(level, message, extra=extra)

    def log_database_query(self, query_type: str, duration_ms: float,
                          table: str = None, row_count: int = None):
        """Log database query performance."""
        metadata = {}
        if table:
            metadata['table'] = table
        if row_count is not None:
            metadata['row_count'] = row_count

        self.log_operation(f"db_query_{query_type}", duration_ms, metadata)

    def log_api_call(self, endpoint: str, method: str, duration_ms: float,
                    status_code: int, response_size: int = None):
        """Log API call performance."""
        metadata = {
            'method': method,
            'status_code': status_code,
            'endpoint': endpoint
        }
        if response_size:
            metadata['response_size_bytes'] = response_size

        self.log_operation(f"api_{method.lower()}", duration_ms, metadata)


class APILogger:
    """Specialized logger for API requests and responses."""

    def __init__(self):
        self.logger = logging.getLogger('api_logger')

    def log_request(self, request: HttpRequest, view_name: str = None):
        """Log incoming API request."""
        user_info = get_user_info(request)

        log_data = {
            'timestamp': time.time(),
            'method': request.method,
            'path': request.path,
            'view_name': view_name,
            'client_ip': get_client_ip(request),
            'user_id': user_info['user_id'],
            'username': user_info['username'],
            'user_agent': request.META.get('HTTP_USER_AGENT', 'unknown'),
            'content_type': request.content_type,
            'query_params': dict(request.GET),
            'content_length': request.META.get('CONTENT_LENGTH', 0)
        }

        self.logger.info(f"API Request: {request.method} {request.path}", extra=log_data)

    def log_response(self, request: HttpRequest, response, duration_ms: float, view_name: str = None):
        """Log API response."""
        user_info = get_user_info(request)

        log_data = {
            'timestamp': time.time(),
            'method': request.method,
            'path': request.path,
            'view_name': view_name,
            'status_code': response.status_code,
            'duration_ms': round(duration_ms, 2),
            'client_ip': get_client_ip(request),
            'user_id': user_info['user_id'],
            'response_size': len(response.content) if hasattr(response, 'content') else 0
        }

        level = logging.ERROR if response.status_code >= 500 else (
            logging.WARNING if response.status_code >= 400 else logging.INFO
        )

        self.logger.log(
            level,
            f"API Response: {request.method} {request.path} - {response.status_code} ({duration_ms:.2f}ms)",
            extra=log_data
        )


# Singleton instances
security_logger = SecurityLogger()
performance_logger = PerformanceLogger()
api_logger = APILogger()


def log_performance(operation_name: str = None, threshold_ms: float = 1000):
    """Decorator to log function execution time."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                performance_logger.log_operation(op_name, duration_ms, threshold_ms=threshold_ms)

                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                op_name = operation_name or f"{func.__module__}.{func.__name__}"

                # Log the failed operation
                performance_logger.logger.error(
                    f"Operation {op_name} failed after {duration_ms:.2f}ms: {str(e)}",
                    extra={'operation': op_name, 'duration': duration_ms}
                )
                raise
        return wrapper
    return decorator


def log_celery_task(task_name: str = None):
    """Decorator to log Celery task execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger('celery.task')

            name = task_name or func.__name__
            start_time = time.time()

            logger.info(f"Starting Celery task: {name}")

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.info(f"Completed Celery task: {name} in {duration:.2f}s")
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed Celery task: {name} after {duration:.2f}s - {str(e)}")
                raise

        return wrapper
    return decorator


class StructuredLogger:
    """General structured logger with context support."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}

    def add_context(self, **kwargs):
        """Add persistent context to all log messages."""
        self.context.update(kwargs)
        return self

    def clear_context(self):
        """Clear all context."""
        self.context = {}
        return self

    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with context."""
        extra = {**self.context, **kwargs}
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """Factory function to create structured loggers."""
    return StructuredLogger(name)