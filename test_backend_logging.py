#!/usr/bin/env python3
"""
Test script for independent backend logging functionality

Run with: python test_backend_logging.py
Note: Backend logging is completely independent and does NOT receive frontend logs
"""

import os
import sys
import django
from pathlib import Path

# Add the project directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'intellivision'))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'intellivision.settings')
django.setup()

# Now we can import our logging utilities
from apps.logging_utils import (
    security_logger,
    performance_logger,
    api_logger,
    get_logger,
    log_performance
)
import logging
import time
from unittest.mock import Mock

def test_basic_logging():
    """Test basic Django logging configuration"""
    print("🧪 Testing Basic Django Logging")

    # Test different loggers
    django_logger = logging.getLogger('django')
    django_logger.info("Django info log test")

    celery_logger = logging.getLogger('celery')
    celery_logger.info("Celery info log test")

    face_auth_logger = logging.getLogger('apps.face_auth')
    face_auth_logger.info("Face auth logger test")

    video_analytics_logger = logging.getLogger('apps.video_analytics')
    video_analytics_logger.info("Video analytics logger test")

    print("✅ Basic logging test completed")

def test_security_logging():
    """Test security logging functionality"""
    print("🔒 Testing Security Logging")

    # Mock request object
    mock_request = Mock()
    mock_request.META = {
        'HTTP_X_FORWARDED_FOR': '192.168.1.100, 10.0.0.1',
        'REMOTE_ADDR': '10.0.0.1',
        'HTTP_USER_AGENT': 'Mozilla/5.0 Test Browser'
    }
    mock_request.user = Mock()
    mock_request.user.is_authenticated = True
    mock_request.user.id = 123
    mock_request.user.username = 'testuser'
    mock_request.user.is_staff = False
    mock_request.user.is_superuser = False

    # Test security logging
    security_logger.log_login_attempt(mock_request, 'testuser', True, 'face_auth')
    security_logger.log_login_attempt(mock_request, 'baduser', False, 'face_auth')
    security_logger.log_face_auth(mock_request, 'testuser', True, 0.95)
    security_logger.log_permission_denied(mock_request, '/admin/', 'access')
    security_logger.log_suspicious_activity(mock_request, 'multiple_failed_logins', {
        'attempts': 5,
        'timeframe': '5 minutes'
    })

    print("✅ Security logging test completed")

def test_performance_logging():
    """Test performance logging functionality"""
    print("⚡ Testing Performance Logging")

    # Test operation logging
    performance_logger.log_operation('database_query', 150.5, {
        'query_type': 'SELECT',
        'table': 'auth_user'
    })

    performance_logger.log_operation('slow_operation', 2500.0, {
        'operation_type': 'image_processing'
    })

    # Test database query logging
    performance_logger.log_database_query('SELECT', 75.2, 'video_jobs', 150)

    # Test API call logging
    performance_logger.log_api_call('/api/jobs/', 'GET', 120.3, 200, 1024)

    print("✅ Performance logging test completed")

def test_api_logging():
    """Test API logging functionality"""
    print("🌐 Testing API Logging")

    # Mock request for API logging
    mock_request = Mock()
    mock_request.method = 'POST'
    mock_request.path = '/api/face-auth/login/'
    mock_request.content_type = 'application/json'
    mock_request.META = {
        'HTTP_USER_AGENT': 'Test Client/1.0',
        'REMOTE_ADDR': '192.168.1.100',
        'CONTENT_LENGTH': '256'
    }
    mock_request.GET = {'debug': 'true'}
    mock_request.user = Mock()
    mock_request.user.is_authenticated = False

    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'{"success": true}'

    # Test API logging
    api_logger.log_request(mock_request, 'face_auth_login')
    api_logger.log_response(mock_request, mock_response, 250.5, 'face_auth_login')

    # Test error response
    mock_response.status_code = 400
    api_logger.log_response(mock_request, mock_response, 100.2, 'face_auth_login')

    print("✅ API logging test completed")

def test_structured_logger():
    """Test structured logger functionality"""
    print("📊 Testing Structured Logger")

    # Create structured logger
    structured_logger = get_logger('test_module')

    # Add context
    structured_logger.add_context(
        user_id=123,
        session_id='sess_abc123',
        request_id='req_xyz789'
    )

    # Test logging with context
    structured_logger.info('Processing user request',
                          operation='face_recognition',
                          image_size='1024x768')

    structured_logger.warning('Slow operation detected',
                             duration_ms=2000,
                             threshold_ms=1000)

    structured_logger.error('Operation failed',
                           error_code='FACE_NOT_DETECTED',
                           retry_count=3)

    # Clear context
    structured_logger.clear_context()
    structured_logger.info('Log without context')

    print("✅ Structured logger test completed")

@log_performance('test_decorated_function')
def test_performance_decorator():
    """Test performance logging decorator"""
    print("🎯 Testing Performance Decorator")
    time.sleep(0.1)  # Simulate work
    return "decorator_test_result"

def main():
    """Run all logging tests"""
    print("🚀 Starting Backend Logging Tests\n")

    try:
        test_basic_logging()
        print()

        test_security_logging()
        print()

        test_performance_logging()
        print()

        test_api_logging()
        print()

        test_structured_logger()
        print()

        # Test decorator
        result = test_performance_decorator()
        print(f"✅ Performance decorator test completed: {result}")
        print()

        print("🎉 All backend logging tests completed successfully!")
        print("ℹ️  Backend logging is independent and does NOT receive frontend logs")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()