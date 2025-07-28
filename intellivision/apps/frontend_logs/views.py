"""
Frontend Logging Views

Handles frontend log submissions and stores them appropriately.
"""

import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status


# Get the frontend logger
frontend_logger = logging.getLogger('frontend_logger')


@api_view(['POST'])
@permission_classes([AllowAny])  # Frontend logs should be accepted even from unauthenticated users
def receive_frontend_logs(request):
    """
    Receive and process frontend log entries.
    
    Expected payload:
    {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "level": 2,
        "levelName": "ERROR",
        "message": "Error message",
        "context": {
            "component": "Dashboard",
            "sessionId": "session-123",
            "userId": "user-456",
            "action": "user_action"
        },
        "error": {
            "name": "Error",
            "message": "Detailed error message",
            "stack": "Error stack trace"
        },
        "url": "https://example.com/dashboard",
        "userAgent": "Mozilla/5.0..."
    }
    """
    try:
        log_entry = request.data
        
        # Validate required fields
        required_fields = ['timestamp', 'level', 'levelName', 'message']
        for field in required_fields:
            if field not in log_entry:
                return Response(
                    {'error': f'Missing required field: {field}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # Extract log data
        timestamp = log_entry.get('timestamp')
        level_name = log_entry.get('levelName')
        message = log_entry.get('message')
        context = log_entry.get('context', {})
        error_info = log_entry.get('error', {})
        url = log_entry.get('url', 'unknown')
        user_agent = log_entry.get('userAgent', 'unknown')
        
        # Build structured log message
        log_message = f"[FRONTEND] {message}"
        
        # Create extra data for structured logging
        extra_data = {
            'frontend_timestamp': timestamp,
            'component': context.get('component', 'unknown'),
            'session_id': context.get('sessionId', 'unknown'),
            'user_id': context.get('userId', 'anonymous'),
            'action': context.get('action', 'unknown'),
            'url': url,
            'user_agent': user_agent,
            'frontend_level': level_name,
        }
        
        # Add error information if present
        if error_info:
            extra_data.update({
                'error_name': error_info.get('name', 'Unknown'),
                'error_message': error_info.get('message', ''),
                'error_stack': error_info.get('stack', '')
            })
            log_message += f" | Error: {error_info.get('message', 'Unknown error')}"
        
        # Add metadata if present
        if 'metadata' in context:
            extra_data['metadata'] = json.dumps(context['metadata'])
        
        # Log based on level (frontend logs are typically ERROR or FATAL)
        level_name_upper = level_name.upper()
        if level_name_upper in ['ERROR', 'FATAL']:
            frontend_logger.error(log_message, extra=extra_data)
        elif level_name_upper == 'WARN':
            frontend_logger.warning(log_message, extra=extra_data)
        else:
            frontend_logger.info(log_message, extra=extra_data)
        
        return Response({'status': 'logged'}, status=status.HTTP_200_OK)
        
    except json.JSONDecodeError:
        return Response(
            {'error': 'Invalid JSON payload'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        # Log the error in processing frontend logs
        logging.getLogger('django').error(f"Error processing frontend log: {str(e)}")
        return Response(
            {'error': 'Internal server error'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Simple health check endpoint for frontend logging service."""
    return Response({
        'status': 'healthy',
        'service': 'frontend-logging',
        'timestamp': json.dumps(str(request.META.get('HTTP_DATE', 'unknown')))
    })