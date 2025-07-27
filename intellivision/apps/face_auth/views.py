# views.py - Face Authentication API Views
# Provides endpoints for registering, logging in, and checking the status of face authentication jobs.

from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from .tasks import register_face_user_task, login_face_user_task
import tempfile
from celery import AsyncResult


class RegisterFaceView(APIView):
    """API endpoint for registering a new face user."""
    parser_classes = [MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request):
        """Accepts a username and image, starts a background registration task, and returns the task ID."""
        try:
            username = request.data['username']
        except KeyError:
            return Response({'error': 'Username is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            image = request.FILES['image']
        except KeyError:
            return Response({'error': 'Image file is required'}, status=status.HTTP_400_BAD_REQUEST)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image.read())
            tmp.flush()
            image_path = tmp.name
        task = register_face_user_task.delay(username, image_path)
        return Response({'task_id': task.id})


class LoginFaceView(APIView):
    """API endpoint for logging in a face user."""
    parser_classes = [MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request):
        """Accepts a username and image, starts a background login task, and returns the task ID."""
        try:
            username = request.data['username']
        except KeyError:
            return Response({'error': 'Username is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            image = request.FILES['image']
        except KeyError:
            return Response({'error': 'Image file is required'}, status=status.HTTP_400_BAD_REQUEST)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image.read())
            tmp.flush()
            image_path = tmp.name
        task = login_face_user_task.delay(username, image_path)
        return Response({'task_id': task.id})


class TaskStatusView(APIView):
    """API endpoint for checking the status of a background face authentication task."""
    permission_classes = [AllowAny]

    def get(self, request, task_id):
        """Returns the current state and result of the given Celery task ID."""
        result = AsyncResult(task_id)

        if not result.ready():
            return Response({"state": result.state})

        res = result.result
        if isinstance(res, dict):
            return Response({
                "state": result.state,
                "success": res.get("success", False),
                "message": res.get("message", ""),
                "token": res.get("token", None),
                "name": res.get("name", ""),
                "image": res.get("image", "")
            })
        else:
            # fallback for legacy results and exceptions
            import traceback
            if isinstance(res, Exception):
                return Response({
                    "state": result.state,
                    "success": False,
                    "message": f"Task failed: {str(res)}",
                    "error_type": res.__class__.__name__,
                    "traceback": traceback.format_exception_only(type(res), res)
                }, status=500)
            return Response({"state": result.state, "result": str(res)})
