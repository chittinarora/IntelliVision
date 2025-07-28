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
from celery.result import AsyncResult
import logging

logger = logging.getLogger(__name__)

def standardized_error_response(message, code, status_code=status.HTTP_400_BAD_REQUEST, guidance=None):
    """Create a standardized error response with consistent format."""
    error_data = {
        "success": False,
        "error": {
            "message": message,
            "code": code,
            "guidance": guidance or "Please check your input and try again."
        }
    }
    return Response(error_data, status=status_code)

def standardized_success_response(task_id, message="Task created successfully"):
    """Create a standardized success response for async tasks."""
    return Response({
        "success": True,
        "task_id": task_id,
        "message": message,
        "status": "processing",
        "estimated_time": "15-30 seconds"
    })


class RegisterFaceView(APIView):
    """API endpoint for registering a new face user."""
    parser_classes = [MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request):
        """Accepts a username and image, starts a background registration task, and returns the task ID."""
        # Validate username
        try:
            username = request.data['username']
        except KeyError:
            return standardized_error_response(
                message="Username is required for account creation",
                code="MISSING_USERNAME",
                guidance="Please enter a username of at least 2 characters to create your account."
            )

        if not username or len(username.strip()) < 2:
            return standardized_error_response(
                message="Username must be at least 2 characters long",
                code="INVALID_USERNAME_LENGTH",
                guidance="Please choose a username with 2 or more characters. Letters, numbers, and underscores are allowed."
            )

        # Validate username format (basic alphanumeric + underscore)
        username = username.strip()
        if not username.replace('_', '').replace('-', '').isalnum():
            return standardized_error_response(
                message="Username contains invalid characters",
                code="INVALID_USERNAME_FORMAT",
                guidance="Username can only contain letters, numbers, hyphens, and underscores."
            )

        # Validate image file
        try:
            image = request.FILES['image']
        except KeyError:
            return standardized_error_response(
                message="Face photo is required for account creation",
                code="MISSING_IMAGE",
                guidance="Please capture a clear photo of your face using the camera above."
            )

        if not image:
            return standardized_error_response(
                message="Face photo is required for account creation",
                code="EMPTY_IMAGE",
                guidance="Please capture a clear photo of your face using the camera above."
            )

        # Check file size (max 10MB for face images)
        if image.size > 10 * 1024 * 1024:
            return standardized_error_response(
                message="Image file is too large (maximum 10MB allowed)",
                code="FILE_TOO_LARGE",
                guidance="Please capture a new photo. If the file is still too large, check your camera settings."
            )

        # Check file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if image.content_type not in allowed_types:
            return standardized_error_response(
                message="Invalid file type. Only JPEG and PNG images are supported",
                code="INVALID_FILE_TYPE",
                guidance="Please capture a new photo using the camera above, which will create a supported image format."
            )

        # Create temporary file for processing
        try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                for chunk in image.chunks():
                    tmp.write(chunk)
            tmp.flush()
            image_path = tmp.name
        except Exception as e:
            logger.error(f"Failed to save uploaded image: {e}")
            return standardized_error_response(
                message="Failed to process uploaded image",
                code="IMAGE_PROCESSING_ERROR",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                guidance="Please try capturing a new photo. If the problem persists, check your internet connection."
            )

        # Start background task
        try:
        task = register_face_user_task.delay(username, image_path)
            logger.info(f"Registration task created for user {username}: {task.id}")
            return standardized_success_response(
                task_id=task.id,
                message=f"Creating account for {username}... This may take 15-30 seconds."
            )
        except Exception as e:
            logger.error(f"Failed to create registration task: {e}")
            return standardized_error_response(
                message="Failed to start account creation process",
                code="TASK_CREATION_ERROR",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                guidance="Please try again. If the problem persists, our system may be experiencing high load."
            )


class LoginFaceView(APIView):
    """API endpoint for logging in a face user."""
    parser_classes = [MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request):
        """Accepts an image, starts a background login task, and returns the task ID."""
        # Username is optional for login - the system will recognize the person from the image
        username = request.data.get('username', '')

        # Validate image file
        try:
            image = request.FILES['image']
        except KeyError:
            return standardized_error_response(
                message="Face photo is required for authentication",
                code="MISSING_IMAGE",
                guidance="Please capture a clear photo of your face using the camera above."
            )

        if not image:
            return standardized_error_response(
                message="Face photo is required for authentication",
                code="EMPTY_IMAGE",
                guidance="Please capture a clear photo of your face using the camera above."
            )

        # Check file size (max 10MB for face images)
        if image.size > 10 * 1024 * 1024:
            return standardized_error_response(
                message="Image file is too large (maximum 10MB allowed)",
                code="FILE_TOO_LARGE",
                guidance="Please capture a new photo. If the file is still too large, check your camera settings."
            )

        # Check file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if image.content_type not in allowed_types:
            return standardized_error_response(
                message="Invalid file type. Only JPEG and PNG images are supported",
                code="INVALID_FILE_TYPE",
                guidance="Please capture a new photo using the camera above, which will create a supported image format."
            )

        # Create temporary file for processing
        try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                for chunk in image.chunks():
                    tmp.write(chunk)
            tmp.flush()
            image_path = tmp.name
        except Exception as e:
            logger.error(f"Failed to save uploaded image: {e}")
            return standardized_error_response(
                message="Failed to process uploaded image",
                code="IMAGE_PROCESSING_ERROR",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                guidance="Please try capturing a new photo. If the problem persists, check your internet connection."
            )

        # Start background task
        try:
        task = login_face_user_task.delay(image_path)
            logger.info(f"Login task created: {task.id}")
            return standardized_success_response(
                task_id=task.id,
                message="Authenticating your face... This may take 15-30 seconds."
            )
        except Exception as e:
            logger.error(f"Failed to create login task: {e}")
            return standardized_error_response(
                message="Failed to start authentication process",
                code="TASK_CREATION_ERROR",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                guidance="Please try again. If the problem persists, our system may be experiencing high load."
            )


class TaskStatusView(APIView):
    """API endpoint for checking the status of a background face authentication task."""
    permission_classes = [AllowAny]

    def get(self, request, task_id):
        """Returns the current state and result of the given Celery task ID."""
        try:
        result = AsyncResult(task_id)
        except Exception as e:
            logger.error(f"Failed to get task result for {task_id}: {e}")
            return standardized_error_response(
                message="Invalid or expired task",
                code="INVALID_TASK_ID",
                guidance="Please try the authentication process again from the beginning."
            )

        # Task still processing
        if not result.ready():
            processing_messages = {
                'PENDING': 'Initializing face recognition...',
                'STARTED': 'Processing your photo...',
                'RETRY': 'Retrying with enhanced image processing...',
            }
            return Response({
                "success": True,
                "state": result.state,
                "status": "processing",
                "message": processing_messages.get(result.state, "Processing your request..."),
                "progress": "Working on it - this may take 15-30 seconds"
            })

        # Task completed
        res = result.result
        if isinstance(res, dict):
            if res.get("success", False):
                # Successful authentication/registration
            return Response({
                    "success": True,
                "state": result.state,
                    "message": res.get("message", "Authentication successful!"),
                "token": res.get("token", None),
                "name": res.get("name", ""),
                    "image": res.get("image", ""),
                    "status": "completed"
                })
            else:
                # Authentication/registration failed
                message = res.get("message", "Authentication failed")

                # Provide better guidance based on common failure reasons
                guidance = "Please try again with a clearer photo."
                if "No face detected" in message:
                    guidance = "Make sure your face is clearly visible and well-lit. Try moving closer to the camera."
                elif "already registered" in message:
                    guidance = "This username is taken. Please try logging in instead or choose a different username."
                elif "No match found" in message:
                    guidance = "Face not recognized. Make sure you're registered, or try adjusting your position and lighting."
                elif "Match too weak" in message:
                    guidance = "Face recognition confidence is low. Please ensure good lighting and look directly at the camera."
                elif "Qdrant" in message:
                    guidance = "System temporarily unavailable. Please try again in a few moments."

                return Response({
                    "success": False,
                    "state": result.state,
                    "error": {
                        "message": message,
                        "code": "AUTHENTICATION_FAILED",
                        "guidance": guidance
                    },
                    "status": "failed"
            })
        else:
            # Handle unexpected result format or exceptions
            if isinstance(res, Exception):
                logger.error(f"Task {task_id} failed with exception: {res}")
                return Response({
                    "success": False,
                    "state": result.state,
                    "error": {
                        "message": "Authentication process encountered an error",
                        "code": "PROCESSING_ERROR",
                        "guidance": "Please try again. If the problem persists, our system may be experiencing issues."
                    },
                    "status": "failed"
                }, status=500)

            # Fallback for unexpected result types
            return Response({
                "success": False,
                "state": result.state,
                "error": {
                    "message": "Unexpected response from authentication system",
                    "code": "UNEXPECTED_RESULT",
                    "guidance": "Please try the authentication process again."
                },
                "status": "failed"
            })
