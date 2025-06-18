from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .tasks import register_face_user_task, login_face_user_task
import tempfile
from celery.result import AsyncResult


class RegisterFaceView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data['username']
        image = request.FILES['image']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image.read())
            tmp.flush()
            image_path = tmp.name
        task = register_face_user_task.delay(username, image_path)
        return Response({'task_id': task.id})


class LoginFaceView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data['username']
        image = request.FILES['image']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image.read())
            tmp.flush()
            image_path = tmp.name
        task = login_face_user_task.delay(username, image_path)
        return Response({'task_id': task.id})


class TaskStatusView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, task_id):
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
            # fallback for legacy results
            return Response({"state": result.state, "result": res})

