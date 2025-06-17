from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .tasks import register_face_user_task, login_face_user_task
import tempfile
from celery.result import AsyncResult


class RegisterFaceView(APIView):
    parser_classes = [MultiPartParser]

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
    def get(self, request, task_id):
        result = AsyncResult(task_id)
        return Response({
            'state': result.state,
            'result': result.result if result.ready() else None
        })
