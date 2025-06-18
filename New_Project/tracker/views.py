from django.shortcuts import render
from rest_framework import viewsets, permissions
from rest_framework.parsers import MultiPartParser, FormParser
from django.contrib.auth import get_user_model

from .models import VideoJob
from .serializers import VideoJobSerializer
from .tasks import process_video_job

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response


class VideoJobViewSet(viewsets.ModelViewSet):
    queryset = VideoJob.objects.all()
    serializer_class = VideoJobSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        """
        Save the job with the current user, and kick off processing in the background.
        """
        User = get_user_model()
        user = self.request.user if self.request.user.is_authenticated else User.objects.first()
        instance = serializer.save(user=user, status='pending')
        process_video_job.delay(instance.id)

    def get_queryset(self):
        """
        Only return jobs belonging to the current authenticated user.
        """
        user = self.request.user
        if user.is_authenticated:
            return VideoJob.objects.filter(user=user).order_by('-created_at')
        return VideoJob.objects.none()


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user_view(request):
    """
    Return basic info about the currently authenticated user.
    """
    user = request.user
    return Response({
        "id": user.id,
        "username": user.username,
    })
