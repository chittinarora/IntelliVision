from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets, permissions
from .models import VideoJob
from .serializers import VideoJobSerializer
from rest_framework.parsers import MultiPartParser, FormParser
from .tasks import process_video_job
from django.contrib.auth import get_user_model

class VideoJobViewSet(viewsets.ModelViewSet):
    queryset = VideoJob.objects.all()
    serializer_class = VideoJobSerializer
    permission_classes = [permissions.AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        User = get_user_model()
        user = self.request.user if self.request.user.is_authenticated else User.objects.first()
        instance = serializer.save(user=user, status='pending')
        process_video_job.delay(instance.id)
        # Optionally: trigger background task here

    def get_queryset(self):
        # Only show jobs for the logged-in user
        return VideoJob.objects.all()
