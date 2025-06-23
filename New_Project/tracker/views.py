from django.shortcuts import render
from rest_framework import viewsets, permissions, serializers
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
        data = self.request.data
        job_type = data.get("job_type", "emergency_count")

        # ROI validation only for emergency_count jobs
        if job_type == "emergency_count":
            try:
                roi_x = float(data['roi_x'])
                roi_y = float(data['roi_y'])
                roi_width = float(data['roi_width'])
                roi_height = float(data['roi_height'])
            except (KeyError, ValueError):
                raise serializers.ValidationError(
                    "ROI fields are required and must be valid floats for emergency_count jobs.")

            if not (0 <= roi_x <= 1 and 0 <= roi_y <= 1 and 0 < roi_width <= 1 and 0 < roi_height <= 1):
                raise serializers.ValidationError("ROI values must be in [0, 1] and greater than zero.")
            if roi_width * roi_height < 0.05 * 0.05:
                raise serializers.ValidationError("ROI too small. Minimum size is 5% x 5% of the frame.")

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
