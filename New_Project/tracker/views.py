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
from rest_framework import status
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone

# Import the food waste estimation function
from tracker.analytics.food_waste_estimation import analyze_food_image, analyze_multiple_food_images


class VideoJobViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing VideoJob objects. Handles creation, listing, and filtering by user.
    """
    queryset = VideoJob.objects.all()
    serializer_class = VideoJobSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        """
        Save the job with the current user, validate ROI for emergency_count jobs, and kick off processing in the background.
        """
        # Extract job type from request data
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

            # Ensure ROI values are within valid range
            if not (0 <= roi_x <= 1 and 0 <= roi_y <= 1 and 0 < roi_width <= 1 and 0 < roi_height <= 1):
                raise serializers.ValidationError("ROI values must be in [0, 1] and greater than zero.")
            if roi_width * roi_height < 0.05 * 0.05:
                raise serializers.ValidationError("ROI too small. Minimum size is 5% x 5% of the frame.")

        # Assign user and save job
        User = get_user_model()
        user = self.request.user if self.request.user.is_authenticated else User.objects.first()
        instance = serializer.save(user=user, status='pending')
        # Start background processing
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


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def food_waste_estimation_view(request):
    """
    Create a single VideoJob for one upload (even with multiple images),
    save all images, analyze them, and store the combined result in the job's results field.
    Returns job info (job_id, status, created_at, etc).
    """
    import tempfile
    import os
    from uuid import uuid4
    from django.core.files.base import ContentFile
    from django.utils import timezone
    from .models import VideoJob
    from django.contrib.auth import get_user_model

    User = get_user_model()
    user = request.user if request.user.is_authenticated else User.objects.first()

    # Gather all images (either 'images' or 'image')
    image_files = request.FILES.getlist('images')
    if not image_files and 'image' in request.FILES:
        image_files = [request.FILES['image']]

    if not image_files:
        return Response({'error': 'No image(s) file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    # Save all images to temp files for analysis
    temp_paths = []
    saved_files = []
    try:
        for image_file in image_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                for chunk in image_file.chunks():
                    tmp.write(chunk)
                temp_paths.append(tmp.name)
            # Save the first image as the input_video for the job (for record-keeping)
            if len(saved_files) == 0:
                image_file.seek(0)
                image_content = ContentFile(image_file.read(), name=f"food_{uuid4()}.jpg")
                saved_files.append(image_content)

        # Create a single VideoJob for this upload
        job = VideoJob.objects.create(
            user=user,
            status='pending',
            input_video=saved_files[0],
            job_type='food_waste_estimation',
            created_at=timezone.now(),
            updated_at=timezone.now(),
        )
        # Analyze all images and store the result in the job's results field
        from tracker.analytics.food_waste_estimation import analyze_multiple_food_images
        result_data = analyze_multiple_food_images(temp_paths)
        job.results = result_data
        job.status = 'done'  # Mark as done since analysis is synchronous here
        job.save()
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

    return Response({
        'job_id': job.id,
        'status': job.status,
        'created_at': job.created_at,
        'job_type': job.job_type,
        'results': job.results,
    })
