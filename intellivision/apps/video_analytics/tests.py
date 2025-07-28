# /apps/video_analytics/tests.py

"""
Unit tests for video analytics functionality.
"""

from django.test import TestCase
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import VideoJob
from .tasks import validate_input_file
from .utils import validate_file_upload_for_job_type
import os


class VideoAnalyticsTests(TestCase):
    """Tests for video analytics app."""

    def setUp(self):
        """Set up test user and data."""
        self.user = User.objects.create_user(username='testuser', password='testpass')

    def test_video_job_creation(self):
        """Test creating a VideoJob instance."""
        job = VideoJob.objects.create(
            user=self.user,
            job_type='people_count',
            input_video=SimpleUploadedFile('test.mp4', b'content', content_type='video/mp4')
        )
        self.assertEqual(job.status, 'pending')
        self.assertEqual(job.user.username, 'testuser')
        self.assertEqual(job.job_type, 'people_count')

    def test_file_validation(self):
        """Test file validation in tasks.py."""
        with open('test.mp4', 'wb') as f:
            f.write(b'content')
        is_valid, msg = validate_input_file('test.mp4')
        self.assertTrue(is_valid, msg)

        with open('test.txt', 'wb') as f:
            f.write(b'content')
        is_valid, msg = validate_input_file('test.txt')
        self.assertFalse(is_valid)
        self.assertIn('Invalid file type', msg)

        os.unlink('test.mp4')
        os.unlink('test.txt')

    def test_job_type_specific_validation(self):
        """Test job-type-specific file validation."""

        # Test pothole detection - should only accept images
        image_file = SimpleUploadedFile('test.jpg', b'image_content', content_type='image/jpeg')
        video_file = SimpleUploadedFile('test.mp4', b'video_content', content_type='video/mp4')

        # Pothole detection should accept images
        is_valid, msg = validate_file_upload_for_job_type(image_file, 'pothole-detection')
        self.assertTrue(is_valid, f"Image should be valid for pothole detection: {msg}")

        # Pothole detection should reject videos
        is_valid, msg = validate_file_upload_for_job_type(video_file, 'pothole-detection')
        self.assertFalse(is_valid, "Video should be rejected for pothole detection")
        self.assertIn('pothole-detection', msg)
        self.assertIn('images (JPG, JPEG, PNG)', msg)

        # Test people counting - should only accept videos (default behavior)
        is_valid, msg = validate_file_upload_for_job_type(video_file, 'people-count')
        self.assertTrue(is_valid, f"Video should be valid for people counting: {msg}")

        is_valid, msg = validate_file_upload_for_job_type(image_file, 'people-count')
        self.assertFalse(is_valid, "Image should be rejected for people counting")
        self.assertIn('videos (MP4)', msg)

        # Test room readiness - should accept both
        is_valid, msg = validate_file_upload_for_job_type(image_file, 'room-readiness')
        self.assertTrue(is_valid, f"Image should be valid for room readiness: {msg}")

        is_valid, msg = validate_file_upload_for_job_type(video_file, 'room-readiness')
        self.assertTrue(is_valid, f"Video should be valid for room readiness: {msg}")

    def test_file_size_validation(self):
        """Test file size validation for job-type-specific validation."""
        # Create a mock large file
        large_file = SimpleUploadedFile('large.jpg', b'x' * (600 * 1024 * 1024), content_type='image/jpeg')  # 600MB

        is_valid, msg = validate_file_upload_for_job_type(large_file, 'pothole-detection')
        self.assertFalse(is_valid, "Large file should be rejected")
        self.assertIn('File size', msg)
        self.assertIn('exceeds', msg)
