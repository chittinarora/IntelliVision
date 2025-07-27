# /apps/video_analytics/tests.py

"""
Unit tests for video analytics functionality.
"""

from django.test import TestCase
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import VideoJob
from .tasks import validate_input_file
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