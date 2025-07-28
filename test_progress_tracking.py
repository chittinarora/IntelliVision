#!/usr/bin/env python3
"""
Test script for progress tracking functionality.
"""

import os
import sys
import django

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'intellivision.settings')
django.setup()

from intellivision.apps.video_analytics.progress_utils import update_job_progress
from intellivision.apps.video_analytics.models import VideoJob
from django.contrib.auth.models import User

def test_progress_tracking():
    """Test the progress tracking functionality."""
    print("🧪 Testing Progress Tracking...")

    # Get or create a test user
    user, created = User.objects.get_or_create(
        username='test_user',
        defaults={'email': 'test@example.com'}
    )

    # Create a test job
    job = VideoJob.objects.create(
        user=user,
        job_type='people-count',
        status='processing',
        input_video='test_video.mp4'
    )

    print(f"✅ Created test job {job.id}")

    # Test progress updates
    test_cases = [
        (10, 100, 5.5),
        (25, 100, 6.2),
        (50, 100, 7.1),
        (75, 100, 6.8),
        (100, 100, 6.5),
    ]

    for processed, total, fps in test_cases:
        print(f"📊 Updating progress: {processed}/{total} frames, {fps} FPS")
        update_job_progress(job.id, processed, total, fps)

        # Refresh from database
        job.refresh_from_db()

        if job.results:
            print(f"   ✅ Results: {job.results.get('processed_frames')}/{job.results.get('total_frames')}")
        if job.meta:
            print(f"   ✅ Meta FPS: {job.meta.get('fps')}")

    # Clean up
    job.delete()
    print("🧹 Cleaned up test job")
    print("✅ Progress tracking test completed!")

if __name__ == '__main__':
    test_progress_tracking()
