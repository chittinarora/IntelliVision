# /apps/video_analytics/progress_utils.py

"""
=====================================
Progress Utilities
=====================================
Utility functions for updating job progress in the database.
"""

import logging
from django.db import transaction
from .models import VideoJob

logger = logging.getLogger(__name__)

def update_job_progress(job_id: int, processed_frames: int, total_frames: int, fps: float = None):
    """
    Update job progress in the database for real-time tracking.

    Args:
        job_id: Job ID
        processed_frames: Number of frames processed
        total_frames: Total number of frames
        fps: Current processing FPS
    """
    try:
        with transaction.atomic():
            job = VideoJob.objects.select_for_update().get(id=job_id)

            # Create or update results with progress
            if not job.results:
                job.results = {}

            job.results.update({
                'processed_frames': processed_frames,
                'total_frames': total_frames,
            })

            # Fix: VideoJob model doesn't have a 'meta' field, store FPS in results instead
            if fps is not None:
                job.results['fps'] = fps

            job.save(update_fields=['results'])

    except Exception as e:
        logger.warning(f"⚠️ Failed to update progress for job {job_id}: {e}")
