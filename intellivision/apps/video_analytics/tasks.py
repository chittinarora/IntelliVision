# /apps/video_analytics/tasks.py

import os
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta

from celery import shared_task
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils import timezone

from .models import VideoJob

# =============================
# Analytics Function Imports (Strategy Pattern)
# =============================
# Import all the specific analytics functions that will be used as "strategies".
from .analytics.people_count import tracking_video as process_people_count
from .analytics.emergency_count import tracking_video as process_emergency_count
from .analytics.car_count import recognize_number_plates as process_car_count, \
    analyze_parking_video as process_parking_analysis
from .analytics.pothole_detection import tracking_video as process_pothole_video, \
    run_pothole_image_detection as process_pothole_image
from .analytics.food_waste_estimation import analyze_food_image as process_food_waste
from .analytics.room_readiness import analyze_room_image, analyze_room_video_multi_zone_only
from .analytics.pest_monitoring import tracking_video as process_wildlife_detection
from .analytics.lobby_detection import run_crowd_analysis as process_lobby_detection

# Set up a logger for this module
logger = logging.getLogger(__name__)


# =============================
# Helper Functions
# =============================

def ensure_api_media_url(url: str) -> str:
    """Ensures that a URL is correctly formatted for API responses."""
    if not url:
        return url
    if url.startswith('/api/media/') or url.startswith('http'):
        return url
    if url.startswith('/media/'):
        return '/api' + url
    return '/api/media/' + url.lstrip('/')


def save_output_and_get_url(job: VideoJob, output_file_path: str) -> str:
    """
    Helper function to save an output file (video or image) to Django's
    default storage and return a web-accessible URL.
    """
    if not output_file_path or not os.path.exists(output_file_path):
        logger.warning(f"Job {job.id}: Output file path '{output_file_path}' not found or is None.")
        return None

    actual_filename = os.path.basename(output_file_path)
    saved_name = f"outputs/{actual_filename}"
    ext = os.path.splitext(actual_filename)[1].lower()

    with open(output_file_path, 'rb') as f:
        saved_path = default_storage.save(saved_name, ContentFile(f.read()))
        output_url = default_storage.url(saved_path)

    # Assign to the correct model field based on file extension
    if ext in ['.mp4', '.webm', '.mov']:
        job.output_video.name = saved_name
    else:
        job.output_image.name = saved_name

    return ensure_api_media_url(output_url)


# =============================
# Celery Task Definition
# =============================

@shared_task
def process_video_job(job_id: int):
    """
    Celery task to process a video analytics job using a strategy pattern.
    """
    logger.info(f"--- Starting Celery job {job_id} ---")
    job = VideoJob.objects.get(id=job_id)

    try:
        job.status = 'processing'
        job.save()

        # --- Strategy Pattern Implementation ---
        # This dictionary maps a job_type to its specific processing function and arguments.
        JOB_PROCESSORS = {
            "people_count": {"func": process_people_count, "args": [job.input_video.path, f"/tmp/output_{job.id}.mp4"]},
            "car_count": {"func": process_car_count, "args": [os.path.basename(job.input_video.path)]},
            "parking_analysis": {"func": process_parking_analysis, "args": [os.path.basename(job.input_video.path)]},
            "wildlife_detection": {"func": process_wildlife_detection,
                                   "args": [job.input_video.path, f"/tmp/output_{job.id}.mp4"]},
            "food_waste_estimation": {"func": process_food_waste, "args": [job.input_video.path]},
            "room_readiness": {"func": analyze_room_video_multi_zone_only, "args": [job.input_video.path]},
            "lobby_detection": {
                "func": process_lobby_detection,
                "args": [job.input_video.path, job.lobby_zones, f"/tmp/output_{job.id}.mp4"]
            },
            "emergency_count": {
                "func": process_emergency_count,
                "args": [job.input_video.path, f"/tmp/output_{job.id}.mp4", job.emergency_lines, job.video_width,
                         job.video_height]
            },
            "pothole_detection": {
                "func": process_pothole_video,
                "args": [job.input_video.path, f"/tmp/output_{job.id}.mp4"]
            },
        }

        # Special handling for jobs that can accept image or video
        ext = os.path.splitext(job.input_video.name)[1].lower()
        if job.job_type == "pothole_detection" and ext in ['.jpg', '.jpeg', '.png']:
            JOB_PROCESSORS["pothole_detection"] = {
                "func": process_pothole_image,
                "args": [job.input_video.path, f"/tmp/output_{job.id}.jpg"]
            }
        elif job.job_type == "room_readiness" and ext in ['.jpg', '.jpeg', '.png']:
            JOB_PROCESSORS["room_readiness"] = {
                "func": analyze_room_image,
                "args": [job.input_video.path]
            }

        # Get the processor for the current job type
        processor_config = JOB_PROCESSORS.get(job.job_type)
        if not processor_config:
            raise ValueError(f"Unknown or unsupported job type: {job.job_type}")

        processor_func = processor_config["func"]
        processor_args = processor_config["args"]

        logger.info(f"Job {job.id}: Executing '{processor_func.__name__}' for job type '{job.job_type}'.")

        # Execute the analytics function
        result_data = processor_func(*processor_args)

        # --- Standardized Output Handling ---
        # The analytics functions should return a dictionary. We look for specific keys
        # to find the path of the generated output file.
        output_path_key = 'output_video' if 'output_video' in result_data else 'output_image' if 'output_image' in result_data else 'output_path'
        output_file_path = result_data.get(output_path_key)

        if output_file_path:
            final_output_url = save_output_and_get_url(job, output_file_path)
            if final_output_url:
                # Update the result data with the final, web-accessible URL
                result_data[output_path_key] = final_output_url
                result_data['output_path'] = final_output_url  # Ensure a consistent key for the frontend
                logger.info(f"Job {job.id}: Saved output to {final_output_url}")

        # Save results and finalize job
        job.results = result_data
        job.status = 'done'
        logger.info(f"--- Finished Celery job {job.id} successfully ---")

    except Exception as e:
        logger.error(f"--- ERROR in Celery job {job.id} ---", exc_info=True)
        job.status = 'failed'
        job.results = {"error": str(e), "traceback": traceback.format_exc()}
        # Re-raise the exception to let Celery know the task failed
        raise

    finally:
        # This block will always run, even if the task fails
        job.save()
