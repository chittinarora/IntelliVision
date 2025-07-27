# /apps/video_analytics/tasks.py

"""
=====================================
Imports
=====================================
Imports for Celery task processing and analytics functions.
"""

import os
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

from celery import shared_task
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils import timezone
import mimetypes

from .models import VideoJob

# Analytics Function Imports (Strategy Pattern)
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

# Set up logger
logger = logging.getLogger(__name__)

# Valid file extensions
VALID_EXTENSIONS = {'.mp4', '.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

"""
=====================================
Helper Functions
=====================================
Utilities for URL formatting, input validation, and output saving.
"""

def ensure_api_media_url(url: str) -> str:
    """Ensure URL is correctly formatted for API responses."""
    if not url:
        return url
    if url.startswith('/api/media/') or url.startswith('http'):
        return url
    if url.startswith('/media/'):
        return '/api' + url
    return '/api/media/' + url.lstrip('/')

def validate_input_file(file_path: str) -> tuple[bool, str]:
    """Validate file type and size."""
    if not file_path or not default_storage.exists(file_path):
        return False, f"File not found: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        return False, f"Invalid file type: {ext}. Allowed: {', '.join(VALID_EXTENSIONS)}"

    size = default_storage.size(file_path)
    if size > MAX_FILE_SIZE:
        return False, f"File size {size / (1024 * 1024):.2f}MB exceeds 500MB limit"

    return True, ""

def save_output_and_get_url(job: VideoJob, output_file_path: str) -> str:
    """
    Save output file to Django storage and return web-accessible URL.

    Args:
        job: VideoJob instance
        output_file_path: Path to the output file

    Returns:
        Web-accessible URL or None if saving fails
    """
    if not output_file_path or not os.path.exists(output_file_path):
        logger.warning(f"Job {job.id}: Output file '{output_file_path}' not found")
        return None

    actual_filename = os.path.basename(output_file_path)
    saved_name = f"outputs/{actual_filename}"
    ext = os.path.splitext(actual_filename)[1].lower()

    with open(output_file_path, 'rb') as f:
        saved_path = default_storage.save(saved_name, ContentFile(f.read()))
        output_url = default_storage.url(saved_path)

    # Assign to correct model field
    if ext in ['.mp4', '.webm', '.mov']:
        job.output_video.name = saved_name
    else:
        job.output_image.name = saved_name

    logger.info(f"Job {job.id}: Saved output to {saved_path}")
    return ensure_api_media_url(output_url)

"""
=====================================
Celery Task Definition
=====================================
Processes video analytics jobs using a strategy pattern.
Updated status to 'completed' instead of 'done' and job types to hyphenated values
to align with models.py and job.ts, fixing status field mismatch (Critical Issue #6)
and ensuring job type consistency (Critical Issue #2).
"""

@shared_task(bind=True)
def process_video_job(self, job_id: int) -> None:
    """
    Celery task to process video analytics jobs using a strategy pattern.
    Includes input validation, GPU monitoring, and enhanced logging.

    Args:
        self: Celery task instance
        job_id: ID of the VideoJob to process
    """
    logger.info(f"🚀 STARTING CELERY JOB {job_id} 🚀")
    try:
        job = VideoJob.objects.get(id=job_id)
    except VideoJob.DoesNotExist:
        logger.error(f"Job {job_id}: Not found in database")
        raise

    # Log job details
    logger.info(
        f"📋 Job {job_id}: Type={job.job_type}, User={job.user.username}, Input={job.input_video.name or 'None'}")
    if job.youtube_url:
        logger.info(f"🎬 Job {job_id}: YouTube URL={job.youtube_url}")

    # Validate input file
    is_valid, error_msg = validate_input_file(job.input_video.path)
    if not is_valid:
        job.status = 'failed'
        job.results = {'error': error_msg, 'error_time': datetime.now().isoformat()}
        job.save()
        logger.error(f"Job {job_id}: {error_msg}")
        raise ValueError(error_msg)

    # GPU memory check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3
            logger.info(f"🎮 Job {job_id}: GPU memory before: {gpu_memory:.2f}GB")
    except Exception as e:
        logger.info(f"💻 Job {job_id}: Running on CPU or GPU check failed: {e}")

    try:
        logger.info(f"🔄 Job {job_id}: Updating status to 'processing'")
        job.status = 'processing'
        job.save()

        # Strategy pattern for job processing
        JOB_PROCESSORS = {
            "people-count": {"func": process_people_count, "args": [job.input_video.path, f"/tmp/output_{job_id}.mp4"]},
            "car-count": {"func": process_car_count, "args": [os.path.basename(job.input_video.path)]},
            "parking-analysis": {"func": process_parking_analysis, "args": [os.path.basename(job.input_video.path)]},
            "wildlife-detection": {"func": process_wildlife_detection,
                                   "args": [job.input_video.path, f"/tmp/output_{job_id}.mp4"]},
            "food-waste-estimation": {"func": process_food_waste, "args": [job.input_video.path]},
            "room-readiness": {"func": analyze_room_video_multi_zone_only, "args": [job.input_video.path]},
            "lobby-detection": {"func": process_lobby_detection,
                                "args": [job.input_video.path, job.lobby_zones, f"/tmp/output_{job_id}.mp4"]},
            "emergency-count": {"func": process_emergency_count,
                                "args": [job.input_video.path, f"/tmp/output_{job_id}.mp4", job.emergency_lines,
                                         job.video_width, job.video_height]},
            "pothole-detection": {"func": process_pothole_video,
                                  "args": [job.input_video.path, f"/tmp/output_{job_id}.mp4"]},
        }

        # Handle image inputs
        ext = os.path.splitext(job.input_video.name)[1].lower()
        if job.job_type == "pothole-detection" and ext in ['.jpg', '.jpeg', '.png']:
            JOB_PROCESSORS["pothole-detection"] = {"func": process_pothole_image,
                                                   "args": [job.input_video.path, f"/tmp/output_{job_id}.jpg"]}
        elif job.job_type == "room-readiness" and ext in ['.jpg', '.jpeg', '.png']:
            JOB_PROCESSORS["room-readiness"] = {"func": analyze_room_image, "args": [job.input_video.path]}

        processor_config = JOB_PROCESSORS.get(job.job_type)
        if not processor_config:
            raise ValueError(f"Unknown job type: {job.job_type}")

        processor_func = processor_config["func"]
        processor_args = processor_config["args"]

        logger.info(f"🎯 Job {job_id}: Initializing '{processor_func.__name__}' with {len(processor_args)} args")

        # Log video specs
        if ext in ['.mp4', '.avi', '.mov', '.webm']:
            try:
                import cv2
                cap = cv2.VideoCapture(job.input_video.path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                logger.info(
                    f"🎬 Job {job_id}: Video specs - {width}x{height}, {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
            except Exception as e:
                logger.warning(f"⚠️ Job {job_id}: Could not get video specs: {e}")

        # Process job
        start_time = time.time()
        logger.info(f"🚀 Job {job_id}: Starting analytics at {datetime.now().strftime('%H:%M:%S')}")
        result_data = processor_func(*processor_args)
        processing_duration = time.time() - start_time
        logger.info(f"✅ Job {job_id}: Analytics completed in {processing_duration:.2f} seconds")

        # Log result summary
        if isinstance(result_data, dict):
            logger.info(f"📊 Job {job_id}: Result keys: {list(result_data.keys())}")
        else:
            logger.warning(f"⚠️ Job {job_id}: Non-dict result: {type(result_data)}")

        # Handle output
        output_path_key = 'output_video' if 'output_video' in result_data else 'output_image' if 'output_image' in result_data else 'output_path'
        output_file_path = result_data.get(output_path_key)
        if output_file_path:
            final_output_url = save_output_and_get_url(job, output_file_path)
            if final_output_url:
                result_data[output_path_key] = final_output_url
                result_data['output_path'] = final_output_url
                logger.info(f"✅ Job {job_id}: Output saved at {final_output_url}")
            else:
                logger.error(f"❌ Job {job_id}: Failed to save output")
        else:
            logger.info(f"ℹ️ Job {job_id}: No output file generated")

        # Save results
        job.results = result_data
        job.status = 'completed'  # Changed from 'done' to 'completed'
        job.save()
        logger.info(f"🎉 Job {job_id}: Completed successfully")

    except Exception as e:
        logger.error(f"❌ Job {job_id}: Failed - {type(e).__name__}: {str(e)}", exc_info=True)
        job.status = 'failed'
        job.results = {
            "error": str(e),
            "error_type": type(e).__name__,
            "error_time": datetime.now().isoformat(),
            "traceback": traceback.format_exc()
        }
        job.save()
        raise

    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"🎮 Job {job_id}: GPU memory cleaned")
        except:
            pass
        logger.info(f"💾 Job {job_id}: Final state saved")