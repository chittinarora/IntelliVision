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
import tempfile
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
from uuid import uuid4
from io import BytesIO

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")

from celery import shared_task
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils import timezone

# Third-party imports for YouTube processing
import cv2
import yt_dlp
from PIL import Image

from .models import VideoJob
from .rate_limiting import release_job_slot

# Add model manager import
from .analytics.model_manager import initialize_models, get_model_with_fallback

# Import exception classes
from .exception_handlers import GPUError, ModelLoadingError

# Analytics Function Imports - Using lazy imports to prevent eager model loading
# Models will only be loaded when the specific task is executed, not on worker startup

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
        logger.error(f"Job {job.id}: Output file '{output_file_path}' not found")
        return None

    try:
        # Check file size and permissions
        file_size = os.path.getsize(output_file_path)
        logger.info(f"Job {job.id}: Saving output file - Size: {file_size / (1024*1024):.2f}MB, Path: {output_file_path}")

        actual_filename = os.path.basename(output_file_path)
        saved_name = f"outputs/{actual_filename}"
        ext = os.path.splitext(actual_filename)[1].lower()

        # Check if output directory exists and is writable
        output_dir = os.path.join(default_storage.location, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Job {job.id}: Output directory: {output_dir}")

        with open(output_file_path, 'rb') as f:
            saved_path = default_storage.save(saved_name, ContentFile(f.read()))
            output_url = default_storage.url(saved_path)

        # Verify the file was actually saved
        if not default_storage.exists(saved_path):
            logger.error(f"Job {job.id}: File save verification failed - {saved_path} does not exist in storage")
            return None

        saved_size = default_storage.size(saved_path)
        logger.info(f"Job {job.id}: File save verified - Saved size: {saved_size / (1024*1024):.2f}MB at {saved_path}")

        # Assign to correct model field
        if ext in ['.mp4', '.webm', '.mov']:
            job.output_video.name = saved_name
        else:
            job.output_image.name = saved_name

        job.save(update_fields=['output_video', 'output_image'])
        logger.info(f"Job {job.id}: Model field updated and saved")

        final_url = ensure_api_media_url(output_url)
        logger.info(f"Job {job.id}: Final URL: {final_url}")
        return final_url

    except Exception as e:
        logger.error(f"Job {job.id}: File save failed with error: {str(e)}", exc_info=True)
        return None

def log_memory_usage(job_id: str = None, stage: str = "unknown"):
    """Log current memory usage for debugging."""
    if not PSUTIL_AVAILABLE:
        return

    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_percent = process.memory_percent()
        job_prefix = f"Job {job_id}: " if job_id else ""
        logger.info(f"{job_prefix}Memory usage at {stage}: {memory_mb:.1f}MB ({memory_percent:.1f}%)")
    except Exception as e:
        logger.warning(f"Failed to log memory usage: {e}")

"""
=====================================
Celery Task Definition
=====================================
Processes video analytics jobs using a strategy pattern.
Updated status to 'completed' instead of 'done' and job types to hyphenated values
to align with models.py and job.ts, fixing status field mismatch (Critical Issue #6)
and ensuring job type consistency (Critical Issue #2).
"""

def get_job_with_retry(job_id: int, max_retries: int = 3, retry_delay: float = 0.5) -> VideoJob:
    """
    Get job from database with retry logic to handle race conditions.

    Args:
        job_id: ID of the job to fetch
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds

    Returns:
        VideoJob instance

    Raises:
        VideoJob.DoesNotExist if job not found after all retries
    """
    for attempt in range(max_retries):
        try:
            return VideoJob.objects.get(id=job_id)
        except VideoJob.DoesNotExist:
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"Job {job_id}: Not found in database after {max_retries} attempts")
                raise
            logger.info(f"Job {job_id}: Not found, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

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
    log_memory_usage(str(job_id), "job_start")

    # Initialize model management system if not already done
    try:
        initialize_models(auto_download=True)
        logger.info("✅ Model management system initialized for job processing")
        log_memory_usage(str(job_id), "after_model_init")
    except Exception as e:
        logger.warning(f"⚠️ Model management initialization failed: {e}")
        # Continue with job processing, models will be resolved individually

    try:
        # Use retry logic when fetching the job
        job = get_job_with_retry(job_id)
    except VideoJob.DoesNotExist:
        logger.error(f"Job {job_id}: Not found in database after retries")
        raise

    # Log job details
    logger.info(
        f"📋 Job {job_id}: Type={job.job_type}, User={job.user.username}, Input={job.input_video.name or 'None'}")
    if job.youtube_url:
        logger.info(f"🎬 Job {job_id}: YouTube URL={job.youtube_url}")

    # ======================================
    # HANDLE YOUTUBE DOWNLOADS IN CELERY
    # ======================================
    if job.youtube_url and not job.input_video:
        logger.info(f"📥 Job {job_id}: Downloading YouTube video for processing")

        import tempfile
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        download_result = download_youtube_video(job.youtube_url, temp_video_path, quality='best')

        if not download_result['success']:
            job.status = 'failed'
            job.results = {'error': f"YouTube download failed: {download_result['error']}", 'error_time': datetime.now().isoformat()}
            job.save()
            logger.error(f"Job {job_id}: YouTube download failed: {download_result['error']}")
            release_job_slot()
            return

        # Save downloaded video to Django storage
        try:
            with open(temp_video_path, 'rb') as f:
                from uuid import uuid4
                file_name = f"yt_{job.job_type}_{uuid4().hex}.mp4"
                input_file_content = ContentFile(f.read(), name=file_name)
                job.input_video = input_file_content
                job.save()
                logger.info(f"✅ Job {job_id}: YouTube video downloaded and saved: {file_name} ({download_result['file_size'] / (1024*1024):.2f}MB)")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except Exception as e:
                    logger.warning(f"Job {job_id}: Failed to clean up temp video: {e}")

    # Validate input file (now works for both uploaded files and downloaded YouTube videos)
    if not job.input_video:
        error_msg = "No input file or YouTube URL provided"
        job.status = 'failed'
        job.results = {'error': error_msg, 'error_time': datetime.now().isoformat()}
        job.save()
        logger.error(f"Job {job_id}: {error_msg}")
        release_job_slot()
        raise ValueError(error_msg)

    is_valid, error_msg = validate_input_file(job.input_video.path)
    if not is_valid:
        job.status = 'failed'
        job.results = {'error': error_msg, 'error_time': datetime.now().isoformat()}
        job.save()
        logger.error(f"Job {job_id}: {error_msg}")
        release_job_slot()
        raise ValueError(error_msg)

    try:
        logger.info(f"🔄 Job {job_id}: Updating status to 'processing'")
        job.status = 'processing'
        job.save()

        # Strategy pattern for job processing with lazy imports
        def get_processor_func(job_type: str):
            """Lazy import analytics functions to prevent model loading on worker startup."""
            if job_type == "people-count":
                from .analytics.people_count import tracking_video
                return tracking_video
            elif job_type == "car-count":
                from .analytics.car_count import recognize_number_plates
                return recognize_number_plates
            elif job_type == "parking-analysis":
                from .analytics.car_count import analyze_parking_video
                return analyze_parking_video
            elif job_type == "wildlife-detection":
                from .analytics.pest_monitoring import tracking_video
                return tracking_video
            elif job_type == "wildlife-detection-image":
                from .analytics.pest_monitoring import tracking_image
                return tracking_image
            elif job_type == "food-waste-estimation":
                from .analytics.food_waste_estimation import analyze_food_image
                return analyze_food_image
            elif job_type == "room-readiness":
                from .analytics.room_readiness import analyze_room_video_multi_zone_only
                return analyze_room_video_multi_zone_only
            elif job_type == "room-readiness-image":
                from .analytics.room_readiness import analyze_room_image
                return analyze_room_image
            elif job_type == "lobby-detection":
                from .analytics.lobby_detection import tracking_video
                return tracking_video
            elif job_type == "emergency-count":
                from .analytics.emergency_count import tracking_video
                return tracking_video
            elif job_type == "pothole-detection":
                from .analytics.pothole_detection import tracking_video
                return tracking_video
            elif job_type == "pothole-detection-image":
                from .analytics.pothole_detection import run_pothole_image_detection
                return run_pothole_image_detection
            else:
                raise ValueError(f"Unknown job type: {job_type}")

        # Create unique output paths with timestamp to prevent conflicts
        timestamp = int(time.time())
        JOB_PROCESSORS = {
            "people-count": {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.mp4"]},
            "car-count": {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.mp4"]},
            "parking-analysis": {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.mp4"]},
            "wildlife-detection": {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.mp4"]},
            "food-waste-estimation": {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.mp4"]},
            "room-readiness": {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.mp4"]},
            "lobby-detection": {"args": [job.input_video.path, job.lobby_zones, f"/tmp/output_{job_id}_{timestamp}.mp4", job.id]},
            "emergency-count": {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.mp4", job.emergency_lines,
                                         job.video_width, job.video_height]},
            "pothole-detection": {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.mp4"]},
        }

        # Handle image inputs - override function for special cases
        ext = os.path.splitext(job.input_video.name)[1].lower()
        if job.job_type == "pothole-detection" and ext in ['.jpg', '.jpeg', '.png']:
            JOB_PROCESSORS["pothole-detection"] = {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.jpg"]}
            processor_func = lambda job_type: get_processor_func("pothole-detection-image")
        elif job.job_type == "room-readiness" and ext in ['.jpg', '.jpeg', '.png']:
            JOB_PROCESSORS["room-readiness"] = {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.jpg"]}
            processor_func = lambda job_type: get_processor_func("room-readiness-image")
        elif job.job_type == "wildlife-detection" and ext in ['.jpg', '.jpeg', '.png']:
            JOB_PROCESSORS["wildlife-detection"] = {"args": [job.input_video.path, f"/tmp/output_{job_id}_{timestamp}.jpg"]}
            processor_func = lambda job_type: get_processor_func("wildlife-detection-image")
        elif job.job_type == "emergency-count" and ext in ['.jpg', '.jpeg', '.png']:
            # Emergency count doesn't support images, return error
            raise ValueError(f"Emergency count job type does not support image files ({ext}). Please use video files.")
        else:
            processor_func = get_processor_func

        processor_config = JOB_PROCESSORS.get(job.job_type)
        if not processor_config:
            raise ValueError(f"Unknown job type: {job.job_type}")

        processor_func_instance = processor_func(job.job_type)
        processor_args = processor_config["args"]

        logger.info(f"🎯 Job {job_id}: Initializing '{processor_func_instance.__name__}' with {len(processor_args)} args")

        # Log video specs
        if ext in ['.mp4', '.avi', '.mov', '.webm']:
            try:
                import cv2
                cap = cv2.VideoCapture(job.input_video.path)
                try:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total_frames / fps if fps > 0 else 0
                    logger.info(
                        f"🎬 Job {job_id}: Video specs - {width}x{height}, {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
                finally:
                    cap.release()
            except Exception as e:
                logger.warning(f"⚠️ Job {job_id}: Could not get video specs: {e}")

        # Process job
        start_time = time.time()
        logger.info(f"🚀 Job {job_id}: Starting analytics at {datetime.now().strftime('%H:%M:%S')}")
        log_memory_usage(str(job_id), "before_analytics")

        # Update job status to processing
        job.status = 'processing'
        job.save(update_fields=['status'])

        # Pass job_id to analytics functions for progress logging
        # The analytics functions are now regular functions, not Celery tasks
        # So we don't need to pass 'self' anymore
        try:
            result_data = processor_func_instance(*processor_args)
            log_memory_usage(str(job_id), "after_analytics")
        except (GPUError, ModelLoadingError) as e:
            logger.error(f"❌ Job {job_id}: Analytics processing failed due to model/GPU error: {e}", exc_info=True)
            log_memory_usage(str(job_id), "after_analytics_error")

            job.status = 'failed'
            job.results = {
                "error": str(e),
                "error_type": type(e).__name__,
                "error_time": datetime.now().isoformat(),
                "traceback": traceback.format_exc()
            }
            job.save()
            release_job_slot()
            return

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

                # Clean up temporary file after successful save
                try:
                    if os.path.exists(output_file_path):
                        os.remove(output_file_path)
                        logger.info(f"🗑️ Job {job_id}: Cleaned up temporary file: {output_file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Job {job_id}: Failed to clean up temporary file {output_file_path}: {e}")
            else:
                logger.error(f"❌ Job {job_id}: Failed to save output - KEEPING TEMP FILE FOR DEBUGGING")
                # DO NOT clean up temporary file if save failed - keep for debugging
                logger.error(f"🚨 Job {job_id}: Temp file preserved at: {output_file_path}")
                result_data[output_path_key] = None
                result_data['output_path'] = None
        else:
            logger.info(f"ℹ️ Job {job_id}: No output file generated")

        # Save results - extract data field for all analytics jobs for consistency
        if isinstance(result_data, dict) and 'data' in result_data:
            job.results = result_data['data']
        else:
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
        # Don't re-raise the exception to prevent task retry loops
        logger.error(f"💾 Job {job_id}: Failed state saved")

    finally:
        # Always release job slot and clean up resources
        log_memory_usage(str(job_id), "job_completion")

        try:
            release_job_slot()
            logger.info(f"💾 Job {job_id}: Job slot released")
        except Exception as e:
            logger.error(f"Failed to release job slot: {e}")

        logger.info(f"💾 Job {job_id}: Final cleanup completed")

"""
=====================================
YouTube Processing Functions
=====================================
Functions for handling YouTube video downloads and frame extraction in Celery workers.
"""

def download_youtube_video(youtube_url: str, temp_path: str, quality: str = 'best') -> Dict[str, Any]:
    """
    Download YouTube video using yt-dlp in Celery worker.
    Supports authenticated downloads if YOUTUBE_COOKIES_PATH is set.
    """
    import yt_dlp
    from django.conf import settings
    import os

    try:
        logger.info(f"🎬 Downloading YouTube video: {youtube_url}")

        if quality == 'worst':
            format_selector = 'worst[ext=mp4]/worst'
        else:
            format_selector = 'bestvideo[ext=mp4][vcodec=h264]+bestaudio[ext=m4a]/best[ext=mp4][vcodec=h264]/best[ext=mp4]'

        ydl_opts = {
            'format': format_selector,
            'outtmpl': temp_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 60,  # Longer timeout for Celery
        }

        # Add cookies if available
        cookies_path = getattr(settings, 'YOUTUBE_COOKIES_PATH', None)
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
            logger.info("Using authentication cookies for YouTube download.")
        else:
            logger.info("No authentication cookies provided for YouTube download.")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(youtube_url, download=False)
            except yt_dlp.utils.DownloadError as e:
                # Check for authentication-required error
                if 'sign in to confirm you’re not a bot' in str(e).lower() or 'cookies' in str(e).lower():
                    return {'success': False, 'error': 'This video requires authentication. Please provide a valid cookies.txt file for YouTube.'}
                return {'success': False, 'error': str(e)}

            if not info:
                return {'success': False, 'error': 'Invalid YouTube URL or video not available'}

            filesize = info.get('filesize') or info.get('filesize_approx', 0)
            if filesize and filesize > MAX_FILE_SIZE:
                return {
                    'success': False,
                    'error': f'Video size {filesize / (1024*1024):.1f}MB exceeds {MAX_FILE_SIZE / (1024*1024):.0f}MB limit'
                }

            # Download the video
            try:
                ydl.download([youtube_url])
            except yt_dlp.utils.DownloadError as e:
                if 'sign in to confirm you’re not a bot' in str(e).lower() or 'cookies' in str(e).lower():
                    return {'success': False, 'error': 'This video requires authentication. Please provide a valid cookies.txt file for YouTube.'}
                return {'success': False, 'error': str(e)}

        # Verify download
        if not os.path.exists(temp_path):
            return {'success': False, 'error': 'Download failed - file not found'}

        actual_size = os.path.getsize(temp_path)
        if actual_size > MAX_FILE_SIZE:
            os.unlink(temp_path)  # Clean up oversized file
            return {
                'success': False,
                'error': f'Downloaded video size {actual_size / (1024*1024):.2f}MB exceeds limit'
            }

        logger.info(f"✅ YouTube download successful: {actual_size / (1024*1024):.2f}MB")
        return {
            'success': True,
            'file_size': actual_size,
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0)
        }

    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        # Clean up partial download
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        return {'success': False, 'error': str(e)}

@shared_task(bind=True)
def extract_youtube_frame(self, job_id: int) -> None:
    """
    Celery task to extract first frame from YouTube video.

    Args:
        self: Celery task instance
        job_id: ID of the VideoJob
    """
    import tempfile
    import cv2
    from uuid import uuid4
    from io import BytesIO
    from PIL import Image
    import requests

    logger.info(f"🚀 STARTING YouTube Frame Extraction Job {job_id} 🚀")

    try:
        job = get_job_with_retry(job_id)
    except VideoJob.DoesNotExist:
        logger.error(f"Job {job_id}: Not found in database")
        return

    if not job.youtube_url:
        logger.error(f"Job {job_id}: No YouTube URL provided")
        job.status = 'failed'
        job.results = {'error': 'No YouTube URL provided'}
        job.save()
        return

    logger.info(f"📋 Job {job_id}: Extracting frame from YouTube URL: {job.youtube_url}")

    temp_video_path = None
    temp_frame_path = None

    try:
        job.status = 'processing'
        job.save()

        # ======================================
        # STRATEGY 1: Try YouTube thumbnail first
        # ======================================
        try:
            logger.info(f"📸 Job {job_id}: Attempting thumbnail extraction")

            import yt_dlp
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(job.youtube_url, download=False)
                if info and 'thumbnails' in info and info['thumbnails']:
                    # Get highest quality thumbnail
                    best_thumbnail = max(info['thumbnails'], key=lambda x: x.get('width', 0) * x.get('height', 0))
                    thumbnail_url = best_thumbnail.get('url')

                    if thumbnail_url:
                        logger.info(f"📸 Job {job_id}: Found thumbnail: {thumbnail_url}")

                        # Download thumbnail
                        response = requests.get(thumbnail_url, timeout=30)
                        response.raise_for_status()

                        if len(response.content) < 50 * 1024 * 1024:  # Under 50MB
                            # Convert thumbnail to standard format
                            img = Image.open(BytesIO(response.content))
                            temp_frame_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name

                            if img.format not in ['JPEG', 'JPG']:
                                img.convert('RGB').save(temp_frame_path, 'JPEG', quality=90)
                            else:
                                with open(temp_frame_path, 'wb') as f:
                                    f.write(response.content)

                            # Save to storage
                            with open(temp_frame_path, 'rb') as f:
                                frame_filename = f"thumbnails/yt_thumb_{uuid4().hex}.jpg"
                                saved_path = default_storage.save(frame_filename, ContentFile(f.read()))
                                frame_url = ensure_api_media_url(default_storage.url(saved_path))

                            # Success with thumbnail
                            job.status = 'completed'
                            job.results = {
                                'frame_url': frame_url,
                                'method': 'thumbnail',
                                'title': info.get('title', 'Unknown'),
                                'processing_time': time.time()
                            }
                            job.output_image = frame_url
                            job.save()

                            logger.info(f"✅ Job {job_id}: Frame extracted via thumbnail: {frame_url}")
                            return

        except Exception as thumbnail_error:
            logger.warning(f"Job {job_id}: Thumbnail extraction failed: {thumbnail_error}, falling back to video download")

        # ======================================
        # STRATEGY 2: Download video and extract frame
        # ======================================
        logger.info(f"📥 Job {job_id}: Downloading video for frame extraction")

        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        download_result = download_youtube_video(job.youtube_url, temp_video_path, quality='worst')

        if not download_result['success']:
            job.status = 'failed'
            job.results = {'error': download_result['error']}
            job.save()
            logger.error(f"Job {job_id}: Download failed: {download_result['error']}")
            return

        # Extract frame using OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize memory usage
            success, frame = cap.read()
            if not success:
                job.status = 'failed'
                job.results = {'error': 'Could not extract frame from video'}
                job.save()
                logger.error(f"Job {job_id}: Frame extraction failed")
                return
        finally:
            cap.release()

        # Save frame
        temp_frame_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(temp_frame_path, frame)
        del frame  # Free memory

        # Save to storage
        with open(temp_frame_path, 'rb') as f:
            frame_filename = f"thumbnails/yt_frame_{uuid4().hex}.jpg"
            saved_path = default_storage.save(frame_filename, ContentFile(f.read()))
            frame_url = ensure_api_media_url(default_storage.url(saved_path))

        # Success with video download
        job.status = 'completed'
        job.results = {
            'frame_url': frame_url,
            'method': 'video_download',
            'title': download_result.get('title', 'Unknown'),
            'file_size_mb': download_result['file_size'] / (1024*1024),
            'processing_time': time.time()
        }
        job.output_image = frame_url
        job.save()

        logger.info(f"✅ Job {job_id}: Frame extracted via video download: {frame_url}")

    except Exception as e:
        logger.error(f"Job {job_id}: Frame extraction error: {e}", exc_info=True)
        job.status = 'failed'
        job.results = {'error': str(e)}
        job.save()

    finally:
        # Clean up temporary files
        for path in [temp_video_path, temp_frame_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.info(f"Job {job_id}: Cleaned up {path}")
                except Exception as e:
                    logger.warning(f"Job {job_id}: Failed to clean up {path}: {e}")

        # Release job slot
        release_job_slot()
