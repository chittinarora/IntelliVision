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
    Enhanced with detailed progress logging and GPU monitoring.
    """
    logger.info(f"🚀 STARTING CELERY JOB {job_id} 🚀")
    job = VideoJob.objects.get(id=job_id)
    
    # Log initial job details
    logger.info(f"📋 Job Details - ID: {job.id}, Type: {job.job_type}, User: {job.user.username}")
    logger.info(f"📁 Input File: {job.input_video.name if job.input_video else 'None'}")
    if job.youtube_url:
        logger.info(f"🎬 YouTube URL: {job.youtube_url}")
    
    # GPU Memory Check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            logger.info(f"🎮 GPU Memory - Allocated: {gpu_memory_allocated:.2f}GB, Reserved: {gpu_memory_reserved:.2f}GB, Total: {gpu_memory_total:.2f}GB")
        else:
            logger.info("💻 Running on CPU (no GPU available)")
    except Exception as e:
        logger.warning(f"⚠️ Could not check GPU memory: {e}")

    try:
        logger.info(f"🔄 Updating job status to 'processing'...")
        job.status = 'processing'
        job.save()
        logger.info(f"✅ Job {job.id} status updated to processing")

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

        logger.info(f"🎯 Job {job.id}: STAGE 1 - Initializing '{processor_func.__name__}' for job type '{job.job_type}'")
        logger.info(f"📊 Job {job.id}: Processing arguments: {len(processor_args)} args provided")
        
        # Log input file details
        if job.input_video and os.path.exists(job.input_video.path):
            file_size = os.path.getsize(job.input_video.path) / (1024*1024)  # MB
            logger.info(f"📈 Job {job.id}: Input file size: {file_size:.2f} MB")
            
            # Try to get video info for video files
            if job.input_video.name.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                try:
                    import cv2
                    cap = cv2.VideoCapture(job.input_video.path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    
                    logger.info(f"🎬 Job {job.id}: Video specs - {width}x{height}, {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
                    
                    # Estimate processing time based on job type and video length
                    if job.job_type in ['people_count', 'wildlife_detection']:
                        estimated_time = duration * 2  # Rough estimate: 2x real-time
                        logger.info(f"⏱️ Job {job.id}: Estimated processing time: {estimated_time:.1f} seconds")
                except Exception as e:
                    logger.warning(f"⚠️ Job {job.id}: Could not get video specs: {e}")

        # GPU memory check before processing
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear cache before processing
                gpu_memory_before = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"🎮 Job {job.id}: GPU memory before processing: {gpu_memory_before:.2f}GB")
        except:
            pass

        # Record start time for performance tracking
        start_time = time.time()
        logger.info(f"🚀 Job {job.id}: STAGE 2 - Starting analytics processing at {datetime.now().strftime('%H:%M:%S')}")

        # Execute the analytics function
        result_data = processor_func(*processor_args)
        
        # Record completion time
        end_time = time.time()
        processing_duration = end_time - start_time
        logger.info(f"✅ Job {job.id}: STAGE 2 COMPLETE - Analytics processing finished in {processing_duration:.2f} seconds")
        
        # GPU memory check after processing
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"🎮 Job {job.id}: GPU memory after processing: {gpu_memory_after:.2f}GB")
        except:
            pass
            
        # Log result summary
        if isinstance(result_data, dict):
            result_keys = list(result_data.keys())
            logger.info(f"📊 Job {job.id}: Analytics returned {len(result_keys)} result fields: {result_keys}")
            
            # Log specific metrics if available
            if 'count' in result_data:
                logger.info(f"🔢 Job {job.id}: Detection count: {result_data['count']}")
            if 'total_detections' in result_data:
                logger.info(f"🔢 Job {job.id}: Total detections: {result_data['total_detections']}")
        else:
            logger.warning(f"⚠️ Job {job.id}: Analytics returned non-dict result: {type(result_data)}")

        logger.info(f"🎯 Job {job.id}: STAGE 3 - Processing output files and saving results")
        
        # --- Standardized Output Handling ---
        # The analytics functions should return a dictionary. We look for specific keys
        # to find the path of the generated output file.
        output_path_key = 'output_video' if 'output_video' in result_data else 'output_image' if 'output_image' in result_data else 'output_path'
        output_file_path = result_data.get(output_path_key)

        if output_file_path:
            logger.info(f"📁 Job {job.id}: Found output file at: {output_file_path}")
            
            # Check if output file exists and get its size
            if os.path.exists(output_file_path):
                output_size = os.path.getsize(output_file_path) / (1024*1024)  # MB
                logger.info(f"📊 Job {job.id}: Output file size: {output_size:.2f} MB")
            else:
                logger.warning(f"⚠️ Job {job.id}: Output file does not exist at path: {output_file_path}")
                
            final_output_url = save_output_and_get_url(job, output_file_path)
            if final_output_url:
                # Update the result data with the final, web-accessible URL
                result_data[output_path_key] = final_output_url
                result_data['output_path'] = final_output_url  # Ensure a consistent key for the frontend
                logger.info(f"✅ Job {job.id}: STAGE 3 COMPLETE - Output saved and accessible at: {final_output_url}")
            else:
                logger.error(f"❌ Job {job.id}: Failed to save output file to storage")
        else:
            logger.info(f"ℹ️ Job {job.id}: No output file generated (this may be normal for some job types)")

        # Final GPU memory cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"🎮 Job {job.id}: Final GPU memory after cleanup: {final_gpu_memory:.2f}GB")
        except:
            pass

        # Calculate total job duration
        total_duration = time.time() - start_time
        logger.info(f"⏱️ Job {job.id}: Total job duration: {total_duration:.2f} seconds")

        # Save results and finalize job
        job.results = result_data
        job.status = 'done'
        job.save()
        
        logger.info(f"🎉 SUCCESS: Job {job.id} ({job.job_type}) completed successfully for user {job.user.username}")
        logger.info(f"🏁 CELERY JOB {job_id} FINISHED 🏁")

    except Exception as e:
        # Enhanced error logging with context
        error_time = datetime.now().strftime('%H:%M:%S')
        logger.error(f"❌ CRITICAL ERROR in Celery job {job.id} at {error_time} ❌")
        logger.error(f"💥 Error Type: {type(e).__name__}")
        logger.error(f"💥 Error Message: {str(e)}")
        logger.error(f"💥 Job Type: {job.job_type}")
        logger.error(f"💥 User: {job.user.username}")
        logger.error(f"💥 Input File: {job.input_video.name if job.input_video else 'None'}")
        
        # Log GPU state on error
        try:
            import torch
            if torch.cuda.is_available():
                error_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                logger.error(f"💥 GPU Memory at Error: {error_gpu_memory:.2f}GB")
                torch.cuda.empty_cache()  # Try to free memory
        except:
            pass
            
        logger.error(f"💥 Full Stack Trace:", exc_info=True)
        
        job.status = 'failed'
        job.results = {
            "error": str(e), 
            "error_type": type(e).__name__,
            "error_time": error_time,
            "traceback": traceback.format_exc()
        }
        
        logger.error(f"🚫 JOB {job.id} FAILED - Status updated to 'failed' 🚫")
        
        # Re-raise the exception to let Celery know the task failed
        raise

    finally:
        # This block will always run, even if the task fails
        try:
            job.save()
            logger.info(f"💾 Job {job.id} final state saved to database")
        except Exception as save_error:
            logger.error(f"❌ Failed to save job {job.id} final state: {save_error}")
            
        # Final cleanup attempt
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
