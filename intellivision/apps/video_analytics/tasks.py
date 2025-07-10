# tracker/tasks.py

# =============================
# Imports and Setup
# =============================
# Celery for async task management
from celery import shared_task
# Django file handling and settings
from django.core.files import File
from apps.video_analytics.models import VideoJob
from django.conf import settings
import os
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from PIL import Image
import io
from pathlib import Path
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)

# All top-level analytics imports that use PyTorch/MPS have been removed.
# They will be imported dynamically inside the task.

# =============================
# Celery Task: process_video_job
# =============================
# This task processes a video analytics job based on its type.
# It dynamically imports the required analytics module, runs the analysis,
# saves the output to the appropriate media folder, and updates the job status/results.
@shared_task
def process_video_job(job_id):
    """
    Celery task to process a video analytics job based on its type.
    Dynamically imports the required analytics module, runs the analysis,
    saves the output to the appropriate media folder, and updates the job status/results.

    Args:
        job_id (int): The ID of the VideoJob to process.
    """
    logger.info(f"=== Celery: process_video_job CALLED with job_id: {job_id}")
    job = VideoJob.objects.get(id=job_id)

    result_data = None  # Initialize to avoid UnboundLocalError
    try:
        # Set job status to processing
        job.status = 'processing'
        job.save()

        # Prepare input and output paths
        input_path = job.input_video.path
        output_filename = f'output_{job.id}.mp4'
        output_path = os.path.join('/tmp', output_filename)

        logger.info(f"Job type is: {job.job_type}")

        # =============================
        # PEOPLE COUNT
        # =============================
        if job.job_type == "people_count":
            # Lazy import for people counting
            from apps.video_analytics.analytics.people_count import tracking_video
            # Run people counting
            result_data = tracking_video(input_path, output_path)
            output_file_path = result_data['output_video']
            actual_filename = os.path.basename(output_file_path)
            saved_name = f"outputs/{actual_filename}"
            # Save output video to media/outputs
            with open(output_file_path, 'rb') as out_f:
                saved_path = default_storage.save(saved_name, ContentFile(out_f.read()))
                output_url = default_storage.url(saved_path)
            job.output_video.name = saved_name  # Store relative path only
            result_data['output_video'] = output_url
            job.results = {**result_data, "output_path": output_url}

        # =============================
        # EMERGENCY COUNT
        # =============================
        elif job.job_type == "emergency_count":
            # Lazy import for emergency counting
            from apps.video_analytics.analytics.emergency_count import tracking_video
            # Build line_coords_dict from emergency_lines JSONField
            line_coords_dict = {}
            if job.emergency_lines:
                for idx, line in enumerate(job.emergency_lines, start=1):
                    line_key = f"line{idx}"
                    start_x = float(line.get("start_x", 0))
                    start_y = float(line.get("start_y", 0))
                    end_x = float(line.get("end_x", 0))
                    end_y = float(line.get("end_y", 0))
                    # Scale normalized coordinates to pixel values if needed
                    if 0 <= start_x <= 1 and 0 <= start_y <= 1 and 0 <= end_x <= 1 and 0 <= end_y <= 1 and job.video_width and job.video_height:
                        start_x *= job.video_width
                        end_x *= job.video_width
                        start_y *= job.video_height
                        end_y *= job.video_height
                    line_coords_dict[line_key] = {
                        "coords": (
                            (start_x, start_y),
                            (end_x, end_y)
                        ),
                        "inDirection": line.get("inDirection", "UP")
                    }
            video_width = int(job.video_width) if job.video_width else None
            video_height = int(job.video_height) if job.video_height else None
            # Run emergency counting
            result_data = tracking_video(input_path, output_path, line_coords_dict=line_coords_dict, video_width=video_width, video_height=video_height)
            output_file_path = result_data['output_video']
            actual_filename = os.path.basename(output_file_path)
            saved_name = f"outputs/{actual_filename}"
            # Save output video to media/outputs
            with open(output_file_path, 'rb') as out_f:
                saved_path = default_storage.save(saved_name, ContentFile(out_f.read()))
                output_url = default_storage.url(saved_path)
            job.output_video.name = saved_name  # Store relative path only
            result_data['output_video'] = output_url
            job.results = {**result_data, "output_path": output_url}

        # =============================
        # CAR COUNT
        # =============================
        elif job.job_type == "car_count":
            from apps.video_analytics.analytics.car_count import recognize_number_plates
            MODELS_DIR = Path(__file__).resolve().parent / 'models'
            input_filename = os.path.basename(input_path)
            models_input_path = MODELS_DIR / input_filename
            if not models_input_path.exists():
                import shutil
                shutil.copy(input_path, models_input_path)
            raw_result = recognize_number_plates(input_filename)
            output_file_path = raw_result.get('output_video')
            output_url = None
            if output_file_path and os.path.exists(output_file_path):
                actual_filename = os.path.basename(output_file_path)
                saved_name = f"outputs/{actual_filename}"
                with open(output_file_path, 'rb') as out_f:
                    saved_path = default_storage.save(saved_name, ContentFile(out_f.read()))
                    output_url = default_storage.url(saved_path)
                job.output_video.name = saved_name
            # Unified result format
            result_data = {
                'status': raw_result.get('status', 'error'),
                'job_type': 'car_count',
                'output_video': output_url,
                'output_path': output_url,
                'data': raw_result.get('summary', {}),
                'meta': {},
                'error': None if raw_result.get('status') == 'completed' else raw_result.get('message', 'Error')
            }
            job.results = result_data

        # =============================
        # PARKING ANALYSIS
        # =============================
        elif job.job_type == "parking_analysis":
            from apps.video_analytics.analytics.car_count import analyze_parking_video
            MODELS_DIR = Path(__file__).resolve().parent / 'models'
            input_filename = os.path.basename(input_path)
            models_input_path = MODELS_DIR / input_filename
            if not models_input_path.exists():
                import shutil
                shutil.copy(input_path, models_input_path)
            raw_result = analyze_parking_video(input_filename)
            output_file_path = raw_result.get('output_video')
            output_url = None
            if output_file_path and os.path.exists(output_file_path):
                actual_filename = os.path.basename(output_file_path)
                saved_name = f"outputs/{actual_filename}"
                with open(output_file_path, 'rb') as out_f:
                    saved_path = default_storage.save(saved_name, ContentFile(out_f.read()))
                    output_url = default_storage.url(saved_path)
                job.output_video.name = saved_name
            result_data = {
                'status': raw_result.get('status', 'error'),
                'job_type': 'parking_analysis',
                'output_video': output_url,
                'output_path': output_url,
                'data': raw_result.get('summary', {}),
                'meta': {},
                'error': None if raw_result.get('status') == 'completed' else raw_result.get('message', 'Error')
            }
            job.results = result_data

        # =============================
        # POTHOLE DETECTION
        # =============================
        elif job.job_type == "pothole_detection":
            # Lazy import for pothole detection
            from apps.video_analytics.analytics.pothole_detection import tracking_video, run_pothole_image_detection
            ext = os.path.splitext(input_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                # Image input: run pothole image detection
                output_filename = f'output_{job.id}.jpg'
                output_path = os.path.join('/tmp', output_filename)
                result_data = run_pothole_image_detection(input_path, output_path)
                output_file_path = result_data['output_video'] if 'output_video' in result_data else output_path
                actual_filename = os.path.basename(output_file_path)
                saved_name = f"outputs/{actual_filename}"
                # Save output image to media/outputs
                with open(output_file_path, 'rb') as out_f:
                    saved_path = default_storage.save(saved_name, ContentFile(out_f.read()))
                    output_url = default_storage.url(saved_path)
                job.output_video.name = saved_name  # Store relative path only
                result_data['output_video'] = output_url
                job.results = {**result_data, "output_path": output_url}
            else:
                # Video input: run pothole video detection
                result_data = tracking_video(input_path, output_path)
                output_file_path = result_data['output_video']
                actual_filename = os.path.basename(output_file_path)
                saved_name = f"outputs/{actual_filename}"
                # Save output video to media/outputs
                with open(output_file_path, 'rb') as out_f:
                    saved_path = default_storage.save(saved_name, ContentFile(out_f.read()))
                    output_url = default_storage.url(saved_path)
                job.output_video.name = saved_name  # Store relative path only
                result_data['output_video'] = output_url
                job.results = {**result_data, "output_path": output_url}

        # =============================
        # FOOD WASTE ESTIMATION
        # =============================
        elif job.job_type == "food_waste_estimation":
            # Lazy import for food waste estimation
            from apps.video_analytics.analytics.food_waste_estimation import analyze_food_image
            # Run food waste estimation (image only)
            result_data = analyze_food_image(input_path)
            job.output_video = None

        # =============================
        # ROOM READINESS
        # =============================
        elif job.job_type == "room_readiness":
            # Lazy import for room readiness
            from apps.video_analytics.analytics.room_readiness import analyze_room_image, analyze_room_video_multi_zone_only
            ext = os.path.splitext(input_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                # Image input: analyze directly
                frame_result = analyze_room_image(input_path)
                # Save analyzed image to outputs and get URL
                with open(input_path, 'rb') as f:
                    saved_name = f"outputs/roomreadiness_{job.id}_frame1.jpg"
                    saved_path = default_storage.save(saved_name, ContentFile(f.read()))
                    output_url = default_storage.url(saved_path)
                # Place output image path inside the frame dict
                frame_result['output_image'] = output_url
                # Set job.results to the unified format (not wrapped in a list)
                result_data = frame_result
                job.output_video = None
            else:
                # Video input: analyze using multi-zone function (returns unified format)
                result_data = analyze_room_video_multi_zone_only(input_path)
                job.output_video = None

        # =============================
        # PEST MONITORING
        # =============================
        elif job.job_type == "pest_monitoring":
            # Lazy import for pest monitoring
            from apps.video_analytics.analytics.pest_monitoring import tracking_video
            # Run pest monitoring (image or video)
            result_data = tracking_video(input_path, output_path)
            ext = os.path.splitext(input_path)[1].lower()
            output_file_path = os.path.join(settings.MEDIA_ROOT, result_data['output_path'])
            if ext in ['.jpg', '.jpeg', '.png']:
                # Image output
                output_filename = f'output_{job.id}.jpg'
                with open(output_file_path, 'rb') as out_f:
                    job.output_image.save(output_filename, File(out_f))
                job.output_video = None
                result_data['media_type'] = 'image'
                result_data['output_url'] = job.output_image.url
            else:
                # Video output
                output_filename = f'output_{job.id}.mp4'
                with open(output_file_path, 'rb') as out_f:
                    job.output_video.save(output_filename, File(out_f))
                job.output_image = None
                result_data['media_type'] = 'video'
                result_data['output_url'] = job.output_video.url

        # =============================
        # WILDLIFE DETECTION
        # =============================
        elif job.job_type == "wildlife_detection":
            # Lazy import for wildlife detection
            from apps.video_analytics.analytics.wildlife_detection import tracking_video
            # Run wildlife detection
            result_data = tracking_video(input_path, output_path)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # =============================
        # LOBBY DETECTION / CROWD ANALYSIS
        # =============================
        elif job.job_type == "lobby_detection":
            # Lazy import for lobby/crowd analysis
            from apps.video_analytics.analytics.lobby_detection import run_crowd_analysis
            # Build zone configs from lobby_zones JSONField
            zone_configs = {}
            if job.lobby_zones:
                for idx, zone in enumerate(job.lobby_zones):
                    name = zone.get('name', f'Zone{idx+1}')
                    zone_configs[name] = {
                        'points': zone['points'],
                        'threshold': zone['threshold']
                    }
            output_path = f"/tmp/output_{job.id}.mp4"
            # Run crowd analysis
            result_data = run_crowd_analysis(input_path, zone_configs, output_path=output_path)
            output_file_path = result_data['output_video']
            actual_filename = os.path.basename(output_file_path)
            saved_name = f"outputs/{actual_filename}"
            # Save output video to media/outputs
            with open(output_file_path, 'rb') as out_f:
                saved_path = default_storage.save(saved_name, ContentFile(out_f.read()))
                output_url = default_storage.url(saved_path)
            job.output_video.name = saved_name  # Store relative path only
            result_data['output_video'] = output_url
            job.results = {**result_data, "output_path": output_url}

        # =============================
        # UNKNOWN JOB TYPE
        # =============================
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")

        # =============================
        # FINALIZE JOB
        # =============================
        # Save results and mark job as done
        if result_data is not None:
            job.results = result_data
        job.status = 'done'
        job.save()

        # Clean up temporary output file (except for pest_monitoring jobs)
        if job.job_type not in ["pest_monitoring"] and os.path.exists(output_path):
            os.remove(output_path)

    except Exception as e:
        logger.error("=== ERROR in process_video_job ===")
        logger.error(e)
        job.status = 'failed'
        job.save()
        raise
