# tracker/tasks.py

from celery import shared_task
from django.core.files import File
from .models import VideoJob
from django.conf import settings
import os
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

# All top-level analytics imports that use PyTorch/MPS have been removed.
# They will be imported dynamically inside the task.

@shared_task
def process_video_job(job_id):
    print(f"=== Celery: process_video_job CALLED with job_id: {job_id}")
    job = VideoJob.objects.get(id=job_id)

    try:
        job.status = 'processing'
        job.save()

        input_path = job.input_video.path
        output_filename = f'output_{job.id}.mp4'
        output_path = os.path.join('/tmp', output_filename)

        print(f"Job type is: {job.job_type}")

        # --- PEOPLE COUNT ---
        if job.job_type == "people_count":
            # Lazy import is correct here
            from .analytics.people_count import tracking_video
            result_data = tracking_video(input_path, output_path)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # --- CAR COUNT ---
        elif job.job_type == "car_count":
            # Lazy import is correct here
            from .analytics.car_count import tracking_video
            result_data = tracking_video(input_path, output_path)
            annotated_video_path = result_data.get('annotated_video_path')
            if annotated_video_path and os.path.exists(annotated_video_path):
                with open(annotated_video_path, 'rb') as out_f:
                    job.output_video.save(os.path.basename(annotated_video_path), File(out_f))

        # --- EMERGENCY COUNT ---
        elif job.job_type == "emergency_count":
            # Lazy import is correct here
            from .analytics.emergency_count import tracking_video
            # Build line_coords_dict from emergency_lines JSONField
            line_coords_dict = {}
            if job.emergency_lines:
                for idx, line in enumerate(job.emergency_lines, start=1):
                    line_key = f"line{idx}"
                    line_coords_dict[line_key] = (
                        (float(line.get("start_x", 0)), float(line.get("start_y", 0))),
                        (float(line.get("end_x", 0)), float(line.get("end_y", 0)))
                    )
            video_width = int(job.video_width) if job.video_width else None
            video_height = int(job.video_height) if job.video_height else None
            result_data = tracking_video(input_path, output_path, line_coords_dict=line_coords_dict, video_width=video_width, video_height=video_height)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # --- POTHOLE DETECTION ---
        elif job.job_type == "pothole_detection":
            # Lazy import is correct here
            from .analytics.pothole_detection import tracking_video, run_pothole_image_detection
            ext = os.path.splitext(input_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                output_filename = f'output_{job.id}.jpg'
                output_path = os.path.join('/tmp', output_filename)
                result_data = run_pothole_image_detection(input_path, output_path)
                with open(output_path, 'rb') as out_f:
                    job.output_video.save(output_filename, File(out_f))
            else:
                result_data = tracking_video(input_path, output_path)
                with open(output_path, 'rb') as out_f:
                    job.output_video.save(output_filename, File(out_f))

        # --- FOOD WASTE ESTIMATION ---
        elif job.job_type == "food_waste_estimation":
            # FIX APPLIED: Moved the import inside the elif block
            from .analytics.food_waste_estimation import analyze_food_image
            result_data = analyze_food_image(input_path)
            job.output_video = None

        # --- ROOM READINESS ---
        elif job.job_type == "room_readiness":
            # Lazy import is correct here
            from .analytics.room_readiness import analyze_room_image
            result_data = analyze_room_image(input_path)
            job.output_video = None

        # --- PEST MONITORING ---
        elif job.job_type == "pest_monitoring":
            # Lazy import is correct here
            from .analytics.pest_monitoring import tracking_video
            result_data = tracking_video(input_path, output_path)
            ext = os.path.splitext(input_path)[1].lower()
            output_file_path = os.path.join(settings.MEDIA_ROOT, result_data['output_path'])
            if ext in ['.jpg', '.jpeg', '.png']:
                output_filename = f'output_{job.id}.jpg'
                with open(output_file_path, 'rb') as out_f:
                    job.output_image.save(output_filename, File(out_f))
                job.output_video = None
                result_data['media_type'] = 'image'
                result_data['output_url'] = job.output_image.url
            else:
                output_filename = f'output_{job.id}.mp4'
                with open(output_file_path, 'rb') as out_f:
                    job.output_video.save(output_filename, File(out_f))
                job.output_image = None
                result_data['media_type'] = 'video'
                result_data['output_url'] = job.output_video.url

        # --- WILDLIFE DETECTION ---
        elif job.job_type == "wildlife_detection":
            # Lazy import is correct here
            from .analytics.wildlife_detection import tracking_video
            result_data = tracking_video(input_path, output_path)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # --- LOBBY DETECTION / CROWD ANALYSIS ---
        elif job.job_type == "lobby_detection":
            from .analytics.lobby_detection import run_crowd_analysis
            zone_configs = {}
            if job.lobby_zones:
                for idx, zone in enumerate(job.lobby_zones):
                    name = zone.get('name', f'Zone{idx+1}')
                    zone_configs[name] = {
                        'points': zone['points'],
                        'threshold': zone['threshold']
                    }
            output_path = f"/tmp/output_{job.id}.mp4"
            result_data = run_crowd_analysis(input_path, zone_configs, output_path=output_path)
            # Save video to media storage
            with open(result_data['output_video_path'], 'rb') as out_f:
                saved_name = f"outputs/output_{job.id}.mp4"
                saved_path = default_storage.save(saved_name, ContentFile(out_f.read()))
                output_url = default_storage.url(saved_path)
            job.output_video = output_url
            job.status = 'done'
            # Set results to include people counts and alerts
            job.results = {
                "zone_counts": result_data.get("zone_counts", {}),
                "alerts": result_data.get("alerts", []),
                "output_video": output_url
            }
            job.save()

        # --- UNKNOWN JOB TYPE ---
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")

        # --- FINALIZE ---
        job.results = result_data
        job.status = 'done'
        job.save()

        if job.job_type not in ["pest_monitoring"] and os.path.exists(output_path):
            os.remove(output_path)

    except Exception as e:
        print("=== ERROR in process_video_job ===")
        print(e)
        job.status = 'failed'
        job.save()
        raise
