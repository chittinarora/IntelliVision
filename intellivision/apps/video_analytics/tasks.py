# tracker/tasks.py

from celery import shared_task
from django.core.files import File
from .models import VideoJob
from django.conf import settings
import os

from apps.video_analytics.analytics.food_waste_estimation import analyze_food_image
# Import other analytics modules as needed at the top, or dynamically within job type blocks.

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
            from .analytics.people_count import tracking_video
            result_data = tracking_video(input_path, output_path)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # --- CAR COUNT ---
        elif job.job_type == "car_count":
            from .analytics.car_count import tracking_video
            result_data = tracking_video(input_path, output_path)
            # Save the annotated video if it was downloaded
            annotated_video_path = result_data.get('annotated_video_path')
            if annotated_video_path and os.path.exists(annotated_video_path):
                with open(annotated_video_path, 'rb') as out_f:
                    job.output_video.save(os.path.basename(annotated_video_path), File(out_f))

        # --- EMERGENCY COUNT (IN/OUT, requires line coordinates) ---
        elif job.job_type == "emergency_count":
            from .analytics.emergency_count import tracking_video
            # Get line coordinates from the job
            line1_start_x = float(job.line1_start_x)
            line1_start_y = float(job.line1_start_y)
            line1_end_x = float(job.line1_end_x)
            line1_end_y = float(job.line1_end_y)
            line2_start_x = float(job.line2_start_x)
            line2_start_y = float(job.line2_start_y)
            line2_end_x = float(job.line2_end_x)
            line2_end_y = float(job.line2_end_y)
            video_width = int(job.video_width) if job.video_width else None
            video_height = int(job.video_height) if job.video_height else None

            # Construct line_coords_dict for the tracking function
            line_coords_dict = {
                'line1': ((float(line1_start_x), float(line1_start_y)), (float(line1_end_x), float(line1_end_y))),
                'line2': ((float(line2_start_x), float(line2_start_y)), (float(line2_end_x), float(line2_end_y)))
            }

            result_data = tracking_video(input_path, output_path, line_coords_dict=line_coords_dict, video_width=video_width, video_height=video_height)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # --- POTHOLE DETECTION ---
        elif job.job_type == "pothole_detection":
            from .analytics.pothole_detection import tracking_video, run_pothole_image_detection
            ext = os.path.splitext(input_path)[1].lower()
            image_exts = ['.jpg', '.jpeg', '.png']
            if ext in image_exts:
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
            result_data = analyze_food_image(input_path)
            job.output_video = None

        # --- PEST MONITORING (IMAGES & VIDEOS) ---
        elif job.job_type == "pest_monitoring":
            from .analytics.pest_monitoring import tracking_video
            result_data = tracking_video(input_path, output_path)
            ext = os.path.splitext(input_path)[1].lower()
            image_exts = ['.jpg', '.jpeg', '.png']
            output_file_path = os.path.join(settings.MEDIA_ROOT, result_data['output_path'])

            if ext in image_exts:
                output_filename = f'output_{job.id}.jpg'
                with open(output_file_path, 'rb') as out_f:
                    job.output_image.save(output_filename, File(out_f))
                job.output_video = None
                result_data['media_type'] = 'image'
                result_data['output_image'] = job.output_image.url
                result_data['output_url'] = job.output_image.url
            else:
                output_filename = f'output_{job.id}.mp4'
                with open(output_file_path, 'rb') as out_f:
                    job.output_video.save(output_filename, File(out_f))
                job.output_image = None
                result_data['media_type'] = 'video'
                result_data['output_video'] = job.output_video.url
                result_data['output_url'] = job.output_video.url

        # --- WILDLIFE DETECTION ---
        elif job.job_type == "wildlife_detection":
            from .analytics.wildlife_detection import tracking_video
            result_data = tracking_video(input_path, output_path)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # --- UNKNOWN JOB TYPE ---
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")

        # --- FINALIZE ---
        job.results = result_data
        job.status = 'done'
        job.save()

        # Clean up temp file for non-pest_monitoring
        if job.job_type not in ["pest_monitoring"] and os.path.exists(output_path):
            os.remove(output_path)

    except Exception as e:
        print("=== ERROR in process_video_job ===")
        print(e)
        job.status = 'failed'
        job.save()
        raise
