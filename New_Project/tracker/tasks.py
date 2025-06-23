from celery import shared_task
from django.core.files import File
import os
from .models import VideoJob
from .analytics.food_waste_estimation import analyze_food_image

"""
Celery tasks for processing video jobs in the tracker app.
Handles video analysis and result saving for various job types.
"""

@shared_task
def process_video_job(job_id):
    """
    Celery task to process a VideoJob by job_id. Handles different job types and saves results/output video.
    """
    print(f"=== Celery: process_video_job CALLED with job_id: {job_id}")
    job = VideoJob.objects.get(id=job_id)

    try:
        job.status = 'processing'
        job.save()

        input_path = job.input_video.path
        output_filename = f'output_{job.id}.mp4'
        output_path = os.path.join('/tmp', output_filename)

        print(f"Job type is: {job.job_type}")

        # Select and run the appropriate analytics pipeline based on job type
        # PEOPLE COUNT
        if job.job_type == "people_count":
            from .analytics.people_count import tracking_video
            result_data = tracking_video(input_path, output_path)
            # Save output video file
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # CAR COUNT
        elif job.job_type == "car_count":
            from .analytics.car_count import tracking_video
            result_data = tracking_video(input_path, output_path)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # EMERGENCY COUNT (IN/OUT, requires ROI)
        elif job.job_type == "emergency_count":
            from .analytics.emergency_count import tracking_video

            # Extract and validate ROI fields
            roi_x = float(job.roi_x)
            roi_y = float(job.roi_y)
            roi_width = float(job.roi_width)
            roi_height = float(job.roi_height)

            # Denormalize ROI to pixel coordinates
            RESIZED_WIDTH = 1280
            RESIZED_HEIGHT = 720

            x1 = int(roi_x * RESIZED_WIDTH)
            y1 = int(roi_y * RESIZED_HEIGHT)
            x2 = int((roi_x + roi_width) * RESIZED_WIDTH)
            y2 = int((roi_y + roi_height) * RESIZED_HEIGHT)
            roi = ((x1, y1), (x2, y2))

            print(f"Using ROI pixel coordinates: {roi}")

            result_data = tracking_video(input_path, output_path, roi=roi)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # POTHOLE DETECTION
        elif job.job_type == "pothole_detection":
            from .analytics.pothole_detection import tracking_video
            result_data = tracking_video(input_path, output_path)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # FOOD WASTE ESTIMATION
        elif job.job_type == "food_waste_estimation":
            # Use the new OpenAI-based food image analysis
            result_data = analyze_food_image(input_path)
            # No output video for food_waste_estimation jobs
            job.output_video = None

        # PEST MONITORING
        elif job.job_type == "pest_monitoring":
            from .analytics.pest_monitoring import tracking_video
            result_data = tracking_video(input_path, output_path)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        # WILDLIFE DETECTION
        elif job.job_type == "wildlife_detection":
            from .analytics.wildlife_detection import tracking_video
            result_data = tracking_video(input_path, output_path)
            with open(output_path, 'rb') as out_f:
                job.output_video.save(output_filename, File(out_f))

        else:
            raise ValueError(f"Unknown job type: {job.job_type}")

        print("Back from job processing, saving output video and results.")

        # Save results to job
        job.results = result_data
        job.status = 'done'
        job.save()
        print(f"Job {job_id} processed and saved.")

        # Clean up output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)

    except Exception as e:
        print("=== ERROR in process_video_job ===")
        print(e)
        job.status = 'failed'
        job.save()
        raise
