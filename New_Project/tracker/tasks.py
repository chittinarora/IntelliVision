from celery import shared_task
from django.core.files import File
import os
from .models import VideoJob

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

        # Dynamically import the appropriate tracking function
        print(f"Job type is: {job.job_type}")
        if job.job_type == "people_count":
            from .analytics.people_count import tracking_video
        elif job.job_type == "car_count":
            from .analytics.car_count import tracking_video
        elif job.job_type == "in_out":
            from .analytics.in_out import tracking_video
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")

        # Run the actual tracking
        print("Calling tracking_video()")
        result_data = tracking_video(input_path, output_path)
        print("Back from tracking_video()!")

        # Save output video
        with open(output_path, 'rb') as out_f:
            job.output_video.save(output_filename, File(out_f))

        # Update result fields based on result_data returned
        for key, value in result_data.items():
            if hasattr(job, key):
                setattr(job, key, value)

        job.status = 'done'
        job.save()
        print(f"Job {job_id} processed and saved.")

        # Clean up
        os.remove(output_path)

    except Exception as e:
        print("=== ERROR in process_video_job ===")
        print(e)
        job.status = 'failed'
        job.save()
        raise