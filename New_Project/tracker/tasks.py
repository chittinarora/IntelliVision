from celery import shared_task
# You only need to import tracking_video once at the top
from .track import tracking_video


@shared_task
def process_video_job(job_id):
    print("=== Celery: process_video_job CALLED with job_id:", job_id)

    # It's better to get the job object outside the try/except initially
    # so you can update its status even if setup fails.
    from .models import VideoJob
    job = VideoJob.objects.get(id=job_id)

    try:
        from django.core.files import File
        import os

        print("Loaded job:", job)

        input_path = job.input_video.path
        output_filename = f'output_{job.id}.mp4'
        output_path = os.path.join('/tmp', output_filename)  # Using a temporary directory is good practice
        print("Paths:", input_path, "->", output_path)

        print("Calling tracking_video()")
        result_data = tracking_video(input_path, output_path)
        print("Back from tracking_video()!")

        person_count = result_data.get('person_count')
        print(f"Tracking complete. Found {person_count} unique persons.")

        # Save the processed video file to the model
        with open(output_path, 'rb') as out_f:
            job.output_video.save(output_filename, File(out_f))

        # Update the job status and the new person count field
        job.status = 'done'
        job.person_count = person_count
        job.save()

        print(f"Job {job_id} processed and saved.")
        # Clean up the temporary file
        os.remove(output_path)

    except Exception as e:
        print("=== ERROR in process_video_job ===")
        print(e)
        # Update job status to 'failed' if anything goes wrong
        job.status = 'failed'
        job.save()
        raise  # Re-raise the exception so Celery knows the task failed