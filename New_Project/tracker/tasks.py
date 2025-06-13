from celery import shared_task

@shared_task
def process_video_job(job_id):
    print("=== Celery: process_video_job CALLED with job_id:", job_id)

    try:
        from .models import VideoJob
        from .track import tracking_video  # make sure this is the right import!
        from django.core.files import File
        import os

        job = VideoJob.objects.get(id=job_id)
        print("Loaded job:", job)

        input_path = job.input_video.path
        output_filename = f'output_{job.id}.mp4'
        output_path = os.path.join('/tmp', output_filename)
        print("Paths:", input_path, "->", output_path)

        print("Calling process_video()")
        tracking_video(input_path, output_path)
        print("Back from process_video()!")

        with open(output_path, 'rb') as out_f:
            job.output_video.save(output_filename, File(out_f))
        job.status = 'done'
        job.save()
        print(f"Job {job_id} processed and saved.")
        os.remove(output_path)

    except Exception as e:
        print("=== ERROR in process_video_job ===")
        print(e)
        job = VideoJob.objects.get(id=job_id)
        job.status = 'failed'
        job.save()
        raise