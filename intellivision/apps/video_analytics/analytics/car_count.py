"""
car_count.py - Video Analytics
Car counting and plate detection using FastAPI backend.
"""

import requests
import os
from django.conf import settings # To access settings
from pathlib import Path
import time

# --- FastAPI Video Detection ---
def call_fastapi_detect_video(video_file_path: str, original_filename: str):
    """
    Upload a video to FastAPI for car counting/plate detection and provide a callback URL.
    Returns a dict with job_id and status for tracking.
    """
    fastapi_url = getattr(settings, 'FASTAPI_ANPR_URL', 'http://localhost:9000')
    upload_endpoint = f"{fastapi_url}/detect/video"
    # The callback URL should be your Django endpoint that will receive the results
    callback_url = getattr(settings, 'ANPR_CALLBACK_URL', 'http://localhost:8000/api/anpr-callback/')

    # Step 1: Upload the video and provide callback_url
    with open(video_file_path, 'rb') as f:
        files = {'video': (original_filename, f, 'video/mp4')}
        data = {'callback_url': callback_url}
        try:
            print(f"Uploading video {original_filename} to FastAPI for processing with callback...")
            response = requests.post(upload_endpoint, files=files, data=data, timeout=300)
            response.raise_for_status()
            upload_results = response.json()
            print("Video uploaded to FastAPI. Processing will be handled asynchronously.")
        except requests.exceptions.RequestException as e:
            print(f"Error calling FastAPI /detect/video: {e}")
            raise

    # Return the job_id and status for tracking
    return {
        "message": "Video uploaded for processing. Results will be sent to callback endpoint.",
        "job_id": upload_results.get('job_id'),
        "status": upload_results.get('status', 'processing_started')
    }

# --- Main Tracking Pipeline ---
def tracking_video(input_path: str, output_path: str) -> dict:
    """
    Wrapper for car counting using FastAPI. Matches the interface used by other analytics modules.
    input_path: Path to the input video file.
    output_path: Path where the annotated video should be saved. (Ignored, as FastAPI determines output location)
    Returns: dict with summary, history, and local paths to downloaded results.
    """
    original_filename = os.path.basename(input_path)
    results = call_fastapi_detect_video(input_path, original_filename)
    return results
