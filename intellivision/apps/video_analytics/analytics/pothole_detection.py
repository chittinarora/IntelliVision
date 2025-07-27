"""
Pothole detection analytics using Roboflow HTTP API.
Supports both video and image input.
"""

# ======================================
# Imports and Setup
# ======================================
import cv2
import os
import base64
import tempfile
import time
import re
import logging

import numpy as np
import requests
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone
import mimetypes
from celery import shared_task

# Try to import convert function, handle gracefully if not available
try:
    from ..convert import convert_to_web_mp4
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("convert_to_web_mp4 not available. Video conversion will be skipped.")
    def convert_to_web_mp4(input_path, output_filename):
        """Fallback function when convert module is not available."""
        return False

# ======================================
# Logger and Constants
# ======================================
logger = logging.getLogger(__name__)
VALID_EXTENSIONS = {'.mp4', '.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ROBOFLOW_API_URL = "https://detect.roboflow.com/pothole-voxrl/1"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
if not ROBOFLOW_API_KEY:
    logger.warning("ROBOFLOW_API_KEY not set. Pothole detection may fail.")
MAX_SIZE = 1024

# Define OUTPUT_DIR with fallback
try:
    OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
except AttributeError:
    logger.warning("JOB_OUTPUT_DIR not defined in settings. Using fallback: MEDIA_ROOT/outputs")
    OUTPUT_DIR = Path(settings.MEDIA_ROOT) / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================
# Helper Functions
# ======================================

def validate_input_file(file_path: str) -> tuple[bool, str]:
    """Validate file type and size."""
    file_path = Path(file_path).name
    if not default_storage.exists(file_path):
        return False, f"File not found: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        return False, f"Invalid file type: {ext}. Allowed: {', '.join(VALID_EXTENSIONS)}"
    size = default_storage.size(file_path)
    if size > MAX_FILE_SIZE:
        return False, f"File size {size / (1024*1024):.2f}MB exceeds 500MB limit"
    return True, ""

def resize_frame_if_needed(frame):
    """Resize frame if it exceeds maximum dimensions."""
    h, w = frame.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame

def send_frame_to_roboflow(frame) -> List[Dict[str, Any]]:
    """Send frame to Roboflow API for pothole detection."""
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        base64_encoded_image = base64.b64encode(img_encoded).decode('utf-8')
        response = requests.post(
            ROBOFLOW_API_URL,
            params={"api_key": ROBOFLOW_API_KEY},
            data=base64_encoded_image,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        result = response.json()
        return result.get("predictions", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Roboflow API request failed: {str(e)}")
        return []

# ======================================
# Main Analysis Functions
# ======================================

def run_pothole_image_detection(input_path: str) -> Dict[str, Any]:
    """
    Process an image for pothole detection.

    Args:
        input_path: Path to input image

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    input_path = Path(input_path).name
    is_valid, error_msg = validate_input_file(input_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'pothole_detection',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    try:
        # Load image
        with default_storage.open(input_path, 'rb') as f:
            frame = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Failed to read image")
            return {
                'status': 'failed',
                'job_type': 'pothole_detection',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': 'Failed to read image'},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': 'Failed to read image', 'code': 'IMAGE_READ_ERROR'}
            }

        # Process image
        resized_frame = resize_frame_if_needed(frame)
        predictions = send_frame_to_roboflow(resized_frame)
        potholes = [
            {
                "x": float(pred['x']),
                "y": float(pred['y']),
                "width": float(pred['width']),
                "height": float(pred['height']),
                "confidence": float(pred.get('confidence', 0)),
                "class": pred.get('class', 'pothole')
            }
            for pred in predictions if pred.get('confidence', 0) >= 0.1
        ]
        alerts = [{"message": f"Pothole detected at ({p['x']}, {p['y']}) with confidence {p['confidence']:.2f}",
                   "timestamp": timezone.now().isoformat()} for p in potholes]

        # Annotate image
        for pred in potholes:
            x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (x + w // 2, y + h // 2)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)
            cv2.putText(frame, pred['class'], (top_left[0], top_left[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"Total Potholes: {len(potholes)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save output
        job_id = re.search(r'(\d+)', input_path)
        job_id = job_id.group(1) if job_id else str(int(time.time()))
        output_filename = f"outputs/pothole_{job_id}.jpg"
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            with open(tmp.name, 'rb') as f:
                default_storage.save(output_filename, f)
        output_url = default_storage.url(output_filename)

        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'pothole_detection',
            'output_image': output_url,
            'output_video': None,
            'data': {
                'total_potholes': len(potholes),
                'potholes': potholes,
                'alerts': alerts
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': 1
            },
            'error': None
        }
    except Exception as e:
        logger.exception(f"Image processing failed: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'pothole_detection',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }
    finally:
        if 'tmp' in locals() and os.path.exists(tmp.name):
            os.remove(tmp.name)

def run_pothole_detection(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Process a video for pothole detection using Roboflow API.

    Args:
        input_path: Path to input video
        output_path: Path to save output video

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    input_path = Path(input_path).name
    is_valid, error_msg = validate_input_file(input_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'pothole_detection',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    try:
        # Open video
        with default_storage.open(input_path, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            logger.error("Failed to open video")
            return {
                'status': 'failed',
                'job_type': 'pothole_detection',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': 'Failed to open video'},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': 'Failed to open video', 'code': 'VIDEO_READ_ERROR'}
            }

        orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(3)), int(cap.get(4))
        FRAME_SKIP = 5
        output_fps = max(1, orig_fps // FRAME_SKIP)
        job_id = re.search(r'(\d+)', input_path)
        job_id = job_id.group(1) if job_id else str(int(time.time()))
        output_filename = output_path if output_path else f"outputs/pothole_{job_id}.mp4"
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
            out = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))
            MAX_FRAMES = 200
            frame_idx = 0
            processed_frames = 0
            total_potholes = 0
            frame_details = []
            last_log_time = start_time
            alerts = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or processed_frames >= MAX_FRAMES:
                    logger.info("Done processing frames or reached max frame limit.")
                    break
                if frame_idx % FRAME_SKIP != 0:
                    frame_idx += 1
                    continue
                logger.debug(f"Processing frame {frame_idx}")
                resized_frame = resize_frame_if_needed(frame)
                predictions = send_frame_to_roboflow(resized_frame)
                potholes_in_frame = [
                    {
                        "x": float(pred['x']),
                        "y": float(pred['y']),
                        "width": float(pred['width']),
                        "height": float(pred['height']),
                        "confidence": float(pred.get('confidence', 0)),
                        "class": pred.get('class', 'pothole')
                    }
                    for pred in predictions if pred.get('confidence', 0) >= 0.1
                ]
                total_potholes += len(potholes_in_frame)
                frame_details.append({
                    "frame_index": frame_idx,
                    "potholes": potholes_in_frame
                })
                for pothole in potholes_in_frame:
                    alerts.append({
                        "message": f"Pothole detected at ({pothole['x']}, {pothole['y']}) with confidence {pothole['confidence']:.2f}",
                        "timestamp": timezone.now().isoformat()
                    })
                for pred in potholes_in_frame:
                    x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                    top_left = (x - w // 2, y - h // 2)
                    bottom_right = (x + w // 2, y + h // 2)
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)
                    cv2.putText(frame, pred['class'], (top_left[0], top_left[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, f"Total Potholes: {total_potholes}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                out.write(frame)
                processed_frames += 1
                frame_idx += 1

                # Periodic logging
                current_time = time.time()
                if current_time - last_log_time >= 5 or processed_frames >= MAX_FRAMES or not ret:
                    progress = (frame_idx / (MAX_FRAMES * FRAME_SKIP)) * 100 if MAX_FRAMES > 0 else 100
                    elapsed_time = current_time - start_time
                    time_remaining = (elapsed_time / frame_idx) * (MAX_FRAMES * FRAME_SKIP - frame_idx) if frame_idx > 0 else 0
                    avg_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
                    logger.info(f"**Job {job_id}**: Progress **{progress:.1f}%** ({frame_idx}/{MAX_FRAMES * FRAME_SKIP}), Status: Processing...")
                    logger.info(f"[{'#' * int(progress // 10)}{'-' * (10 - int(progress // 10))}] Done: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d} | Left: {int(time_remaining // 60):02d}:{int(time_remaining % 60):02d} | Avg FPS: {avg_fps:.1f}")
                    last_log_time = current_time

            cap.release()
            out.release()
            with open(tmp_out.name, 'rb') as f:
                default_storage.save(output_filename, f)
            output_url = default_storage.url(output_filename)
            web_output_filename = output_filename.replace('.mp4', '_web.mp4')
            if convert_to_web_mp4(tmp_out.name, web_output_filename):
                output_url = default_storage.url(web_output_filename)
                os.remove(tmp_out.name)
            processing_time = time.time() - start_time
            return {
                'status': 'completed',
                'job_type': 'pothole_detection',
                'output_image': None,
                'output_video': output_url,
                'data': {
                    'total_potholes': total_potholes,
                    'frames': frame_details,
                    'alerts': alerts
                },
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'processing_time_seconds': processing_time,
                    'fps': output_fps,
                    'frame_count': processed_frames
                },
                'error': None
            }
    except Exception as e:
        logger.exception(f"Video processing failed: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'pothole_detection',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }
    finally:
        if 'cap' in locals() and cap:
            cap.release()
        if 'out' in locals() and out:
            out.release()
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if 'tmp_out' in locals() and os.path.exists(tmp_out.name):
            os.remove(tmp_out.name)

# ======================================
# Celery Integration
# ======================================

@shared_task(bind=True)
def tracking_video(self, input_path: str, output_path: str = None, job_id: str = None) -> Dict[str, Any]:
    """
    Celery task for pothole detection.

    Args:
        self: Celery task instance
        input_path: Path to input video or image
        output_path: Path to save output
        job_id: VideoJob ID

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    logger.info(f"🚀 Starting pothole detection job {job_id}")
    ext = os.path.splitext(input_path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png']
    if ext in image_exts:
        result = run_pothole_image_detection(input_path)
    else:
        result = run_pothole_detection(input_path, output_path)
    self.update_state(
        state='PROGRESS',
        meta={
            'progress': 100.0,
            'time_remaining': 0,
            'frame': result['meta'].get('frame_count', 1),
            'total_frames': result['meta'].get('frame_count', 1),
            'status': result['status'],
            'job_id': job_id
        }
    )
    processing_time = time.time() - start_time
    result['meta']['processing_time_seconds'] = processing_time
    result['meta']['timestamp'] = timezone.now().isoformat()
    logger.info(f"**Job {job_id}**: Progress **100.0%** ({result['meta'].get('frame_count', 1)}/{result['meta'].get('frame_count', 1)}), Status: {result['status']}...")
    logger.info(f"[##########] Done: {int(processing_time // 60):02d}:{int(processing_time % 60):02d} | Left: 00:00 | Avg FPS: {result['meta'].get('fps', 'N/A')}")
    return result