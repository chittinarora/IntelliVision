"""
Pest monitoring analytics using YOLO and BotSort for wildlife detection (snakes).
Logs results to MongoDB and supports image and video inputs.
"""

# ======================================
# Imports and Setup
# ======================================
import cv2
import tempfile
import time
import re
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
from PIL import Image
import numpy as np
from ultralytics import YOLO
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone
import mimetypes
from celery import shared_task
import torch

# Try to import optional dependencies
try:
    from boxmot import BotSort
    BOTSORT_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("BotSort not available. Install with: pip install boxmot")
    BOTSORT_AVAILABLE = False

try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("MongoDB not available. Install with: pip install pymongo")
    MONGODB_AVAILABLE = False

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
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://toram444444:06nJTevaUItCDpd9@cluster01.lemxesc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster01")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set!")
DETECTION_CONFIDENCE = 0.4
EMBEDDER_FILE = MODELS_DIR / "osnet_ibn_x1_0_msmt17.pth"

# Define OUTPUT_DIR with fallback
try:
    OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
except AttributeError:
    logger.warning("JOB_OUTPUT_DIR not defined in settings. Using fallback: MEDIA_ROOT/outputs")
    OUTPUT_DIR = Path(settings.MEDIA_ROOT) / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MongoDB setup
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["snake_db"]
snake_collection = db["snake_detections"]

# Check model existence
MODEL_FILES = ["best_animal.pt", "osnet_ibn_x1_0_msmt17.pth"]
for model_file in MODEL_FILES:
    if not (MODELS_DIR / model_file).exists():
        logger.error(f"Model file missing: {model_file}")
        raise FileNotFoundError(f"Model file {model_file} not found in {MODELS_DIR}")

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

def get_timestamp():
    """Generate timestamp for file naming."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ======================================
# Main Analysis Functions
# ======================================

def detect_snakes_in_image(image_path: str) -> Dict:
    """
    Detect snakes in an image.

    Args:
        image_path: Path to input image

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    image_path = Path(image_path).name
    is_valid, error_msg = validate_input_file(image_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'pest_monitoring',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    try:
        # Load image
        with default_storage.open(image_path, 'rb') as f:
            img = Image.open(f).convert("RGB")
            img_array = np.array(img)

        # Run detection
        model = YOLO(str(MODELS_DIR / "best_animal.pt"))
        results = model(img_array, conf=DETECTION_CONFIDENCE)
        plotted = results[0].plot()
        num_snakes = len(results[0].boxes)

        # Collect detected animals
        detected_animals = []
        alerts = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detected_animals.append({"name": class_name, "confidence": confidence})
            alerts.append({
                "message": f"{class_name} detected with confidence {confidence:.2f}",
                "timestamp": timezone.now().isoformat()
            })

        # Save annotated image
        timestamp = get_timestamp()
        output_filename = f"outputs/detection_img_{timestamp}.jpg"
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR))
            with open(tmp.name, 'rb') as f:
                default_storage.save(output_filename, f)
        output_url = default_storage.url(output_filename)

        # Log to MongoDB
        mongo_doc = {
            "type": "image",
            "file_name": image_path,
            "detected_snakes": num_snakes,
            "detected_animals": detected_animals,
            "timestamp": datetime.now(),
            "output_url": output_url
        }
        result = snake_collection.insert_one(mongo_doc)

        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'pest_monitoring',
            'output_image': output_url,
            'output_video': None,
            'data': {
                'detected_snakes': num_snakes,
                'detected_animals': detected_animals,
                'mongo_id': str(result.inserted_id),
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
            'job_type': 'pest_monitoring',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }
    finally:
        if 'tmp' in locals() and os.path.exists(tmp.name):
            os.remove(tmp.name)

def detect_snakes_in_video(video_path: str) -> Dict:
    """
    Detect snakes in a video using YOLO and BotSort.

    Args:
        video_path: Path to input video

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    video_path = Path(video_path).name
    is_valid, error_msg = validate_input_file(video_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'pest_monitoring',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    try:
        # Open video
        with default_storage.open(video_path, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            logger.error("Failed to open video")
            return {
                'status': 'failed',
                'job_type': 'pest_monitoring',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': 'Failed to open video'},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': 'Failed to open video', 'code': 'VIDEO_READ_ERROR'}
            }

        # Video setup
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        timestamp = get_timestamp()
        job_id = re.search(r'(\d+)', video_path)
        job_id = job_id.group(1) if job_id else str(int(time.time()))
        output_filename = f"outputs/detection_vid_{job_id}.mp4"
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
            out = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            # Tracker setup
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            tracker = BotSort(
                track_high_thresh=0.15,
                new_track_thresh=0.15,
                track_buffer=30,
                device=device,
                half=False,
                reid_weights=EMBEDDER_FILE
            )
            model = YOLO(str(MODELS_DIR / "best_animal.pt"))

            total_detected = 0
            detected_animals = []
            frame_number = 0
            last_log_time = start_time
            alerts = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_number += 1

                # Detection
                results = model(frame, conf=DETECTION_CONFIDENCE, imgsz=640)
                boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = float(box.cls[0])
                    class_id = int(cls)
                    class_name = model.names[class_id]
                    boxes.append([x1, y1, x2, y2, conf, cls])
                    detected_animals.append({"name": class_name, "confidence": conf})
                    alerts.append({
                        "message": f"{class_name} detected with confidence {conf:.2f} at frame {frame_number}",
                        "timestamp": timezone.now().isoformat()
                    })
                boxes = np.array(boxes) if boxes else np.zeros((0, 6))
                tracks = tracker.update(boxes, frame)

                # Draw tracked boxes
                for t in tracks:
                    x1, y1, x2, y2, track_id = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                out.write(frame)
                total_detected += len(tracks)

                # Periodic logging
                current_time = time.time()
                if current_time - last_log_time >= 5 or frame_number == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    progress = (frame_number / cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100
                    elapsed_time = current_time - start_time
                    time_remaining = (elapsed_time / frame_number) * (cap.get(cv2.CAP_PROP_FRAME_COUNT) - frame_number) if frame_number > 0 else 0
                    avg_fps = frame_number / elapsed_time if elapsed_time > 0 else 0
                    logger.info(f"**Job {job_id}**: Progress **{progress:.1f}%** ({frame_number}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}), Status: Processing...")
                    logger.info(f"[{'#' * int(progress // 10)}{'-' * (10 - int(progress // 10))}] Done: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d} | Left: {int(time_remaining // 60):02d}:{int(time_remaining % 60):02d} | Avg FPS: {avg_fps:.1f}")
                    last_log_time = current_time

            cap.release()
            out.release()

            # Save output
            with open(tmp_out.name, 'rb') as f:
                default_storage.save(output_filename, f)
            output_url = default_storage.url(output_filename)
            web_output_filename = output_filename.replace('.mp4', '_web.mp4')
            if convert_to_web_mp4(tmp_out.name, web_output_filename):
                output_url = default_storage.url(web_output_filename)
                os.remove(tmp_out.name)

            # Log to MongoDB
            mongo_doc = {
                "type": "video",
                "file_name": video_path,
                "detected_snakes": total_detected,
                "detected_animals": detected_animals,
                "timestamp": datetime.now(),
                "output_url": output_url
            }
            result = snake_collection.insert_one(mongo_doc)

            processing_time = time.time() - start_time
            return {
                'status': 'completed',
                'job_type': 'pest_monitoring',
                'output_image': None,
                'output_video': output_url,
                'data': {
                    'detected_snakes': total_detected,
                    'detected_animals': detected_animals,
                    'mongo_id': str(result.inserted_id),
                    'alerts': alerts
                },
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'processing_time_seconds': processing_time,
                    'fps': fps,
                    'frame_count': frame_number
                },
                'error': None
            }
    except Exception as e:
        logger.exception(f"Video processing failed: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'pest_monitoring',
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
def tracking_video(self, input_path: str, output_path: str = None, job_id: str = None) -> Dict:
    """
    Celery task for pest monitoring.

    Args:
        self: Celery task instance
        input_path: Path to input video or image
        output_path: Path to save output
        job_id: VideoJob ID

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    logger.info(f"🚀 Starting pest monitoring job {job_id}")
    ext = os.path.splitext(input_path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png']
    if ext in image_exts:
        result = detect_snakes_in_image(input_path)
    else:
        result = detect_snakes_in_video(input_path)
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