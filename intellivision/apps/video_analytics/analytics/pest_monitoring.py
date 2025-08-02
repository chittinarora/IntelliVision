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

# Import progress logger
try:
    from ..progress_logger import create_progress_logger
except ImportError:
    def create_progress_logger(job_id, total_items, job_type, logger_name=None):
        """Fallback progress logger if module not available."""
        class DummyLogger:
            def __init__(self, job_id, total_items, job_type, logger_name=None):
                self.job_id = job_id
                self.total_items = total_items
                self.job_type = job_type
                self.logger = logging.getLogger(logger_name or job_type)

            def update_progress(self, processed_count, status=None, force_log=False):
                self.logger.info(f"Job {self.job_id}: Progress {processed_count}/{self.total_items}")

            def log_completion(self, final_count=None):
                self.logger.info(f"Job {self.job_id}: Completed {self.job_type}")

        return DummyLogger(job_id, total_items, job_type, logger_name)

VALID_EXTENSIONS = {'.mp4', '.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# ======================================
# Model Management Integration
# ======================================
try:
    from .model_manager import get_model_with_fallback

    # Get model paths with automatic fallback
    YOLO_MODEL_PATH = str(get_model_with_fallback("best_animal"))  # Use best_animal for wildlife detection
    REID_MODEL_PATH = str(get_model_with_fallback("osnet_reid"))

    logger.info(f"Resolved YOLO model: {YOLO_MODEL_PATH}")
    logger.info(f"Resolved Re-ID model: {REID_MODEL_PATH}")

except Exception as e:
    logger.error(f"Failed to resolve models with fallback: {e}")
    # Fallback to old hardcoded paths as last resort
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR = BASE_DIR / 'video_analytics' / 'models'
    YOLO_MODEL_PATH = str(MODELS_DIR / 'best_animal.pt')
    REID_MODEL_PATH = str(MODELS_DIR / 'osnet_x0_25_msmt17.pt')

    # Check model existence the old way
    MODEL_FILES = ["best_animal.pt", "osnet_x0_25_msmt17.pt"]
    for model_file in MODEL_FILES:
        if not (MODELS_DIR / model_file).exists():
            logger.error(f"Model file missing: {model_file}")
            raise FileNotFoundError(f"Model file {model_file} not found in {MODELS_DIR}")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://toram444444:06nJTevaUItCDpd9@cluster01.lemxesc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster01")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set!")
DETECTION_CONFIDENCE = 0.4

# Define OUTPUT_DIR with fallback
try:
    OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
except AttributeError:
    logger.warning("JOB_OUTPUT_DIR not defined in settings. Using fallback: MEDIA_ROOT/outputs")
    OUTPUT_DIR = Path(settings.MEDIA_ROOT) / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MongoDB setup with availability check
if MONGODB_AVAILABLE:
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client["snake_db"]
        snake_collection = db["snake_detections"]
        # Test connection
        mongo_client.admin.command('ping')
        logger.info("MongoDB connection established")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        MONGODB_AVAILABLE = False
        mongo_client = None
        db = None
        snake_collection = None
else:
    mongo_client = None
    db = None
    snake_collection = None

# ======================================
# Model Caching
# ======================================

# Thread-safe global model cache to prevent repeated loading
import threading
_model_cache = {}
_model_cache_lock = threading.Lock()

def get_cached_model(model_type: str):
    """Get cached model instance or create new one (thread-safe)."""
    with _model_cache_lock:
        if model_type not in _model_cache:
            try:
                from .model_manager import get_model_with_fallback
                model_path = str(get_model_with_fallback(model_type))
                _model_cache[model_type] = YOLO(model_path)
                logger.info(f"Loaded and cached {model_type} model: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load {model_type} model: {e}")
                raise
        return _model_cache[model_type]

# ======================================
# Helper Functions
# ======================================

def validate_input_file(file_path: str) -> tuple[bool, str]:
    """Validate file type and size."""
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
# Helper Functions for Improvements
# ======================================

def initialize_progress_logger(job_id, job_type, total_items=1):
    """Standardized progress logger initialization."""
    return create_progress_logger(
        job_id=str(job_id) if job_id else "0",
        total_items=total_items,
        job_type=job_type
    )

def create_error_response(error_msg, error_code, start_time, job_type="pest_monitoring"):
    """Standardized error response creation."""
    return {
        'status': 'failed',
        'job_type': job_type,
        'output_image': None,
        'output_video': None,
        'data': {'alerts': [], 'error': error_msg},
        'meta': {
            'timestamp': timezone.now().isoformat(),
            'processing_time_seconds': time.time() - start_time
        },
        'error': {'message': error_msg, 'code': error_code}
    }

def cleanup_video_resources(cap=None, out=None, temp_files=None):
    """Centralized video resource cleanup."""
    if cap is not None:
        try:
            cap.release()
            logger.debug("Released video capture")
        except Exception as e:
            logger.warning(f"Failed to release video capture: {e}")

    if out is not None:
        try:
            out.release()
            logger.debug("Released video writer")
        except Exception as e:
            logger.warning(f"Failed to release video writer: {e}")

    if temp_files:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")

def log_with_level(level, message, job_id=None):
    """Standardized logging without emojis."""
    prefix = f"[Job {job_id}] " if job_id else ""
    if level == "info":
        logger.info(f"{prefix}{message}")
    elif level == "warning":
        logger.warning(f"{prefix}{message}")
    elif level == "error":
        logger.error(f"{prefix}{message}")

# ======================================
# Main Analysis Functions
# ======================================

def detect_snakes_in_image(image_path: str, output_path: str = None, job_id: str = None) -> Dict:
    """
    Detect snakes in an image.

    Args:
        image_path: Path to input image
        output_path: Path to save output image (for tasks.py integration)
        job_id: VideoJob ID for progress tracking

    Returns:
        Standardized response dictionary with filesystem paths
    """
    start_time = time.time()

    # Add job_id logging for progress tracking
    if job_id:
        log_with_level("info", "Starting snake detection job", job_id)

    is_valid, error_msg = validate_input_file(image_path)
    if not is_valid:
        log_with_level("error", f"Invalid input: {error_msg}", job_id)
        return create_error_response(error_msg, 'INVALID_INPUT', start_time)

    try:
        # Load image
        with default_storage.open(image_path, 'rb') as f:
            img = Image.open(f).convert("RGB")
            img_array = np.array(img)

        # Run detection using cached model
        model = get_cached_model("best_animal")
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

        # Create temporary output file for tasks.py integration
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR))
            final_output_path = tmp.name

        log_with_level("info", f"Snake detection completed, output saved to {final_output_path}", job_id)

        # Log to MongoDB if available
        mongo_result = None
        if MONGODB_AVAILABLE and snake_collection is not None:
            try:
                mongo_doc = {
                    "type": "image",
                    "file_name": image_path,
                    "detected_snakes": num_snakes,
                    "detected_animals": detected_animals,
                    "timestamp": datetime.now(),
                    "output_url": output_path
                }
                mongo_result = snake_collection.insert_one(mongo_doc)
                log_with_level("info", f"Logged to MongoDB: {mongo_result.inserted_id}", job_id)
            except Exception as e:
                log_with_level("warning", f"MongoDB logging failed: {e}", job_id)

        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'pest_monitoring',
            'output_image': final_output_path,
            'output_video': None,
            'data': {
                'detected_snakes': num_snakes,
                'detected_animals': detected_animals,
                'mongo_id': str(mongo_result.inserted_id) if mongo_result else None,
                'alerts': alerts
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': 1
            },
            'processed_frames': 1,
            'total_frames': 1,
            'error': None
        }
    except Exception as e:
        log_with_level("error", f"Image processing failed: {str(e)}", job_id)
        return create_error_response(str(e), 'PROCESSING_ERROR', start_time)
    finally:
        # Safe cleanup for temporary files
        temp_files_to_clean = []

        # Check for temporary file from image processing
        if 'tmp' in locals() and hasattr(tmp, 'name'):
            temp_files_to_clean.append(tmp.name)

        # Clean up all temporary files safely
        for temp_file in temp_files_to_clean:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")

def detect_snakes_in_video(video_path: str, output_path: str = None, job_id: str = None, progress_logger=None) -> Dict:
    """
    Detect snakes in a video using YOLO and BotSort.

    Args:
        video_path: Path to input video
        output_path: Path to save output video (for tasks.py integration)
        job_id: VideoJob ID for progress tracking

    Returns:
        Standardized response dictionary with filesystem paths
    """
    start_time = time.time()

    # Add job_id logging for progress tracking
    if job_id:
        log_with_level("info", "Starting snake detection video job", job_id)

    is_valid, error_msg = validate_input_file(video_path)
    if not is_valid:
        log_with_level("error", f"Invalid input: {error_msg}", job_id)
        return create_error_response(error_msg, 'INVALID_INPUT', start_time)

    try:
        # Open video
        with default_storage.open(video_path, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            log_with_level("error", "Failed to open video", job_id)
            return create_error_response('Failed to open video', 'VIDEO_READ_ERROR', start_time)

        # Video setup
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Use provided job_id, don't overwrite it
        output_job_id = job_id or str(int(time.time()))
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
            out = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            # Tracker setup with availability check
            if not BOTSORT_AVAILABLE:
                raise RuntimeError("BotSort not available. Install with: pip install boxmot")

            device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            tracker = BotSort(
                track_high_thresh=0.15,
                new_track_thresh=0.15,
                track_buffer=30,
                device=device,
                half=False,
                reid_weights=Path(REID_MODEL_PATH)
            )
            model = get_cached_model("best_animal")

            total_detected = 0
            detected_animals = []
            frame_number = 0
            last_log_time = start_time
            alerts = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

                # Update progress logger with database fields for frontend
                current_time = time.time()
                if progress_logger and (current_time - last_log_time >= 5 or frame_number == total_frames):
                    progress_logger.update_progress(
                        frame_number, 
                        status=f"Processing frame {frame_number}/{total_frames}",
                        processed_frames=frame_number,
                        total_frames=total_frames,
                        force_log=True
                    )
                    last_log_time = current_time
                elif current_time - last_log_time >= 5 or frame_number == total_frames:
                    # Fallback logging when no progress_logger
                    progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                    elapsed_time = current_time - start_time
                    time_remaining = (elapsed_time / frame_number) * (total_frames - frame_number) if frame_number > 0 and total_frames > 0 else 0
                    avg_fps = frame_number / elapsed_time if elapsed_time > 0 else 0
                    logger.info(f"**Job {output_job_id}**: Progress **{progress:.1f}%** ({frame_number}/{total_frames}), Status: Processing...")
                    logger.info(f"[{'#' * int(progress // 10)}{'-' * (10 - int(progress // 10))}] Done: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d} | Left: {int(time_remaining // 60):02d}:{int(time_remaining % 60):02d} | Avg FPS: {avg_fps:.1f}")
                    last_log_time = current_time

            cap.release()
            out.release()

            # Create temporary file for web conversion
            with tempfile.NamedTemporaryFile(suffix='_web.mp4', delete=False) as web_tmp:
                web_tmp_path = web_tmp.name

            log_with_level("info", f"Attempting ffmpeg conversion: {tmp_out.name} -> {web_tmp_path}", job_id)
            if convert_to_web_mp4(tmp_out.name, web_tmp_path):
                final_output_path = web_tmp_path  # Use converted file
                log_with_level("info", "FFmpeg conversion successful", job_id)
            else:
                final_output_path = tmp_out.name  # Fallback to original
                log_with_level("warning", "FFmpeg conversion failed, using original file", job_id)

            # Log to MongoDB if available
            mongo_result = None
            if MONGODB_AVAILABLE and snake_collection is not None:
                try:
                    mongo_doc = {
                        "type": "video",
                        "file_name": video_path,
                        "detected_snakes": total_detected,
                        "detected_animals": detected_animals,
                        "timestamp": datetime.now(),
                        "output_url": output_path
                    }
                    mongo_result = snake_collection.insert_one(mongo_doc)
                    log_with_level("info", f"Logged to MongoDB: {mongo_result.inserted_id}", job_id)
                except Exception as e:
                    log_with_level("warning", f"MongoDB logging failed: {e}", job_id)

            processing_time = time.time() - start_time
            return {
                'status': 'completed',
                'job_type': 'pest_monitoring',
                'output_image': None,
                'output_video': final_output_path,
                'data': {
                    'detected_snakes': total_detected,
                    'detected_animals': detected_animals,
                    'mongo_id': str(mongo_result.inserted_id) if mongo_result else None,
                    'alerts': alerts
                },
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'processing_time_seconds': processing_time,
                    'fps': fps,
                    'frame_count': frame_number
                },
                'processed_frames': frame_number,
                'total_frames': total_frames,
                'error': None
            }
    except Exception as e:
        log_with_level("error", f"Video processing failed: {str(e)}", job_id)
        return create_error_response(str(e), 'PROCESSING_ERROR', start_time)
    finally:
        # Centralized cleanup for video processing
        temp_files_to_clean = []

        if 'tmp_path' in locals():
            temp_files_to_clean.append(tmp_path)

        # Only clean up tmp_out if it's not the final_output_path
        if ('tmp_out' in locals() and 'final_output_path' in locals() and
            hasattr(tmp_out, 'name') and tmp_out.name != final_output_path):
            temp_files_to_clean.append(tmp_out.name)
        elif 'tmp_out' in locals() and 'final_output_path' not in locals() and hasattr(tmp_out, 'name'):
            temp_files_to_clean.append(tmp_out.name)

        cleanup_video_resources(
            cap=locals().get('cap'),
            out=locals().get('out'),
            temp_files=temp_files_to_clean
        )

        # Note: final_output_path is not cleaned up here as tasks.py needs it
        # tasks.py will handle cleanup after saving to Django storage

# ======================================
# Celery Integration
# ======================================

def tracking_image(input_path: str, output_path: str = None, job_id: str = None) -> Dict:
    """
    Process image for wildlife detection.

    Args:
        input_path: Path to input image
        output_path: Path to save output image (for tasks.py integration)
        job_id: Optional job ID for progress tracking

    Returns:
        Dict with detection results and filesystem paths
    """
    start_time = time.time()

    # Validate input
    is_valid, error_msg = validate_input_file(input_path)
    if not is_valid:
        log_with_level("error", f"Invalid input: {error_msg}", job_id)
        return create_error_response(error_msg, 'INVALID_INPUT', start_time, 'pest_monitoring')

    try:
        # Process image for wildlife detection
        result = detect_snakes_in_image(input_path, output_path, job_id)

        # Add metadata
        result['meta'] = {
            'timestamp': timezone.now().isoformat(),
            'processing_time_seconds': time.time() - start_time,
            'job_id': job_id
        }

        log_with_level("info", f"Wildlife detection completed for image: {input_path}", job_id)
        return result

    except Exception as e:
        log_with_level("error", f"Wildlife detection failed for image {input_path}: {str(e)}", job_id)
        return create_error_response(str(e), 'PROCESSING_ERROR', start_time, 'pest_monitoring')

def tracking_video(input_path: str, output_path: str = None, job_id: str = None) -> Dict:
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
    log_with_level("info", "Starting pest monitoring job", job_id)

    ext = os.path.splitext(input_path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png']

    if ext in image_exts:
        # Initialize progress logger for image processing
        progress_logger = initialize_progress_logger(job_id, "pest_monitoring", 1)

        progress_logger.update_progress(0, status="Processing image for pest monitoring...", processed_frames=0, total_frames=1, force_log=True)
        result = detect_snakes_in_image(input_path, output_path, job_id)
        progress_logger.update_progress(1, status="Pest monitoring completed", processed_frames=1, total_frames=1, force_log=True)
        progress_logger.log_completion(1)
    else:
        # Initialize progress logger for video processing
        progress_logger = initialize_progress_logger(job_id, "pest_monitoring", 100)

        progress_logger.update_progress(0, status="Starting video processing for pest monitoring...", processed_frames=0, total_frames=100, force_log=True)
        result = detect_snakes_in_video(input_path, output_path, job_id, progress_logger)
        progress_logger.update_progress(100, status="Video processing completed", processed_frames=100, total_frames=100, force_log=True)
        progress_logger.log_completion(100)

    processing_time = time.time() - start_time
    result['meta']['processing_time_seconds'] = processing_time
    result['meta']['timestamp'] = timezone.now().isoformat()

    return result
