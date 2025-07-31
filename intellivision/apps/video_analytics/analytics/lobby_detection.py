"""
Lobby detection analytics using YOLOv8x and BotSort/ByteTrack for crowd detection and zone analysis.
Optimized for close proximity tracking with enhanced zone detection.
"""

# ======================================
# Imports and Setup
# ======================================
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import torch
import os
import json
import logging
import time
import re
import tempfile
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone
import mimetypes
from celery import shared_task

logger = logging.getLogger(__name__)

# Try to import utility functions, handle gracefully if not available
try:
    from .model_manager import get_model_with_fallback
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("load_yolo_model not available. Using YOLO directly.")
    def get_model_with_fallback(model_name):
        """Fallback function when utils module is not available."""
        logger.error(f"ERROR:model_manager not available for model: {model_name}")
        raise ImportError(f"ERROR: model_manager not available for model: {model_name}")

try:
    from ..convert import convert_to_web_mp4
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("ERROR: convert_to_web_mp4 not available. Video conversion will be skipped.")
    def convert_to_web_mp4(input_path, output_filename):
        """Fallback function when convert module is not available."""
        return False

try:
    from boxmot import BotSort, DeepOcSort
    BOTSORT_AVAILABLE = True
    DEEPOCSORT_AVAILABLE = True
    logger.info("SUCCESS: boxmot library found - BotSort/DeepOcSort available")
except ImportError as e:
    logger.warning(f"ERROR: boxmot library import failed: {e}")
    logger.warning("Install with: pip install boxmot")
    BOTSORT_AVAILABLE = False
    DEEPOCSORT_AVAILABLE = False

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
                self.logger.info(f"**Job {self.job_id}**: Progress {processed_count}/{self.total_items}")

            def log_completion(self, final_count=None):
                self.logger.info(f"**Job {self.job_id}**: Completed {self.job_type}")

        return DummyLogger(job_id, total_items, job_type, logger_name)

VALID_EXTENSIONS = {'.mp4'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# ======================================
# Model Management Integration
# ======================================
try:
    from .model_manager import get_model_with_fallback

    # Get model paths with automatic fallback
    YOLO_MODEL_PATH = str(get_model_with_fallback("yolov8x"))
    REID_MODEL_PATH = str(get_model_with_fallback("osnet_reid"))

    logger.info(f"SUCCESS: Resolved YOLO model: {YOLO_MODEL_PATH}")
    logger.info(f"SUCCESS: Resolved Re-ID model: {REID_MODEL_PATH}")

except Exception as e:
    logger.error(f"ERROR: Failed to resolve models with fallback: {e}")
    # Fallback to old hardcoded paths as last resort
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR = BASE_DIR / 'video_analytics' / 'models'
    YOLO_MODEL_PATH = str(MODELS_DIR / 'yolov8x.pt')
    REID_MODEL_PATH = str(MODELS_DIR / 'osnet_x0_25_msmt17.pt')

    # Check model existence the old way
    MODEL_FILES = ["yolov8x.pt", "osnet_x0_25_msmt17.pt"]
    for model_file in MODEL_FILES:
        if not (MODELS_DIR / model_file).exists():
            logger.error(f"ERROR: Model file missing: {model_file}")
            raise FileNotFoundError(f"ERROR: Model file {model_file} not found in {MODELS_DIR}")

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
CONFIG_FILE = "lobby_zone_configs.json"

# Define OUTPUT_DIR with fallback
try:
    OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
except AttributeError:
    logger.warning("JOB_OUTPUT_DIR not defined in settings. Using fallback: MEDIA_ROOT/outputs")
    OUTPUT_DIR = Path(settings.MEDIA_ROOT) / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================
# Utility Functions
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

def get_next_filename(base_path):
    """Generate next sequential filename."""
    base_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    name, ext = os.path.splitext(base_name)
    if not default_storage.exists(base_path):
        return base_path
    i = 1
    max_attempts = 1000  # Prevent infinite loop
    while i <= max_attempts:
        new_name = f"{name}_{i}{ext}"
        new_path = os.path.join(base_dir, new_name)
        if not default_storage.exists(new_path):
            return new_path
        i += 1

    # If we can't find a unique name after 1000 attempts, add timestamp
    timestamp = int(time.time())
    new_name = f"{name}_{timestamp}{ext}"
    new_path = os.path.join(base_dir, new_name)
    return new_path

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def bbox_polygon_overlap(bbox, polygon):
    """Check if a bounding box overlaps with a polygon."""
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if any(point_in_polygon(corner, polygon) for corner in corners):
        return True
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    if point_in_polygon((center_x, center_y), polygon):
        return True
    bottom_center = (center_x, y2)
    if point_in_polygon(bottom_center, polygon):
        return True
    return False

# ======================================
# Enhanced Zone Class
# ======================================

class EnhancedZone:
    """Enhanced zone with better detection logic."""
    def __init__(self, polygon, name, threshold):
        self.polygon = polygon
        self.name = name
        self.threshold = threshold

    def count_detections(self, detections):
        """Count detections that overlap with the zone."""
        if len(detections) == 0:
            return 0
        count = 0
        for bbox in detections.xyxy:
            if bbox_polygon_overlap(bbox, self.polygon):
                count += 1
        return count

# ======================================
# Tracker Setup
# ======================================

def setup_best_tracker(device, fps):
    """Setup the best available tracker with Re-ID model, falling back to ByteTrack."""
    tracker = None
    tracker_name = "ByteTrack"  # Default tracker

    if BOTSORT_AVAILABLE:
        try:
            reid_model_path = Path(REID_MODEL_PATH)
            if reid_model_path.exists():
                logger.info(f"SUCCESS: Re-ID model found at {reid_model_path}. Using BotSort with Re-ID.")
                tracker = BotSort(
                    model_weights=YOLO_MODEL_PATH,
                    device=device,
                    fp16=False,
                    track_high_thresh=0.5,
                    track_low_thresh=0.1,
                    new_track_thresh=0.6,
                    track_buffer=30,
                    match_thresh=0.8,
                    frame_rate=fps,
                    reid_weights=str(reid_model_path),
                    proximity_thresh=0.5,
                    appearance_thresh=0.25
                )
                tracker_name = "BotSort"
            else:
                logger.warning(f"Re-ID model not found at {reid_model_path}. Using BotSort without Re-ID.")
                tracker = BotSort(
                    model_weights=YOLO_MODEL_PATH,
                    device=device,
                    fp16=False,
                    track_high_thresh=0.5,
                    track_low_thresh=0.1,
                    new_track_thresh=0.6,
                    track_buffer=30,
                    match_thresh=0.8,
                    frame_rate=fps
                )
                tracker_name = "BotSort"
        except Exception as e:
            logger.warning(f"BotSort setup failed: {e}. Falling back to ByteTrack.")
            tracker = None  # Force fallback to ByteTrack

    if tracker is None:
        logger.info("Using ByteTrack as the tracker.")
        tracker = sv.ByteTrack(frame_rate=fps)
        tracker_name = "ByteTrack"

    return tracker, tracker_name

# ======================================
# Main Analysis Function
# ======================================

def run_crowd_analysis(source_path: str, zone_configs: dict, output_path: str = None, job_id: str = None) -> Dict:
    """
    Run crowd analysis on video with zone-based counting.

    Args:
        source_path: Path to input video
        zone_configs: Dictionary containing zone definitions
        output_path: Path to save output video
        job_id: Job ID for progress tracking

    Returns:
        Dictionary with analysis results
    """
    start_time = time.time()

    # Validate input
    is_valid, error_msg = validate_input_file(source_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'lobby_detection',
            'output_image': None,
            'output_video': None,
            'data': {'zones': {}, 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    try:
        # Setup model
        model_name = YOLO_MODEL_PATH
        model = YOLO(model_name)

        # Device setup
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        logger.info(f"Using model: {model_name} on device: {device}")

        # Setup output
        # Extract job ID from source path if not provided as parameter
        extracted_job_id = re.search(r'(\d+)', source_path)
        file_job_id = extracted_job_id.group(1) if extracted_job_id else str(int(time.time()))
        output_filename = f"outputs/output_crowd_{file_job_id}.mp4"
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
            video_info = sv.VideoInfo.from_video_path(source_path)
            writer = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, video_info.resolution_wh)

            # Setup tracker
            tracker, tracker_name = setup_best_tracker(device, video_info.fps)
            logger.info(f"🔍 ACTIVE TRACKER: {tracker_name}")

            # Setup zones with validation
            try:
                zones = []
                for name, data in zone_configs.items():
                    if not isinstance(data, dict):
                        raise ValueError(f"Zone '{name}' configuration must be a dictionary")
                    if 'points' not in data:
                        raise ValueError(f"Zone '{name}' missing required 'points' field")
                    if 'threshold' not in data:
                        raise ValueError(f"Zone '{name}' missing required 'threshold' field")
                    if not isinstance(data['points'], list) or len(data['points']) < 3:
                        raise ValueError(f"Zone '{name}' must have at least 3 points")
                    if not isinstance(data['threshold'], (int, float)) or data['threshold'] <= 0:
                        raise ValueError(f"Zone '{name}' threshold must be a positive number")

                    zones.append(EnhancedZone(data['points'], name, data['threshold']))

                if not zones:
                    raise ValueError("At least one zone must be configured")

            except Exception as e:
                error_msg = f"Zone configuration error: {str(e)}"
                logger.error(error_msg)
                return {
                    'status': 'failed',
                    'job_type': 'lobby_detection',
                    'output_image': None,
                    'output_video': None,
                    'data': {'alerts': [], 'error': error_msg},
                    'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                    'error': {'message': error_msg, 'code': 'ZONE_CONFIG_ERROR'}
                }

            cap = cv2.VideoCapture(source_path)
            box_annotator = sv.BoxAnnotator(thickness=1, color=sv.Color(r=218, g=165, b=32))
            alert_log = []
            last_alert_times = {}
            frame_number = 0
            last_log_time = start_time
            ALERTS_DIR = OUTPUT_DIR / 'alerts'
            ALERTS_DIR.mkdir(parents=True, exist_ok=True)

            with tqdm(total=video_info.total_frames, desc="Processing") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_number += 1

                    # Detection
                    results = model(frame, verbose=False, classes=[0], imgsz=1280)[0]
                    detections = sv.Detections.from_ultralytics(results)
                    detections = detections[detections.confidence > 0.3]
                    if len(detections) > 0:
                        areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
                        min_area = 1000
                        detections = detections[areas > min_area]

                    # Tracking
                    if tracker_name in ["BotSort", "DeepOcSort"]:
                        if len(detections) > 0:
                            dets_for_tracker = np.column_stack([detections.xyxy, detections.confidence, detections.class_id])
                            outputs = tracker.update(dets_for_tracker, frame)
                            if len(outputs) > 0:
                                tracked_detections = sv.Detections(
                                    xyxy=outputs[:, :4],
                                    class_id=outputs[:, 5].astype(int),
                                    tracker_id=outputs[:, 4].astype(int)
                                )
                            else:
                                tracked_detections = sv.Detections.empty()
                        else:
                            tracker.update(np.empty((0, 6)), frame)
                            tracked_detections = sv.Detections.empty()
                    else:  # ByteTrack
                        tracked_detections = tracker.update_with_detections(detections)

                    # Debug logging
                    if frame_number % 30 == 0 and len(tracked_detections) > 0:
                        max_id = max(tracked_detections.tracker_id) if len(tracked_detections.tracker_id) > 0 else 0
                        logger.info(f"Frame {frame_number}: {len(tracked_detections)} tracks, max ID: {max_id}")

                    # Visualization
                    annotated_frame = frame.copy()
                    zone_shading_overlay = annotated_frame.copy()
                    for zone in zones:
                        count = zone.count_detections(tracked_detections)
                        alert_triggered = count > zone.threshold
                        zone_color = (0, 0, 255) if alert_triggered else (0, 255, 0)
                        pts = np.array(zone.polygon, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(zone_shading_overlay, [pts], zone_color)
                        if alert_triggered and (zone.name not in last_alert_times or
                                               (frame_number - last_alert_times.get(zone.name, 0)) > video_info.fps * 10):
                            video_timestamp_seconds = frame_number / video_info.fps
                            video_timestamp = str(timedelta(seconds=int(video_timestamp_seconds)))
                            system_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            alert_msg = f"🚨 ALERT: {zone.name} has {count} people (threshold: {zone.threshold})"
                            logger.warning(alert_msg)
                            alert_data = {
                                'zone_name': zone.name,
                                'count': count,
                                'threshold': zone.threshold,
                                'video_timestamp': video_timestamp,
                                'system_timestamp': system_timestamp,
                                'frame_number': frame_number,
                                'message': alert_msg
                            }
                            alert_log.append(alert_data)
                            alert_img_path = ALERTS_DIR / f"alert_{zone.name}_{system_timestamp}.jpg"
                            cv2.imwrite(str(alert_img_path), frame)
                            with open(str(alert_img_path), 'rb') as f:
                                default_storage.save(f"alerts/alert_{zone.name}_{system_timestamp}.jpg", f)
                            last_alert_times[zone.name] = frame_number
                    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, zone_shading_overlay, 0.3, 0)
                    for zone in zones:
                        pts = np.array(zone.polygon, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pts], isClosed=True, color=(218, 165, 32), thickness=1)
                    if zones:
                        info_texts = [f"{z.name}: {z.count_detections(tracked_detections)}/{z.threshold}" for z in zones]
                        max_text_width = max(cv2.getTextSize(text, FONT_FACE, 0.4, 1)[0][0] for text in info_texts) if info_texts else 0
                        box_width = max_text_width + 12
                        box_height = len(zones) * 18 + 8
                        box_x = annotated_frame.shape[1] - box_width - 8
                        box_y = 8
                        overlay = annotated_frame.copy()
                        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
                        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
                        for i, zone in enumerate(zones):
                            count = zone.count_detections(tracked_detections)
                            text_color = (0, 0, 255) if count > zone.threshold else (255, 255, 255)
                            cv2.putText(annotated_frame, f"{zone.name}: {count}/{zone.threshold}",
                                        (box_x + 6, box_y + (i + 1) * 18 - 4), FONT_FACE, 0.4, text_color, 1)
                    if len(tracked_detections) > 0:
                        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
                        for i, (bbox, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                            x1, y1, x2, y2 = bbox
                            id_text = str(int(tracker_id))
                            (text_width, text_height), baseline = cv2.getTextSize(id_text, FONT_FACE, 0.6, 2)
                            cv2.rectangle(annotated_frame, (int(x1), int(y1) - text_height - 5),
                                          (int(x1) + text_width + 4, int(y1)), (0, 0, 0), -1)
                            cv2.putText(annotated_frame, id_text, (int(x1) + 2, int(y1) - 3),
                                        FONT_FACE, 0.6, (255, 255, 255), 2)
                    writer.write(annotated_frame)
                    pbar.update(1)

                    # Periodic logging
                    current_time = time.time()
                    if current_time - last_log_time >= 5 or frame_number == video_info.total_frames:
                        progress = (frame_number / video_info.total_frames) * 100
                        elapsed_time = current_time - start_time
                        time_remaining = (elapsed_time / frame_number) * (video_info.total_frames - frame_number) if frame_number > 0 else 0
                        avg_fps = frame_number / elapsed_time if elapsed_time > 0 else 0
                        logger.info(f"**Job {job_id}**: Progress **{progress:.1f}%** ({frame_number}/{video_info.total_frames}), Status: Processing...")
                        logger.info(f"[{'#' * int(progress // 10)}{'-' * (10 - int(progress // 10))}] Done: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d} | Left: {int(time_remaining // 60):02d}:{int(time_remaining % 60):02d} | Avg FPS: {avg_fps:.1f}")
                        last_log_time = current_time

            cap.release()
            writer.release()

            # Create temporary file for web conversion
            with tempfile.NamedTemporaryFile(suffix='_web.mp4', delete=False) as web_tmp:
                web_tmp_path = web_tmp.name

            logger.info(f"🔄 Attempting ffmpeg conversion: {tmp_out.name} -> {web_tmp_path}")
            if convert_to_web_mp4(tmp_out.name, web_tmp_path):
                final_output_path = web_tmp_path  # Use converted file
                logger.info(f"✅ FFmpeg conversion successful")
            else:
                final_output_path = tmp_out.name  # Fallback to original
                logger.warning(f"⚠️ FFmpeg conversion failed, using original file")
            final_zone_counts = {}
            alerts_by_zone = {}
            for zone in zones:
                count = zone.count_detections(tracked_detections) if 'tracked_detections' in locals() else 0
                final_zone_counts[zone.name] = count
                alerts_by_zone[zone.name] = sum(1 for alert in alert_log if alert['zone_name'] == zone.name)
            processing_time = time.time() - start_time
            return {
                'status': 'completed',
                'job_type': 'lobby_detection',
                'output_image': None,
                'output_video': final_output_path,
                'data': {
                    'zone_counts': final_zone_counts,
                    'alerts': alert_log,
                    'alert_summary': {
                        'total_alerts': len(alert_log),
                        'alerts_by_zone': alerts_by_zone
                    }
                },
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'processing_time_seconds': processing_time,
                    'fps': video_info.fps,
                    'frame_count': frame_number
                },
                'error': None
            }
    except Exception as e:
        logger.exception(f"Video processing failed: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'lobby_detection',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }
    finally:
        if 'cap' in locals() and cap:
            cap.release()
        if 'writer' in locals() and writer:
            writer.release()
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        # Note: final_output_path is not cleaned up here as tasks.py needs it
        # tasks.py will handle cleanup after saving to Django storage
        # Only clean up tmp_out if it's not the final_output_path
        if ('tmp_out' in locals() and 'final_output_path' in locals() and
            tmp_out.name != final_output_path and os.path.exists(tmp_out.name)):
            os.remove(tmp_out.name)

# ======================================
# Celery Integration
# ======================================

def tracking_video(source_path: str, zone_configs: dict, output_path: str = None, job_id: str = None) -> Dict:
    """
    Celery task for lobby crowd detection.

    Args:
        self: Celery task instance
        source_path: Path to input video
        zone_configs: Dictionary of zone configurations
        output_path: Path to save output video
        job_id: VideoJob ID

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    logger.info(f"🚀 Starting lobby detection job {job_id}")

    # Initialize progress logger for video processing
    progress_logger = create_progress_logger(
        job_id=str(job_id) if job_id else "unknown",
        total_items=100,  # Estimate for video frames
        job_type="lobby_detection"
    )

    progress_logger.update_progress(0, status="Starting lobby crowd analysis...", force_log=True)
    result = run_crowd_analysis(source_path, zone_configs, output_path, job_id)
    progress_logger.update_progress(100, status="Lobby detection completed", force_log=True)
    progress_logger.log_completion(100)

    result['meta']['processing_time_seconds'] = time.time() - start_time
    result['meta']['timestamp'] = timezone.now().isoformat()

    return result
