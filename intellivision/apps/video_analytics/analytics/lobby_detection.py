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

# Try to import utility functions, handle gracefully if not available
try:
    from ..utils import load_yolo_model
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("load_yolo_model not available. Using YOLO directly.")
    def load_yolo_model(model_path):
        """Fallback function when utils module is not available."""
        return YOLO(model_path)

try:
    from ..convert import convert_to_web_mp4
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("convert_to_web_mp4 not available. Video conversion will be skipped.")
    def convert_to_web_mp4(input_path, output_filename):
        """Fallback function when convert module is not available."""
        return False

try:
    from boxmot import BotSort, DeepOcSort
    BOTSORT_AVAILABLE = True
    DEEPOCSORT_AVAILABLE = True
    logger.info("✅ boxmot library found - BotSort/DeepOcSort available")
except ImportError as e:
    logger.warning(f"❌ boxmot library import failed: {e}")
    logger.warning("Install with: pip install boxmot")
    BOTSORT_AVAILABLE = False
    DEEPOCSORT_AVAILABLE = False

# ======================================
# Logger and Constants
# ======================================
logger = logging.getLogger(__name__)
VALID_EXTENSIONS = {'.mp4'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REID_MODEL_PATH = MODELS_DIR / "osnet_x0_25_msmt17.pt"
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
CONFIG_FILE = "lobby_zone_configs.json"

# Define OUTPUT_DIR with fallback
try:
    OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
except AttributeError:
    logger.warning("JOB_OUTPUT_DIR not defined in settings. Using fallback: MEDIA_ROOT/outputs")
    OUTPUT_DIR = Path(settings.MEDIA_ROOT) / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check model existence
MODEL_FILES = ["yolov8x.pt", "osnet_x0_25_msmt17.pt"]
for model_file in MODEL_FILES:
    if not (MODELS_DIR / model_file).exists():
        logger.error(f"Model file missing: {model_file}")
        raise FileNotFoundError(f"Model file {model_file} not found in {MODELS_DIR}")

# ======================================
# Utility Functions
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

def get_next_filename(base_path):
    """Generate next sequential filename."""
    base_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    name, ext = os.path.splitext(base_name)
    if not default_storage.exists(base_path):
        return base_path
    i = 1
    while True:
        new_name = f"{name}_{i}{ext}"
        new_path = os.path.join(base_dir, new_name)
        if not default_storage.exists(new_path):
            return new_path
        i += 1

def download_reid_model():
    """Download Re-ID model if not present."""
    if not REID_MODEL_PATH.exists():
        logger.info("Downloading Re-ID model for better tracking...")
        url = "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0.0/osnet_x0_25_msmt17.pt"
        try:
            urllib.request.urlretrieve(url, str(REID_MODEL_PATH))
            logger.info(f"✅ Re-ID model downloaded: {REID_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to download Re-ID model: {e}")
            return None
    return REID_MODEL_PATH

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
    """Setup tracker optimized for stable ID scenarios."""
    reid_model = download_reid_model()
    logger.info(f"🔍 Tracker availability - BotSort: {BOTSORT_AVAILABLE}, DeepOCSORT: {DEEPOCSORT_AVAILABLE}")
    if BOTSORT_AVAILABLE and reid_model:
        try:
            logger.info("🔄 Attempting to initialize BotSort...")
            logger.info(f"🔍 Re-ID model path: {reid_model}")
            torch_device = torch.device(device)
            try:
                tracker = BotSort(
                    reid_weights=REID_MODEL_PATH,
                    device=torch_device,
                    half=False,
                    per_class=False,
                    track_high_thresh=0.6,
                    track_low_thresh=0.2,
                    new_track_thresh=0.8,
                    track_buffer=fps * 2,
                    match_thresh=0.7,
                    proximity_thresh=0.5,
                    appearance_thresh=0.1,
                    frame_rate=fps,
                    with_reid=True
                )
                logger.info("✅ Using BotSort tracker with optimized ID stability parameters.")
                return tracker, "BotSort"
            except Exception as e1:
                logger.warning(f"BotSort optimized parameters failed: {e1}")
                logger.info("🔄 Trying BotSort with basic parameters...")
                tracker = BotSort(
                    reid_weights=REID_MODEL_PATH,
                    device=torch_device,
                    half=False
                )
                logger.info("✅ Using BotSort tracker with basic parameters.")
                return tracker, "BotSort"
        except Exception as e:
            logger.warning(f"BotSort setup completely failed: {e}. Falling back to ByteTrack.")
    elif BOTSORT_AVAILABLE and not reid_model:
        logger.warning("BotSort available but Re-ID model failed to download. Using ByteTrack.")
    logger.info("Using ByteTrack as fallback tracker.")
    try:
        tracker = sv.ByteTrack(
            frame_rate=fps,
            track_activation_threshold=0.8,
            lost_track_buffer=150,
            minimum_matching_threshold=0.9,
            minimum_consecutive_frames=3
        )
        logger.info("✅ Using advanced ByteTrack parameters")
    except TypeError:
        tracker = sv.ByteTrack(
            frame_rate=fps,
            track_thresh=0.8,
            track_buffer=150,
            match_thresh=0.9
        )
        logger.info("✅ Using basic ByteTrack parameters (older supervision)")
    return tracker, "ByteTrack"

# ======================================
# Main Analysis Function
# ======================================

def run_crowd_analysis(source_path: str, zone_configs: dict) -> Dict:
    """
    Process video for crowd analysis in lobby zones.

    Args:
        source_path: Path to input video
        zone_configs: Dictionary of zone configurations

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    source_path = Path(source_path).name
    is_valid, error_msg = validate_input_file(source_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'lobby_detection',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    try:
        # Open video
        with default_storage.open(source_path, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
        video_info = sv.VideoInfo.from_video_path(tmp_path)
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            logger.error("Failed to open video")
            return {
                'status': 'failed',
                'job_type': 'lobby_detection',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': 'Failed to open video'},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': 'Failed to open video', 'code': 'VIDEO_READ_ERROR'}
            }

        # Setup model
        model_name = str(MODELS_DIR / 'yolov8x.pt')
        model = load_yolo_model(model_name)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        logger.info(f"Using model: {model_name} on device: {device}")

        # Setup output
        job_id = re.search(r'(\d+)', source_path)
        job_id = job_id.group(1) if job_id else str(int(time.time()))
        output_filename = f"outputs/output_crowd_{job_id}.mp4"
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
            writer = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, video_info.resolution_wh)

            # Setup tracker
            tracker, tracker_name = setup_best_tracker(device, video_info.fps)
            logger.info(f"🔍 ACTIVE TRACKER: {tracker_name}")

            # Setup zones
            zones = [EnhancedZone(data['points'], name, data['threshold']) for name, data in zone_configs.items()]
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
                    else:
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
            with open(tmp_out.name, 'rb') as f:
                default_storage.save(output_filename, f)
            output_url = default_storage.url(output_filename)
            web_output_filename = output_filename.replace('.mp4', '_web.mp4')
            if convert_to_web_mp4(tmp_out.name, web_output_filename):
                output_url = default_storage.url(web_output_filename)
                os.remove(tmp_out.name)
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
                'output_video': output_url,
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
        if 'tmp_out' in locals() and os.path.exists(tmp_out.name):
            os.remove(tmp_out.name)

# ======================================
# Celery Integration
# ======================================

@shared_task(bind=True)
def tracking_video(self, source_path: str, zone_configs: dict, output_path: str = None, job_id: str = None) -> Dict:
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
    result = run_crowd_analysis(source_path, zone_configs)
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