import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import torch
import os
import json
import argparse
import logging
import datetime
import time
import pathlib
import urllib.request
from django.conf import settings
from apps.video_analytics.models import load_yolo_model

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
CONFIG_FILE = "lobby_zone_configs.json"

# Try to import tracking libraries
try:
    # Use the correct class names from your boxmot version
    from boxmot import BotSort, DeepOcSort  # Note: DeepOcSort, not DeepOCSORT

    BOTSORT_AVAILABLE = True
    DEEPOCSORT_AVAILABLE = True
    logger.info("✅ boxmot library found - BotSort/DeepOcSort available")
except ImportError as e:
    logger.warning(f"❌ boxmot library import failed: {e}")
    logger.warning("   Install with: pip install boxmot")
    BOTSORT_AVAILABLE = False
    DEEPOCSORT_AVAILABLE = False

# Canonical models directory for all analytics jobs
from pathlib import Path
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REID_MODEL_PATH = MODELS_DIR / "osnet_x0_25_msmt17.pt"


# =================================================================================
# UTILITY FUNCTIONS
# =================================================================================
def get_next_filename(base_path):
    """
    Checks for existing files and returns the next available sequential filename.
    e.g., output.mp4 -> output_1.mp4 -> output_2.mp4
    """
    base_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    name, ext = os.path.splitext(base_name)

    if not os.path.exists(base_path):
        return base_path

    i = 1
    while True:
        new_name = f"{name}_{i}{ext}"
        new_path = os.path.join(base_dir, new_name)
        if not os.path.exists(new_path):
            return new_path
        i += 1


def download_reid_model():
    """Download Re-ID model if not present"""
    model_path = REID_MODEL_PATH
    if not model_path.exists():
        logger.info("Downloading Re-ID model for better tracking...")
        url = "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0.0/osnet_x0_25_msmt17.pt"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(model_path))
            logger.info(f"✅ Re-ID model downloaded: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to download Re-ID model: {e}")
            return None
    return model_path


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def bbox_polygon_overlap(bbox, polygon):
    """Check if a bounding box overlaps with a polygon (more flexible zone detection)"""
    x1, y1, x2, y2 = bbox

    # Check if any corner of the bbox is inside the polygon
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if any(point_in_polygon(corner, polygon) for corner in corners):
        return True

    # Check if center point is inside (traditional method)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    if point_in_polygon((center_x, center_y), polygon):
        return True

    # Check if any part of the person overlaps (bottom center for standing people)
    bottom_center = (center_x, y2)
    if point_in_polygon(bottom_center, polygon):
        return True

    return False


# =================================================================================
# ENHANCED ZONE CLASS
# =================================================================================
class EnhancedZone:
    """Enhanced zone with better detection logic"""

    def __init__(self, polygon, name, threshold):
        self.polygon = polygon
        self.name = name
        self.threshold = threshold

    def count_detections(self, detections):
        """Count detections that overlap with the zone"""
        if len(detections) == 0:
            return 0

        count = 0
        for bbox in detections.xyxy:
            if bbox_polygon_overlap(bbox, self.polygon):
                count += 1

        return count


# =================================================================================
# ADVANCED TRACKING SETUP - OPTIMIZED FOR STABILITY
# =================================================================================
def setup_best_tracker(device, fps):
    """Setup tracker specifically optimized for STABLE ID scenarios"""

    reid_model = download_reid_model()

    logger.info(f"🔍 Tracker availability - BotSort: {BOTSORT_AVAILABLE}, DeepOCSORT: {DEEPOCSORT_AVAILABLE}")

    if BOTSORT_AVAILABLE and reid_model:
        try:
            logger.info("🔄 Attempting to initialize BotSort...")
            logger.info(f"🔍 Re-ID model path: {reid_model}")

            # Convert parameters to correct types
            import torch
            import pathlib

            reid_path = REID_MODEL_PATH
            torch_device = torch.device(device)

            # Try BotSort with correct parameter types
            try:
                tracker = BotSort(
                    reid_weights=reid_path,  # pathlib.Path object
                    device=torch_device,  # torch.device object
                    half=False,  # Required bool parameter
                    per_class=False,
                    # ID stability parameters
                    track_high_thresh=0.6,  # Higher threshold for new tracks
                    track_low_thresh=0.2,  # Lower threshold to keep existing tracks
                    new_track_thresh=0.8,  # Much harder to create new IDs
                    track_buffer=fps * 2,  # 5 second memory at 30fps
                    match_thresh=0.7,  # Very strict matching
                    proximity_thresh=0.5,  # Closer proximity required
                    appearance_thresh=0.1,  # Very strict appearance matching
                    frame_rate=fps,  # Use actual fps
                    with_reid=True
                )
                logger.info("✅ Using BotSort tracker with optimized ID stability parameters.")
                return tracker, "BotSort"
            except Exception as e1:
                logger.warning(f"BotSort optimized parameters failed: {e1}")
                logger.info("🔄 Trying BotSort with basic parameters...")

                # Fallback with minimal parameters
                tracker = BotSort(
                    reid_weights=reid_path,
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

    # Check supervision version for parameter compatibility
    try:
        tracker = sv.ByteTrack(
            frame_rate=fps,
            track_activation_threshold=0.8,  # Higher threshold for new tracks
            lost_track_buffer=150,  # 5 second buffer for lost tracks
            minimum_matching_threshold=0.9,  # Stricter matching
            minimum_consecutive_frames=3  # Require 3 consecutive detections
        )
        logger.info("✅ Using advanced ByteTrack parameters")
    except TypeError:
        # Fallback for older supervision versions
        tracker = sv.ByteTrack(
            frame_rate=fps,
            track_thresh=0.8,  # Higher threshold for new tracks
            track_buffer=150,  # Longer buffer
            match_thresh=0.9  # Stricter matching
        )
        logger.info("✅ Using basic ByteTrack parameters (older supervision)")

    return tracker, "ByteTrack"


# =================================================================================
# MAIN ANALYSIS FUNCTION
# =================================================================================
def run_crowd_analysis(source_path, zone_configs, output_path=None):
    model_name = str(MODELS_DIR / 'yolov8x.pt')
    model = load_yolo_model(model_name)
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    logger.info(f"Using model: {model_name} on device: {device}")

    # Video setup
    video_info = sv.VideoInfo.from_video_path(source_path)
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    OUTPUT_DIR = settings.JOB_OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_path is None:
        output_path = get_next_filename(os.path.join(OUTPUT_DIR, f"output_crowd_{base_name}.mp4"))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             video_info.fps, video_info.resolution_wh)

    # Setup tracker with optimized settings for stability
    tracker, tracker_name = setup_best_tracker(device, video_info.fps)
    logger.info(f"🔍 ACTIVE TRACKER: {tracker_name}")

    # --- AESTHETICS: Golden bounding boxes ---
    # Corrected Code for the Old Library
    box_annotator = sv.BoxAnnotator(thickness=1, color=sv.Color(r=218, g=165, b=32))

    # Setup enhanced zones
    zones = [EnhancedZone(data['points'], name, data['threshold']) for name, data in zone_configs.items()]

    # Main processing loop
    cap = cv2.VideoCapture(source_path)
    # --- FIX: Use a list to correctly log all alerts with timestamps ---
    alert_log = []
    last_alert_times = {}
    frame_number = 0

    ALERTS_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, '../alerts'))
    os.makedirs(ALERTS_DIR, exist_ok=True)

    with tqdm(total=video_info.total_frames, desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1

            # --- Detection ---
            results = model(frame, verbose=False, classes=[0], imgsz=1280)[0]
            detections = sv.Detections.from_ultralytics(results)

            # --- FIX: Lower confidence threshold to feed more consistent data to tracker ---
            # More consistent detections = better ID stability
            detections = detections[detections.confidence > 0.3]  # Lowered from 0.4

            # Additional filtering for stability
            if len(detections) > 0:
                # Filter out very small detections (likely false positives)
                areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (
                            detections.xyxy[:, 3] - detections.xyxy[:, 1])
                min_area = 1000  # Minimum person area in pixels
                detections = detections[areas > min_area]

            # --- Tracking ---
            if tracker_name in ["DeepOcSort", "BotSort"]:  # Updated DeepOCSORT to DeepOcSort
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
            else:  # Fallback for ByteTrack
                tracked_detections = tracker.update_with_detections(detections)

            # Debug: Log tracking stats every 30 frames
            if frame_number % 30 == 0 and len(tracked_detections) > 0:
                max_id = max(tracked_detections.tracker_id) if len(tracked_detections.tracker_id) > 0 else 0
                logger.info(f"Frame {frame_number}: {len(tracked_detections)} tracks, max ID: {max_id}")

            # --- Visualization ---
            annotated_frame = frame.copy()
            zone_shading_overlay = annotated_frame.copy()

            for zone in zones:
                count = zone.count_detections(tracked_detections)
                alert_triggered = count > zone.threshold

                # Zone shading color based on alert
                zone_color = (0, 0, 255) if alert_triggered else (0, 255, 0)
                pts = np.array(zone.polygon, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(zone_shading_overlay, [pts], zone_color)

                # Alert logic
                if alert_triggered and (zone.name not in last_alert_times or
                                        (frame_number - last_alert_times.get(zone.name, 0)) > video_info.fps * 10):
                    # Calculate video timestamp based on frame number and fps
                    video_timestamp_seconds = frame_number / video_info.fps
                    video_timestamp = datetime.timedelta(seconds=int(video_timestamp_seconds))
                    video_timestamp_str = str(video_timestamp)

                    # Current system timestamp
                    system_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    alert_msg = f"🚨 ALERT: {zone.name} has {count} people (threshold: {zone.threshold})"
                    logger.warning(alert_msg)

                    # Store structured alert data with timestamps
                    alert_data = {
                        'zone_name': zone.name,
                        'count': count,
                        'threshold': zone.threshold,
                        'video_timestamp': video_timestamp_str,
                        'system_timestamp': system_timestamp,
                        'frame_number': frame_number,
                        'message': alert_msg
                    }
                    alert_log.append(alert_data)

                    alert_img_path = os.path.join(ALERTS_DIR, f"alert_{zone.name}_{system_timestamp}.jpg")
                    cv2.imwrite(alert_img_path, frame)
                    last_alert_times[zone.name] = frame_number

            # Blend the overlay with the frame
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, zone_shading_overlay, 0.3, 0)

            # Draw zone outlines on top of shading
            for zone in zones:
                pts = np.array(zone.polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], isClosed=True, color=(218, 165, 32), thickness=1)

            # --- COMPACT COUNT DISPLAY ---
            if zones:
                info_texts = [f"{z.name}: {z.count_detections(tracked_detections)}/{z.threshold}" for z in zones]
                max_text_width = max(
                    cv2.getTextSize(text, FONT_FACE, 0.4, 1)[0][0] for text in info_texts) if info_texts else 0
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

            # Draw detections with stable IDs
            if len(tracked_detections) > 0:
                # in your old version of the 'supervision' library.
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)

                # Add ID numbers to each detection
                for i, (bbox, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                    x1, y1, x2, y2 = bbox
                    # Position ID text at top-left of bounding box
                    id_text = str(int(tracker_id))
                    # Get text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(id_text, FONT_FACE, 0.6, 2)
                    # Draw background rectangle for better visibility
                    cv2.rectangle(annotated_frame,
                                  (int(x1), int(y1) - text_height - 5),
                                  (int(x1) + text_width + 4, int(y1)),
                                  (0, 0, 0), -1)
                    # Draw ID text
                    cv2.putText(annotated_frame, id_text,
                                (int(x1) + 2, int(y1) - 3),
                                FONT_FACE, 0.6, (255, 255, 255), 2)

            writer.write(annotated_frame)
            pbar.update(1)

    cap.release()
    writer.release()
    logger.info(f"✅ Analysis complete. Output saved to: {output_path}")

    if alert_log:
        logger.info(f"📊 Total alerts triggered: {len(alert_log)}")
        for alert_data in alert_log:
            logger.info(f"   • {alert_data['message']} at {alert_data['video_timestamp']} (frame {alert_data['frame_number']})")

    # Convert to web-friendly MP4 (like people_count)
    from apps.video_analytics.convert import convert_to_web_mp4
    web_output_path = output_path.replace('.mp4', '_web.mp4')
    if convert_to_web_mp4(output_path, web_output_path):
        final_output_path = web_output_path
    else:
        final_output_path = output_path

    # After processing loop, compute final zone counts and alert summary
    final_zone_counts = {}
    alerts_by_zone = {}

    if zones:
        # Use last frame's tracked detections for final count (or 0 if none)
        for zone in zones:
            # If tracked_detections is not defined (e.g. no frames), set to 0
            try:
                count = zone.count_detections(tracked_detections)
            except Exception:
                count = 0
            final_zone_counts[zone.name] = count
            alerts_by_zone[zone.name] = 0

    # Count alerts by zone
    for alert_data in alert_log:
        zone_name = alert_data['zone_name']
        if zone_name in alerts_by_zone:
            alerts_by_zone[zone_name] += 1

    return {
        'status': 'completed',
        'job_type': 'lobby_detection',
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
            'video_fps': video_info.fps,
            'total_frames': video_info.total_frames,
            'video_duration_seconds': video_info.total_frames / video_info.fps
        },
        'error': None
    }

'''
🎯 OPTIMIZED FOR CLOSE PROXIMITY TRACKING

Key Features:
✅ Tiny UI box (160x35) in top-left corner - doesn't block scene marking
✅ Close proximity tracking optimizations:
   - appearance_thresh=0.1 (very strict Re-ID)
   - new_track_thresh=0.8 (hard to create new IDs)
   - track_buffer=120 (4 second memory)
   - match_thresh=0.95 (prefer existing IDs)
✅ Enhanced zone detection (partial overlap counting)
✅ YOLOv11m model (best for person detection)
✅ Professional tracker parameters (no custom interference)
✅ ID numbers displayed on each tracking box (just the number like 1, 2, etc.)
'''
