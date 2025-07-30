"""
Emergency counting analytics using YOLO and BotSort for people detection and movement analysis.
Counts people crossing defined lines with FastCounter and EnhancedCleanAnalyzer.
"""

# ======================================
# Imports and Setup
# ======================================
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
import os
import logging
import tempfile
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone
import mimetypes
from celery import shared_task
from boxmot import BotSort
from tqdm import tqdm

from ..utils import load_yolo_model

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
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REID_MODEL_PATH = MODELS_DIR / "osnet_x0_25_msmt17.pt"

# Define OUTPUT_DIR with fallback
try:
    OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
except AttributeError:
    logger.warning("JOB_OUTPUT_DIR not defined in settings. Using fallback: MEDIA_ROOT/outputs")
    OUTPUT_DIR = Path(settings.MEDIA_ROOT) / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check model existence
MODEL_FILES = ["yolo12x.pt", "osnet_x0_25_msmt17.pt"]
for model_file in MODEL_FILES:
    if not (MODELS_DIR / model_file).exists():
        logger.error(f"Model file missing: {model_file}")
        raise FileNotFoundError(f"Model file {model_file} not found in {MODELS_DIR}")


# ======================================
# Parameter Management
# ======================================

class OptimalParams:
    """Stores optimal parameters for YOLOv12x emergency counting."""

    def __init__(self):
        self.min_crossing_distance = 50
        self.proximity_threshold = 25
        self.track_timeout_frames = 45
        self.track_buffer = 120
        self.match_thresh = 0.7
        self.confidence_threshold = 0.10
        self.enable_proximity_rule = True

    def log_params(self):
        """Log the current optimal parameters for debugging."""
        logger.info("  OPTIMAL YOLOv12x PARAMETERS (Proven to work):")
        logger.info(f"   Confidence: {self.confidence_threshold} (KEY SUCCESS FACTOR)")
        logger.info(f"   Proximity Threshold: {self.proximity_threshold}")
        logger.info(f"   Track Timeout: {self.track_timeout_frames} frames")
        logger.info(f"   Track Buffer: {self.track_buffer}")
        logger.info(f"   Match Threshold: {self.match_thresh}")
        logger.info(f"   Proximity Rule: {'Enabled' if self.enable_proximity_rule else 'Disabled'}")


# ======================================
# Helper Classes
# ======================================

class FastCounter:
    """Real-time counter for line crossings."""

    def __init__(self, line_definitions: dict):
        self.line_defs = line_definitions
        self.fast_in_count = 0
        self.fast_out_count = 0
        self.last_positions = {}
        self.crossed_tracks = set()
        self.line_orientations = {}
        self.line_in_directions = {}
        for line_name, line_data in line_definitions.items():
            coords = line_data['coords']
            orientation = self._determine_line_orientation(coords[0], coords[1])
            self.line_orientations[line_name] = orientation
            self.line_in_directions[line_name] = line_data.get('inDirection', 'UP')

    def _determine_line_orientation(self, start_point, end_point):
        """Determine if line is horizontal or vertical."""
        x1, y1 = start_point
        x2, y2 = end_point
        horizontal_distance = abs(x2 - x1)
        vertical_distance = abs(y2 - y1)
        return "horizontal" if horizontal_distance > vertical_distance else "vertical"

    def update_tracks(self, detections: sv.Detections):
        if len(detections) == 0:
            return

        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            xyxy = detections.xyxy[i]
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2

            if track_id in self.last_positions:
                last_x, last_y = self.last_positions[track_id]
                for line_name, data in self.line_defs.items():
                    orientation = self.line_orientations[line_name]
                    if orientation == "horizontal":
                        line_coord = (data['coords'][0][1] + data['coords'][1][1]) / 2
                        last_coord = last_y
                        current_coord = center_y
                    else:
                        line_coord = (data['coords'][0][0] + data['coords'][1][0]) / 2
                        last_coord = last_x
                        current_coord = center_x

                    if (last_coord < line_coord < current_coord) or (last_coord > line_coord > current_coord):
                        track_key = f"{track_id}_{line_name}"
                        if track_key not in self.crossed_tracks:
                            if orientation == "horizontal":
                                travel_direction = "DOWN" if last_coord < line_coord else "UP"
                            else:
                                travel_direction = "LR" if last_coord < line_coord else "RL"
                            if travel_direction == self.line_in_directions[line_name]:
                                self.fast_in_count += 1
                            else:
                                self.fast_out_count += 1
                            self.crossed_tracks.add(track_key)

            self.last_positions[track_id] = (center_x, center_y)

    def get_counts(self):
        return self.fast_in_count, self.fast_out_count


class EnhancedCleanAnalyzer:
    """Analyzes tracks for line crossings with proximity and lenient rules."""

    def __init__(self, line_definitions: dict, params: OptimalParams):
        self.line_defs = line_definitions
        self.all_tracks = {}
        self.processed_tracks = set()
        self.clean_in_count, self.clean_out_count = 0, 0
        self.params = params
        self.line_orientations = {}
        self.line_in_directions = {}
        for line_name, line_data in line_definitions.items():
            coords = line_data['coords']
            orientation = self._determine_line_orientation(coords[0], coords[1])
            self.line_orientations[line_name] = orientation
            self.line_in_directions[line_name] = line_data.get('inDirection', 'UP')
            logger.info(
                f"📏 {line_name}: {orientation} line detected, in_direction={self.line_in_directions[line_name]}")

    def _determine_line_orientation(self, start_point, end_point):
        """Determine if a line is horizontal or vertical."""
        x1, y1 = start_point
        x2, y2 = end_point
        horizontal_distance = abs(x2 - x1)
        vertical_distance = abs(y2 - y1)
        return "horizontal" if horizontal_distance > vertical_distance else "vertical"

    def update_tracks(self, detections: sv.Detections, current_frame: int):
        """Update track positions and process timed-out tracks."""
        current_track_ids = set(detections.tracker_id) if len(detections) > 0 else set()
        for i in range(len(detections)):
            tracker_id, xyxy = detections.tracker_id[i], detections.xyxy[i]
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            if tracker_id not in self.all_tracks:
                self.all_tracks[tracker_id] = {'positions': [], 'last_frame': 0, 'analyzed': False}
            self.all_tracks[tracker_id]['positions'].append((current_frame, center_x, center_y))
            self.all_tracks[tracker_id]['last_frame'] = current_frame

        for track_id, data in self.all_tracks.items():
            if track_id not in current_track_ids and track_id not in self.processed_tracks and \
                    current_frame - data['last_frame'] > self.params.track_timeout_frames:
                self.process_track(track_id)

    def process_track(self, track_id: int):
        """Analyze a single track for line crossings."""
        track_data = self.all_tracks[track_id]
        if track_data['analyzed'] or len(track_data['positions']) < 2:
            track_data['analyzed'] = True
            self.processed_tracks.add(track_id)
            return

        positions = track_data['positions']
        start_pos, end_pos = positions[0], positions[-1]

        crossing_made = False
        for line_name, data in self.line_defs.items():
            orientation = self.line_orientations[line_name]
            if orientation == "horizontal":
                start_coord = start_pos[2]  # Y coordinate
                end_coord = end_pos[2]  # Y coordinate
                line_coord = (data['coords'][0][1] + data['coords'][1][1]) / 2
            else:
                start_coord = start_pos[1]  # X coordinate
                end_coord = end_pos[1]  # X coordinate
                line_coord = (data['coords'][0][0] + data['coords'][1][0]) / 2

            start_side = "after" if start_coord > line_coord else "before"
            end_side = "after" if end_coord > line_coord else "before"

            crossing_dist = abs(end_coord - start_coord)
            if start_side != end_side and crossing_dist >= self.params.min_crossing_distance:
                if orientation == "horizontal":
                    travel_direction = "DOWN" if start_coord < line_coord else "UP"
                else:
                    travel_direction = "LR" if start_coord < line_coord else "RL"
                if travel_direction == self.line_in_directions[line_name]:
                    self.clean_in_count += 1
                    logger.info(
                        f"✅ CLEAN IN count: Track {track_id} (moved {crossing_dist:.1f}px). Total: {self.clean_in_count}")
                else:
                    self.clean_out_count += 1
                    logger.info(
                        f"❌ CLEAN OUT count: Track {track_id} (moved {crossing_dist:.1f}px). Total: {self.clean_out_count}")
                crossing_made = True
                break
            elif start_side != end_side:
                if orientation == "horizontal":
                    travel_direction = "DOWN" if start_coord < line_coord else "UP"
                else:
                    travel_direction = "LR" if start_coord < line_coord else "RL"
                if travel_direction == self.line_in_directions[line_name]:
                    self.clean_in_count += 1
                    logger.info(f"✅ CLEAN IN count (Lenient): Track {track_id}. Total: {self.clean_in_count}")
                else:
                    self.clean_out_count += 1
                    logger.info(f"❌ CLEAN OUT count (Lenient): Track {track_id}. Total: {self.clean_out_count}")
                crossing_made = True
                break
            elif start_side == end_side and self.params.enable_proximity_rule and self.params.proximity_threshold > 0:
                if orientation == "horizontal":
                    min_prox_dist = min(abs(pos[2] - line_coord) for pos in positions)
                else:
                    min_prox_dist = min(abs(pos[1] - line_coord) for pos in positions)
                if min_prox_dist < self.params.proximity_threshold:
                    if start_side == "before":
                        in_direction = self.line_in_directions[line_name]
                        if in_direction in ["UP", "DOWN", "LR", "RL"]:
                            self.clean_in_count += 1
                            logger.info(f"✅ CLEAN IN count (Proximity): Track {track_id}. Total: {self.clean_in_count}")
                        else:
                            self.clean_out_count += 1
                            logger.info(
                                f"❌ CLEAN OUT count (Proximity): Track {track_id}. Total: {self.clean_out_count}")
                    else:
                        in_direction = self.line_in_directions[line_name]
                        if in_direction in ["UP", "DOWN", "LR", "RL"]:
                            self.clean_in_count += 1
                            logger.info(f"✅ CLEAN IN count (Proximity): Track {track_id}. Total: {self.clean_in_count}")
                        else:
                            self.clean_out_count += 1
                            logger.info(
                                f"❌ CLEAN OUT count (Proximity): Track {track_id}. Total: {self.clean_out_count}")
                    crossing_made = True
                    break

        track_data['analyzed'] = True
        self.processed_tracks.add(track_id)

    def finalize_analysis(self):
        """Finalize analysis for all tracks."""
        for track_id, track in self.all_tracks.items():
            if track_id not in self.processed_tracks:
                self.process_track(track_id)

    def get_counts(self):
        return self.clean_in_count, self.clean_out_count


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
        return False, f"File size {size / (1024 * 1024):.2f}MB exceeds 500MB limit"

    return True, ""


# ======================================
# Main Analysis Function
# ======================================

def run_optimal_yolov12x_counting(video_path: str, line_definitions: dict, custom_params: OptimalParams = None, output_path: str = None, job_id: str = None) -> Dict:
    """
    Count people crossing lines in a video for emergency analysis using YOLOv12x and BotSort.

    Args:
        video_path: Path to input video
        line_definitions: Dictionary of line configs (coords and direction)
        custom_params: Optional custom parameters
        output_path: Path to save output video (for tasks.py integration)
        job_id: VideoJob ID for progress tracking

    Returns:
        Standardized response dictionary with filesystem paths
    """
    start_time = time.time()

    # Add job_id logging for progress tracking
    if job_id:
        logger.info(f"🚀 Starting emergency count job {job_id}")

    # Validate file exists and is accessible
    try:
        if not default_storage.exists(video_path):
            error_msg = f"Video file not found: {video_path}"
            logger.error(f"Invalid input: {error_msg}")
            return {
                'status': 'failed',
                'job_type': 'emergency_count',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': error_msg},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': error_msg, 'code': 'FILE_NOT_FOUND'}
            }
    except Exception as e:
        error_msg = f"Error accessing video file: {str(e)}"
        logger.error(f"File access error: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'emergency_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'FILE_ACCESS_ERROR'}
        }

    # Validate file format
    is_valid, error_msg = validate_input_file(video_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'emergency_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    # Validate emergency lines configuration
    try:
        if not emergency_lines or not isinstance(emergency_lines, list):
            raise ValueError("Emergency lines configuration is required and must be a list")

        if len(emergency_lines) != 2:
            raise ValueError("Exactly 2 emergency lines are required (entry and exit)")

        for i, line in enumerate(emergency_lines):
            if not isinstance(line, dict):
                raise ValueError(f"Emergency line {i+1} must be a dictionary")
            required_fields = ['startX', 'startY', 'endX', 'endY']
            if not all(field in line for field in required_fields):
                raise ValueError(f"Emergency line {i+1} missing required fields: {required_fields}")
            if not all(isinstance(line[field], (int, float)) for field in required_fields):
                raise ValueError(f"Emergency line {i+1} coordinates must be numbers")

        # Validate video dimensions if provided
        if video_width is not None and (not isinstance(video_width, (int, float)) or video_width <= 0):
            raise ValueError("Video width must be a positive number")
        if video_height is not None and (not isinstance(video_height, (int, float)) or video_height <= 0):
            raise ValueError("Video height must be a positive number")

    except Exception as e:
        error_msg = f"Line configuration error: {str(e)}"
        logger.error(error_msg)
        return {
            'status': 'failed',
            'job_type': 'emergency_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'LINE_CONFIG_ERROR'}
        }

    try:
        # Open video
        with default_storage.open(video_path, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

        video_info = sv.VideoInfo.from_video_path(tmp_path)
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video
        # Extract job ID from video path if not provided as parameter
        extracted_job_id = re.search(r'(\d+)', video_path)
        file_job_id = extracted_job_id.group(1) if extracted_job_id else str(int(time.time()))
        output_filename = f"outputs/output_{file_job_id}.mp4"
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
            writer = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            # Setup parameters
            params = custom_params if custom_params else OptimalParams()
            params.log_params()

            # Load model
            model = load_yolo_model(str(MODELS_DIR / "yolo12x.pt"))
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(device)
            logger.info(f"Using device: {device}")

            # Initialize tracker
            tracker = BotSort(
                reid_weights=REID_MODEL_PATH,
                device=device,
                half=False,
                track_buffer=params.track_buffer,
                match_thresh=params.match_thresh
            )

            # Initialize counters
            fast_counter = FastCounter(line_definitions)
            clean_analyzer = EnhancedCleanAnalyzer(line_definitions, params)
            box_annotator = sv.BoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=5)
            frame_number = 0
            last_log_time = start_time
            alerts = []

            with tqdm(total=video_info.total_frames, desc="Optimal YOLOv12x Processing") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_number += 1

                    # YOLO prediction
                    yolo_results = model.predict(
                        source=frame,
                        verbose=False,
                        device=device,
                        conf=params.confidence_threshold,
                        classes=[0]
                    )[0]

                    detections_for_tracker = yolo_results.boxes.data.cpu().numpy()
                    tracked_detections = sv.Detections.empty()
                    if len(detections_for_tracker) > 0:
                        tracks = tracker.update(detections_for_tracker, frame)
                        if tracks.shape[0] > 0:
                            tracked_detections = sv.Detections(
                                xyxy=tracks[:, :4],
                                class_id=tracks[:, 6].astype(int),
                                confidence=tracks[:, 5],
                                tracker_id=tracks[:, 4].astype(int)
                            )
                    else:
                        tracker.update(np.empty((0, 6)), frame)

                    # Update counters
                    fast_counter.update_tracks(tracked_detections)
                    clean_analyzer.update_tracks(tracked_detections, frame_number)

                    # Check for fast movement
                    if len(tracked_detections) > 0:
                        speeds = [np.linalg.norm(np.array(bbox[:2]) - np.array(bbox[2:])) for bbox in
                                  tracked_detections.xyxy]
                        if any(speed > 10 for speed in speeds):
                            alerts.append({
                                "message": f"Fast movement detected at frame {frame_number}",
                                "timestamp": timezone.now().isoformat()
                            })

                    # Visualization
                    annotated_frame = frame.copy()
                    if len(tracked_detections) > 0:
                        labels = [f"ID:{tracker_id}" for tracker_id in tracked_detections.tracker_id]
                        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
                        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections,
                                                                   labels=labels)

                    h, w = annotated_frame.shape[:2]
                    colors = [(0, 0, 255), (0, 255, 255)]
                    for idx, (name, data) in enumerate(line_definitions.items()):
                        pt1 = list(map(int, data['coords'][0]))
                        pt2 = list(map(int, data['coords'][1]))
                        pt1[0] = max(0, min(pt1[0], w - 1))
                        pt1[1] = max(0, min(pt1[1], h - 1))
                        pt2[0] = max(0, min(pt2[0], w - 1))
                        pt2[1] = max(0, min(pt2[1], h - 1))
                        color = colors[idx % len(colors)]
                        cv2.line(annotated_frame, tuple(pt1), tuple(pt2), color, 6)
                        cv2.line(annotated_frame, tuple(pt1), tuple(pt2), (255, 255, 255), 2)
                        mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                        cv2.putText(
                            annotated_frame,
                            f"{data.get('inDirection', 'UP')} = IN",
                            (mid_x, mid_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            color,
                            3
                        )

                    fast_in, fast_out = fast_counter.get_counts()
                    total_in_out = f"OPTIMAL YOLOv12x - IN: {fast_in} | OUT: {fast_out} | Total: {fast_in + fast_out}"
                    cv2.putText(
                        annotated_frame,
                        total_in_out,
                        (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )

                    writer.write(annotated_frame)
                    pbar.update(1)

                    # Periodic logging
                    current_time = time.time()
                    if current_time - last_log_time >= 5 or frame_number == total_frames:
                        progress = (frame_number / total_frames) * 100
                        elapsed_time = current_time - start_time
                        time_remaining = (elapsed_time / frame_number) * (
                                    total_frames - frame_number) if frame_number > 0 else 0
                        avg_fps = frame_number / elapsed_time if elapsed_time > 0 else 0
                        logger.info(
                            f"**Job {job_id}**: Progress **{progress:.1f}%** ({frame_number}/{total_frames}), Status: Processing...")
                        logger.info(
                            f"[{'#' * int(progress // 10)}{'-' * (10 - int(progress // 10))}] Done: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d} | Left: {int(time_remaining // 60):02d}:{int(time_remaining % 60):02d} | Avg FPS: {avg_fps:.1f}")
                        last_log_time = current_time

            cap.release()
            writer.release()

            # Create temporary file for web conversion
            from ..convert import convert_to_web_mp4
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='_web.mp4', delete=False) as web_tmp:
                web_tmp_path = web_tmp.name

            logger.info(f"🔄 Attempting ffmpeg conversion: {tmp_out.name} -> {web_tmp_path}")
            if convert_to_web_mp4(tmp_out.name, web_tmp_path):
                final_output_path = web_tmp_path  # Use converted file
                logger.info(f"✅ FFmpeg conversion successful")
            else:
                final_output_path = tmp_out.name  # Fallback to original
                logger.warning(f"⚠️ FFmpeg conversion failed, using original file")

            # Finalize analysis
            clean_analyzer.finalize_analysis()
            final_clean_in, final_clean_out = clean_analyzer.get_counts()
            fast_in, fast_out = fast_counter.get_counts()

            processing_time = time.time() - start_time
            return {
                'status': 'completed',
                'job_type': 'emergency_count',
                'output_image': None,
                'output_video': final_output_path,
                'data': {
                    'in_count': final_clean_in,
                    'out_count': final_clean_out,
                    'fast_in_count': fast_in,
                    'fast_out_count': fast_out,
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
            'job_type': 'emergency_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }
    finally:
        # Note: final_output_path is not cleaned up here as tasks.py needs it
        # tasks.py will handle cleanup after saving to Django storage
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        # Only clean up tmp_out if it's not the final_output_path
        if ('tmp_out' in locals() and 'final_output_path' in locals() and
            tmp_out.name != final_output_path and os.path.exists(tmp_out.name)):
            os.remove(tmp_out.name)


def tracking_video(video_path: str, output_path: str, line_configs: dict, video_width: int = None,
                   video_height: int = None, job_id: str = None) -> Dict:
    """
    Celery task for emergency counting.

    Args:
        self: Celery task instance
        video_path: Path to input video
        output_path: Path to save output video
        line_configs: Dictionary of line configs
        video_width: Video width
        video_height: Video height
        job_id: VideoJob ID

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    logger.info(f"🚀 Starting emergency count job {job_id}")

    # Initialize progress logger for video processing
    progress_logger = create_progress_logger(
        job_id=str(job_id) if job_id else "unknown",
        total_items=100,  # Estimate for video frames
        job_type="emergency_count"
    )

    progress_logger.update_progress(0, status="Starting emergency counting analysis...", force_log=True)
    result = run_optimal_yolov12x_counting(video_path, line_configs, None, output_path, job_id)
    progress_logger.update_progress(100, status="Emergency counting completed", force_log=True)
    progress_logger.log_completion(100)

    # Update Celery task state
    result['meta']['processing_time_seconds'] = time.time() - start_time
    result['meta']['timestamp'] = timezone.now().isoformat()

    return result
