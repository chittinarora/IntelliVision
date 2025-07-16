"""
emergency_count.py - Emergency Counting Analytics

Implements YOLO-based emergency counting and tracking logic for video analytics jobs.
Includes optimal parameter tuning, line crossing logic, and track analysis for emergency scenarios.
"""

# === Imports ===
import os
import pathlib
import time
import json
import math
import argparse
import logging
import random
import re

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import torch
from boxmot import BotSort
from apps.video_analytics.models.utils import load_yolo_model

# --- Setup logger for this module ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =================================================================================
# OPTIMAL PARAMETERS FOR YOLOv12x (BASED ON SUCCESSFUL EXPERIMENT)
# =================================================================================
class OptimalParams:
    """
    Stores optimal parameters for YOLOv12x emergency counting, based on experimental results.
    """
    def __init__(self):
        self.min_crossing_distance = 50
        self.proximity_threshold = 25
        self.track_timeout_frames = 45
        self.track_buffer = 120
        self.match_thresh = 0.7
        self.confidence_threshold = 0.10
        self.enable_proximity_rule = True

    def log_params(self):
        """
        Log the current optimal parameters for debugging and reproducibility.
        """
        logger.info("  OPTIMAL YOLOv12x PARAMETERS (Proven to work):")
        logger.info(f"   Confidence: {self.confidence_threshold} (KEY SUCCESS FACTOR)")
        logger.info(f"   Proximity Threshold: {self.proximity_threshold}")
        logger.info(f"   Track Timeout: {self.track_timeout_frames} frames")
        logger.info(f"   Track Buffer: {self.track_buffer}")
        logger.info(f"   Match Threshold: {self.match_thresh}")
        logger.info(f"   Proximity Rule: {'Enabled' if self.enable_proximity_rule else 'Disabled'}")

# =================================================================================
# ENHANCED CLEAN ANALYZER
# =================================================================================
class EnhancedCleanAnalyzer:
    """
    Analyzes tracks and line crossings for emergency counting.
    Determines in/out counts based on line orientation, direction, and proximity rules.
    """
    def __init__(self, line_definitions: dict, params: OptimalParams):
        self.line_defs = line_definitions
        self.all_tracks = {}
        self.processed_tracks = set()
        self.clean_in_count, self.clean_out_count = 0, 0
        self.params = params

        # Analyze line orientations for proper tracking
        self.line_orientations = {}
        self.line_in_directions = {}
        for line_name, line_data in line_definitions.items():
            coords = line_data['coords']
            orientation = self._determine_line_orientation(coords[0], coords[1])
            self.line_orientations[line_name] = orientation
            self.line_in_directions[line_name] = line_data.get('inDirection', 'UP')
            logger.info(f"📏 {line_name}: {orientation} line detected, in_direction={self.line_in_directions[line_name]}")

    def _determine_line_orientation(self, start_point, end_point):
        """
        Determine if a line is horizontal or vertical based on its endpoints.
        """
        x1, y1 = start_point
        x2, y2 = end_point

        horizontal_distance = abs(x2 - x1)
        vertical_distance = abs(y2 - y1)

        return "horizontal" if horizontal_distance > vertical_distance else "vertical"

    def update_tracks(self, detections: sv.Detections, current_frame: int):
        """
        Update track positions and process tracks that have timed out.
        """
        current_track_ids = set(detections.tracker_id) if len(detections) > 0 else set()
        for i in range(len(detections)):
            tracker_id, xyxy = detections.tracker_id[i], detections.xyxy[i]

            # Store both X and Y coordinates for proper tracking
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2

            if tracker_id not in self.all_tracks:
                self.all_tracks[tracker_id] = {'positions': [], 'last_frame': 0, 'analyzed': False}
            self.all_tracks[tracker_id]['positions'].append((current_frame, center_x, center_y))
            self.all_tracks[tracker_id]['last_frame'] = current_frame

        # Process tracks that have timed out
        for track_id, data in self.all_tracks.items():
            if track_id not in current_track_ids and track_id not in self.processed_tracks and current_frame - data[
                'last_frame'] > self.params.track_timeout_frames:
                self.process_track(track_id)

    def process_track(self, track_id: int):
        """
        Analyze a single track for line crossings and update in/out counts.
        """
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
                # For horizontal lines, track Y-movement (UP/DOWN)
                start_coord = start_pos[2]  # Y coordinate
                end_coord = end_pos[2]  # Y coordinate
                line_coord = (data['coords'][0][1] + data['coords'][1][1]) / 2  # Line Y position
            else:
                # For vertical lines, track X-movement (LR/RL)
                start_coord = start_pos[1]  # X coordinate
                end_coord = end_pos[1]  # X coordinate
                line_coord = (data['coords'][0][0] + data['coords'][1][0]) / 2  # Line X position

            # Determine which side of line the track started and ended
            start_side = "after" if start_coord > line_coord else "before"
            end_side = "after" if end_coord > line_coord else "before"

            # Rule 1: Standard Crossing
            crossing_dist = abs(end_coord - start_coord)
            if start_side != end_side and crossing_dist >= self.params.min_crossing_distance:

                if orientation == "horizontal":
                    # UP/DOWN movement
                    travel_direction = "DOWN" if start_coord < line_coord else "UP"
                else:
                    # LR/RL movement
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

            # Rule 2: Lenient crossing
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

            # Rule 3: Proximity rule
            elif start_side == end_side and self.params.enable_proximity_rule and self.params.proximity_threshold > 0:
                if orientation == "horizontal":
                    min_prox_dist = min(abs(pos[2] - line_coord) for pos in positions)  # Y distance
                else:
                    min_prox_dist = min(abs(pos[1] - line_coord) for pos in positions)  # X distance

                if min_prox_dist < self.params.proximity_threshold:
                    if start_side == "before":
                        # For proximity rule, use the inDirection to determine if movement is "in"
                        in_direction = self.line_in_directions[line_name]
                        if in_direction in ["UP", "DOWN", "LR", "RL"]:
                            self.clean_in_count += 1
                            logger.info(f"✅ CLEAN IN count (Proximity): Track {track_id}. Total: {self.clean_in_count}")
                        else:
                            self.clean_out_count += 1
                            logger.info(f"❌ CLEAN OUT count (Proximity): Track {track_id}. Total: {self.clean_out_count}")
                    else:
                        # For proximity rule, use the inDirection to determine if movement is "in"
                        in_direction = self.line_in_directions[line_name]
                        if in_direction in ["UP", "DOWN", "LR", "RL"]:
                            self.clean_in_count += 1
                            logger.info(f"✅ CLEAN IN count (Proximity): Track {track_id}. Total: {self.clean_in_count}")
                        else:
                            self.clean_out_count += 1
                            logger.info(f"❌ CLEAN OUT count (Proximity): Track {track_id}. Total: {self.clean_out_count}")
                    crossing_made = True
                    break

        track_data['analyzed'] = True
        self.processed_tracks.add(track_id)

    def finalize_analysis(self):
        """
        Finalize analysis for all tracks that have not yet been processed.
        """
        for track_id, track in self.all_tracks.items():
            if track_id not in self.processed_tracks:
                self.process_track(track_id)

    def get_counts(self):
        return self.clean_in_count, self.clean_out_count


# =================================================================================
# FAST COUNTER FOR REAL-TIME DISPLAY
# =================================================================================
class FastCounter:
    def __init__(self, line_definitions: dict):
        self.line_defs = line_definitions
        self.fast_in_count = 0
        self.fast_out_count = 0
        self.last_positions = {}
        self.crossed_tracks = set()

        # Analyze line orientations for proper tracking
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
                        # For horizontal lines, check Y-coordinate crossings
                        line_coord = (data['coords'][0][1] + data['coords'][1][1]) / 2
                        last_coord = last_y
                        current_coord = center_y
                    else:
                        # For vertical lines, check X-coordinate crossings
                        line_coord = (data['coords'][0][0] + data['coords'][1][0]) / 2
                        last_coord = last_x
                        current_coord = center_x

                    # Check if crossed the line
                    if (last_coord < line_coord < current_coord) or (last_coord > line_coord > current_coord):
                        track_key = f"{track_id}_{line_name}"
                        if track_key not in self.crossed_tracks:

                            if orientation == "horizontal":
                                # UP/DOWN movement
                                travel_direction = "DOWN" if last_coord < line_coord else "UP"
                            else:
                                # LR/RL movement
                                travel_direction = "LR" if last_coord < line_coord else "RL"

                            if travel_direction == self.line_in_directions[line_name]:
                                self.fast_in_count += 1
                            else:
                                self.fast_out_count += 1

                            self.crossed_tracks.add(track_key)

            self.last_positions[track_id] = (center_x, center_y)

    def get_counts(self):
        return self.fast_in_count, self.fast_out_count


# =================================================================================
# MAIN PROCESSING FUNCTION WITH OPTIMAL PARAMETERS
# =================================================================================
def run_optimal_yolov12x_counting(video_path: str, line_definitions: dict, custom_params: OptimalParams = None):
    """Run people counting with proven optimal parameters."""

    params = custom_params if custom_params else OptimalParams()

    logger.info("🚀 RUNNING OPTIMAL YOLOv12x PEOPLE COUNTING")
    params.log_params()

    # Load YOLOv12x model
    model = load_yolo_model('../models/yolo12x.pt')
    logger.info("🔧 Using YOLOv12x (Optimal Configuration)")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Using device: {device}")

    video_info = sv.VideoInfo.from_video_path(video_path)

    # Initialize counters
    fast_counter = FastCounter(line_definitions)
    clean_analyzer = EnhancedCleanAnalyzer(line_definitions, params)

    # Initialize tracker
    reid_path = pathlib.Path("osnet_x0_25_msmt17.pt")
    tracker = BotSort(
        reid_weights=reid_path,
        device=device,
        half=False,
        track_buffer=params.track_buffer,
        match_thresh=params.match_thresh
    )

    # Video processing setup
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=5)

    # Use job ID for output naming if available
    job_id = None
    # Try to extract job_id from video_path or line_definitions if possible
    match = re.search(r'(\d+)', os.path.basename(video_path))
    if match:
        job_id = match.group(1)
    else:
        job_id = str(int(time.time()))  # fallback to timestamp
    output_name = f"output_{job_id}.mp4"

    writer = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, video_info.resolution_wh)

    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    frame_count = 0
    with tqdm(total=video_info.total_frames, desc="Optimal YOLOv12x Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            frame_count += 1

            # YOLO prediction with optimal confidence
            yolo_results = model.predict(
                source=frame,
                verbose=False,
                device=device,
                conf=params.confidence_threshold,
                classes=[0]  # Only detect persons
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

            # Get current counts
            fast_in, fast_out = fast_counter.get_counts()

            # Visualization
            annotated_frame = frame.copy()

            # Draw detections (boxes, labels, etc.)
            if len(tracked_detections) > 0:
                labels = [f"ID:{tracker_id}" for tracker_id in tracked_detections.tracker_id]
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections,
                                                           labels=labels)

            # Draw counting lines as the LAST overlay, with high visibility
            h, w = annotated_frame.shape[:2]
            colors = [(0, 0, 255), (0, 255, 255)]  # Red, Yellow
            for idx, (name, data) in enumerate(line_definitions.items()):
                pt1 = list(map(int, data['coords'][0]))
                pt2 = list(map(int, data['coords'][1]))
                # Clamp coordinates to frame bounds
                pt1[0] = max(0, min(pt1[0], w-1))
                pt1[1] = max(0, min(pt1[1], h-1))
                pt2[0] = max(0, min(pt2[0], w-1))
                pt2[1] = max(0, min(pt2[1], h-1))
                color = colors[idx % len(colors)]
                cv2.line(annotated_frame, tuple(pt1), tuple(pt2), color, 6)
                # Draw a contrasting border for extra visibility
                cv2.line(annotated_frame, tuple(pt1), tuple(pt2), (255,255,255), 2)
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

            # Show counts
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

    cap.release()
    writer.release()

    # Convert to web-friendly MP4
    from apps.video_analytics.convert import convert_to_web_mp4
    web_output_path = output_name.replace('.mp4', '_web.mp4')
    if convert_to_web_mp4(output_name, web_output_path):
        final_output_path = web_output_path
    else:
        final_output_path = output_name

    # Finalize analysis
    clean_analyzer.finalize_analysis()
    final_clean_in, final_clean_out = clean_analyzer.get_counts()

    # Results logging with safe handling for possible None values and detailed comments

    # Log the results of the optimal YOLOv12x analysis
    logger.info("\n📊 OPTIMAL YOLOV12x RESULTS:")

    # Safely compute totals, treating None as 0 to avoid TypeError
    safe_fast_in = fast_in if fast_in is not None else 0
    safe_fast_out = fast_out if fast_out is not None else 0
    total_fast = safe_fast_in + safe_fast_out

    logger.info(
        f"🚀 Real-time Counter -> IN: {safe_fast_in} | OUT: {safe_fast_out} | Total: {total_fast}"
    )

    # Safely compute clean counter totals
    safe_clean_in = final_clean_in if final_clean_in is not None else 0
    safe_clean_out = final_clean_out if final_clean_out is not None else 0
    total_clean = safe_clean_in + safe_clean_out

    logger.info(
        f"✨ Clean Counter    -> IN: {safe_clean_in} | OUT: {safe_clean_out} | Total: {total_clean}"
    )

    # Log the final output video path
    logger.info(f"📹 Output video saved to: {final_output_path}")

    # Return a unified result dictionary
    return {
        'status': 'completed',
        'job_type': 'emergency_count',
        'output_video': final_output_path,
        'data': {
            'in_count': final_clean_in,
            'out_count': final_clean_out,
            'fast_in_count': fast_in,
            'fast_out_count': fast_out,
        },
        'meta': {},
        'error': None
    }


# =================================================================================
# FRESH LINE DRAWING WORKFLOW - ALWAYS DRAW NEW LINES
# =================================================================================
def fresh_line_workflow(video_path: str):
    """Always draw fresh lines, ignore any saved configurations."""

    video_name = os.path.basename(video_path)
    logger.info(f"🎯 Drawing fresh lines for {video_name} (ignoring any saved config)")

    # Always launch interactive line placement
    try:
        logger.info("🖊️ Starting interactive line placement...")
        # Import InteractiveLinePlacer here to avoid NameError if not already imported
        from apps.video_analytics.interactive.line_placer import InteractiveLinePlacer
        placer = InteractiveLinePlacer(video_path)
        lines = placer.draw_lines_interactive()

        if lines:
            logger.info("✅ Lines configured successfully. Starting counting...")
            return lines
        else:
            logger.warning("❌ No lines configured. Cannot proceed with counting.")
            return None

    except Exception as e:
        logger.error(f"❌ Error during line placement: {e}")
        import traceback
        traceback.print_exc()
        return None


# =================================================================================
# MAIN FUNCTION - FRESH LINES EVERY TIME
# =================================================================================
def main():
    """Main function with fresh line drawing every time."""
    parser = argparse.ArgumentParser(description="Smart YOLOv12x People Counter - Fresh Lines Every Time")
    parser.add_argument("--video", required=True, help="Path to the video file")

    # Optional parameter overrides (for advanced users)
    parser.add_argument("--confidence", type=float, help="Override optimal confidence (default: 0.10)")
    parser.add_argument("--proximity", type=int, help="Override proximity threshold (default: 25)")
    parser.add_argument("--timeout", type=int, help="Override track timeout (default: 45)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        logger.error(f"❌ Video not found: {args.video}")
        return

    logger.info("🎨 FRESH LINE COUNTER - Draw Lines & Count")
    logger.info("=" * 45)

    # Always draw fresh lines
    line_definitions = fresh_line_workflow(args.video)

    if not line_definitions:
        logger.error("❌ Failed to configure lines. Exiting.")
        return

    # Set up parameters (use optimal defaults or user overrides)
    params = OptimalParams()
    if args.confidence:
        params.confidence_threshold = args.confidence
        logger.info(f"🔧 Overriding confidence: {args.confidence}")
    if args.proximity:
        params.proximity_threshold = args.proximity
        logger.info(f"🔧 Overriding proximity: {args.proximity}")
    if args.timeout:
        params.track_timeout_frames = args.timeout
        logger.info(f"🔧 Overriding timeout: {args.timeout}")

    # Run the counting automatically
    try:
        logger.info("🚀 Starting people counting...")
        result = run_optimal_yolov12x_counting(args.video, line_definitions, params)

        logger.info(f"\n🎯 FINAL RESULTS:")
        logger.info(f"   IN: {result['data']['in_count']}")
        logger.info(f"   OUT: {result['data']['out_count']}")
        logger.info(f"   NET: {result['data']['in_count'] - result['data']['out_count']}")

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)


def tracking_video(input_path: str, output_path: str, line_coords_dict: dict, video_width=None, video_height=None) -> dict:
    """
    Wrapper for Celery integration. Runs emergency counting and returns results in a standard format.
    Args:
        input_path: Path to the input video file
        output_path: Path where the annotated video should be saved
        line_coords_dict: Dictionary of line configs (with coords and direction)
        video_width: Optional video width
        video_height: Optional video height
    Returns:
        dict with unified result structure
    """
    # Pass all arguments to the main function
    return run_optimal_yolov12x_counting(
        video_path=input_path,
        line_definitions=line_coords_dict,
    )


if __name__ == "__main__":
    main()
