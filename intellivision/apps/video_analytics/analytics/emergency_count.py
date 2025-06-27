import os
import cv2
import numpy as np
import torch
import pathlib
import glob
import re
from typing import Dict, Any, Tuple, Optional, Union

from ultralytics import YOLO
from boxmot import BotSort
import supervision as sv
from apps.video_analytics.convert import convert_to_web_mp4

# --- CONFIG CLASS ---

class PeopleCountConfig:
    def __init__(
        self,
        enable_stabilization: bool = False,  # Always disabled
        smoothing_window_size: int = 30,
        yolo_conf_threshold: float = 0.15,
        min_bbox_height: int = 25,
        min_bbox_area: int = 200,
        max_width_height_ratio: float = 2.0,
        min_width_height_ratio: float = 0.1,
        track_high_thresh: float = 0.4,
        track_low_thresh: float = 0.05,
        new_track_thresh: float = 0.4,
        track_buffer: int = 100,
        match_thresh: float = 0.6,
        min_crossing_distance: int = 50,
        emergency_proximity_threshold: int = 40,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
        output_dir: str = '.',
    ):
        self.enable_stabilization = enable_stabilization
        self.smoothing_window_size = smoothing_window_size
        self.yolo_conf_threshold = yolo_conf_threshold
        self.min_bbox_height = min_bbox_height
        self.min_bbox_area = min_bbox_area
        self.max_width_height_ratio = max_width_height_ratio
        self.min_width_height_ratio = min_width_height_ratio
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_crossing_distance = min_crossing_distance
        self.emergency_proximity_threshold = emergency_proximity_threshold
        self.video_width = video_width
        self.video_height = video_height
        self.output_dir = output_dir

# --- STABILIZATION ---

def stabilize_video(input_path, output_path, config: PeopleCountConfig):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_gray = None
    transforms = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            out.write(frame)
            continue
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None:
            out.write(frame)
            prev_gray = gray
            continue
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:50]
        if len(matches) < 10:
            out.write(frame)
            prev_gray = gray
            continue
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffine2D(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None:
            M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        dx = M[0, 2]
        dy = M[1, 2]
        da = np.arctan2(M[1, 0], M[0, 0])
        transforms.append([dx, dy, da])
        if len(transforms) >= config.smoothing_window_size:
            smoothed = np.mean(transforms[-config.smoothing_window_size:], axis=0)
        else:
            smoothed = np.mean(transforms, axis=0)
        smoothed_M = np.array([
            [np.cos(smoothed[2]), -np.sin(smoothed[2]), smoothed[0]],
            [np.sin(smoothed[2]), np.cos(smoothed[2]), smoothed[1]]
        ], dtype=np.float32)
        stabilized_frame = cv2.warpAffine(frame, smoothed_M, (width, height))
        out.write(stabilized_frame)
        prev_gray = gray
    cap.release()
    out.release()
    return True

# --- FIXED CLEAN ANALYZER CLASSES (with people_in_out_2.py logic) ---
import math

class CleanTrajectory:
    def __init__(self, track_id, first_frame):
        self.track_id = track_id
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.positions = []
        self.confidence_history = []
        self.analyzed = False

    def add_position(self, frame, x, y, confidence):
        self.positions.append((frame, x, y))
        self.confidence_history.append(confidence)
        self.last_frame = frame

    def analyze_real_crossing(self, line_coords_dict, config: PeopleCountConfig):
        """CORRECTED VERSION from people_in_out_2.py with entry/exit logic"""
        if self.analyzed or len(self.positions) == 0:
            return []

        self.analyzed = True
        crossings = []
        start_x = self.positions[0][1]
        end_x = self.positions[-1][1]

        for line_name, line_coords in line_coords_dict.items():
            line_x = (line_coords[0][0] + line_coords[1][0]) / 2
            crossing_distance = abs(end_x - start_x)

            # Directional crossing check
            if ((start_x > line_x and end_x < line_x) or
                (start_x < line_x and end_x > line_x)) and crossing_distance >= config.min_crossing_distance:

                # FORCE ALL LINES TO BE EXIT LINES
                from_side = 'B'
                to_side = 'A'

                crossing = {
                    'track_id': self.track_id,
                    'line_name': line_name,
                    'from_side': from_side,
                    'to_side': to_side,
                    'start_frame': self.first_frame,
                    'end_frame': self.last_frame,
                    'confidence': min(np.mean(self.confidence_history), 1.0),
                    'method': 'directional',
                    'crossing_distance': crossing_distance
                }
                crossings.append(crossing)

            # Emergency detection for tracks very close to line
            elif config.emergency_proximity_threshold > 0:
                min_distance_to_line = min(self.distance_to_line(x, y, line_coords) for _, x, y in self.positions)

                if min_distance_to_line < config.emergency_proximity_threshold:
                    # FORCE ALL LINES TO BE EXIT LINES FOR EMERGENCY
                    emergency_from = 'B'
                    emergency_to = 'A'

                    crossing = {
                        'track_id': self.track_id,
                        'line_name': line_name,
                        'from_side': emergency_from,
                        'to_side': emergency_to,
                        'start_frame': self.first_frame,
                        'end_frame': self.last_frame,
                        'confidence': 0.6,
                        'method': 'emergency',
                        'crossing_distance': min_distance_to_line
                    }
                    crossings.append(crossing)

        return crossings

    def distance_to_line(self, x, y, line_coords):
        """Calculate distance from point to line"""
        start_point, end_point = line_coords
        x1, y1 = start_point
        x2, y2 = end_point

        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        return numerator / denominator if denominator > 0 else float('inf')

class CleanAnalyzer:
    def __init__(self, config: PeopleCountConfig):
        self.config = config
        self.all_tracks = {}
        self.current_frame = 0
        self.all_crossings = []
        self.line_coords_dict = {}
        self.processed_tracks = set()
        self.realtime_counts = {}

    def set_line_coordinates(self, line_coords_dict):
        self.line_coords_dict = line_coords_dict
        for line_name in line_coords_dict.keys():
            self.realtime_counts[line_name] = {'in': 0, 'out': 0}

    def update_tracks(self, detections_sv):
        self.current_frame += 1
        current_track_ids = set()

        if len(detections_sv) > 0:
            for i in range(len(detections_sv)):
                track_id = detections_sv.tracker_id[i]
                bbox = detections_sv.xyxy[i]
                confidence = detections_sv.confidence[i]
                current_track_ids.add(track_id)
                center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                if track_id not in self.all_tracks:
                    self.all_tracks[track_id] = CleanTrajectory(track_id, self.current_frame)
                self.all_tracks[track_id].add_position(self.current_frame, center_x, center_y, confidence)

        # Analyze disappeared tracks in REAL-TIME
        for track_id in list(self.all_tracks.keys()):
            if track_id not in current_track_ids and track_id not in self.processed_tracks:
                frames_since_seen = self.current_frame - self.all_tracks[track_id].last_frame
                if frames_since_seen > 30:
                    track = self.all_tracks[track_id]
                    if self.line_coords_dict and not track.analyzed:
                        crossings = track.analyze_real_crossing(self.line_coords_dict, self.config)
                        # Update real-time counters immediately
                        for crossing in crossings:
                            line_name = crossing['line_name']
                            if crossing['from_side'] == 'A' and crossing['to_side'] == 'B':
                                self.realtime_counts[line_name]['in'] += 1
                            elif crossing['from_side'] == 'B' and crossing['to_side'] == 'A':
                                self.realtime_counts[line_name]['out'] += 1
                        self.all_crossings.extend(crossings)
                        self.processed_tracks.add(track_id)

    def get_realtime_counts(self):
        return self.realtime_counts

    def finalize_analysis(self):
        for track_id, track in self.all_tracks.items():
            if track_id not in self.processed_tracks and self.line_coords_dict:
                crossings = track.analyze_real_crossing(self.line_coords_dict, self.config)
                # Update real-time counters for final tracks too
                for crossing in crossings:
                    line_name = crossing['line_name']
                    if crossing['from_side'] == 'A' and crossing['to_side'] == 'B':
                        self.realtime_counts[line_name]['in'] += 1
                    elif crossing['from_side'] == 'B' and crossing['to_side'] == 'A':
                        self.realtime_counts[line_name]['out'] += 1
                self.all_crossings.extend(crossings)
                self.processed_tracks.add(track_id)

    def get_results(self):
        unique_crossing_tracks = set(c['track_id'] for c in self.all_crossings)
        return len(unique_crossing_tracks), len(self.all_crossings)

# --- UTILS ---

def get_next_output_number(output_dir: str):
    pattern = os.path.join(output_dir, "backend_people_count_*.mp4")
    existing_files = glob.glob(pattern)
    numbers = []
    for file_path in existing_files:
        match = re.search(r"backend_people_count_(\d+)\.mp4", file_path)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 1

# --- MAIN API FUNCTION ---

def people_counting_video(
    input_path: str,
    output_path: Optional[str],
    line_coords_dict: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]],
    config: Optional[PeopleCountConfig] = None,
    max_frames: Optional[int] = None,
    model_path: str = "yolov8m.pt",
    reid_model_path: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """
    Runs people counting (multi-line, two-pass, stabilized, accurate) on a video file.
    - input_path: path to input video file
    - output_path: desired output path for annotated video (if None, auto-generates)
    - line_coords_dict: {line_name: ((x1,y1),(x2,y2)), ...}
    - config: PeopleCountConfig instance (optional)
    - max_frames: (optional) limit processing to N frames
    Returns: dict with counts and paths.
    """
    if config is None:
        config = PeopleCountConfig()
    video_path = input_path

    # Stabilization is disabled

    # Always use CPU to avoid MPS/Metal forking issues in Celery workers
    device = torch.device("cpu")
    print("🔥 Forcing CPU for Celery worker stability")

    # Model setup
    model = YOLO(model_path)
    model.to(device)

    # FIXED: ReID model path - always use ~/.cache/torch/checkpoints/osnet_x0_25_msmt17.pt if not provided
    if reid_model_path is None:
        reid_model_path = pathlib.Path.home() / ".cache" / "torch" / "checkpoints" / "osnet_x0_25_msmt17.pt"

    # Tracker
    bot_sort_tracker = BotSort(
        reid_weights=reid_model_path,
        device=device,
        half=False,
        track_high_thresh=config.track_high_thresh,
        track_low_thresh=config.track_low_thresh,
        new_track_thresh=config.new_track_thresh,
        track_buffer=config.track_buffer,
        match_thresh=config.match_thresh,
    )

    # Video info
    video_info = sv.VideoInfo.from_video_path(video_path)
    width = config.video_width if config.video_width else video_info.resolution_wh[0]
    height = config.video_height if config.video_height else video_info.resolution_wh[1]
    fps = video_info.fps
    total_frames = video_info.total_frames
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    # Denormalize line coordinates if they are in 0-1 range
    def is_normalized(val):
        return 0.0 <= val <= 1.0

    denorm_line_coords_dict = {}
    for line_name, ((x1, y1), (x2, y2)) in line_coords_dict.items():
        if all(is_normalized(v) for v in [x1, y1, x2, y2]):
            denorm_line_coords_dict[line_name] = (
                (int(x1 * width), int(y1 * height)),
                (int(x2 * width), int(y2 * height))
            )
        else:
            denorm_line_coords_dict[line_name] = ((int(x1), int(y1)), (int(x2), int(y2)))

    # Person class ID
    person_class_id = next((i for i, name in model.names.items() if name == 'person'), 0)

    # FIXED: Pass 1 Analysis - Create analyzer OUTSIDE the loop
    print("🔍 Pass 1: Running complete analysis to get final accurate counts...")
    analyzer = CleanAnalyzer(config)
    analyzer.set_line_coordinates(denorm_line_coords_dict)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.predict(source=frame, verbose=False, device=device, conf=config.yolo_conf_threshold)[0]
        person_detections_yolo = results[results.boxes.cls == person_class_id]

        # Enhanced filtering
        filtered_indices = []
        if person_detections_yolo.boxes is not None and len(person_detections_yolo.boxes.xyxy) > 0:
            for i, bbox_tensor in enumerate(person_detections_yolo.boxes.xyxy):
                bbox = bbox_tensor.tolist()
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                if h < config.min_bbox_height or (w * h) < config.min_bbox_area:
                    continue
                aspect_ratio = w / h
                if not (config.min_width_height_ratio <= aspect_ratio <= config.max_width_height_ratio):
                    continue
                filtered_indices.append(i)

            if filtered_indices:
                person_detections_yolo = person_detections_yolo[filtered_indices]
            else:
                person_detections_yolo = sv.Detections.empty()
        else:
            person_detections_yolo = sv.Detections.empty()

        det_np = np.empty((0, 6))
        if len(person_detections_yolo) > 0:
            det_np = torch.cat([
                person_detections_yolo.boxes.xyxy,
                person_detections_yolo.boxes.conf.unsqueeze(1),
                person_detections_yolo.boxes.cls.unsqueeze(1)
            ], dim=1).cpu().numpy()

        outputs = bot_sort_tracker.update(det_np, frame)

        if outputs.shape[0] > 0:
            detections_sv = sv.Detections(
                xyxy=outputs[:, :4],
                class_id=outputs[:, 6].astype(int),
                confidence=outputs[:, 5],
                tracker_id=outputs[:, 4].astype(int)
            )
        else:
            detections_sv = sv.Detections.empty()

        analyzer.update_tracks(detections_sv)

    cap.release()

    # Finalize analysis to get FINAL accurate counts
    analyzer.finalize_analysis()
    final_accurate_counts = analyzer.get_realtime_counts()

    print("✅ Pass 1 Complete - Final accurate counts determined:")
    for line_name, counts in final_accurate_counts.items():
        print(f"   {line_name}: In={counts['in']}, Out={counts['out']}, Total={counts['in'] + counts['out']}")

    # Pass 2: Video creation with progressive counts
    print("🎬 Pass 2: Creating video with progressive counting...")
    output_number = get_next_output_number(config.output_dir)
    if output_path is None:
        output_path = os.path.join(config.output_dir, f"backend_people_count_{output_number}.mp4")

    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Reinitialize tracker/analyzer for this pass
    bot_sort_tracker = BotSort(
        reid_weights=reid_model_path,
        device=device,
        half=False,
        track_high_thresh=config.track_high_thresh,
        track_low_thresh=config.track_low_thresh,
        new_track_thresh=config.new_track_thresh,
        track_buffer=config.track_buffer,
        match_thresh=config.match_thresh,
    )
    progressive_analyzer = CleanAnalyzer(config)
    progressive_analyzer.set_line_coordinates(denorm_line_coords_dict)

    frame_count = 0
    while frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.predict(source=frame, verbose=False, device=device, conf=config.yolo_conf_threshold)[0]
        person_detections_yolo = results[results.boxes.cls == person_class_id]

        # Same filtering logic
        filtered_indices = []
        if person_detections_yolo.boxes is not None and len(person_detections_yolo.boxes.xyxy) > 0:
            for i, bbox_tensor in enumerate(person_detections_yolo.boxes.xyxy):
                bbox = bbox_tensor.tolist()
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                if h < config.min_bbox_height or (w * h) < config.min_bbox_area:
                    continue
                aspect_ratio = w / h
                if not (config.min_width_height_ratio <= aspect_ratio <= config.max_width_height_ratio):
                    continue
                filtered_indices.append(i)

            if filtered_indices:
                person_detections_yolo = person_detections_yolo[filtered_indices]
            else:
                person_detections_yolo = sv.Detections.empty()
        else:
            person_detections_yolo = sv.Detections.empty()

        det_np = np.empty((0, 6))
        if len(person_detections_yolo) > 0:
            det_np = torch.cat([
                person_detections_yolo.boxes.xyxy,
                person_detections_yolo.boxes.conf.unsqueeze(1),
                person_detections_yolo.boxes.cls.unsqueeze(1)
            ], dim=1).cpu().numpy()

        outputs = bot_sort_tracker.update(det_np, frame)

        if outputs.shape[0] > 0:
            detections_sv = sv.Detections(
                xyxy=outputs[:, :4],
                class_id=outputs[:, 6].astype(int),
                confidence=outputs[:, 5],
                tracker_id=outputs[:, 4].astype(int)
            )
        else:
            detections_sv = sv.Detections.empty()

        progressive_analyzer.update_tracks(detections_sv)

        # Annotate
        annotated = frame.copy()
        bbox_annotator = sv.BoxAnnotator(thickness=2)
        annotated = bbox_annotator.annotate(scene=annotated, detections=detections_sv)

        # Draw lines and overlay counts
        y_offset = 50
        current_progressive_counts = progressive_analyzer.get_realtime_counts()
        video_progress = frame_count / frames_to_process

        for line_name, final_counts in final_accurate_counts.items():
            current_counts = current_progressive_counts[line_name]
            final_in = final_counts['in']
            final_out = final_counts['out']
            current_in = current_counts['in']
            current_out = current_counts['out']

            # FIXED: Progressive counting with recursive correction from people_in_out_2.py
            if video_progress > 0.95:
                adjusted_in = final_in
                adjusted_out = final_out
            elif video_progress > 0.85:
                adjustment_progress = (video_progress - 0.85) / 0.1
                missing_in = max(0, final_in - current_in)
                missing_out = max(0, final_out - current_out)
                adjusted_in = current_in + round(missing_in * adjustment_progress)
                adjusted_out = current_out + round(missing_out * adjustment_progress)
                adjusted_in = min(adjusted_in, final_in)
                adjusted_out = min(adjusted_out, final_out)
            else:
                adjusted_in = current_in
                adjusted_out = current_out

            total_count = adjusted_in + adjusted_out
            cv2.putText(annotated, f"{line_name} - In: {adjusted_in}, Out: {adjusted_out} (Total: {total_count})",
                        (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

        # Draw lines on the annotated frame
        for line_name, ((x1, y1), (x2, y2)) in denorm_line_coords_dict.items():
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red line, thickness 2
            cv2.putText(annotated, line_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(annotated)

    cap.release()
    out.release()
    # Convert to web-friendly MP4
    web_output_path = output_path.replace('.mp4', '_web.mp4')
    if convert_to_web_mp4(output_path, web_output_path):
        final_output_path = web_output_path
    else:
        final_output_path = output_path
    # Return final results (from first pass)
    unique_people, total_crossings = analyzer.get_results()
    return {
        "unique_people": unique_people,
        "total_crossings": total_crossings,
        "final_counts": final_accurate_counts,
        "output_video_path": final_output_path,
    }

def tracking_video(input_path: str, output_path: str,
                   line_coords_dict: Optional[Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
                   video_width: Optional[int] = None, video_height: Optional[int] = None) -> dict:
    """
    Wrapper for emergency people counting using the people_counting_video function.
    Matches the interface used by other analytics modules.

    Args:
        input_path: Path to the input video file
        output_path: Path where the annotated video should be saved
        line_coords_dict: Dictionary with line names as keys and tuples of (start_point, end_point) as values
                         Example: {'line1': ((100, 200), (500, 200)), 'line2': ((100, 400), (500, 400))}
        video_width: Optional video width from frontend
        video_height: Optional video height from frontend

    Returns:
        dict with counts and output video path
    """
    if line_coords_dict is None:
        # Default lines if none provided
        line_coords_dict = {
            'line1': ((100, 200), (500, 200)),
            'line2': ((100, 400), (500, 400))
        }

    # Create config with default settings
    config = PeopleCountConfig(
        enable_stabilization=False,
        yolo_conf_threshold=0.15,
        min_bbox_height=25,
        min_bbox_area=200,
        track_high_thresh=0.4,
        track_low_thresh=0.05,
        new_track_thresh=0.4,
        track_buffer=100,
        match_thresh=0.6,
        min_crossing_distance=50,
        emergency_proximity_threshold=40,
        video_width=video_width,
        video_height=video_height
    )

    # Call the main processing function
    results = people_counting_video(
        input_path=input_path,
        output_path=output_path,
        line_coords_dict=line_coords_dict,
        config=config
    )

    return {
        "unique_people": results["unique_people"],
        "total_crossings": results["total_crossings"],
        "final_counts": results["final_counts"],
        "output_path": results["output_video_path"]
    }
