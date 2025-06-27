"""
people_count.py - Video Analytics
Robust two-pass people tracking and counting using YOLO and BotSort.
"""

import os
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import BotSort
from tqdm import tqdm
from pathlib import Path

# Use convert_to_web_mp4 from your existing project import
from ..convert import convert_to_web_mp4

# --- Logger setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("people_count_two_pass")

# --- Constants ---
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
MODEL_FILE = "yolo12m.pt"    # Update as needed
EMBEDDER_FILE = Path(os.path.expanduser("~/.cache/torch/checkpoints/osnet_ibn_x1_0_msmt17.pth"))
DETECTION_CONFIDENCE = 0.2
MIN_BOX_AREA = 1000

# --- Global cache for singleton model/tracker ---
LOADED_OBJECTS = {}

# --- Model and Tracker Loader ---
def get_model_and_tracker():
    """Singleton loader for YOLO and BotSort models."""
    if "model" not in LOADED_OBJECTS:
        logger.info(f"Loading YOLO model: {MODEL_FILE}")
        model = YOLO(MODEL_FILE)
        tracker = BotSort(
            reid_weights=EMBEDDER_FILE, device='cpu', half=False,
            track_buffer=150, appearance_thresh=0.20, new_track_thresh=0.85
        )
        LOADED_OBJECTS["model"] = model
        LOADED_OBJECTS["tracker"] = tracker
    return LOADED_OBJECTS["model"], LOADED_OBJECTS["tracker"]

# --- Tracking Pass ---
def run_tracking_pass(input_path, model, tracker):
    """First pass: collects tracks frame by frame."""
    logger.info("Pass 1: Collecting tracking data")
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        logger.error(f"Cannot open video file: {input_path}")
        raise IOError(f"Cannot open video file: {input_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    all_tracks_by_frame = {}
    frame_number = 0
    tracker.reset()
    with tqdm(total=total_frames, desc="Tracking Pass", unit="frame") as pbar:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame_number += 1
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            results = model(frame_resized, conf=DETECTION_CONFIDENCE, classes=[0], verbose=False)[0]
            filtered_detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                if (x2 - x1) * (y2 - y1) > MIN_BOX_AREA:
                    filtered_detections.append(box.data[0].cpu().numpy())
            filtered_detections = np.array(filtered_detections) if filtered_detections else np.empty((0, 6))
            tracks = tracker.update(filtered_detections, frame_resized)
            if tracks.shape[0] > 0:
                all_tracks_by_frame[frame_number] = []
                for track in tracks:
                    x1, y1, x2, y2, track_id = track[0], track[1], track[2], track[3], int(track[4])
                    all_tracks_by_frame[frame_number].append({'id': track_id, 'bbox': (x1, y1, x2, y2)})
            pbar.update(1)
    capture.release()
    logger.info("Tracking pass complete.")
    return all_tracks_by_frame, fps

# --- Post-processing ---
def post_process_and_create_final_map(all_tracks_by_frame, fps):
    """Remap fragmented IDs and assign final sequential IDs."""
    logger.info("Post-processing: Merging track IDs")
    if not all_tracks_by_frame:
        return {}, 0
    tracks_by_id = {}
    for frame_num, frame_data in all_tracks_by_frame.items():
        for track_info in frame_data:
            track_id = track_info['id']
            if track_id not in tracks_by_id:
                tracks_by_id[track_id] = []
            tracks_by_id[track_id].append({'frame': frame_num, **track_info})
    if not tracks_by_id:
        return {}, 0
    for track_id in tracks_by_id:
        tracks_by_id[track_id].sort(key=lambda x: x['frame'])
    merge_map = {}
    sorted_ids = sorted(tracks_by_id.keys())
    time_threshold_frames, dist_threshold_px = 7.0 * fps, 250
    for i in range(len(sorted_ids)):
        for j in range(i + 1, len(sorted_ids)):
            id1, id2 = sorted_ids[i], sorted_ids[j]
            if id1 in merge_map or id2 in merge_map:
                continue
            track1, track2 = tracks_by_id[id1], tracks_by_id[id2]
            end_of_track1, start_of_track2 = track1[-1], track2[0]
            frame_gap = start_of_track2['frame'] - end_of_track1['frame']
            if 0 < frame_gap <= time_threshold_frames:
                bbox1 = end_of_track1['bbox']
                center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
                bbox2 = start_of_track2['bbox']
                center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
                distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
                if distance < dist_threshold_px:
                    merge_map[id2] = id1
    final_id_map, sequential_id_counter, root_to_final_id = {}, 1, {}
    for original_id in sorted_ids:
        current_id = original_id
        while current_id in merge_map:
            current_id = merge_map[current_id]
        if current_id not in root_to_final_id:
            root_to_final_id[current_id] = sequential_id_counter
            sequential_id_counter += 1
        final_id_map[original_id] = root_to_final_id[current_id]
    final_person_count = sequential_id_counter - 1
    logger.info(f"Post-processing complete. Final count: {final_person_count}")
    return final_id_map, final_person_count

# --- Video Rendering Pass ---
def render_final_video_pass(input_path, output_path, fps, all_tracks_by_frame, final_id_map):
    """Second pass: renders video with corrected IDs."""
    logger.info("Rendering video with corrected IDs")
    capture = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_path = output_path.replace(".mp4", "_temp.mp4")
    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    with tqdm(total=total_frames, desc="Rendering Pass", unit="frame") as pbar:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame_number += 1
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            if frame_number in all_tracks_by_frame:
                frame_data = all_tracks_by_frame[frame_number]
                for track_info in frame_data:
                    original_id = track_info['id']
                    clean_id = final_id_map.get(original_id)
                    if clean_id:
                        x1, y1, x2, y2 = track_info['bbox']
                        cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame_resized, f'ID:{clean_id}', (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            writer.write(frame_resized)
            pbar.update(1)
    capture.release()
    writer.release()
    return temp_output_path

# --- Main Tracking Pipeline ---
def tracking_video(input_path: str, output_path: str) -> dict:
    """
    Runs robust two-pass people tracking with post-process ID cleanup.
    Returns: {'person_count': int, 'output_path': str}
    """
    model, tracker = get_model_and_tracker()
    all_tracks_data, video_fps = run_tracking_pass(input_path, model, tracker)
    final_id_map, final_count = post_process_and_create_final_map(all_tracks_data, video_fps)
    temp_output_path = render_final_video_pass(
        input_path=input_path,
        output_path=output_path,
        fps=video_fps,
        all_tracks_by_frame=all_tracks_data,
        final_id_map=final_id_map
    )
    # Finalize for web
    if convert_to_web_mp4(temp_output_path, output_path):
        os.remove(temp_output_path)
    else:
        logger.warning("Web conversion failed. Using original output.")
        os.rename(temp_output_path, output_path)
    logger.info(f"Process finished. Total unique persons tracked: {final_count}")
    return {'person_count': final_count, 'output_path': output_path}
