import os
import subprocess
import warnings
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from ..convert import convert_to_web_mp4


# --- Utility Functions ---


def setup_models_and_tracker(model_path: str, embedder_wts_path: str) -> Tuple[YOLO, DeepSort]:
    """
    Initializes and returns the YOLO model and DeepSORT tracker.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="deep_sort_realtime.embedder.embedder_pytorch")

    print("Loading YOLO model...")
    model = YOLO(model_path)

    print("Initializing DeepSORT tracker...")
    # Tracker parameters are now hardcoded here
    tracker = DeepSort(
        max_age=60,
        n_init=4,
        max_iou_distance=0.4,
        nn_budget=1500,
        embedder="torchreid",
        embedder_model_name="osnet_x1_0",
        embedder_wts=embedder_wts_path,
    )
    return model, tracker


def run_tracking_loop(capture: cv2.VideoCapture, writer: cv2.VideoWriter, model: YOLO, tracker: DeepSort) -> int:
    """
    Contains the main loop to process video frames, perform tracking, and write the output.

    Returns:
        int: The total count of unique persons tracked.
    """
    compact_id_map = {}
    next_compact_id = 1

    # Processing parameters are now hardcoded here
    frame_width, frame_height = 1280, 720
    yolo_conf, yolo_iou = 0.6, 0.4
    target_class = 'person'

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (frame_width, frame_height))

        # Get detections from YOLO
        results = model(frame_resized, conf=yolo_conf, iou=yolo_iou)[0]
        detections = []
        for box in results.boxes:
            class_name = model.names[int(box.cls[0])]
            if class_name == target_class:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                detections.append(([x1, y1, w, h], conf, class_name))

        # Update tracker and draw results
        tracks = tracker.update_tracks(detections, frame=frame_resized)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            if track_id not in compact_id_map:
                compact_id_map[track_id] = next_compact_id
                next_compact_id += 1
            display_id = compact_id_map[track_id]

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f'ID: {display_id}', (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        writer.write(frame_resized)

    return next_compact_id - 1


# --- Main Entry Point ---

def tracking_video(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Orchestrates the video tracking process by calling helper functions for each step.
    """
    # --- 1. Define Paths ---
    # Using the flexible paths for better portability
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    # NOTE: This path points to a "models" subfolder for better organization
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo12m.pt")

    # NOTE: This path automatically finds the current user's home directory
    EMBEDDER_WTS_PATH = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_msmt17.pth")

    # --- 2. Setup Models and Video I/O ---
    model, tracker = setup_models_and_tracker(MODEL_PATH, EMBEDDER_WTS_PATH)

    temp_output_path = output_path.replace(".mp4", "_temp.mp4")
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25  # Fallback fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (1280, 720))

    person_count = 0
    try:
        # --- 3. Run Core Processing Loop ---
        person_count = run_tracking_loop(capture, writer, model, tracker)
    finally:
        # --- 4. Release Resources (this is a critical safety feature) ---
        print("Releasing video resources...")
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    # --- 5. Post-process and Finalize ---
    if convert_to_web_mp4(temp_output_path, output_path):
        os.remove(temp_output_path)
    else:
        print("Warning: Web conversion failed. Using the original temp output.")
        os.rename(temp_output_path, output_path)

    print(f"=== process_video FINISHED ===")
    print(f"Total unique persons tracked: {person_count}")

    print(person_count)

    return {'person_count': person_count, 'output_path': output_path}
