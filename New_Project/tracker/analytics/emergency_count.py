import os
import warnings
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

try:
    from ..convert import convert_to_web_mp4
except ImportError:
    print("Warning: 'convert_to_web_mp4' could not be imported from the parent directory.")
    print("The output video will not be web-optimized, and the temp file will be renamed.")


"""
Analytics module for emergency in/out people counting using YOLO and DeepSORT.
Provides tracking and counting logic for emergency_count jobs.
"""

# --- 1. SETUP WITH IMPROVED PARAMETERS ---
def setup_models_and_tracker(model_path: str, embedder_wts_path: str) -> Tuple[YOLO, DeepSort]:
    """
    Loads the YOLO model and initializes the DeepSORT tracker with improved parameters.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="deep_sort_realtime.embedder.embedder_pytorch")
    print("Loading YOLO model...")
    model = YOLO(model_path)

    print("Initializing DeepSORT tracker with improved parameters...")
    tracker = DeepSort(
        max_age=60,
        n_init=3,
        max_iou_distance=0.7,
        nn_budget=100,
        embedder="torchreid",
        embedder_model_name="osnet_x1_0",
        embedder_wts=embedder_wts_path,
    )
    return model, tracker


# --- 2. TRACKING LOGIC WITH CORRECTED COUNTING ---
def run_in_out_tracking(
        capture: cv2.VideoCapture,
        writer: cv2.VideoWriter,
        model: YOLO,
        tracker: DeepSort,
        roi: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Dict[str, int]:
    """
    Processes frames, counts in/out based on ROI with corrected logic,
    and writes annotated output video.
    Returns: {'in': int, 'out': int}
    """
    # ROI coordinates
    (x1, y1), (x2, y2) = roi

    track_states = {}
    counts = {"in": 0, "out": 0}
    compact_id_map = {}
    next_compact_id = 1

    frame_width, frame_height = 1280, 720
    yolo_conf, yolo_iou = 0.7, 0.5
    target_class = 'person'

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (frame_width, frame_height))
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)

        results = model(frame_resized, conf=yolo_conf, iou=yolo_iou, classes=[0])[0]

        detections = []
        for box in results.boxes:
            class_name = model.names[int(box.cls[0])]
            if class_name == target_class:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                w, h = bx2 - bx1, by2 - by1
                conf = float(box.conf[0])
                detections.append(([bx1, by1, w, h], conf, class_name))

        tracks = tracker.update_tracks(detections, frame=frame_resized)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id

            if track_id not in compact_id_map:
                compact_id_map[track_id] = next_compact_id
                next_compact_id += 1
            display_id = compact_id_map[track_id]

            px1, py1, px2, py2 = map(int, track.to_ltrb())
            cx, cy = (px1 + px2) // 2, py2

            is_currently_in = (x1 <= cx <= x2) and (y1 <= cy <= y2)
            previous_state_in = track_states.get(track_id)

            if previous_state_in is None:
                track_states[track_id] = is_currently_in
                continue

            if not previous_state_in and is_currently_in:
                counts["in"] += 1
            elif previous_state_in and not is_currently_in:
                counts["out"] += 1

            track_states[track_id] = is_currently_in

            if is_currently_in:
                cv2.rectangle(frame_resized, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(
                    frame_resized,
                    f'ID: {display_id}',
                    (px1, max(py1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        counter_bg_color = (0, 0, 0)
        counter_text_color = (255, 255, 255)
        cv2.rectangle(frame_resized, (10, 10), (220, 80), counter_bg_color, -1)
        cv2.putText(frame_resized, f"IN: {counts['in']}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, counter_text_color, 2)
        cv2.putText(frame_resized, f"OUT: {counts['out']}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, counter_text_color,
                    2)

        writer.write(frame_resized)

    return counts


# --- 3. MAIN ORCHESTRATION FUNCTION ---
def tracking_video(
        input_path: str,
        output_path: str,
        roi: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Dict[str, Any]:
    """
    Runs the full in/out people counting pipeline on a video file.
    """
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo12m.pt")
    EMBEDDER_WTS_PATH = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_msmt17.pth")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}. Please download it first.")
    if not os.path.exists(EMBEDDER_WTS_PATH):
        raise FileNotFoundError(
            f"Embedder weights not found at {EMBEDDER_WTS_PATH}. "
            "They should be downloaded automatically by deep_sort_realtime on the first run."
        )

    model, tracker = setup_models_and_tracker(MODEL_PATH, EMBEDDER_WTS_PATH)

    temp_output_path = output_path.replace(".mp4", "_temp.mp4")

    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (1280, 720))

    counts = {"in": 0, "out": 0}
    try:
        print("Starting tracking process...")
        counts = run_in_out_tracking(capture, writer, model, tracker, roi)
    finally:
        print("Releasing video resources...")
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    # Restore original post-processing logic
    if convert_to_web_mp4(temp_output_path, output_path):
        os.remove(temp_output_path)
    else:
        print("Warning: Web conversion failed or was skipped. Using original temp output.")
        os.rename(temp_output_path, output_path)

    print("\n=== People Counting FINISHED ===")
    print(f"Final Counts -> IN: {counts['in']}, OUT: {counts['out']}")
    print(f"Output video saved to: {output_path}")

    return {"in": counts["in"], "out": counts["out"], "output_path": output_path}


# --- Example Usage ---
if __name__ == '__main__':
    input_video_path = 'path/to/your/video.mp4'
    output_video_path = 'result_video.mp4'
    region_of_interest = ((850, 150), (1250, 700))

    if not os.path.exists(input_video_path):
        print(f"FATAL ERROR: The input video was not found at '{input_video_path}'")
        print("Please update the 'input_video_path' variable in the `if __name__ == '__main__':` block.")
    else:
        tracking_video(
            input_path=input_video_path,
            output_path=output_video_path,
            roi=region_of_interest
        )
