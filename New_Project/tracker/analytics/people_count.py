import os
import logging
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import BotSort
from tqdm import tqdm

# Assuming this is in a sub-directory, otherwise adjust the path
from ..convert import convert_to_web_mp4

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("people_count")

# --- Module Constants ---
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
YOLO_CONF = 0.4
MODEL_FILE = "best.pt"
EMBEDDER_FILE = Path(os.path.expanduser("~/.cache/torch/checkpoints/osnet_ibn_x1_0_msmt17.pth"))

# --- Global Placeholder for Models ---
# This dictionary will hold the models once loaded, making them a singleton per worker.
LOADED_MODELS = {}

def get_models() -> tuple[YOLO, BotSort]:
    """
    Initializes and returns the models, ensuring they are loaded only once per worker process.
    """
    if "yolo" not in LOADED_MODELS:
        logger.info("Models not found in this worker. Initializing for the first time...")

        try:
            PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        except NameError:
            # Fallback for interactive environments
            PROJECT_ROOT = os.getcwd()

        model_path = os.path.join(PROJECT_ROOT, "models", MODEL_FILE)

        yolo_model = YOLO(model_path)

        # Initialize BotSort with tuned parameters
        botsort_tracker = BotSort(
            reid_weights=EMBEDDER_FILE,
            device='cpu',
            half=False,
            # Tuned parameters to balance accuracy and reduce flickering
            new_track_thresh=0.8,
            appearance_thresh=0.35,
            track_buffer=45
        )

        LOADED_MODELS["yolo"] = yolo_model
        LOADED_MODELS["tracker"] = botsort_tracker
        logger.info("Models loaded and cached in this worker process.")

    return LOADED_MODELS["yolo"], LOADED_MODELS["tracker"]


# --- Main Tracking and Rendering Function ---
def tracking_video(input_path: str, output_path: str) -> dict:
    """
    Performs people tracking on a video using YOLOv8 and BoTSORT,
    and renders the output video in a single pass.
    """
    # --- 1. Get Models using the Lazy Loader ---
    model, tracker = get_models()

    # --- 2. Setup Video I/O ---
    logger.info("Input: %s", input_path)
    logger.info("Output: %s", output_path)

    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        logger.error("Cannot open video file: %s", input_path)
        raise IOError(f"Cannot open video file: {input_path}")

    # Get video properties for writer and progress bar
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception:
        total_frames = 0

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_path = output_path.replace(".mp4", "_temp.mp4")
    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    # --- 3. Main Tracking Loop ---
    person_count = 0
    tracked_ids = set()

    # Initialize tqdm progress bar
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        try:
            tracker.reset()

            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                else:
                    frame_resized = frame

                # Run YOLO detection
                results = model(frame_resized, conf=YOLO_CONF, classes=[0], verbose=False)[0]

                # Update tracker with detections
                if results.boxes.data.shape[0] > 0:
                    tracks = tracker.update(results.boxes.data.cpu().numpy(), frame_resized)
                else:
                    tracks = tracker.update(np.empty((0, 6)), frame_resized)

                # Process and draw tracks directly onto the frame
                if tracks.shape[0] > 0:
                    for track in tracks:
                        x1, y1, x2, y2 = track[:4]
                        track_id = track[4]
                        tracked_ids.add(int(track_id))

                        cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f'ID:{int(track_id)}'
                        cv2.putText(frame_resized, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                writer.write(frame_resized)
                pbar.update(1) # Update the progress bar

            person_count = len(tracked_ids)

        finally:
            logger.info("Releasing video resources...")
            capture.release()
            writer.release()
            cv2.destroyAllWindows()

    # --- 4. Finalize Video ---
    if convert_to_web_mp4(temp_output_path, output_path):
        os.remove(temp_output_path)
    else:
        logger.warning("Web conversion failed. Using original output.")
        os.rename(temp_output_path, output_path)

    logger.info("Process finished. Total unique persons tracked: %d", person_count)

    return {'person_count': person_count, 'output_path': output_path}
