import cv2
import tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
import os
from django.conf import settings
from apps.video_analytics.convert import convert_to_web_mp4
from boxmot import BotSort
import numpy as np

# ---------------------- MongoDB Connection ----------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["snake_db"]
collection = db["detections"]

# ---------------------- YOLO + File Settings ---------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "models" / "best.pt")
SAVE_DIR = Path(settings.MEDIA_ROOT) / "snake_detections"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
model = YOLO(MODEL_PATH)
DETECTION_CONFIDENCE = 0.15
EMBEDDER_FILE = Path(os.path.expanduser("~/.cache/torch/checkpoints/osnet_ibn_x1_0_msmt17.pth"))

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def detect_snakes_in_image(image_path):
    """
    Detect snakes in an image (input as file path).
    Returns: dict with count, save path, and Mongo ID.
    """
    # Load image into numpy array
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Run detection
    results = model(img_array, )
    plotted = results[0].plot()
    num_snakes = len(results[0].boxes)

    # Save annotated image to disk
    timestamp = get_timestamp()
    save_path = SAVE_DIR / f"detection_img_{timestamp}.jpg"
    cv2.imwrite(str(save_path), cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR))

    # Log to MongoDB
    mongo_doc = {
        "type": "image",
        "file_name": os.path.basename(image_path),
        "detected_snakes": num_snakes,
        "timestamp": datetime.now()
    }
    result = collection.insert_one(mongo_doc)

    # Return result as dict
    return {
        "status": "success",
        "detected_snakes": num_snakes,
        "saved_path": str(save_path),
        "mongo_id": str(result.inserted_id)
    }

def detect_snakes_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = get_timestamp()
    save_path = SAVE_DIR / f"detection_vid_{timestamp}.mp4"
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # --- Tracker Setup ---
    tracker = BotSort(track_high_thresh=0.15, new_track_thresh=0.15, track_buffer=30, device="cpu", half=False, reid_weights=EMBEDDER_FILE)  # tune thresholds as needed

    total_detected = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=DETECTION_CONFIDENCE)
        boxes = []
        # Convert YOLO results to tracker format: [x1, y1, x2, y2, conf, cls]
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = float(box.cls[0])
            boxes.append([x1, y1, x2, y2, conf, cls])
        boxes = np.array(boxes) if boxes else np.zeros((0, 6))
        tracks = tracker.update(boxes, frame)

        # Draw tracked boxes with ID
        for t in tracks:
            x1, y1, x2, y2, track_id = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out.write(frame)
        total_detected += len(tracks)

    cap.release()
    out.release()

    # Log to MongoDB
    mongo_doc = {
        "type": "video",
        "file_name": os.path.basename(video_path),
        "detected_snakes": total_detected,
        "timestamp": datetime.now()
    }
    result = collection.insert_one(mongo_doc)

    # Return result as dict
    return {
        "status": "success",
        "detected_snakes": total_detected,
        "saved_path": str(save_path),
        "mongo_id": str(result.inserted_id)
    }

def tracking_video(input_path: str, output_path: str) -> dict:
    """
    Main entry point for pest monitoring jobs. Handles both images and videos.
    For images: runs detection, saves annotated image to MEDIA_ROOT, returns result with web URL.
    For videos: runs detection, converts to web-friendly MP4, returns result with web URL.
    """
    ext = os.path.splitext(input_path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png']
    if ext in image_exts:
        # Image job
        result = detect_snakes_in_image(input_path)
        # Save to web-accessible location
        rel_path = Path(result['saved_path']).relative_to(settings.MEDIA_ROOT)
        url = settings.MEDIA_URL + str(rel_path).replace(os.sep, '/')
        result['output_url'] = url
        result['output_path'] = str(rel_path)
        return result
    else:
        # Video job
        result = detect_snakes_in_video(input_path)
        # Convert to web-friendly MP4
        web_output_path = str(Path(result['saved_path']).with_name(Path(result['saved_path']).stem + '_web.mp4'))
        if convert_to_web_mp4(result['saved_path'], web_output_path):
            if os.path.exists(result['saved_path']):
                os.remove(result['saved_path'])
            rel_path = Path(web_output_path).relative_to(settings.MEDIA_ROOT)
            url = settings.MEDIA_URL + str(rel_path).replace(os.sep, '/')
            result['output_url'] = url
            result['output_path'] = str(rel_path)
        else:
            rel_path = Path(result['saved_path']).relative_to(settings.MEDIA_ROOT)
            url = settings.MEDIA_URL + str(rel_path).replace(os.sep, '/')
            result['output_url'] = url
            result['output_path'] = str(rel_path)
        return result
