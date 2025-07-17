import cv2
import tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
from django.conf import settings
from apps.video_analytics.convert import convert_to_web_mp4
from boxmot import BotSort
import numpy as np
from pymongo import MongoClient

# Canonical models directory for all analytics jobs
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Read the exact URI you injected into Docker
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://toram444444:06nJTevaUItCDpd9@cluster01.lemxesc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster01")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set!")

# Create the client against your Atlas cluster
mongo_client = MongoClient(MONGO_URI)

# Choose your DB name (if not embedded in the URI)
# For a srv URI without DB, you can pick one here:
db = mongo_client["snake_db"]

# And reference the collection you need
snake_collection = db["snake_detections"]

# ---------------------- YOLO + File Settings ---------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = OUTPUT_DIR  # No subfolder
model = YOLO(str(MODELS_DIR / "best_animal.pt"))
DETECTION_CONFIDENCE = 0.4
EMBEDDER_FILE = MODELS_DIR / "osnet_ibn_x1_0_msmt17.pth"

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def detect_snakes_in_image(image_path):
    """
    Detect snakes in an image (input as file path).
    Returns: dict with count, save path, Mongo ID, and detected animals with confidence.
    """
    # Load image into numpy array
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Run detection
    results = model(img_array, )
    plotted = results[0].plot()
    num_snakes = len(results[0].boxes)

    # Collect detected animals and confidence
    detected_animals = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        detected_animals.append({"name": class_name, "confidence": confidence})

    # Save annotated image to disk
    timestamp = get_timestamp()
    save_path = SAVE_DIR / f"detection_img_{timestamp}.jpg"
    cv2.imwrite(str(save_path), cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR))

    # Log to MongoDB
    mongo_doc = {
        "type": "image",
        "file_name": os.path.basename(image_path),
        "detected_snakes": num_snakes,
        "detected_animals": detected_animals,
        "timestamp": datetime.now()
    }
    result = snake_collection.insert_one(mongo_doc)

    # Return result as dict
    return {
        "status": "success",
        "detected_snakes": num_snakes,
        "detected_animals": detected_animals,
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
    import torch
    # Device selection: prefer cuda > mps > cpu
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    tracker = BotSort(track_high_thresh=0.15, new_track_thresh=0.15, track_buffer=30, device=device, half=False, reid_weights=EMBEDDER_FILE)  # tune thresholds as needed

    total_detected = 0
    detected_animals = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=DETECTION_CONFIDENCE, imgsz=640)
        boxes = []
        # Convert YOLO results to tracker format: [x1, y1, x2, y2, conf, cls]
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = float(box.cls[0])
            boxes.append([x1, y1, x2, y2, conf, cls])
            # Collect detected animal info
            class_id = int(cls)
            class_name = model.names[class_id]
            detected_animals.append({"name": class_name, "confidence": conf})
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
        "detected_animals": detected_animals,
        "timestamp": datetime.now()
    }
    result = snake_collection.insert_one(mongo_doc)

    # Return result as dict
    return {
        'status': 'completed',
        'job_type': 'wildlife_detection',
        'output_video': str(save_path),
        'data': {
            'detected_snakes': total_detected,
            'detected_animals': detected_animals,
            'mongo_id': str(result.inserted_id),
            'alerts': []
        },
        'meta': {},
        'error': None
    }

def tracking_video(input_path: str, output_path: str) -> dict:
    """
    Main entry point for wildlife detection jobs. Handles both images and videos.
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
        # Ensure MEDIA_URL always ends with a single slash
        if not url.startswith('/api/media/'):
            url = '/api/media/' + str(rel_path).replace(os.sep, '/')
        return {
            'status': 'completed',
            'job_type': 'wildlife_detection',
            'output_image': str(rel_path),
            'output_url': url,
            'data': {
                'detected_snakes': result['detected_snakes'],
                'detected_animals': result['detected_animals'],
                'mongo_id': result['mongo_id'],
                'alerts': []
            },
            'meta': {},
            'error': None
        }
    else:
        # Video job
        result = detect_snakes_in_video(input_path)
        # Convert to web-friendly MP4
        output_video_path = result.get('output_video')
        if output_video_path:
            web_output_path = str(output_video_path).replace('.mp4', '_web.mp4')
            if convert_to_web_mp4(str(output_video_path), web_output_path):
                result['output_video'] = web_output_path
        return result
