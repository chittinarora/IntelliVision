import os
import shutil
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from threading import Thread

from pymongo import MongoClient
import cloudinary
import cloudinary.uploader

from dotenv import load_dotenv
from anpr.detector import LicensePlateDetector
from anpr.tracker import VehicleTracker
from anpr.ocr import PlateOCR
from anpr.processor import ANPRProcessor

# Load environment
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

# Configure Cloudinary & MongoDB
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)
mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/anpr')
mongo_client = MongoClient(mongo_uri)
db = mongo_client['anpr']
parking_log = db['parking_log']

# Initialize processor
plate_model = BASE_DIR / 'models' / 'best.pt'
car_model   = BASE_DIR / 'models' / 'yolo11m.pt'
processor   = ANPRProcessor(str(plate_model), str(car_model))

# State for parking logic
track_centroids = defaultdict(list)
car_entries = set()
car_exits   = set()
entry_line_y = 200
exit_line_y  = 400

# === Modular functions ===

def upload_video_file(local_path: str, filename: str) -> dict:
    """
    Copy an uploaded video from local_path to data/input and start async processing.
    """
    input_dir = BASE_DIR / 'data' / 'input'
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / filename
    shutil.copy(local_path, dest)
    # kick off async task
    Thread(target=processor.process_video, args=(str(dest),), daemon=True).start()
    return {'status': 'started', 'path': str(dest)}


def analyze_video_file(file_path: str) -> dict:
    """
    Synchronously process a local video file. Returns summary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    out_filename, summary = processor.process_video(file_path)
    return {'output': out_filename, 'summary': summary}


def get_preview_path(filename: str) -> str:
    """
    Return absolute path to annotated video for preview.
    """
    path = BASE_DIR / 'data' / 'output' / filename
    if not path.exists():
        raise FileNotFoundError(f"Preview not found: {filename}")
    return str(path)


def get_download_path(filename: str) -> str:
    """
    Return absolute path to a result file (video, csv, xlsx).
    """
    output_dir = BASE_DIR / 'data' / 'output'
    candidate = output_dir / filename
    if not candidate.exists() and candidate.suffix == '':
        for ext in ('.mp4', '.csv', '.xlsx'):
            alt = output_dir / f"{filename}{ext}"
            if alt.exists():
                candidate = alt
                break
    if not candidate.exists():
        raise FileNotFoundError(f"Download not found: {filename}")
    return str(candidate)


def get_detection_history() -> list:
    """
    Fetch all final plate detections from MongoDB.
    """
    records = list(db['final_plate_output'].find({}, {'_id': 0}))
    return records


def start_live_camera(entry_y: int = 200, exit_y: int = 400):
    """
    Begin live camera loop on default device and log entry/exit.
    """
    def loop():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break
            cars = processor.car_detector.detect_plates(frame, classes=[2])
            car_dets = [(x1,y1,x2,y2,conf,2) for x1,y1,x2,y2,conf in cars]
            tracks = processor.tracker.update(car_dets, frame)
            for tr in tracks:
                tid = tr['track_id']
                bx1,by1,bx2,by2 = tr['bbox']
                cy = (by1+by2)//2
                track_centroids[tid].append(cy)
                if len(track_centroids[tid])>1:
                    prev, curr = track_centroids[tid][-2:]
                    if prev<entry_y<=curr and tid not in car_entries:
                        car_entries.add(tid)
                        parking_log.insert_one({'track_id':tid,'event':'entry','time':datetime.now()})
                    if prev>exit_y>=curr and tid not in car_exits:
                        car_exits.add(tid)
                        parking_log.insert_one({'track_id':tid,'event':'exit','time':datetime.now()})
    Thread(target=loop, daemon=True).start()
    return {'status':'camera_started'}


def set_parking_lines(entry_y: int, exit_y: int):
    global entry_line_y, exit_line_y
    entry_line_y, exit_line_y = entry_y, exit_y
    return {'entry_line': entry_line_y, 'exit_line': exit_line_y}


def get_parking_stats(total_slots: int = None) -> dict:
    """
    Compute current parking occupancy.
    """
    slots = total_slots if total_slots is not None else 50
    occupied = len(car_entries) - len(car_exits)
    available = max(slots - occupied, 0)
    return {'total': slots, 'occupied': occupied, 'available': available, 'timestamp': datetime.now().isoformat()}


def get_parking_dashboard() -> list:
    """
    Return raw parking events log.
    """
    return list(parking_log.find({}, {'_id':0}))


def export_parking_logs() -> str:
    """
    Write parking log to CSV and return file path.
    """
    logs = list(parking_log.find({}, {'_id':0}))
    df = pd.DataFrame(logs)
    out = BASE_DIR / 'data' / 'output' / f"parking_{datetime.now():%Y%m%d%H%M%S}.csv"
    df.to_csv(out, index=False)
    return str(out)
