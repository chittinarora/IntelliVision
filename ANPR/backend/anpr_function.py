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
from anpr.processor import ANPRProcessor, ParkingProcessor

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

# Initialize processors
plate_model = BASE_DIR / 'models' / 'best.pt'
car_model   = BASE_DIR / 'models' / 'yolo11m.pt'
anpr_processor = ANPRProcessor(str(plate_model), str(car_model))
parking_processor = ParkingProcessor(str(plate_model), str(car_model), total_slots=50)

# === Modular functions ===

def upload_video_file(local_path: str, filename: str) -> dict:
    """
    Copy an uploaded video from local_path to data/input and start async processing.
    Returns job ID and status.
    """
    try:
        input_dir = BASE_DIR / 'data' / 'input'
        input_dir.mkdir(parents=True, exist_ok=True)
        dest = input_dir / filename
        shutil.copy(local_path, dest)
        
        # Start processing in background thread
        Thread(target=anpr_processor.process_video, args=(str(dest),), daemon=True).start()
        
        return {
            'status': 'started',
            'message': 'Processing started in background',
            'file_path': str(dest)
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def analyze_video_file(file_path: str) -> dict:
    """
    Synchronously process a local video file. Returns summary.
    """
    if not os.path.exists(file_path):
        return {'status': 'error', 'message': f"File not found: {file_path}"}
    
    try:
        out_filename, summary = anpr_processor.process_video(file_path)
        return {
            'status': 'completed',
            'output': out_filename,
            'summary': summary
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


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
    
    # Check for common extensions if no extension provided
    if not candidate.exists() and not candidate.suffix:
        for ext in ('.mp4', '.csv', '.xlsx', '.jpg', '.png'):
            alt = output_dir / f"{filename}{ext}"
            if alt.exists():
                return str(alt)
    
    if candidate.exists():
        return str(candidate)
    
    raise FileNotFoundError(f"Download not found: {filename}")


def get_detection_history(limit: int = 100) -> list:
    """
    Fetch plate detection history from MongoDB.
    """
    try:
        records = list(db.detections.find({}, {'_id': 0}).sort("detected_at", -1).limit(limit))
        return records
    except Exception as e:
        return []


def start_parking_system(entry_cam: int = 0, exit_cam: int = 1):
    """
    Start the parking system with configured cameras
    """
    try:
        # Configure and start
        parking_processor.configure_cameras(entry_cam_id=entry_cam, exit_cam_id=exit_cam)
        parking_processor.start_processing()
        return {'status': 'success', 'message': 'Parking system started'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def stop_parking_system():
    """Stop the parking system"""
    try:
        parking_processor.stop_processing()
        return {'status': 'success', 'message': 'Parking system stopped'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def get_parking_status() -> dict:
    """Get current parking status"""
    try:
        return parking_processor.get_parking_status()
    except Exception as e:
        return {
            'total_slots': 0,
            'available': 0,
            'occupied': 0,
            'updated_at': datetime.utcnow().isoformat()
        }


def get_parking_events(limit: int = 100) -> list:
    """Get recent parking events"""
    try:
        return list(db.parking_events.find({}, {'_id': 0}).sort("timestamp", -1).limit(limit))
    except Exception as e:
        return []


def manual_parking_action(plate: str, action: str) -> dict:
    """
    Perform manual parking action (entry or exit)
    """
    plate = plate.upper().strip()
    try:
        if action == 'entry':
            slot_id = parking_processor.assign_slot(plate)
            if slot_id:
                return {'status': 'success', 'slot_id': slot_id}
            return {'status': 'error', 'message': 'No available slots'}
        
        elif action == 'exit':
            if parking_processor.release_slot(plate):
                return {'status': 'success'}
            return {'status': 'error', 'message': 'Vehicle not found'}
        
        return {'status': 'error', 'message': 'Invalid action'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def export_parking_logs() -> str:
    """
    Export parking logs to CSV and return file path
    """
    try:
        logs = list(db.parking_events.find({}, {'_id': 0}))
        if not logs:
            return ""
            
        df = pd.DataFrame(logs)
        filename = f"parking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_path = BASE_DIR / 'data' / 'output' / filename
        df.to_csv(out_path, index=False)
        return str(out_path)
    except Exception as e:
        return ""
