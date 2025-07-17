import os
import uuid
import time
import logging
import csv
from pathlib import Path
from threading import Lock
from datetime import datetime
from pymongo import MongoClient
import cloudinary
from dotenv import load_dotenv
from .anpr.processor import ANPRProcessor, ParkingProcessor
from apps.video_analytics.convert import convert_to_web_mp4
from django.conf import settings

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("anpr_functions")

# Load environment
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
MODELS_DIR = BASE_DIR / 'video_analytics' / 'models'
plate_model = MODELS_DIR / 'best_car.pt'
car_model = MODELS_DIR / 'yolo11m_car.pt'
load_dotenv(BASE_DIR / '.env')

# Define canonical output directory for all outputs
OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure Cloudinary & MongoDB
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)
mongo_uri = os.environ.get('MONGO_URI', 'mongodb+srv://toram444444:06nJTevaUItCDpd9@cluster01.lemxesc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster01')
mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
db = mongo_client['anpr']

# Initialize processors
sync_anpr_processor = ANPRProcessor(str(plate_model), str(car_model))
sync_parking_processor = ParkingProcessor(str(plate_model), str(car_model), total_slots=50)

# Processor locks for thread safety
anpr_lock = Lock()
parking_lock = Lock()

# === Synchronous Processing Functions ===

def recognize_number_plates(video_path: str) -> dict:
    """
    Process video for number plate recognition (blocking operation)
    Returns: {
        'status': 'completed'|'error',
        'summary': {
            'total_frames': int,
            'processing_fps': float,
            'detected_plates': list,
            'recognized_plates': list
        },
        'output_video': str,
        'preview_url': str,
        'download_url': str,
        'processing_time': float,
        'message': str
    }
    """
    start_time = time.time()
    try:
        # Validate input
        video_path = str(MODELS_DIR / video_path)
        if not Path(video_path).exists():
            return {
                'status': 'error',
                'message': f"File not found: {video_path}",
                'processing_time': time.time() - start_time
            }

        # Process with thread lock
        with anpr_lock:
            logger.info(f"Starting plate recognition: {Path(video_path).name}")
            output, summary = sync_anpr_processor.process_video(video_path)

        # === Convert output video to web-friendly MP4 ===
        if output and Path(output).exists():
            web_output = str(Path(output).with_name(Path(output).stem + '_web.mp4'))
            if convert_to_web_mp4(str(output), web_output):
                output = web_output
                logger.info(f"Converted output to web mp4: {web_output}")
            else:
                logger.warning(f"Failed to convert output to web mp4, using original: {output}")

        # Generate URLs
        output_filename = Path(output).name
        result = {
            'status': 'completed',
            'summary': summary,
            'output_video': output,
            'preview_url': f"/preview/{output_filename}",
            'download_url': f"/download/{output_filename}",
            'processing_time': time.time() - start_time,
            'message': 'Plate recognition completed successfully'
        }

        # Save to database
        db.plate_results.insert_one({
            'video_path': video_path,
            'timestamp': datetime.utcnow(),
            'result': result,
            'type': 'plate_recognition'
        })

        return result

    except Exception as e:
        logger.error(f"Plate recognition failed: {str(e)}")
        return {
            'status': 'error',
            'message': f"Processing error: {str(e)}",
            'processing_time': time.time() - start_time
        }


def analyze_parking_video(video_path: str) -> dict:
    """
    Process video for parking analysis (blocking operation)
    Returns: {
        'status': 'completed'|'error',
        'summary': {
            'entries': int,
            'exits': int,
            'max_occupancy': int,
            'final_occupancy': int,
            'recognized_plates': list,
            'processing_fps': float,
            'total_frames': int,
            'processing_time': float
        },
        'output_video': str,
        'preview_url': str,
        'download_url': str,
        'processing_time': float,
        'message': str
    }
    """
    start_time = time.time()
    try:
        # Validate input
        video_path = str(MODELS_DIR / video_path)
        if not Path(video_path).exists():
            return {
                'status': 'error',
                'message': f"File not found: {video_path}",
                'processing_time': time.time() - start_time
            }

        # Process with thread lock
        with parking_lock:
            logger.info(f"Starting parking analysis: {Path(video_path).name}")
            output, summary = sync_parking_processor.process_video(video_path)

        # === Convert output video to web-friendly MP4 ===
        if output and Path(output).exists():
            web_output = str(Path(output).with_name(Path(output).stem + '_web.mp4'))
            if convert_to_web_mp4(str(output), web_output):
                output = web_output
                logger.info(f"Converted output to web mp4: {web_output}")
            else:
                logger.warning(f"Failed to convert output to web mp4, using original: {output}")

        # Ensure all summary fields are present
        summary = {
            'entries': summary.get('entries', 0),
            'exits': summary.get('exits', 0),
            'max_occupancy': summary.get('max_occupancy', 0),
            'final_occupancy': summary.get('final_occupancy', 0),
            'recognized_plates': summary.get('recognized_plates', []),
            'processing_fps': summary.get('processing_fps', 0),
            'total_frames': summary.get('total_frames', 0),
            'processing_time': summary.get('processing_time', 0),
            'vehicle_count': summary.get('vehicle_count', 0)  # Add vehicle_count to summary
        }

        # Generate URLs
        output_filename = Path(output).name
        result = {
            'status': 'completed',
            'summary': summary,
            'output_video': output,
            'preview_url': f"/preview/{output_filename}",
            'download_url': f"/download/{output_filename}",
            'processing_time': time.time() - start_time,
            'message': 'Parking analysis completed successfully'
        }

        # Save to database
        db.parking_results.insert_one({
            'video_path': video_path,
            'timestamp': datetime.utcnow(),
            'result': result,
            'type': 'parking_analysis'
        })

        return result

    except Exception as e:
        logger.error(f"Parking analysis failed: {str(e)}")
        return {
            'status': 'error',
            'message': f"Processing error: {str(e)}",
            'processing_time': time.time() - start_time
        }


# === Real-time Parking System Functions ===

def process_parking_stream(entry_cam: int = 0, exit_cam: int = 1) -> dict:
    """
    Start parking system processing with webcams
    Returns: {
        'status': 'running'|'error',
        'message': str
    }
    """
    try:
        sync_parking_processor.configure_cameras(entry_cam_id=entry_cam, exit_cam_id=exit_cam)
        sync_parking_processor.start_processing()
        return {
            'status': 'running',
            'message': f'Parking system started with cameras: Entry={entry_cam}, Exit={exit_cam}'
        }
    except Exception as e:
        logger.error(f"Parking stream failed: {str(e)}")
        return {'status': 'error', 'message': str(e)}


def stop_parking_system() -> dict:
    """Stop the parking system processing"""
    try:
        sync_parking_processor.stop_processing()
        return {
            'status': 'success',
            'message': 'Parking system stopped successfully'
        }
    except Exception as e:
        logger.error(f"Stop parking error: {str(e)}")
        return {'status': 'error', 'message': str(e)}


def get_parking_status() -> dict:
    """Get current parking status"""
    try:
        status = sync_parking_processor.get_parking_status()
        status['updated_at'] = datetime.utcnow().isoformat()
        return status
    except Exception as e:
        logger.error(f"Parking status error: {str(e)}")
        return {
            'total_slots': 0,
            'available': 0,
            'occupied': 0,
            'updated_at': datetime.utcnow().isoformat(),
            'status': 'error',
            'message': str(e)
        }


def manual_parking_action(plate: str, action: str) -> dict:
    """
    Perform manual parking action (entry or exit)
    Returns: Operation status
    """
    plate = plate.upper().strip()
    try:
        if action == 'entry':
            slot_id = sync_parking_processor.assign_slot(plate)
            if slot_id:
                return {
                    'status': 'success',
                    'slot_id': slot_id,
                    'message': f"Vehicle {plate} assigned to slot {slot_id}"
                }
            return {'status': 'error', 'message': 'No available slots'}

        elif action == 'exit':
            if sync_parking_processor.release_slot(plate):
                return {'status': 'success', 'message': f"Vehicle {plate} exited"}
            return {'status': 'error', 'message': 'Vehicle not found'}

        return {'status': 'error', 'message': 'Invalid action'}
    except Exception as e:
        logger.error(f"Manual action error: {str(e)}")
        return {'status': 'error', 'message': str(e)}


# === Data Access Functions ===

def get_preview_path(filename: str) -> str:
    """
    Get path to preview file
    Returns: Absolute file path
    """
    try:
        path = OUTPUT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Preview not found: {filename}")
        return str(path)
    except Exception as e:
        logger.error(f"Preview error: {str(e)}")
        return ""


def get_download_path(filename: str) -> str:
    """
    Get path to result file for download
    Returns: Absolute file path
    """
    try:
        candidate = OUTPUT_DIR / filename

        # Check for common extensions if no extension provided
        if not candidate.exists() and not candidate.suffix:
            for ext in ('.mp4', '.csv', '.xlsx', '.jpg', '.png'):
                alt = OUTPUT_DIR / f"{filename}{ext}"
                if alt.exists():
                    return str(alt)

        if candidate.exists():
            return str(candidate)

        raise FileNotFoundError(f"Download not found: {filename}")
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return ""


def get_detection_history(limit: int = 100) -> list:
    """
    Get plate detection history
    Returns: List of detection records
    """
    try:
        records = list(db.detections.find(
            {},
            {'_id': 0}
        ).sort("detected_at", -1).limit(limit))
        return records
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return []


def get_parking_events(limit: int = 100) -> list:
    """Get recent parking events"""
    try:
        return list(db.parking_events.find(
            {},
            {'_id': 0}
        ).sort("timestamp", -1).limit(limit))
    except Exception as e:
        logger.error(f"Parking events error: {str(e)}")
        return []


def export_parking_logs() -> str:
    """
    Export parking logs to CSV
    Returns: Path to exported file
    """
    try:
        logs = list(db.parking_events.find({}, {'_id': 0}))
        if not logs:
            return ""

        # Simplified export without pandas
        filename = f"parking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_path = OUTPUT_DIR / filename

        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)

        return str(out_path)
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return ""


# === Configuration Functions ===

def configure_parking_zones(zones: list) -> dict:
    """
    Configure parking zones
    Returns: Operation status
    """
    try:
        sync_parking_processor.configure_zones(zones)
        return {
            'status': 'success',
            'message': f'Updated {len(zones)} parking zones'
        }
    except Exception as e:
        logger.error(f"Zone config error: {str(e)}")
        return {'status': 'error', 'message': str(e)}


def get_system_config() -> dict:
    """
    Get current system configuration
    Returns: Configuration settings
    """
    try:
        return {
            'debug_mode': os.getenv('DEBUG_VIDEO', 'false').lower() == 'true',
            'plate_confidence': float(os.getenv('PLATE_CONFIDENCE', '0.75')),
            'total_slots': sync_parking_processor.total_slots,
            'active_zones': sync_parking_processor.zones,
            'mongodb_connected': mongo_client.server_info() is not None
        }
    except Exception as e:
        logger.error(f"Config error: {str(e)}")
        return {'status': 'error', 'message': str(e)}


def process_image_file(image_path: str) -> dict:
    """
    Process an image file for license plate detection
    Returns: {
        'status': 'completed'|'error',
        'annotated_image': str,
        'detections': list,
        'preview_url': str,
        'download_url': str,
        'processing_time': float,
        'message': str
    }
    """
    start_time = time.time()
    try:
        # Validate input
        image_path = str(MODELS_DIR / image_path)
        if not Path(image_path).exists():
            return {
                'status': 'error',
                'message': f"File not found: {image_path}",
                'processing_time': time.time() - start_time
            }

        # Process with thread lock
        with anpr_lock:
            logger.info(f"Processing image: {Path(image_path).name}")
            output, detections = sync_anpr_processor.process_image(image_path)

        result = {
            'status': 'completed',
            'annotated_image': output,
            'detections': detections,
            'preview_url': f"/preview/{Path(output).name}",
            'download_url': f"/download/{Path(output).name}",
            'processing_time': time.time() - start_time,
            'message': 'Image processing completed'
        }

        # Save to database
        db.image_results.insert_one({
            'image_path': image_path,
            'timestamp': datetime.utcnow(),
            'result': result
        })

        return result

    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        return {
            'status': 'error',
            'message': f"Processing error: {str(e)}",
            'processing_time': time.time() - start_time
        }
