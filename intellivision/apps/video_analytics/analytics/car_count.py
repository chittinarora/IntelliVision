"""
Car counting and ANPR analytics using YOLO-based detection and OCR.
Supports video/image plate recognition and parking analysis with MongoDB and Cloudinary integration.
"""

# ======================================
# Imports and Setup
# ======================================
import os
import re
import tempfile
import uuid
import time
import logging
import csv
from pathlib import Path
from threading import Lock
from datetime import datetime, timedelta
from typing import Dict

from pymongo import MongoClient
import cloudinary
from dotenv import load_dotenv
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone
import mimetypes
from celery import shared_task
from .anpr.processor import ANPRProcessor, ParkingProcessor
from ..convert import convert_to_web_mp4

# ======================================
# Logger and Constants
# ======================================
logger = logging.getLogger("anpr_functions")

# Import progress logger
try:
    from ..progress_logger import create_progress_logger
except ImportError:
    def create_progress_logger(job_id, total_items, job_type, logger_name=None):
        """Fallback progress logger if module not available."""
        class DummyLogger:
            def __init__(self, job_id, total_items, job_type, logger_name=None):
                self.job_id = job_id
                self.total_items = total_items
                self.job_type = job_type
                self.logger = logging.getLogger(logger_name or job_type)

            def update_progress(self, processed_count, status=None, force_log=False):
                self.logger.info(f"**Job {self.job_id}**: Progress {processed_count}/{self.total_items}")

            def log_completion(self, final_count=None):
                self.logger.info(f"**Job {self.job_id}**: Completed {self.job_type}")

        return DummyLogger(job_id, total_items, job_type, logger_name)

VALID_EXTENSIONS = {'.mp4', '.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Define OUTPUT_DIR with fallback
try:
    OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
except AttributeError:
    logger.warning("JOB_OUTPUT_DIR not defined in settings. Using fallback: MEDIA_ROOT/outputs")
    OUTPUT_DIR = Path(settings.MEDIA_ROOT) / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize model paths using model manager with fallback support
try:
    from .model_manager import get_model_with_fallback

    # Get model paths with automatic fallback
    PLATE_MODEL = str(get_model_with_fallback("best_car"))
    CAR_MODEL = str(get_model_with_fallback("yolo11m_car"))

    logger.info(f"✅ Resolved plate model: {PLATE_MODEL}")
    logger.info(f"✅ Resolved car model: {CAR_MODEL}")

except Exception as e:
    logger.error(f"❌ Failed to resolve models with fallback: {e}")
    # Fallback to old hardcoded paths as last resort
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR = BASE_DIR / 'video_analytics' / 'models'
    PLATE_MODEL = str(MODELS_DIR / 'best_car.pt')
    CAR_MODEL = str(MODELS_DIR / 'yolo11m_car.pt')

    # Check model existence the old way
    MODEL_FILES = ["best_car.pt", "yolo11m_car.pt"]
    for model_file in MODEL_FILES:
        if not (MODELS_DIR / model_file).exists():
            logger.error(f"Model file missing: {model_file}")
            raise FileNotFoundError(f"Model file {model_file} not found in {MODELS_DIR}")

# Configure Cloudinary & MongoDB
try:
    # Try to get base directory from previously defined variable
    base_dir = BASE_DIR if 'BASE_DIR' in locals() else Path(__file__).resolve().parent.parent.parent
except NameError:
    base_dir = Path(__file__).resolve().parent.parent.parent

load_dotenv(base_dir / '.env')
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb+srv://toram444444:06nJTevaUItCDpd9@cluster01.lemxesc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster01')
mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = mongo_client['anpr']

# ======================================
# LAZY LOADING PROCESSOR MANAGER
# ======================================
class ProcessorManager:
    """
    Thread-safe lazy loading manager for ANPR processors.
    Maintains full backward compatibility with existing global processor pattern.
    """
    _instance = None
    _init_lock = Lock()
    
    def __init__(self):
        # Initialize lightweight components only
        self._anpr_processor = None
        self._parking_processor = None
        self._anpr_lock = Lock()
        self._parking_lock = Lock()
        self._anpr_init_attempted = False
        self._parking_init_attempted = False
        self._anpr_init_error = None
        self._parking_init_error = None
        logger.info("🔄 ProcessorManager initialized (models will load on demand)")
    
    @classmethod
    def get_instance(cls):
        """Thread-safe singleton pattern"""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _lazy_load_anpr(self):
        """Lazy load ANPR processor with same error handling as original"""
        if self._anpr_init_attempted:
            return self._anpr_processor
        
        with self._anpr_lock:
            if self._anpr_init_attempted:
                return self._anpr_processor
            
            self._anpr_init_attempted = True
            logger.info("🔄 Loading ANPR processor (first request)...")
            start_time = time.time()
            
            try:
                self._anpr_processor = ANPRProcessor(str(PLATE_MODEL), str(CAR_MODEL))
                load_time = time.time() - start_time
                logger.info(f"✅ ANPR processor loaded in {load_time:.2f}s")
            except Exception as e:
                self._anpr_init_error = e
                logger.error(f"❌ Failed to initialize ANPR processor: {e}")
                logger.warning("⚠️ ANPR functionality will be disabled")
                self._anpr_processor = None
        
        return self._anpr_processor
    
    def _lazy_load_parking(self):
        """Lazy load parking processor with same error handling as original"""
        if self._parking_init_attempted:
            return self._parking_processor
        
        with self._parking_lock:
            if self._parking_init_attempted:
                return self._parking_processor
            
            self._parking_init_attempted = True
            logger.info("🔄 Loading parking processor (first request)...")
            start_time = time.time()
            
            try:
                # Get total slots from settings or use default
                total_slots = getattr(settings, 'PARKING_TOTAL_SLOTS', 50)
                self._parking_processor = ParkingProcessor(str(PLATE_MODEL), str(CAR_MODEL), total_slots=total_slots)
                load_time = time.time() - start_time
                logger.info(f"✅ Parking processor loaded in {load_time:.2f}s")
            except Exception as e:
                self._parking_init_error = e
                logger.error(f"❌ Failed to initialize parking processor: {e}")
                logger.warning("⚠️ Parking functionality will be disabled")
                self._parking_processor = None
        
        return self._parking_processor
    
    @property
    def sync_anpr_processor(self):
        """Property that maintains exact same interface as global variable"""
        return self._lazy_load_anpr()
    
    @property
    def sync_parking_processor(self):
        """Property that maintains exact same interface as global variable"""
        return self._lazy_load_parking()
    
    @property
    def anpr_lock(self):
        """Thread lock for ANPR operations - maintains compatibility"""
        return self._anpr_lock
    
    @property
    def parking_lock(self):
        """Thread lock for parking operations - maintains compatibility"""
        return self._parking_lock

# Create global processor manager instance
_processor_manager = ProcessorManager.get_instance()

# Maintain backward compatibility with existing global variable pattern
# These will be accessed as properties, ensuring lazy loading works
class _ProcessorProxy:
    \"\"\"Proxy class to maintain global variable compatibility with lazy loading\"\"\"\n    @property\n    def sync_anpr_processor(self):\n        return _processor_manager.sync_anpr_processor\n    \n    @property\n    def sync_parking_processor(self):\n        return _processor_manager.sync_parking_processor\n    \n    @property\n    def anpr_lock(self):\n        return _processor_manager.anpr_lock\n    \n    @property\n    def parking_lock(self):\n        return _processor_manager.parking_lock\n\n_proxy = _ProcessorProxy()\n\n# Global variable access functions (to replace direct global variable usage)\ndef get_anpr_processor():\n    \"\"\"Get ANPR processor with lazy loading\"\"\"\n    return _processor_manager.sync_anpr_processor\n\ndef get_parking_processor():\n    \"\"\"Get parking processor with lazy loading\"\"\"\n    return _processor_manager.sync_parking_processor\n\ndef get_anpr_lock():\n    \"\"\"Get ANPR lock\"\"\"\n    return _processor_manager.anpr_lock\n\ndef get_parking_lock():\n    \"\"\"Get parking lock\"\"\"\n    return _processor_manager.parking_lock

# For debugging and monitoring
def get_processor_status():
    """Get current processor loading status for monitoring"""
    return {
        'anpr_loaded': _processor_manager._anpr_processor is not None,
        'parking_loaded': _processor_manager._parking_processor is not None,
        'anpr_init_attempted': _processor_manager._anpr_init_attempted,
        'parking_init_attempted': _processor_manager._parking_init_attempted,
        'anpr_error': str(_processor_manager._anpr_init_error) if _processor_manager._anpr_init_error else None,
        'parking_error': str(_processor_manager._parking_init_error) if _processor_manager._parking_init_error else None
    }

# ======================================
# Helper Functions
# ======================================

def validate_input_file(file_path: str) -> tuple[bool, str]:
    """Validate file type and size."""
    if not default_storage.exists(file_path):
        return False, f"File not found: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        return False, f"Invalid file type: {ext}. Allowed: {', '.join(VALID_EXTENSIONS)}"
    size = default_storage.size(file_path)
    if size > MAX_FILE_SIZE:
        return False, f"File size {size / (1024*1024):.2f}MB exceeds 500MB limit"
    return True, ""

# ======================================
# Main Analysis Functions
# ======================================

def recognize_number_plates(video_path: str, output_path: str) -> Dict:
    """
    Process video for number plate recognition.

    Args:
        video_path: Path to input video
        output_path: Path for output video

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()

    is_valid, error_msg = validate_input_file(video_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    try:
        # Extract job_id from output_path to use for filename
        extracted_job_id = re.search(r'output_(\d+)_', output_path)
        output_job_id = extracted_job_id.group(1) if extracted_job_id else str(int(time.time()))
        output_filename = f"outputs/annotated_{output_job_id}.mp4"

        with get_anpr_lock():
            logger.info(f"Starting plate recognition: {video_path}")
            anpr_processor = get_anpr_processor()
            if anpr_processor is None:
                raise RuntimeError("ANPR processor not available - initialization failed")
            output, summary = anpr_processor.process_video(video_path)

        if output and default_storage.exists(output):
            # Extract filename for Django storage (avoid path traversal)
            output_filename = os.path.basename(output)
            web_output_filename = output_filename.replace('.mp4', '_web.mp4')
            
            with default_storage.open(output, 'rb') as f:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
            
            # Convert and save with relative filename
            web_output_path = f"outputs/{web_output_filename}"
            converted_tmp_path = tmp_path.replace('.mp4', '_converted.mp4')
            
            if convert_to_web_mp4(tmp_path, converted_tmp_path):
                with open(converted_tmp_path, 'rb') as converted_file:
                    default_storage.save(web_output_path, converted_file)
                output = web_output_path
                os.remove(tmp_path)
                os.remove(converted_tmp_path)
            else:
                # Fallback to original if conversion fails
                original_output_path = f"outputs/{output_filename}"
                with open(tmp_path, 'rb') as original_file:
                    default_storage.save(original_output_path, original_file)
                output = original_output_path
                os.remove(tmp_path)
            output_url = default_storage.url(output)
        else:
            logger.error(f"Output video not created: {output}")
            return {
                'status': 'failed',
                'job_type': 'car_count',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': 'Output video not created'},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': 'Output video not created', 'code': 'OUTPUT_ERROR'}
            }

        processing_time = time.time() - start_time
        result = {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': output_url,
            'data': {
                'summary': summary,
                'preview_url': output_url,
                'download_url': output_url,
                'alerts': [{"message": f"Plate {p} detected", "timestamp": timezone.now().isoformat()} for p in summary.get('recognized_plates', [])]
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': summary.get('processing_fps', 0),
                'frame_count': summary.get('total_frames', 0)
            },
            'error': None
        }

        db.plate_results.insert_one({
            'video_path': video_path,
            'timestamp': datetime.utcnow(),
            'result': result,
            'type': 'plate_recognition'
        })

        return result

    except Exception as e:
        logger.exception(f"Plate recognition failed: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def analyze_parking_video(video_path: str, output_path: str = None, job_id: str = None) -> Dict:
    """
    Process video for parking analysis.

    Args:
        video_path: Path to input video
        output_path: Path to save output video (for tasks.py integration)
        job_id: VideoJob ID for progress tracking

    Returns:
        Standardized response dictionary with filesystem paths
    """
    start_time = time.time()

    # Add job_id logging for progress tracking
    if job_id:
        logger.info(f"🚀 Starting car detection video job {job_id}")

    is_valid, error_msg = validate_input_file(video_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    try:
        # Use provided job_id parameter, fallback to extracting from video path
        if job_id:
            output_job_id = str(job_id)
        else:
            extracted_job_id = re.search(r'(\d+)', video_path)
            output_job_id = extracted_job_id.group(1) if extracted_job_id else str(int(time.time()))
        output_filename = f"outputs/parking_analysis_{output_job_id}.mp4"

        with get_parking_lock():
            logger.info(f"Starting parking analysis: {video_path}")
            parking_processor = get_parking_processor()
            if parking_processor is None:
                raise RuntimeError("Parking processor not available - initialization failed")
            output, summary = parking_processor.process_video(video_path)

        if output and os.path.exists(output):
            # Create temporary file for web conversion
            with tempfile.NamedTemporaryFile(suffix='_web.mp4', delete=False) as web_tmp:
                web_tmp_path = web_tmp.name

            logger.info(f"🔄 Attempting ffmpeg conversion: {output} -> {web_tmp_path}")
            if convert_to_web_mp4(output, web_tmp_path):
                final_output_path = web_tmp_path  # Use converted file
                logger.info(f"✅ FFmpeg conversion successful")
            else:
                final_output_path = output  # Fallback to original
                logger.warning(f"⚠️ FFmpeg conversion failed, using original file")
        else:
            logger.error(f"Output video not created: {output}")
            return {
                'status': 'failed',
                'job_type': 'car_count',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': 'Output video not created'},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': 'Output video not created', 'code': 'OUTPUT_ERROR'}
            }

        summary = {
            'entries': summary.get('entries', 0),
            'exits': summary.get('exits', 0),
            'max_occupancy': summary.get('max_occupancy', 0),
            'final_occupancy': summary.get('final_occupancy', 0),
            'recognized_plates': summary.get('recognized_plates', []),
            'processing_fps': summary.get('processing_fps', 0),
            'total_frames': summary.get('total_frames', 0),
            'processing_time': summary.get('processing_time', 0),
            'vehicle_count': summary.get('vehicle_count', 0)
        }

        processing_time = time.time() - start_time
        result = {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': final_output_path,
            'data': {
                'summary': summary,
                'preview_url': final_output_path,
                'download_url': final_output_path,
                'alerts': [{"message": f"Plate {p} detected", "timestamp": timezone.now().isoformat()} for p in summary.get('recognized_plates', [])]
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': summary.get('processing_fps', 0),
                'frame_count': summary.get('total_frames', 0)
            },
            'error': None
        }

        db.parking_results.insert_one({
            'video_path': video_path,
            'timestamp': datetime.utcnow(),
            'result': result,
            'type': 'parking_analysis'
        })

        return result

    except Exception as e:
        logger.exception(f"Parking analysis failed: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def process_image_file(image_path: str, output_path: str = None, job_id: str = None) -> Dict:
    """
    Process an image for license plate detection.

    Args:
        image_path: Path to input image
        output_path: Path to save output image (for tasks.py integration)
        job_id: VideoJob ID for progress tracking

    Returns:
        Standardized response dictionary with filesystem paths
    """
    start_time = time.time()

    # Add job_id logging for progress tracking
    if job_id:
        logger.info(f"🚀 Starting car detection job {job_id}")

    is_valid, error_msg = validate_input_file(image_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    try:
        # Use provided job_id parameter, fallback to extracting from image path
        if job_id:
            output_job_id = str(job_id)
        else:
            extracted_job_id = re.search(r'(\d+)', image_path)
            output_job_id = extracted_job_id.group(1) if extracted_job_id else str(int(time.time()))
        output_filename = f"outputs/annotated_{output_job_id}.jpg"

        with get_anpr_lock():
            logger.info(f"Processing image: {image_path}")
            anpr_processor = get_anpr_processor()
            if anpr_processor is None:
                raise RuntimeError("ANPR processor not available - initialization failed")
            output, detections = anpr_processor.process_image(image_path)

        if output and os.path.exists(output):
            final_output_path = output
            # Generate URL for the output file
            output_url = default_storage.url(output) if hasattr(default_storage, 'url') else f"/media/{output}"
            logger.info(f"✅ Car detection completed, output saved to {final_output_path}")
        else:
            logger.error(f"Output image not created: {output}")
            return {
                'status': 'failed',
                'job_type': 'car_count',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': 'Output image not created'},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': 'Output image not created', 'code': 'OUTPUT_ERROR'}
            }

        processing_time = time.time() - start_time
        result = {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': final_output_path,
            'output_video': None,
            'data': {
                'annotated_image': output,
                'detections': detections,
                'preview_url': output_url,
                'download_url': output_url,
                'alerts': [{"message": f"Plate {d['plate']} detected with confidence {d['confidence']:.2f}", "timestamp": timezone.now().isoformat()} for d in detections]
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': 1
            },
            'error': None
        }

        db.image_results.insert_one({
            'image_path': image_path,
            'timestamp': datetime.utcnow(),
            'result': result
        })

        return result

    except Exception as e:
        logger.exception(f"Image processing failed: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def process_parking_stream(entry_cam: int = 0, exit_cam: int = 1) -> Dict:
    """
    Start parking system processing with webcams.

    Args:
        entry_cam: Entry camera ID
        exit_cam: Exit camera ID

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    try:
        parking_processor = get_parking_processor()
        if parking_processor is None:
            raise RuntimeError("Parking processor not available - initialization failed")
        parking_processor.configure_cameras(entry_cam_id=entry_cam, exit_cam_id=exit_cam)
        parking_processor.start_processing()
        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'message': f'Parking system started with cameras: Entry={entry_cam}, Exit={exit_cam}',
                'alerts': []
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': None
            },
            'error': None
        }
    except Exception as e:
        logger.error(f"Parking stream failed: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def stop_parking_system() -> Dict:
    """
    Stop the parking system processing.

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    try:
        parking_processor = get_parking_processor()
        if parking_processor is None:
            raise RuntimeError("Parking processor not available - initialization failed")
        parking_processor.stop_processing()
        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'message': 'Parking system stopped successfully',
                'alerts': []
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': None
            },
            'error': None
        }
    except Exception as e:
        logger.error(f"Stop parking error: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def get_parking_status() -> Dict:
    """
    Get current parking status.

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    try:
        parking_processor = get_parking_processor()
        if parking_processor is None:
            raise RuntimeError("Parking processor not available - initialization failed")
        status = parking_processor.get_parking_status()
        status['updated_at'] = timezone.now().isoformat()
        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'parking_status': status,
                'alerts': []
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': None
            },
            'error': None
        }
    except Exception as e:
        logger.error(f"Parking status error: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'parking_status': {
                    'total_slots': 0,
                    'available': 0,
                    'occupied': 0,
                    'updated_at': timezone.now().isoformat()
                },
                'alerts': [],
                'error': str(e)
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': time.time() - start_time,
                'fps': None,
                'frame_count': None
            },
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def manual_parking_action(plate: str, action: str) -> Dict:
    """
    Perform manual parking action (entry or exit).

    Args:
        plate: License plate number
        action: 'entry' or 'exit'

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    plate = plate.upper().strip()
    try:
        if action == 'entry':
            parking_processor = get_parking_processor()
            if parking_processor is None:
                raise RuntimeError("Parking processor not available - initialization failed")
            slot_id = parking_processor.assign_slot(plate)
            if slot_id:
                processing_time = time.time() - start_time
                return {
                    'status': 'completed',
                    'job_type': 'car_count',
                    'output_image': None,
                    'output_video': None,
                    'data': {
                        'slot_id': slot_id,
                        'message': f"Vehicle {plate} assigned to slot {slot_id}",
                        'alerts': [{"message": f"Vehicle {plate} assigned to slot {slot_id}", "timestamp": timezone.now().isoformat()}]
                    },
                    'meta': {
                        'timestamp': timezone.now().isoformat(),
                        'processing_time_seconds': processing_time,
                        'fps': None,
                        'frame_count': None
                    },
                    'error': None
                }
            return {
                'status': 'failed',
                'job_type': 'car_count',
                'output_image': None,
                'output_video': None,
                'data': {
                    'alerts': [],
                    'error': 'No available slots'
                },
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'processing_time_seconds': time.time() - start_time,
                    'fps': None,
                    'frame_count': None
                },
                'error': {'message': 'No available slots', 'code': 'NO_SLOTS'}
            }
        elif action == 'exit':
            parking_processor = get_parking_processor()
            if parking_processor is None:
                raise RuntimeError("Parking processor not available - initialization failed")
            if parking_processor.release_slot(plate):
                processing_time = time.time() - start_time
                return {
                    'status': 'completed',
                    'job_type': 'car_count',
                    'output_image': None,
                    'output_video': None,
                    'data': {
                        'message': f"Vehicle {plate} exited",
                        'alerts': [{"message": f"Vehicle {plate} exited", "timestamp": timezone.now().isoformat()}]
                    },
                    'meta': {
                        'timestamp': timezone.now().isoformat(),
                        'processing_time_seconds': processing_time,
                        'fps': None,
                        'frame_count': None
                    },
                    'error': None
                }
            return {
                'status': 'failed',
                'job_type': 'car_count',
                'output_image': None,
                'output_video': None,
                'data': {
                    'alerts': [],
                    'error': 'Vehicle not found'
                },
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'processing_time_seconds': time.time() - start_time,
                    'fps': None,
                    'frame_count': None
                },
                'error': {'message': 'Vehicle not found', 'code': 'VEHICLE_NOT_FOUND'}
            }
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'alerts': [],
                'error': 'Invalid action'
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': time.time() - start_time,
                'fps': None,
                'frame_count': None
            },
            'error': {'message': 'Invalid action', 'code': 'INVALID_ACTION'}
        }
    except Exception as e:
        logger.error(f"Manual action error: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'alerts': [],
                'error': str(e)
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': time.time() - start_time,
                'fps': None,
                'frame_count': None
            },
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def get_preview_path(filename: str) -> Dict:
    """
    Get path to preview file.

    Args:
        filename: Name of the file to preview

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    try:
        path = Path(filename)
        if not default_storage.exists(path):
            logger.error(f"Preview not found: {filename}")
            return {
                'status': 'failed',
                'job_type': 'car_count',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': f"Preview not found: {filename}"},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': f"Preview not found: {filename}", 'code': 'FILE_NOT_FOUND'}
            }
        output_url = default_storage.url(path)
        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': output_url if path.suffix in ['.jpg', '.jpeg', '.png'] else None,
            'output_video': output_url if path.suffix == '.mp4' else None,
            'data': {
                'preview_path': str(path),
                'preview_url': output_url,
                'alerts': []
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': None
            },
            'error': None
        }
    except Exception as e:
        logger.error(f"Preview error: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def get_download_path(filename: str) -> Dict:
    """
    Get path to result file for download.

    Args:
        filename: Name of the file to download

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    try:
        candidate = Path(filename)
        if not default_storage.exists(candidate):
            for ext in ('.mp4', '.csv', '.jpg', '.png'):
                alt = candidate.with_suffix(ext)
                if default_storage.exists(alt):
                    candidate = alt
                    break
            else:
                logger.error(f"Download not found: {filename}")
                return {
                    'status': 'failed',
                    'job_type': 'car_count',
                    'output_image': None,
                    'output_video': None,
                    'data': {'alerts': [], 'error': f"Download not found: {filename}"},
                    'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                    'error': {'message': f"Download not found: {filename}", 'code': 'FILE_NOT_FOUND'}
                }
        output_url = default_storage.url(candidate)
        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': output_url if candidate.suffix in ['.jpg', '.jpeg', '.png'] else None,
            'output_video': output_url if candidate.suffix == '.mp4' else None,
            'data': {
                'download_path': str(candidate),
                'download_url': output_url,
                'alerts': []
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': None
            },
            'error': None
        }
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def get_detection_history(limit: int = 100) -> Dict:
    """
    Get plate detection history.

    Args:
        limit: Number of records to retrieve

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    try:
        records = list(db.detections.find({}, {'_id': 0}).sort("detected_at", -1).limit(limit))
        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'records': records,
                'alerts': []
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': None
            },
            'error': None
        }
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'DATABASE_ERROR'}
        }

def get_parking_events(limit: int = 100) -> Dict:
    """
    Get recent parking events.

    Args:
        limit: Number of events to retrieve

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    try:
        events = list(db.parking_events.find({}, {'_id': 0}).sort("timestamp", -1).limit(limit))
        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'events': events,
                'alerts': []
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': None
            },
            'error': None
        }
    except Exception as e:
        logger.error(f"Parking events error: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'DATABASE_ERROR'}
        }

def export_parking_logs() -> Dict:
    """
    Export parking logs to CSV.

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    try:
        logs = list(db.parking_events.find({}, {'_id': 0}))
        if not logs:
            return {
                'status': 'failed',
                'job_type': 'car_count',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': 'No logs found'},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': 'No logs found', 'code': 'NO_LOGS'}
            }
        filename = f"parking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_filename = f"outputs/{filename}"
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            with open(tmp.name, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
            default_storage.save(output_filename, open(tmp.name, 'rb'))
        output_url = default_storage.url(output_filename)
        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'export_path': output_filename,
                'export_url': output_url,
                'alerts': []
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': None
            },
            'error': None
        }
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def configure_parking_zones(zones: list) -> Dict:
    """
    Configure parking zones.

    Args:
        zones: List of zone configurations

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    try:
        parking_processor = get_parking_processor()
        if parking_processor is None:
            raise RuntimeError("Parking processor not available - initialization failed")
        parking_processor.configure_zones(zones)
        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'message': f'Updated {len(zones)} parking zones',
                'alerts': []
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': None
            },
            'error': None
        }
    except Exception as e:
        logger.error(f"Zone config error: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

def get_system_config() -> Dict:
    """
    Get current system configuration.

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    try:
        config = {
            'debug_mode': os.getenv('DEBUG_VIDEO', 'false').lower() == 'true',
            'plate_confidence': float(os.getenv('PLATE_CONFIDENCE', '0.75')),
            'total_slots': get_parking_processor().total_slots if get_parking_processor() else 0,
            'active_zones': get_parking_processor().zones if get_parking_processor() else [],
            'mongodb_connected': mongo_client.server_info() is not None
        }
        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {
                'config': config,
                'alerts': []
            },
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'fps': None,
                'frame_count': None
            },
            'error': None
        }
    except Exception as e:
        logger.error(f"Config error: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'car_count',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }

# ======================================
# Celery Integration
# ======================================

def tracking_video(input_path: str, output_path: str = None, job_id: str = None) -> Dict:
    """
    Celery task for car counting and ANPR.

    Args:
        self: Celery task instance
        input_path: Path to input video or image
        output_path: Path to save output
        job_id: VideoJob ID

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    logger.info(f"🚀 Starting car count job {job_id}")

    ext = os.path.splitext(input_path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png']

    if ext in image_exts:
        # Initialize progress logger for image processing
        progress_logger = create_progress_logger(
            job_id=str(job_id) if job_id else "0",
            total_items=1,  # Single image
            job_type="car-count"
        )

        progress_logger.update_progress(0, status="Processing image...", force_log=True)
        result = process_image_file(input_path, output_path, job_id)
        progress_logger.update_progress(1, status="Analysis completed", force_log=True)
        progress_logger.log_completion(1)
    else:
        # Initialize progress logger for video processing
        progress_logger = create_progress_logger(
            job_id=str(job_id) if job_id else "0",
            total_items=100,  # Estimate for video frames
            job_type="car-count"
        )

        progress_logger.update_progress(0, status="Starting video processing...", force_log=True)
        result = analyze_parking_video(input_path, output_path, job_id)
        progress_logger.update_progress(100, status="Video processing completed", force_log=True)
        progress_logger.log_completion(100)

    processing_time = time.time() - start_time
    result['meta']['processing_time_seconds'] = processing_time
    result['meta']['timestamp'] = timezone.now().isoformat()

    return result
