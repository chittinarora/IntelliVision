import os
import cv2
import numpy as np
import pandas as pd
import re
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from fastapi import HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import logging
import urllib.parse

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("anpr.processor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define absolute output directory from environment
OUTPUT_DIR = Path(os.getenv("ANPR_OUTPUT_DIR", Path(__file__).parent.parent / "data" / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"All output will be saved to: {OUTPUT_DIR}")

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/anpr")
parking_db_client = MongoClient(mongo_uri)
parking_db = parking_db_client['anpr']

# Initialize parking collections
parking_events = parking_db['parking_events']
parking_slots = parking_db['parking_slots']

# Relative imports
from .detector import LicensePlateDetector
from .tracker import VehicleTracker
from .ocr import PlateOCR

class ANPRProcessor:
    """
    ANPRProcessor handles video/image plate detection with:
    - Output saved to configurable directory
    - Automatic state reset between processing jobs
    - Robust file path handling
    - Enhanced file existence verification
    """

    # Detection thresholds & smoothing
    CAR_DET_CONF = 0.6
    PLATE_DET_CONF = 0.6
    DET_IOU = 0.5
    LOCK_CONF_THRESHOLD = 0.75
    MIN_ROI_SIZE = 10  # Minimum pixel size for ROI width/height

    def __init__(self, plate_model_path: str, car_model_path: str):
        # Verify model paths exist
        if not Path(plate_model_path).exists():
            raise FileNotFoundError(f"Plate model not found: {plate_model_path}")
        if not Path(car_model_path).exists():
            raise FileNotFoundError(f"Car model not found: {car_model_path}")

        # Cloudinary configuration
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True
        )

        # Initialize detectors, OCR, and tracker
        self.plate_detector = LicensePlateDetector(plate_model_path, freeze_conf=self.LOCK_CONF_THRESHOLD)
        self.car_detector = LicensePlateDetector(car_model_path, freeze_conf=self.LOCK_CONF_THRESHOLD)
        self.ocr = PlateOCR()

        # Initialize with empty state
        self.reset()

        # MongoDB connection with improved handling
        self.setup_mongodb()

    def setup_mongodb(self):
        """Configure MongoDB connection with robust error handling"""
        raw_uri = os.getenv("MONGODB_URI")
        if not raw_uri:
            logger.error("MONGODB_URI environment variable not set")
            # Create dummy collection to prevent crashes
            self.create_dummy_collection()
            return

        try:
            # Sanitize and parse URI
            raw_uri = raw_uri.strip().strip('"').strip("'")

            # Handle MongoDB Atlas connection strings
            if "mongodb+srv://" in raw_uri:
                # For Atlas, we don't need to modify the URI
                logger.info("Using MongoDB Atlas connection")
            else:
                # Add directConnection option for replica set issues
                parsed_uri = urllib.parse.urlparse(raw_uri)
                if "directConnection" not in parsed_uri.query:
                    connector = "&" if parsed_uri.query else "?"
                    raw_uri += f"{connector}directConnection=false"

            # Create client with appropriate options
            self.mongo_client = MongoClient(
                raw_uri,
                serverSelectionTimeoutMS=15000,  # 15 seconds
                connectTimeoutMS=10000,          # 10 seconds
                socketTimeoutMS=30000            # 30 seconds
            )

            # Verify connection
            self.mongo_client.admin.command('ping')
            logger.info("MongoDB connection established")

            db_name = os.getenv("MONGODB_DB", "anpr")
            self.db = self.mongo_client[db_name]
            self.collection = self.db["detections"]

        except Exception as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            # Fallback to dummy client to prevent crashes
            self.create_dummy_collection()

    def create_dummy_collection(self):
        """Create a dummy collection to prevent crashes when MongoDB is unavailable"""
        logger.warning("Using dummy MongoDB collection")
        class DummyCollection:
            def insert_one(self, *args, **kwargs):
                logger.debug("DummyCollection.insert_one called")
            def find(self, *args, **kwargs):
                logger.debug("DummyCollection.find called")
                return []
            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        self.collection = DummyCollection()

    def reset(self):
        """Reset all stateful variables for a new processing job"""
        self.tracker = VehicleTracker()
        self.plate_history = defaultdict(lambda: defaultdict(int))
        self.plate_timestamps = defaultdict(dict)
        self.locked_plate = {}
        logger.info("Processor state reset for new job")

    def is_valid_plate(self, text: str) -> bool:
        cleaned = re.sub(r"\s+", "", text).upper()
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', cleaned))

    def validate_roi(self, roi: np.ndarray, bbox: tuple) -> bool:
        """Validate ROI meets minimum requirements"""
        if roi is None or roi.size == 0:
            logger.debug(f"Empty ROI at bbox {bbox}")
            return False
        if roi.shape[0] < self.MIN_ROI_SIZE or roi.shape[1] < self.MIN_ROI_SIZE:
            logger.debug(f"ROI too small: {roi.shape} at bbox {bbox}")
            return False
        return True

    def clamp_bbox(self, x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int) -> tuple:
        """Ensure bounding box stays within frame boundaries"""
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width - 1, x2)
        y2 = min(frame_height - 1, y2)
        return x1, y1, x2, y2

    def process_video(self, video_path: str):
        """Process a video file with output saved to output directory"""
        # Reset state for new video
        self.reset()

        # Verify input file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        filename = os.path.basename(video_path)
        base, ext = os.path.splitext(filename)
        logger.info(f"Starting video processing: {filename}")

        # Create annotated filename with prefix
        annotated_filename = f"annotated_{filename}"
        annotated_video = OUTPUT_DIR / annotated_filename

        # Cleanup any existing output file
        if annotated_video.exists():
            try:
                annotated_video.unlink()
                logger.warning(f"Deleted existing output file: {annotated_video}")
            except Exception as e:
                logger.error(f"Could not delete existing file: {str(e)}")

        # Open video capture
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
        except Exception as e:
            logger.error(f"Video open failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Cannot open video file")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video properties: {width}x{height} @ {fps:.1f}FPS, {frame_count} frames")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {annotated_video}")

        frame_idx = 0
        processing_error = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % 100 == 0:
                    logger.info(f"Processing frame {frame_idx}/{frame_count}")

                # Detect and track cars
                car_dets = self.car_detector.detect_plates(frame,
                                                          conf=self.CAR_DET_CONF,
                                                          iou=self.DET_IOU,
                                                          classes=[2])
                tracks = self.tracker.update([(x1,y1,x2,y2,conf,2) for x1,y1,x2,y2,conf in car_dets], frame)

                # For each tracked car, detect plates and run OCR
                for tr in tracks:
                    tid = tr['track_id']
                    x1,y1,x2,y2 = tr['bbox']

                    # Clamp and validate bounding box
                    x1, y1, x2, y2 = self.clamp_bbox(x1, y1, x2, y2, width, height)
                    if (x2 - x1) < self.MIN_ROI_SIZE or (y2 - y1) < self.MIN_ROI_SIZE:
                        continue

                    roi = frame[y1:y2, x1:x2]

                    # Skip invalid ROIs
                    if not self.validate_roi(roi, (x1,y1,x2,y2)):
                        continue

                    try:
                        plates = self.plate_detector.detect_plates(
                            roi,
                            conf=self.PLATE_DET_CONF,
                            iou=self.DET_IOU
                        )
                    except Exception as e:
                        logger.error(f"Plate detection failed for track {tid}: {str(e)}")
                        continue

                    for px1,py1,px2,py2,score in plates:
                        rx1,ry1 = x1+px1, y1+py1
                        rx2,ry2 = x1+px2, y1+py2

                        # Clamp plate coordinates
                        rx1, ry1, rx2, ry2 = self.clamp_bbox(rx1, ry1, rx2, ry2, width, height)
                        if (rx2 - rx1) < self.MIN_ROI_SIZE or (ry2 - ry1) < self.MIN_ROI_SIZE:
                            continue

                        plate_img = frame[ry1:ry2, rx1:rx2]

                        # Skip invalid plate images
                        if not self.validate_roi(plate_img, (rx1,ry1,rx2,ry2)):
                            continue

                        try:
                            text, conf_txt = self.ocr.read_plate(plate_img)
                        except Exception as e:
                            logger.error(f"OCR failed for plate at {rx1},{ry1}-{rx2},{ry2}: {str(e)}")
                            continue

                        # Lock in a high-confidence read
                        if conf_txt >= self.LOCK_CONF_THRESHOLD and tid not in self.locked_plate:
                            self.locked_plate[tid] = text

                        # If not locked, track frequency
                        if text and tid not in self.locked_plate:
                            self.plate_history[tid][text] += 1
                            self.plate_timestamps[tid].setdefault(text, datetime.now())

                        # Decide final text and styling
                        if tid in self.locked_plate:
                            final_text, fs, th = self.locked_plate[tid], 1.0, 3
                        else:
                            hist = self.plate_history[tid]
                            final_text = max(hist.items(), key=lambda x: x[1])[0] if hist else text
                            fs, th = 0.6, 2

                        # Draw plate box and label
                        cv2.rectangle(frame, (rx1,ry1), (rx2,ry2), (255,0,0), 2)
                        cv2.putText(frame,
                                    f"{final_text} ({conf_txt:.2f})",
                                    (rx1, max(ry1-10,0)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    fs,
                                    (255,255,255),
                                    th)

                # Draw car tracking boxes
                for tr in tracks:
                    bx1,by1,bx2,by2 = tr['bbox']
                    tid = tr['track_id']
                    cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,255,0), 2)
                    cv2.putText(frame,
                                f"Car {tid}",
                                (bx1, max(by1-10,0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0,255,0),
                                2)

                writer.write(frame)
        except Exception as e:
            logger.exception("Video processing failed")
            processing_error = True
        finally:
            cap.release()
            writer.release()

            # Verify output file was created
            if annotated_video.exists():
                file_size = annotated_video.stat().st_size
                if file_size > 1024:  # Minimum 1KB
                    logger.info(f"Saved annotated video: {annotated_video} ({file_size} bytes)")
                else:
                    logger.error(f"Output video too small: {annotated_video} ({file_size} bytes)")
                    annotated_video.unlink()
                    processing_error = True
            else:
                logger.error(f"Output video not created: {annotated_video}")
                processing_error = True

        if processing_error:
            raise HTTPException(status_code=500, detail="Video processing failed")

        # Summarize OCR results and persist
        rows, seen = [], set()
        for tid, hist in self.plate_history.items():
            if tid in self.locked_plate:
                plate = self.locked_plate[tid]
                ts = self.plate_timestamps[tid].get(plate, datetime.now())
            else:
                plate,_ = max(hist.items(), key=lambda x: x[1])
                ts = self.plate_timestamps[tid].get(plate, datetime.now())
            if plate and plate not in seen:
                seen.add(plate)
                count = hist.get(plate,1)
                rows.append({
                    "Plate Number": plate,
                    "Detected At": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "OCR Count": count
                })
                # Use the collection (will work with dummy if needed)
                self.collection.insert_one({
                    "plate_number": plate,
                    "detected_at": ts,
                    "ocr_count": count,
                    "video_file": filename,
                    "output_path": str(annotated_video)
                })

        # Create CSV and Excel reports
        try:
            df = pd.DataFrame(rows)
            csv_file = OUTPUT_DIR / f"annotated_{base}.csv"
            xlsx_file = OUTPUT_DIR / f"annotated_{base}.xlsx"
            df.to_csv(csv_file, index=False)
            df.to_excel(xlsx_file, index=False)
            logger.info(f"Created reports: {csv_file}, {xlsx_file}")
        except Exception as e:
            logger.error(f"Failed to create reports: {str(e)}")
            csv_file = None
            xlsx_file = None

        summary = {
            "plates_detected": df.get("Plate Number", []).tolist() if not df.empty else [],
            "plate_detection_times": {r["Plate Number"]: r["Detected At"] for r in rows},
            "plate_count": len(rows),
            "vehicle_count": self.tracker.get_total_unique_ids(),
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "output_video": str(annotated_video),
            "csv_file": str(csv_file) if csv_file else None,
            "xlsx_file": str(xlsx_file) if xlsx_file else None
        }
        return str(annotated_video), summary

    def process_image(self, image_path: str):
        """Process an image file with output saved to output directory"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        base = os.path.splitext(os.path.basename(image_path))[0]
        # Create annotated filename with prefix
        annotated_filename = f"annotated_{base}.jpg"
        out_file = OUTPUT_DIR / annotated_filename

        # Cleanup any existing output file
        if out_file.exists():
            try:
                out_file.unlink()
                logger.warning(f"Deleted existing output file: {out_file}")
            except Exception as e:
                logger.error(f"Could not delete existing file: {str(e)}")

        try:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read image: {image_path}")
        except Exception as e:
            logger.error(f"Image read failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Cannot read image file")

        results = []
        try:
            plates = self.plate_detector.detect_plates(
                frame,
                conf=self.PLATE_DET_CONF,
                iou=self.DET_IOU
            )
        except Exception as e:
            logger.error(f"Plate detection failed for image: {str(e)}")
            plates = []

        for x1,y1,x2,y2,score in plates:
            if (x2 - x1) < self.MIN_ROI_SIZE or (y2 - y1) < self.MIN_ROI_SIZE:
                continue
            crop = frame[y1:y2, x1:x2]
            if not self.validate_roi(crop, (x1,y1,x2,y2)):
                continue
            try:
                text, conf_txt = self.ocr.read_plate(crop)
            except Exception as e:
                logger.error(f"OCR failed: {str(e)}")
                continue
            fs, th = (1.0,3) if conf_txt>=self.LOCK_CONF_THRESHOLD else (0.6,2)
            label = f"{text} ({conf_txt:.2f})"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, label, (x1, max(y1-10,0)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th)
            results.append({"bbox": [x1,y1,x2,y2], "plate": text, "confidence": round(conf_txt,2)})

        # Save output image
        try:
            cv2.imwrite(str(out_file), frame)
            if not out_file.exists():
                raise RuntimeError("Failed to save output image")
            logger.info(f"Saved annotated image: {out_file}")
        except Exception as e:
            logger.error(f"Image save failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save output image")

        return str(out_file), results

    def get_all_detections(self):
        try:
            return list(self.collection.find({}, {"_id":0}))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


class ParkingProcessor:
    """
    Real-time parking management processor with:
    - Dual camera support (entry and exit)
    - Slot assignment and release
    - Barrier control integration
    - State persistence in MongoDB
    """

    def __init__(self, plate_model_path: str, car_model_path: str, total_slots: int = 50):
        # Initialize detectors
        self.plate_detector = LicensePlateDetector(plate_model_path)
        self.car_detector = LicensePlateDetector(car_model_path)
        self.ocr = PlateOCR()

        # Camera configuration
        self.cameras = {
            "entry": None,
            "exit": None
        }

        # Parking state
        self.total_slots = total_slots
        self.running = False
        self.entry_line_y = 200
        self.exit_line_y = 400
        self.track_history = defaultdict(list)

        # Initialize parking slots
        self.init_parking_slots()

        logger.info("Parking Processor initialized")

    def init_parking_slots(self):
        """Initialize parking slots in MongoDB if not exists"""
        if parking_slots.count_documents({}) == 0:
            for i in range(1, self.total_slots + 1):
                parking_slots.insert_one({
                    "slot_id": f"A-{i}",
                    "status": "AVAILABLE",
                    "plate": None,
                    "entry_time": None
                })
            logger.info(f"Initialized {self.total_slots} parking slots")

    def configure_cameras(self, entry_cam_id: int = 0, exit_cam_id: int = 1):
        """Configure entry and exit cameras"""
        self.cameras = {
            "entry": entry_cam_id,
            "exit": exit_cam_id
        }
        logger.info(f"Cameras configured - Entry: {entry_cam_id}, Exit: {exit_cam_id}")
        return self.cameras

    def assign_slot(self, plate: str) -> str:
        """Assign available slot to vehicle"""
        slot = parking_slots.find_one({"status": "AVAILABLE"})
        if slot:
            parking_slots.update_one(
                {"_id": slot["_id"]},
                {"$set": {
                    "status": "OCCUPIED",
                    "plate": plate,
                    "entry_time": datetime.utcnow()
                }}
            )
            return slot["slot_id"]
        # Fix: Always return a string as per function signature
        return ""
    def release_slot(self, plate: str) -> bool:
        """Release slot occupied by vehicle"""
        slot = parking_slots.find_one({"plate": plate, "status": "OCCUPIED"})
        if slot:
            parking_slots.update_one(
                {"_id": slot["_id"]},
                {"$set": {
                    "status": "AVAILABLE",
                    "plate": None,
                    "entry_time": None
                }}
            )
            return True
        return False

    def open_barrier(self, gate_type: str):
        """Control barrier (stub implementation)"""
        logger.info(f"Opening {gate_type} barrier")
        # Actual GPIO control would go here
        # GPIO.output(BARRIER_PINS[gate_type], GPIO.HIGH)

    def log_parking_event(self, plate: str, event_type: str, slot_id: str = ""):
        """Log parking event to database"""
        event = {
            "plate": plate,
            "event_type": event_type,
            "timestamp": datetime.utcnow(),
            "slot_id": slot_id
        }
        parking_events.insert_one(event)
        logger.info(f"Logged event: {plate} {event_type}")

    def process_frame(self, frame, cam_type: str):
        """Process a single frame for parking system"""
        # Detect vehicles
        car_dets = self.car_detector.detect_plates(frame, classes=[2])
        if not car_dets:
            return

        # Process each vehicle detection
        for (x1, y1, x2, y2, conf) in car_dets:
            # Calculate centroid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Process based on camera type
            if cam_type == "entry":
                self.process_entry(frame, x1, y1, x2, y2)
            elif cam_type == "exit":
                self.process_exit(frame, x1, y1, x2, y2)

    def process_entry(self, frame, x1: int, y1: int, x2: int, y2: int):
        """Process vehicle at entry point"""
        # Extract vehicle ROI
        vehicle_roi = frame[y1:y2, x1:x2]

        # Detect license plate
        plates = self.plate_detector.detect_plates(vehicle_roi)
        if not plates:
            return

        # Get best plate detection
        best_plate = max(plates, key=lambda x: x[4])
        px1, py1, px2, py2, score = best_plate

        # Extract plate image
        plate_img = vehicle_roi[py1:py2, px1:px2]
        plate_text, conf = self.ocr.read_plate(plate_img)

        if plate_text and conf > 0.7:
            # Assign slot and open barrier
            slot_id = self.assign_slot(plate_text)
            if slot_id:
                self.open_barrier("entry")
                self.log_parking_event(plate_text, "ENTRY", slot_id)
                logger.info(f"Assigned slot {slot_id} to {plate_text}")

    def process_exit(self, frame, x1: int, y1: int, x2: int, y2: int):
        """Process vehicle at exit point"""
        # Extract vehicle ROI
        vehicle_roi = frame[y1:y2, x1:x2]

        # Detect license plate
        plates = self.plate_detector.detect_plates(vehicle_roi)
        if not plates:
            return

        # Get best plate detection
        best_plate = max(plates, key=lambda x: x[4])
        px1, py1, px2, py2, score = best_plate

        # Extract plate image
        plate_img = vehicle_roi[py1:py2, px1:px2]
        plate_text, conf = self.ocr.read_plate(plate_img)

        if plate_text and conf > 0.7:
            # Release slot and open barrier
            if self.release_slot(plate_text):
                self.open_barrier("exit")
                self.log_parking_event(plate_text, "EXIT")
                logger.info(f"Released slot for {plate_text}")

    def start_processing(self):
        """Start real-time processing for both cameras"""
        if not self.cameras["entry"] or not self.cameras["exit"]:
            logger.error("Cameras not configured")
            return

        self.running = True

        # Start entry camera thread
        threading.Thread(target=self.process_camera_stream, args=("entry",), daemon=True).start()

        # Start exit camera thread
        threading.Thread(target=self.process_camera_stream, args=("exit",), daemon=True).start()

        logger.info("Started parking processing threads")

    def process_camera_stream(self, cam_type: str):
        """Process video stream from specified camera"""
        cam_id = self.cameras[cam_type]
        cap = cv2.VideoCapture(cam_id)

        if not cap.isOpened():
            logger.error(f"Could not open {cam_type} camera (ID: {cam_id})")
            return

        logger.info(f"Processing {cam_type} camera stream")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Frame read error from {cam_type} camera")
                    continue

                self.process_frame(frame, cam_type)

        except Exception as e:
            logger.error(f"Error processing {cam_type} stream: {str(e)}")
        finally:
            cap.release()
            logger.info(f"Stopped {cam_type} camera processing")

    def stop_processing(self):
        """Stop all processing threads"""
        self.running = False
        logger.info("Parking processing stopped")

    def get_parking_status(self) -> dict:
        """Get current parking status"""
        available = parking_slots.count_documents({"status": "AVAILABLE"})
        occupied = self.total_slots - available
        return {
            "total_slots": self.total_slots,
            "available": available,
            "occupied": occupied,
            "updated_at": datetime.utcnow().isoformat()
        }
