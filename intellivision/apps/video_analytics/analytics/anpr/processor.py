import os
import cv2
import numpy as np
import pandas as pd
import re
import logging
import threading
import time
import asyncio
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from pymongo import MongoClient
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from django.conf import settings

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("anpr.processor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define canonical output directory
OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"All output will be saved to: {OUTPUT_DIR}")

# Relative imports
from .detector import LicensePlateDetector
from .tracker import VehicleTracker
from .ocr import PlateOCR

class ANPRProcessor:
    """
    ANPRProcessor handles video/image plate detection with:
    - Enhanced debugging capabilities
    - Real-time visual feedback
    - Robust error handling
    - Comprehensive logging
    - Video processing diagnostics
    """

    # Detection thresholds & smoothing
    CAR_DET_CONF = 0.6
    PLATE_DET_CONF = 0.6
    DET_IOU = 0.5
    LOCK_CONF_THRESHOLD = 0.6
    MIN_ROI_SIZE = 10  # Minimum pixel size for ROI width/height

    def __init__(self, plate_model_path: str, car_model_path: str):
        # The model paths may come from model manager with fallback, so they should already be valid
        # But let's still verify them as a safety check
        if not Path(plate_model_path).exists():
            logger.error(f"Plate model not found: {plate_model_path}")
            raise FileNotFoundError(f"Plate model not found: {plate_model_path}")
        if not Path(car_model_path).exists():
            logger.error(f"Car model not found: {car_model_path}")
            raise FileNotFoundError(f"Car model not found: {car_model_path}")

        logger.info(f"✅ Initializing ANPR processor with plate model: {plate_model_path}")
        logger.info(f"✅ Initializing ANPR processor with car model: {car_model_path}")

        # Initialize detectors, OCR, and tracker
        self.plate_detector = LicensePlateDetector(plate_model_path, freeze_conf=self.LOCK_CONF_THRESHOLD)
        self.car_detector = LicensePlateDetector(car_model_path, freeze_conf=self.LOCK_CONF_THRESHOLD)
        self.ocr = PlateOCR()

        # Initialize with empty state
        self.reset()

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

    def process_video(self, video_path: str, output_path: str = None):
        """Process a video file with enhanced debugging and diagnostics"""
        # Reset state for new video
        self.reset()
        start_time = time.time()

        # Verify input file exists
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Input video not found: {video_path}")
            return None, {"error": "Input video not found", "path": str(video_path)}

        filename = video_path.name
        base, ext = os.path.splitext(filename)
        logger.info(f"Starting video processing: {filename}")

        # Use provided output_path or create default
        if output_path is None:
            annotated_filename = f"annotated_{filename}"
            annotated_video = OUTPUT_DIR / annotated_filename
        else:
            annotated_video = Path(output_path)

        # Cleanup any existing output file
        if annotated_video.exists():
            try:
                annotated_video.unlink()
                logger.warning(f"Deleted existing output file: {annotated_video}")
            except Exception as e:
                logger.error(f"Could not delete existing file: {str(e)}")

        # Open video capture
        try:
            logger.debug(f"Opening video: {video_path}")
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                error_msg = f"Could not open video: {video_path}"
                logger.error(error_msg)
                return None, {"error": error_msg}
        except Exception as e:
            error_msg = f"Video open failed: {str(e)}"
            logger.error(error_msg)
            return None, {"error": error_msg}

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video properties: {width}x{height} @ {fps:.1f}FPS, {frame_count} frames")

        # Create video writer with fallback codecs
        writer = None
        codecs = ["mp4v", "avc1", "MJPG", "XVID", "H264", "VP80"]
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(annotated_video), fourcc, fps, (width, height))
            if writer.isOpened():
                logger.info(f"Video writer initialized with codec: {codec}")
                break
            else:
                logger.warning(f"Codec {codec} failed, trying next option")

        if not writer or not writer.isOpened():
            cap.release()
            error_msg = f"Could not create output video: {annotated_video}"
            logger.error(error_msg)
            return None, {"error": error_msg}

        frame_idx = 0
        processing_error = False
        error_details = ""

        # Debug visualization setup (disabled in Docker/headless environments)
        debug_mode = (os.getenv("DEBUG_VIDEO", "false").lower() == "true" and 
                     os.getenv("ENVIRONMENT", "production") != "production" and
                     os.getenv("DISPLAY") is not None)
        debug_window_name = f"Processing: {filename}"

        if debug_mode:
            try:
                cv2.namedWindow(debug_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(debug_window_name, 800, 600)
            except cv2.error as e:
                logger.warning(f"Could not create debug window (headless environment?): {e}")
                debug_mode = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Reached end of video")
                    break

                frame_idx += 1
                if frame_idx % 10 == 0:  # Log every 10 frames
                    logger.info(f"Processing frame {frame_idx}/{frame_count}")

                # Create debug frame if needed
                debug_frame = frame.copy() if debug_mode else None

                # Detect and track cars
                car_dets = self.car_detector.detect_plates(frame,
                                                          conf=self.CAR_DET_CONF,
                                                          iou=self.DET_IOU,
                                                          classes=[2])

                # DEBUG: Show raw detections
                if debug_mode:
                    for (x1, y1, x2, y2, conf) in car_dets:
                        cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(debug_frame, f"Car: {conf:.2f}", (int(x1), int(y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                logger.debug(f"Frame {frame_idx}: Found {len(car_dets)} car detections")

                # Convert detections to tracker format: [x, y, w, h]
                tracker_input = []
                for (x1, y1, x2, y2, conf) in car_dets:
                    w = x2 - x1
                    h = y2 - y1
                    # Only include valid detections
                    if w > 0 and h > 0:
                        tracker_input.append(([x1, y1, w, h], conf, 0))

                tracks = self.tracker.update(tracker_input, frame)

                logger.debug(f"Frame {frame_idx}: Tracking {len(tracks)} vehicles")

                # For each tracked car, detect plates and run OCR
                for tr in tracks:
                    tid = tr['track_id']
                    x1,y1,x2,y2 = tr['bbox']

                    # DEBUG: Draw tracking box
                    if debug_mode:
                        cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                        cv2.putText(debug_frame, f"Track {tid}", (int(x1), int(max(y1-10,0))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # Clamp and validate bounding box
                    x1, y1, x2, y2 = self.clamp_bbox(x1, y1, x2, y2, width, height)
                    if (x2 - x1) < self.MIN_ROI_SIZE or (y2 - y1) < self.MIN_ROI_SIZE:
                        logger.debug(f"Track {tid}: Skipping small ROI")
                        continue

                    roi = frame[int(y1):int(y2), int(x1):int(x2)]

                    # Skip invalid ROIs
                    if not self.validate_roi(roi, (x1,y1,x2,y2)):
                        logger.debug(f"Track {tid}: Invalid ROI")
                        continue

                    try:
                        plates = self.plate_detector.detect_plates(
                            roi,
                            conf=self.PLATE_DET_CONF,
                            iou=self.DET_IOU
                        )
                        logger.debug(f"Track {tid}: Found {len(plates)} plates")
                    except Exception as e:
                        logger.error(f"Plate detection failed for track {tid}: {str(e)}")
                        continue

                    for px1,py1,px2,py2,score in plates:
                        rx1,ry1 = x1+px1, y1+py1
                        rx2,ry2 = x1+px2, y1+py2

                        # Clamp plate coordinates
                        rx1, ry1, rx2, ry2 = self.clamp_bbox(rx1, ry1, rx2, ry2, width, height)
                        if (rx2 - rx1) < self.MIN_ROI_SIZE or (ry2 - ry1) < self.MIN_ROI_SIZE:
                            logger.debug(f"Track {tid}: Skipping small plate")
                            continue

                        plate_img = frame[int(ry1):int(ry2), int(rx1):int(rx2)]

                        # Skip invalid plate images
                        if not self.validate_roi(plate_img, (rx1,ry1,rx2,ry2)):
                            logger.debug(f"Track {tid}: Invalid plate image")
                            continue

                        try:
                            text, conf_txt = self.ocr.read_plate(plate_img)
                            logger.info(f"Track {tid}: OCR result: {text} (conf={conf_txt:.2f})")

                            # DEBUG: Show OCR result on debug frame
                            if debug_mode:
                                cv2.rectangle(debug_frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255,0,0), 2)
                                cv2.putText(debug_frame, f"{text} {conf_txt:.2f}",
                                            (int(rx1), int(max(ry1-10,0))),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                        except Exception as e:
                            logger.error(f"OCR failed for plate at {rx1},{ry1}-{rx2},{ry2}: {str(e)}")
                            continue

                        # Only track high-quality plate detections in history
                        if text and conf_txt >= 0.5:  # Minimum confidence threshold for history
                            # Lock in high-confidence reads first
                            if conf_txt >= self.LOCK_CONF_THRESHOLD and tid not in self.locked_plate:
                                self.locked_plate[tid] = text
                                self.plate_timestamps[tid][text] = datetime.now()
                                logger.info(f"Track {tid}: Locked plate: {text} (conf={conf_txt:.2f})")
                            
                            # Only update history if not already locked (prevents overwriting good detections)
                            elif tid not in self.locked_plate:
                                self.plate_history[tid][text] += 1
                                self.plate_timestamps[tid].setdefault(text, datetime.now())
                                logger.debug(f"Track {tid}: Added plate candidate: {text} (conf={conf_txt:.2f})")
                            
                            # If locked, just log but don't update history
                            elif tid in self.locked_plate:
                                logger.debug(f"Track {tid}: Ignoring detection {text} - already locked to {self.locked_plate[tid]}")

                        # Decide final text and styling
                        if tid in self.locked_plate:
                            final_text, fs, th = self.locked_plate[tid], 1.0, 3
                        else:
                            hist = self.plate_history[tid]
                            if hist:
                                final_text = max(hist.items(), key=lambda x: x[1])[0]
                            else:
                                final_text = text
                            fs, th = 0.6, 2

                        # Draw plate box and label
                        cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255,0,0), 2)
                        cv2.putText(frame,
                                    f"{final_text} ({conf_txt:.2f})",
                                    (int(rx1), int(max(ry1-10,0))),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    fs,
                                    (255,255,255),
                                    th)

                # Draw car tracking boxes on output frame
                for tr in tracks:
                    bx1,by1,bx2,by2 = tr['bbox']
                    tid = tr['track_id']
                    cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0,255,0), 2)
                    cv2.putText(frame,
                                f"Car {tid}",
                                (int(bx1), int(max(by1-10,0))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0,255,0),
                                2)

                # Show debug frame
                if debug_mode:
                    try:
                        cv2.imshow(debug_window_name, debug_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("User interrupted processing")
                            break
                    except cv2.error as e:
                        logger.warning(f"Debug display failed: {e}")
                        debug_mode = False

                writer.write(frame)
                
                # Memory cleanup every 50 frames to prevent memory leaks
                if frame_idx % 50 == 0:
                    if debug_mode and 'debug_frame' in locals():
                        del debug_frame
                    # Force garbage collection periodically
                    import gc
                    gc.collect()
        except Exception as e:
            logger.exception("Video processing failed")
            processing_error = True
            error_details = str(e)
        finally:
            cap.release()
            writer.release()
            if debug_mode:
                try:
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    logger.warning(f"Could not destroy debug windows: {e}")
            
            # GPU memory cleanup if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass  # PyTorch not available
            
            # Final memory cleanup
            import gc
            gc.collect()
            logger.info("Released video resources and cleaned up memory")

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
            logger.error(f"Video processing failed: {error_details}")
            return None, {"error": "Video processing failed", "details": error_details}

        # Debug logging before building summary
        logger.info(f"=== SUMMARY DEBUG ===")
        logger.info(f"locked_plate: {self.locked_plate}")
        logger.info(f"plate_history keys: {list(self.plate_history.keys())}")
        for tid, hist in self.plate_history.items():
            logger.info(f"Track {tid} history: {dict(hist)}")
        logger.info(f"plate_timestamps keys: {list(self.plate_timestamps.keys())}")

        # Summarize OCR results - prioritize locked plates, fallback to best candidates
        rows, seen = [], set()
        
        # First pass: Collect all locked plates (highest confidence)
        for tid in self.locked_plate:
            plate = self.locked_plate[tid]
            ts = self.plate_timestamps[tid].get(plate, datetime.now())
            if plate and plate not in seen:
                seen.add(plate)
                # Get count from history if available, otherwise default to 1
                count = self.plate_history[tid].get(plate, 1) if tid in self.plate_history else 1
                rows.append({
                    "Plate Number": plate,
                    "Detected At": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "OCR Count": count,
                    "Source": "locked"
                })
                logger.info(f"Added locked plate to summary: {plate}")
        
        # Second pass: Add best candidates from unlocked tracks
        for tid, hist in self.plate_history.items():
            if tid not in self.locked_plate and hist:
                # Get the most frequent plate for this track
                plate, count = max(hist.items(), key=lambda x: x[1])
                ts = self.plate_timestamps[tid].get(plate, datetime.now())
                if plate and plate not in seen and count >= 2:  # Require at least 2 detections
                    seen.add(plate)
                    rows.append({
                        "Plate Number": plate,
                        "Detected At": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "OCR Count": count,
                        "Source": "candidate"
                    })
                    logger.info(f"Added candidate plate to summary: {plate} (count={count})")
        
        logger.info(f"Summary generation complete: {len(rows)} plates, {len(self.locked_plate)} locked, {len(self.plate_history)} tracked")

        # Create CSV report only (removed Excel functionality)
        try:
            df = pd.DataFrame(rows)
            csv_file = OUTPUT_DIR / f"annotated_{base}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Created CSV report: {csv_file}")
        except Exception as e:
            logger.error(f"Failed to create CSV report: {str(e)}")
            csv_file = None

        # Extract plate numbers for summary
        plate_numbers = [r["Plate Number"] for r in rows]
        locked_plates = [r["Plate Number"] for r in rows if r.get("Source") == "locked"]
        candidate_plates = [r["Plate Number"] for r in rows if r.get("Source") == "candidate"]
        
        summary = {
            "plates_detected": plate_numbers,
            "recognized_plates": plate_numbers,  # alias for compatibility
            "detected_plates": plate_numbers,  # alias for consistency
            "locked_plates": locked_plates,  # high-confidence detections
            "candidate_plates": candidate_plates,  # lower-confidence but frequent detections
            "plate_detection_times": {r["Plate Number"]: r["Detected At"] for r in rows},
            "plate_count": len(rows),
            "vehicle_count": self.tracker.get_total_unique_ids(),
            "total_frames": frame_idx,
            "processing_fps": frame_idx / (time.time() - start_time) if (time.time() - start_time) > 0 else 0,
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "output_video": str(annotated_video),
            "csv_file": str(csv_file) if csv_file else None,
            "detection_stats": {
                "locked_count": len(self.locked_plate),
                "tracked_vehicles": len(self.plate_history),
                "total_unique_vehicles": self.tracker.get_total_unique_ids()
            }
        }
        return str(annotated_video), summary

    def process_image(self, image_path: str, output_path: str = None):
        """Process an image file with output saved to output directory"""
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Input image not found: {image_path}")
            return None, {"error": "Input image not found", "path": str(image_path)}

        base = image_path.stem
        # Use provided output_path or create default
        if output_path is None:
            annotated_filename = f"annotated_{base}.jpg"
            out_file = OUTPUT_DIR / annotated_filename
        else:
            out_file = Path(output_path)

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
                error_msg = f"Could not read image: {image_path}"
                logger.error(error_msg)
                return None, {"error": error_msg}
        except Exception as e:
            error_msg = f"Image read failed: {str(e)}"
            logger.error(error_msg)
            return None, {"error": error_msg}

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
            success = cv2.imwrite(str(out_file), frame)
            if not success or not out_file.exists():
                error_msg = f"Failed to save output image: {out_file}"
                logger.error(error_msg)
                return None, {"error": error_msg}
            logger.info(f"Saved annotated image: {out_file}")
        except Exception as e:
            error_msg = f"Image save failed: {str(e)}"
            logger.error(error_msg)
            return None, {"error": error_msg}

        return str(out_file), results

class ParkingProcessor:
    """
    Enhanced real-time parking management processor with:
    - Zone-based detection
    - Real-time event broadcasting
    - State management for occupancy counts
    - Optimized for local development
    - Video file processing capability
    """
    MIN_ROI_SIZE = 10  # Minimum pixel size for ROI width/height

    def __init__(self, plate_model_path: str, car_model_path: str, total_slots: int = 50):
        # The model paths may come from model manager with fallback, so they should already be valid
        # But let's still verify them as a safety check
        if not Path(plate_model_path).exists():
            logger.error(f"Plate model not found: {plate_model_path}")
            raise FileNotFoundError(f"Plate model not found: {plate_model_path}")
        if not Path(car_model_path).exists():
            logger.error(f"Car model not found: {car_model_path}")
            raise FileNotFoundError(f"Car model not found: {car_model_path}")

        logger.info(f"✅ Initializing parking processor with plate model: {plate_model_path}")
        logger.info(f"✅ Initializing parking processor with car model: {car_model_path}")

        # Initialize detectors
        self.plate_detector = LicensePlateDetector(plate_model_path)
        self.car_detector = LicensePlateDetector(car_model_path)
        self.ocr = PlateOCR()
        self.total_slots = total_slots
        # Camera configuration
        self.cameras = {"entry": None, "exit": None}
        # Zone configuration
        self.zones = defaultdict(list)  # camera_id -> list of zones
        # Parking state
        self.running = False
        self.track_history = defaultdict(list)
        # Event management and state tracking
        self.event_manager = None
        self.parking_state = None
        # Collections to be set later
        self.parking_events_collection = None
        self.parking_slots_collection = None
        self.zone_configs_collection = None
        # Thread locks
        self.lock = threading.Lock()
        logger.info("Parking Processor initialized")

    def clamp_bbox(self, x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int) -> tuple:
        """Ensure bounding box stays within frame boundaries"""
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width - 1, x2)
        y2 = min(frame_height - 1, y2)
        return x1, y1, x2, y2

    def validate_roi(self, roi: np.ndarray, bbox: tuple) -> bool:
        """Validate ROI meets minimum requirements"""
        if roi is None or roi.size == 0:
            return False
        if roi.shape[0] < self.MIN_ROI_SIZE or roi.shape[1] < self.MIN_ROI_SIZE:
            return False
        return True

    def process_video(self, video_path: str, output_path: str = None, progress_callback: callable = None) -> tuple:
        """
        Process a video file for parking analysis
        Returns: (output_video_path, summary_dict)
        """
        from .tracker import VehicleTracker
        import cv2
        import os
        # Create temporary tracker
        tracker = VehicleTracker()
        # Prepare output path
        video_path = Path(video_path)
        if output_path is None:
            output_path = OUTPUT_DIR / f"parking_analysis_{int(time.time())}_{video_path.name}"
        else:
            output_path = Path(output_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Parking analysis: {width}x{height} @ {fps:.1f}FPS, {total_frames} frames")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        frame_idx = 0
        plate_history = {}  # plate -> last_seen_frame (for occupancy tracking)
        all_detected_plates = set()  # all unique plates ever detected (for vehicle count)
        confirmed_entries = set()  # plates that have confirmed entry (prevents duplicate entries)
        confirmed_exits = set()  # plates that have confirmed exit
        entry_frames = {}  # plate -> frame when first detected (for entry validation)
        exit_threshold = 30  # frames absent before considering exit
        
        summary = {
            'entries': 0,
            'exits': 0,
            'max_occupancy': 0,
            'recognized_plates': [],
            'processing_fps': 0,
            'total_frames': total_frames,
            'processing_time': 0
        }
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if progress_callback and frame_idx % 10 == 0:
                progress = frame_idx / total_frames
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed
                progress_callback(progress, f"Frame {frame_idx}/{total_frames} | FPS: {current_fps:.1f}")
            
            # Detect vehicles
            car_dets = self.car_detector.detect_plates(frame, classes=[2])
            tracker_input = []
            for (x1, y1, x2, y2, conf) in car_dets:
                w = x2 - x1
                h = y2 - y1
                if w > 0 and h > 0:
                    tracker_input.append(([x1, y1, w, h], conf, 0))
            
            tracks = tracker.update(tracker_input, frame)
            current_plates = set()
            
            # Process each tracked vehicle
            for tr in tracks:
                tid = tr['track_id']
                x1, y1, x2, y2 = tr['bbox']
                x1, y1, x2, y2 = self.clamp_bbox(x1, y1, x2, y2, width, height)
                if (x2 - x1) < self.MIN_ROI_SIZE or (y2 - y1) < self.MIN_ROI_SIZE:
                    continue
                    
                vehicle_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                if not self.validate_roi(vehicle_roi, (x1, y1, x2, y2)):
                    continue
                    
                plates = self.plate_detector.detect_plates(vehicle_roi)
                if not plates:
                    continue
                    
                best_plate = max(plates, key=lambda x: x[4])
                px1, py1, px2, py2, _ = best_plate
                plate_img = vehicle_roi[int(py1):int(py2), int(px1):int(px2)]
                if not self.validate_roi(plate_img, (px1, py1, px2, py2)):
                    continue
                    
                plate_text, conf = self.ocr.read_plate(plate_img)
                if plate_text and conf > 0.7:
                    current_plates.add(plate_text)
                    all_detected_plates.add(plate_text)  # Track all unique plates
                    
                    # Handle new vehicle entry (only count true first entries)
                    if plate_text not in plate_history:
                        # First time seeing this plate
                        plate_history[plate_text] = frame_idx
                        entry_frames[plate_text] = frame_idx
                        
                        # Count as entry only if not previously exited and re-entered quickly
                        if plate_text not in confirmed_entries:
                            confirmed_entries.add(plate_text)
                            summary['entries'] += 1
                            summary['recognized_plates'].append(plate_text)
                            cv2.putText(frame, "ENTRY", (int(x1), int(y1-30)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                            logger.info(f"Entry detected: {plate_text}")
                    else:
                        # Update last seen frame for existing plate
                        plate_history[plate_text] = frame_idx
            
            # Check for exits (vehicles absent for more than threshold frames)
            exited_plates = []
            for plate, last_seen in list(plate_history.items()):
                if plate not in current_plates and (frame_idx - last_seen) > exit_threshold:
                    # Only count as exit if we haven't already counted this exit
                    if plate not in confirmed_exits:
                        confirmed_exits.add(plate)
                        summary['exits'] += 1
                        logger.info(f"Exit detected: {plate}")
                        
                        # Add exit annotation on frame if possible
                        cv2.putText(frame, f"EXIT: {plate}", (10, 110 + len(exited_plates) * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    
                    exited_plates.append(plate)
            
            # Remove exited plates from current occupancy tracking
            for plate in exited_plates:
                plate_history.pop(plate, None)
            
            # Update occupancy stats
            current_occupancy = len(plate_history)
            summary['max_occupancy'] = max(summary['max_occupancy'], current_occupancy)
            
            # Draw UI elements
            cv2.putText(frame, f"Occupancy: {current_occupancy}/{self.total_slots}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Entries: {summary['entries']} | Exits: {summary['exits']}",
                        (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        elapsed = time.time() - start_time
        summary['processing_fps'] = frame_idx / elapsed if elapsed > 0 else 0
        summary['processing_time'] = elapsed
        summary['final_occupancy'] = len(plate_history)
        summary['vehicle_count'] = len(all_detected_plates)  # Fixed: total unique vehicles detected
        
        # Add ANPR-style detection statistics for consistency with car-count jobs
        high_confidence_plates = []  # In parking analysis, all recognized plates are considered high-confidence
        candidate_plates = []
        
        # Since parking analysis uses confidence threshold of 0.7, treat all recognized plates as "locked"
        high_confidence_plates = list(all_detected_plates)
        
        # Add detection statistics
        summary['locked_plates'] = high_confidence_plates
        summary['candidate_plates'] = candidate_plates  # Empty for parking analysis
        summary['detection_stats'] = {
            'locked_count': len(high_confidence_plates),
            'tracked_vehicles': len(plate_history),  # Current occupancy 
            'total_unique_vehicles': len(all_detected_plates)  # Total unique vehicles detected
        }
        
        logger.info(f"Parking analysis complete: {summary}")
        return str(output_path), summary
