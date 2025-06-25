import os
import cv2
import pandas as pd
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from fastapi import HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

# External model and OCR imports
from anpr.detector import LicensePlateDetector
from anpr.tracker import VehicleTracker
from anpr.ocr import PlateOCR

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

class ANPRProcessor:
    # Lock plate text once OCR confidence reaches this threshold
    CONFIDENCE_LOCK_THRESHOLD = 0.70

    def __init__(self, plate_model_path: str, car_model_path: str):
        # Configure Cloudinary
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True
        )
        # Initialize detectors and OCR
        self.plate_detector = LicensePlateDetector(plate_model_path)
        self.car_detector = LicensePlateDetector(car_model_path)
        self.ocr = PlateOCR()
        self.tracker = VehicleTracker()

        # Data structures for smoothing and locking
        self.plate_history = defaultdict(lambda: defaultdict(int))  # {bbox_key: {text: count}}
        self.plate_timestamps = {}  # {bbox_key: {text: first_detected_datetime}}
        self.locked_plate = {}      # {bbox_key: locked_text}

        # MongoDB connection
        raw_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/anpr").strip().strip('"').strip("'")
        try:
            self.mongo_client = MongoClient(raw_uri)
            self.mongo_client.admin.command("ping")
        except Exception as e:
            raise Exception(f"Could not connect to MongoDB: {e}")
        db_name = os.getenv("MONGODB_DB", "anpr")
        self.db = self.mongo_client[db_name]
        self.collection = self.db["detections"]

    def is_valid_indian_plate(self, text: str) -> bool:
        cleaned = text.replace(" ", "").upper()
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', cleaned))

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        filename = os.path.basename(video_path)
        base, _ = os.path.splitext(filename)

        # Prepare output paths
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_video = output_dir / f"annotated_{filename}"

        # Setup video writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(annotated_video), fourcc, fps, (width, height))

        self.tracker.reset()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect plates and cars
            plates = self.plate_detector.detect_plates(frame)
            cars = self.car_detector.detect_plates(frame, classes=[2])
            car_dets = [(x1, y1, x2, y2, conf, 2) for x1, y1, x2, y2, conf in cars]
            tracks = self.tracker.update(car_dets, frame)

            # OCR smoothing and locking
            for x1, y1, x2, y2, _ in plates:
                crop = frame[y1:y2, x1:x2]
                text, score = self.ocr.read_plate(crop)
                key = f"{x1}_{y1}_{x2}_{y2}"

                # Lock plate if confidence threshold met
                if score >= self.CONFIDENCE_LOCK_THRESHOLD and key not in self.locked_plate:
                    self.locked_plate[key] = text

                # Only update history if not locked
                if text and key not in self.locked_plate:
                    self.plate_history[key][text] += 1
                    self.plate_timestamps.setdefault(key, {})
                    if text not in self.plate_timestamps[key]:
                        self.plate_timestamps[key][text] = datetime.now()

                # Determine which text to display
                if key in self.locked_plate:
                    display_text = self.locked_plate[key]
                    font_scale = 1.2
                    thickness = 3
                else:
                    if self.plate_history[key]:
                        display_text = max(self.plate_history[key].items(), key=lambda x: x[1])[0]
                    else:
                        display_text = text or ""
                    font_scale = 0.6
                    thickness = 2

                label = f"{display_text} ({score:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )

            # Draw car tracks
            for tr in tracks:
                bx1, by1, bx2, by2 = tr['bbox']
                tid = tr['track_id']
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Car {tid}",
                    (bx1, max(by1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            out.write(frame)
        cap.release()
        out.release()

        # Compile final unique plate results
        rows = []
        seen_texts = set()
        keys = list(self.locked_plate.keys()) + [k for k, hist in self.plate_history.items() if hist]
        for key in keys:
            if key in self.locked_plate:
                txt = self.locked_plate[key]
                ts = datetime.now()
            else:
                hist = self.plate_history[key]
                if not hist:
                    continue
                txt = max(hist.items(), key=lambda x: x[1])[0]
                ts = self.plate_timestamps.get(key, {}).get(txt, datetime.now())

            if txt in seen_texts or not txt:
                continue
            seen_texts.add(txt)
            count = self.plate_history[key].get(txt, 1)
            rows.append({
                "Plate Number": txt,
                "Detected At": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "OCR Count": count
            })
            # Save to Mongo
            self.collection.insert_one({
                "plate_number": txt,
                "detected_at": ts,
                "ocr_count": count,
                "video_file": filename
            })

        df = pd.DataFrame(rows)
        csv_file = output_dir / f"annotated_{base}.csv"
        xlsx_file = output_dir / f"annotated_{base}.xlsx"
        df.to_csv(csv_file, index=False)
        df.to_excel(xlsx_file, index=False)

        plates_list = df["Plate Number"].tolist() if not df.empty else []
        times = dict(zip(df["Plate Number"], df["Detected At"]))
        summary = {
            "plates_detected": plates_list,
            "plate_detection_times": times,
            "plate_count": len(df),
            "vehicle_count": self.tracker.get_total_unique_ids(),
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "csv_file": csv_file.name,
            "xlsx_file": xlsx_file.name
        }
        return f"annotated_{filename}", summary

    def get_all_detections(self):
        try:
            return list(self.collection.find({}, {"_id": 0}))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

