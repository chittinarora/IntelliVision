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

# Load environment variables from project .env explicitly
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

class ANPRProcessor:
    def __init__(self, plate_model_path: str, car_model_path: str):
        # Configure Cloudinary using environment variables
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True
        )

        # Initialize detectors, OCR, and tracker
        self.plate_detector = LicensePlateDetector(plate_model_path)
        self.car_detector = LicensePlateDetector(car_model_path)
        self.ocr = PlateOCR()
        self.tracker = VehicleTracker()

        # OCR history per detected plate bbox
        self.plate_history = defaultdict(lambda: defaultdict(int))
        self.plate_timestamps = {}

        # Retrieve MongoDB URI from env, require it
        raw_uri = os.getenv("MONGODB_URI")
        if not raw_uri:
            raise Exception("MONGODB_URI environment variable is not set")
        uri = raw_uri.strip().strip('"').strip("'")

        # Connect to MongoDB (Atlas or local)
        try:
            self.mongo_client = MongoClient(uri)
            # test connection
            self.mongo_client.admin.command('ping')
        except Exception as e:
            raise Exception(f"Could not connect to MongoDB: {e}")

        self.db = self.mongo_client[os.getenv("MONGODB_DB", "anpr")]
        self.collection = self.db["detections"]

    def is_valid_indian_plate(self, text: str) -> bool:
        cleaned = text.replace(" ", "").upper()
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', cleaned))

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        filename = os.path.basename(video_path)
        base, _ = os.path.splitext(filename)

        # Prepare output directory
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

            plates = self.plate_detector.detect_plates(frame)
            cars = self.car_detector.detect_plates(frame, classes=[2])
            car_dets = [(x1, y1, x2, y2, conf, 2) for x1, y1, x2, y2, conf in cars]
            tracks = self.tracker.update(car_dets, frame)

            # OCR on each plate bbox
            for x1, y1, x2, y2, _ in plates:
                crop = frame[y1:y2, x1:x2]
                text, score = self.ocr.read_plate(crop)
                key = f"{x1}_{y1}_{x2}_{y2}"
                # record every OCR result
                if text:
                    self.plate_history[key][text] += 1
                    self.plate_timestamps.setdefault(key, {})
                    if text not in self.plate_timestamps[key]:
                        self.plate_timestamps[key][text] = datetime.now()
                # Annotate frame
                label = f"{text} ({score:.2f})" if text else "Plate"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw car tracking boxes
            for tr in tracks:
                bx1, by1, bx2, by2 = tr['bbox']
                tid = tr['track_id']
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, f"Car {tid}", (bx1, max(by1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        rows = []
        for key, texts in self.plate_history.items():
            best_text, count = max(texts.items(), key=lambda x: x[1])
            first_time = self.plate_timestamps[key][best_text]
            rows.append({
                "Plate Number": best_text,
                "Detected At": first_time.strftime("%Y-%m-%d %H:%M:%S"),
                "OCR Count": count
            })
            self.collection.insert_one({
                "plate_number": best_text,
                "detected_at": first_time,
                "ocr_count": count,
                "video_file": filename
            })

        df = pd.DataFrame(rows)
        csv_name = f"annotated_{base}.csv"
        xlsx_name = f"annotated_{base}.xlsx"
        df.to_csv(output_dir / csv_name, index=False)
        df.to_excel(output_dir / xlsx_name, index=False)

        plates_list = df["Plate Number"].tolist() if "Plate Number" in df.columns else []
        times = dict(zip(df["Plate Number"], df["Detected At"])) if "Plate Number" in df.columns and "Detected At" in df.columns else {}

        summary = {
            "plates_detected": plates_list,
            "plate_detection_times": times,
            "plate_count": len(df),
            "vehicle_count": self.tracker.get_total_unique_ids(),
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "csv_file": csv_name,
            "xlsx_file": xlsx_name
        }

        return f"annotated_{filename}", summary

    def get_all_detections(self):
        try:
            return list(self.collection.find({}, {"_id": 0}))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

