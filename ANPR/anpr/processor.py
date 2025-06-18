import os
import cv2
import pandas as pd
import re
from datetime import datetime
from collections import defaultdict
from pymongo import MongoClient
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from anpr.detector import LicensePlateDetector
from anpr.tracker import VehicleTracker
from anpr.ocr import PlateOCR

class ANPRProcessor:
    def __init__(self, plate_model_path, car_model_path):
        self.plate_detector = LicensePlateDetector(plate_model_path)
        self.car_detector = LicensePlateDetector(car_model_path)
        self.ocr = PlateOCR()
        self.tracker = VehicleTracker()

        # Track OCR history and timestamps
        self.plate_history = defaultdict(lambda: defaultdict(int))   # {box_key: {text: count}}
        self.plate_timestamps = {}                                   # {box_key: {text: timestamp}}

        # Initialize MongoDB
        self.mongo_client = MongoClient("mongodb://localhost:27017")
        self.db = self.mongo_client["anpr"]
        self.collection = self.db["detections"]

    def is_valid_indian_plate(self, text: str) -> bool:
        cleaned = text.replace(" ", "").upper()
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', cleaned))

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        filename = os.path.basename(video_path)
        name_no_ext = os.path.splitext(filename)[0]

        output_path = os.path.join("data/output", f"annotated_{filename}")
        os.makedirs("data/output", exist_ok=True)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.tracker.reset()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            plate_detections = self.plate_detector.detect_plates(frame)

            # PATCH: Add dummy class_id (e.g., 2 for car) to each car detection tuple
            raw_car_detections = self.car_detector.detect_plates(frame)
            car_detections = [
                (x1, y1, x2, y2, conf, 2)
                for (x1, y1, x2, y2, conf) in raw_car_detections
            ]

            tracks = self.tracker.update(car_detections, frame)

            for (x1, y1, x2, y2, conf) in plate_detections:
                plate_img = frame[y1:y2, x1:x2]
                text, text_conf = self.ocr.read_plate(plate_img)
                box_key = f"{x1}_{y1}_{x2}_{y2}"

                if text and self.is_valid_indian_plate(text):
                    self.plate_history[box_key][text] += 1
                    if box_key not in self.plate_timestamps:
                        self.plate_timestamps[box_key] = {}
                    if text not in self.plate_timestamps[box_key]:
                        self.plate_timestamps[box_key][text] = datetime.now()

                label = f"{text} ({text_conf:.2f})" if text else "Plate"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for track in tracks:
                x1, y1, x2, y2 = track['bbox']
                track_id = track['track_id']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Car {track_id}", (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        # Build final table with best OCR text + timestamp + count
        csv_rows = []
        for box_key, text_dict in self.plate_history.items():
            if not text_dict:
                continue
            best_text, count = max(text_dict.items(), key=lambda x: x[1])
            first_time = self.plate_timestamps[box_key][best_text]
            csv_rows.append({
                "Plate Number": best_text,
                "Detected At": first_time.strftime("%Y-%m-%d %H:%M:%S"),
                "OCR Count": count
            })

            # Log to MongoDB
            self.collection.insert_one({
                "plate_number": best_text,
                "detected_at": first_time,
                "ocr_count": count,
                "video_file": filename
            })

        df = pd.DataFrame(csv_rows)
        csv_name = f"annotated_{name_no_ext}.csv"
        xlsx_name = f"annotated_{name_no_ext}.xlsx"
        csv_path = os.path.join("data/output", csv_name)
        xlsx_path = os.path.join("data/output", xlsx_name)
        df.to_csv(csv_path, index=False)
        df.to_excel(xlsx_path, index=False)

        summary = {
            "plates_detected": list(df["Plate Number"]),
            "plate_detection_times": dict(zip(df["Plate Number"], df["Detected At"])),
            "plate_count": len(df),
            "vehicle_count": self.tracker.get_total_unique_ids(),
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "csv_file": csv_name,
            "xlsx_file": xlsx_name
        }

        return f"annotated_{filename}", summary

    def get_all_detections(self):
        try:
            records = list(self.collection.find({}, {"_id": 0}))
            return records
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MongoDB error: {str(e)}")

