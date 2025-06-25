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
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

class ANPRProcessor:
    # Detection & OCR thresholds
    CAR_DET_CONF = 0.6
    PLATE_DET_CONF = 0.6
    DET_IOU = 0.5
    LOCK_CONF_THRESHOLD = 0.75

    def __init__(self, plate_model_path: str, car_model_path: str):
        # Cloudinary config
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True
        )
        # Initialize components
        self.plate_detector = LicensePlateDetector(plate_model_path)
        self.car_detector = LicensePlateDetector(car_model_path)
        self.ocr = PlateOCR()
        self.tracker = VehicleTracker()

        # Smoothing state per track
        self.plate_history = defaultdict(lambda: defaultdict(int))
        self.plate_timestamps = {}
        self.locked_plate = {}

        # MongoDB connection
        raw_uri = os.getenv("MONGODB_URI")
        if not raw_uri:
            raise Exception("MONGODB_URI is not set")
        raw_uri = raw_uri.strip().strip('"').strip("'")
        try:
            self.mongo_client = MongoClient(raw_uri)
            self.mongo_client.admin.command('ping')
        except Exception as e:
            raise Exception(f"MongoDB connection failed: {e}")
        db_name = os.getenv("MONGODB_DB", "anpr")
        self.db = self.mongo_client[db_name]
        self.collection = self.db["detections"]

        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def is_valid_plate(self, text: str) -> bool:
        s = re.sub(r"\s+", "", text.upper())
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', s))

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        filename = os.path.basename(video_path)
        base, _ = os.path.splitext(filename)

        # Output setup
        out_dir = Path("data/output")
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated = out_dir / f"annotated_{filename}"
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated), fourcc, fps, (width, height))

        self.tracker.reset()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Enhance contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cl = self.clahe.apply(gray)
            proc = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

            # Car detection
            raw_cars = self.car_detector.detect_plates(
                proc, conf=self.CAR_DET_CONF, iou=self.DET_IOU, classes=[2]
            )
            car_boxes = [(x1, y1, x2, y2, conf, 2) for x1, y1, x2, y2, conf in raw_cars]
            tracks = self.tracker.update(car_boxes, frame)

            # Plate detection per track
            for tr in tracks:
                tid = tr['track_id']
                x1, y1, x2, y2 = tr['bbox']

                # Skip invalid track bbox
                if x2 <= x1 or y2 <= y1:
                    continue

                car_crop = proc[y1:y2, x1:x2]

                plates = self.plate_detector.detect_plates(
                    car_crop, conf=self.PLATE_DET_CONF, iou=self.DET_IOU
                )
                for px1, py1, px2, py2, score in plates:
                    # Skip invalid plate bbox
                    if px2 <= px1 or py2 <= py1:
                        continue
                    rx1, ry1 = x1 + px1, y1 + py1
                    rx2, ry2 = x1 + px2, y1 + py2

                    # Denoise & binarize
                    plate_img = proc[ry1:ry2, rx1:rx2]
                    if plate_img.size == 0:
                        continue
                    blur = cv2.GaussianBlur(plate_img, (5,5), 0)
                    gray_p = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(
                        gray_p, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    text, conf_txt = self.ocr.read_plate(thresh)

                    # Lock if confident
                    if conf_txt >= self.LOCK_CONF_THRESHOLD and tid not in self.locked_plate:
                        self.locked_plate[tid] = text

                    # Update history if not locked
                    if text and tid not in self.locked_plate:
                        self.plate_history[tid][text] += 1
                        self.plate_timestamps.setdefault(tid, {})
                        if text not in self.plate_timestamps[tid]:
                            self.plate_timestamps[tid][text] = datetime.now()

                    # Choose final text
                    if tid in self.locked_plate:
                        final_text = self.locked_plate[tid]
                        font_scale, thickness = 1.2, 3
                    else:
                        hist = self.plate_history[tid]
                        final_text = max(hist.items(), key=lambda x: x[1])[0] if hist else text
                        font_scale, thickness = 0.6, 2

                    # Draw plate bbox & label
                    cv2.rectangle(
                        frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"{final_text} ({conf_txt:.2f})",
                        (rx1, max(ry1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness
                    )

            # Draw track boxes
            for tr in tracks:
                bx1, by1, bx2, by2 = tr['bbox']
                tid = tr['track_id']
                cv2.rectangle(
                    frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Car {tid}",
                    (bx1, max(by1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            writer.write(frame)

        cap.release()
        writer.release()

        # Compile summary
        rows = []
        seen = set()
        for tid, hist in self.plate_history.items():
            if tid in self.locked_plate:
                plate = self.locked_plate[tid]
                ts = self.plate_timestamps.get(tid, {}).get(plate, datetime.now())
            else:
                plate, _ = max(hist.items(), key=lambda x: x[1])
                ts = self.plate_timestamps[tid].get(plate, datetime.now())
            if plate and plate not in seen:
                seen.add(plate)
                count = self.plate_history[tid].get(plate, 1)
                rows.append({
                    "Plate Number": plate,
                    "Detected At": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "OCR Count": count
                })
                self.collection.insert_one({
                    "plate_number": plate,
                    "detected_at": ts,
                    "ocr_count": count,
                    "video_file": filename
                })

        df = pd.DataFrame(rows)
        csv_path = out_dir / f"annotated_{base}.csv"
        xlsx_path = out_dir / f"annotated_{base}.xlsx"
        df.to_csv(csv_path, index=False)
        df.to_excel(xlsx_path, index=False)

        plates_list = df.get("Plate Number", []).tolist()
        times = {r["Plate Number"]: r["Detected At"] for r in rows}
        summary = {
            "plates_detected": plates_list,
            "plate_detection_times": times,
            "plate_count": len(rows),
            "vehicle_count": self.tracker.get_total_unique_ids(),
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "csv_file": csv_path.name,
            "xlsx_file": xlsx_path.name
        }
        return annotated.name, summary

    def process_image(self, image_path: str):
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        fn = os.path.basename(image_path)
        base, _ = os.path.splitext(fn)
        out_dir = Path("data/output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"annotated_{base}.jpg"
        results = []

        plates = self.plate_detector.detect_plates(
            frame, conf=self.PLATE_DET_CONF, iou=self.DET_IOU
        )
        for x1, y1, x2, y2, conf_txt in plates:
            # Skip invalid
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            text, score = self.ocr.read_plate(crop)
            if score >= self.LOCK_CONF_THRESHOLD:
                font_scale, thickness = 1.2, 3
            else:
                font_scale, thickness = 0.6, 2
            label = f"{text} ({score:.2f})"
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
            results.append({"bbox": [x1, y1, x2, y2], "plate": text, "confidence": round(score, 2)})

        cv2.imwrite(str(out_file), frame)
        return out_file.name, results

    def get_all_detections(self):
        try:
            return list(self.collection.find({}, {"_id": 0}))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

