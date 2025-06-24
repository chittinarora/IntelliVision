from fastapi import APIRouter, File, UploadFile, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect, Form, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import os
import traceback
import asyncio
import cv2
import pandas as pd
from datetime import datetime
from collections import defaultdict
from anpr.processor import ANPRProcessor
from loguru import logger
from pymongo import MongoClient
from Levenshtein import distance as levenshtein_distance
from threading import Thread

# Create router
router = APIRouter()

# === Resolve absolute model paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
plate_model_path = BASE_DIR / "models" / "best.pt"
car_model_path = BASE_DIR / "models" / "yolo11m.pt"

# Initialize processor
processor = ANPRProcessor(
    plate_model_path=str(plate_model_path),
    car_model_path=str(car_model_path)
)

# MongoDB client for live logging
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["anpr"]
live_collection = db["live_ocr_stream"]
final_plate_collection = db["final_plate_output"]
parking_log = db["parking_log"]

# Store WebSocket connections and track history
websocket_clients = set()
track_plate_history = defaultdict(list)
track_best_crop = {}
track_centroids = {}
car_entries = set()
car_exits = set()
entry_line_y = 200
exit_line_y = 400

# === Video upload endpoint ===
@router.post("/video")
async def upload_video(video: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        input_dir = BASE_DIR / "data" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video.filename}"
        input_path = input_dir / filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        background_tasks.add_task(processor.process_video, str(input_path))
        return {"message": "Processing started", "filename": filename}
    except Exception as e:
        logger.exception("Video upload failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Analyze existing video ===
@router.post("/analyze")
def analyze_video(file_path: str = Query(...)):
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        processor.process_video(file_path)
        return {"message": "Analysis complete", "file": file_path}
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Preview processed video ===
@router.get("/preview/{filename}")
async def preview_video(filename: str):
    video_path = BASE_DIR / "data" / "output" / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path=video_path, filename=filename, media_type="video/mp4")

# === Download results ===
@router.get("/download/{filename}")
async def download_file(filename: str):
    output_path = BASE_DIR / "data" / "output" / filename
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=output_path, filename=filename, media_type="application/octet-stream")

# === WebSocket endpoint ===
@router.websocket("/ws")
async def websocket_connection(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)
    try:
        await websocket.send_text("WebSocket connected ✅")
        while True:
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        websocket_clients.discard(websocket)
        logger.info("WebSocket disconnected")

# === Fetch detection history with image ===
@router.get("/history")
def get_detection_history():
    try:
        records = list(final_plate_collection.find({}, {"_id": 0}))
        for r in records:
            if r.get("crop_image") and os.path.exists(r["crop_image"]):
                with open(r["crop_image"], "rb") as img_file:
                    r["image_data"] = img_file.read().hex()
        return JSONResponse(content={"history": records})
    except Exception as e:
        logger.exception("Failed to fetch detection history")
        raise HTTPException(status_code=500, detail=f"History fetch error: {str(e)}")

# === Live parking logic ===
def start_camera_loop():
    cap = cv2.VideoCapture(0)
    logger.info("[Camera] Live feed started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        car_detections = processor.car_detector.detect_plates(frame)
        car_detections = [(x1, y1, x2, y2, conf, 2) for (x1, y1, x2, y2, conf) in car_detections]
        tracks = processor.tracker.update(car_detections, frame)

        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            cy = (y1 + y2) // 2

            if track_id not in track_centroids:
                track_centroids[track_id] = []
            track_centroids[track_id].append(cy)

            if len(track_centroids[track_id]) >= 2:
                prev_y = track_centroids[track_id][-2]
                curr_y = track_centroids[track_id][-1]
                if prev_y < entry_line_y <= curr_y and track_id not in car_entries:
                    car_entries.add(track_id)
                    logger.info(f"Vehicle {track_id} entered")
                    parking_log.insert_one({"track_id": track_id, "event": "entry", "timestamp": datetime.now().isoformat()})
                elif prev_y > exit_line_y >= curr_y and track_id not in car_exits:
                    car_exits.add(track_id)
                    logger.info(f"Vehicle {track_id} exited")
                    parking_log.insert_one({"track_id": track_id, "event": "exit", "timestamp": datetime.now().isoformat()})

        cv2.line(frame, (0, entry_line_y), (frame.shape[1], entry_line_y), (255, 0, 0), 2)
        cv2.line(frame, (0, exit_line_y), (frame.shape[1], exit_line_y), (0, 0, 255), 2)
        cv2.imshow("Live Parking Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("[Camera] Live feed stopped")

@router.post("/start_camera")
def start_camera():
    try:
        Thread(target=start_camera_loop, daemon=True).start()
        return {"message": "Live camera started"}
    except Exception as e:
        logger.exception("Failed to start camera")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/set_lines")
def set_lines(entry_y: int = Form(...), exit_y: int = Form(...)):
    global entry_line_y, exit_line_y
    entry_line_y = entry_y
    exit_line_y = exit_y
    return {"message": "Lines updated", "entry": entry_line_y, "exit": exit_line_y}

@router.get("/stats")
def get_parking_stats(slots: int = Query(None, description="Optional override for total parking slots")):
    """
    Returns parking occupancy using:
    - ?slots=... from user input if provided
    - MongoDB configured slots if set
    - Defaults to 50 if neither is set
    """
    def get_configured_slots():
        config = settings_collection.find_one({"key": "total_slots"})
        return config["value"] if config else 50

    total_slots = slots if slots is not None else get_configured_slots()
    occupied = len(car_entries) - len(car_exits)
    available = max(total_slots - occupied, 0)
    
    return {
        "total_slots": total_slots,
        "occupied": occupied,
        "available": available,
        "last_updated": datetime.now().isoformat()
    }

@router.get("/dashboard")
def get_parking_dashboard():
    try:
        logs = list(parking_log.find({}, {"_id": 0}))
        return JSONResponse(content={"dashboard": logs})
    except Exception as e:
        logger.exception("Dashboard fetch error")
        raise HTTPException(status_code=500, detail="Dashboard error")

@router.get("/export_logs")
def export_logs():
    try:
        logs = list(parking_log.find({}, {"_id": 0}))
        df = pd.DataFrame(logs)
        file_path = BASE_DIR / "data" / "output" / f"parking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_path, index=False)
        return FileResponse(str(file_path), media_type="text/csv", filename=file_path.name)
    except Exception as e:
        logger.exception("Log export failed")
        raise HTTPException(status_code=500, detail="Failed to export logs")

