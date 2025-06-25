import os
import shutil
import asyncio
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from threading import Thread

from fastapi import (
    APIRouter, File, UploadFile, HTTPException, Query,
    BackgroundTasks, WebSocket, WebSocketDisconnect, Form
)
from fastapi.responses import JSONResponse, FileResponse
from pymongo import MongoClient
from Levenshtein import distance as levenshtein_distance
from loguru import logger

from anpr.processor import ANPRProcessor

# Create router
router = APIRouter()

# === Resolve absolute model paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
plate_model_path = BASE_DIR / "models" / "best.pt"
car_model_path   = BASE_DIR / "models" / "yolo11m.pt"

# Initialize processor
processor = ANPRProcessor(
    plate_model_path=str(plate_model_path),
    car_model_path=str(car_model_path)
)

# MongoDB client for live logging & final plates
mongo_client           = MongoClient(os.getenv("mongodb+srv://PDkPsssBV2iZkEU5:Abhi%401801@aionos.jkvscg5.mongodb.net/anpr?retryWrites=true&w=majority&appName=Aionos", "mongodb://localhost:27017/anpr"))
db                     = mongo_client["anpr"]
live_collection        = db["live_ocr_stream"]
final_plate_collection = db["final_plate_output"]
parking_log            = db["parking_log"]  # for parking entry/exit events

# WebSocket state and parking trackers
websocket_clients = set()
track_plate_history = defaultdict(list)
track_best_crop     = {}
track_centroids     = {}
car_entries         = set()
car_exits           = set()
entry_line_y        = 200
exit_line_y         = 400

# === Video upload endpoint ===
@router.post("/video")
async def upload_video(video: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        input_dir = BASE_DIR / "data" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{datetime.now():%Y%m%d%H%M%S}_{video.filename}"
        input_path = input_dir / filename
        with open(input_path, "wb") as buf:
            shutil.copyfileobj(video.file, buf)
        background_tasks.add_task(processor.process_video, str(input_path))
        return {"message": "Processing started", "filename": filename}
    except Exception as e:
        logger.exception("Video upload failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Analyze existing video (synchronous) ===
@router.post("/analyze")
def analyze_video(file_path: str = Query(..., description="Local video file path")):
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        output_filename, summary = processor.process_video(file_path)
        return {
            "message": "Analysis complete",
            "file": output_filename,
            "summary": summary
        }
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Preview processed video ===
@router.get("/preview/{filename}")
async def preview_video(filename: str):
    video_path = BASE_DIR / "data" / "output" / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path=str(video_path), filename=filename, media_type="video/mp4")

# === Download results (video, CSV, Excel) ===
@router.get("/download/{filename}")
async def download_file(filename: str):
    output_dir = BASE_DIR / "data" / "output"
    candidate = output_dir / filename
    # try common extensions
    if not candidate.exists() and Path(filename).suffix == "":
        for ext in [".mp4", ".csv", ".xlsx"]:
            alt = output_dir / f"{filename}{ext}"
            if alt.exists():
                candidate = alt
                break
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    media_type = {
        ".mp4": "video/mp4",
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }.get(candidate.suffix.lower(), "application/octet-stream")
    return FileResponse(path=str(candidate), filename=candidate.name, media_type=media_type)

# === WebSocket endpoint for live updates ===
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

# === Fetch detection history ===
@router.get("/history")
def get_detection_history():
    try:
        records = list(final_plate_collection.find({}, {"_id": 0}))
        for r in records:
            img_path = r.get("crop_image")
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    r["image_data"] = f.read().hex()
        return JSONResponse(content={"history": records})
    except Exception as e:
        logger.exception("Failed to fetch detection history")
        raise HTTPException(status_code=500, detail=str(e))

# === Parking control endpoints ===
@router.post("/start_camera")
def start_camera():
    Thread(target=_camera_loop, daemon=True).start()
    return {"message": "Live camera started"}

@router.post("/set_lines")
def set_lines(entry_y: int = Form(...), exit_y: int = Form(...)):
    global entry_line_y, exit_line_y
    entry_line_y, exit_line_y = entry_y, exit_y
    return {"message": "Lines updated", "entry": entry_line_y, "exit": exit_line_y}

@router.get("/stats")
def get_parking_stats(slots: int = Query(None, description="Optional override total slots")):
    total = slots if slots is not None else 50
    occupied = len(car_entries) - len(car_exits)
    available = max(total - occupied, 0)
    return {"total_slots": total, "occupied": occupied, "available": available, "last_updated": datetime.now().isoformat()}

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
        file_path = BASE_DIR / "data" / "output" / f"parking_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
        df.to_csv(file_path, index=False)
        return FileResponse(str(file_path), media_type="text/csv", filename=file_path.name)
    except Exception as e:
        logger.exception("Log export failed")
        raise HTTPException(status_code=500, detail="Failed to export logs")

# Internal camera loop function
def _camera_loop():
    cap = cv2.VideoCapture(0)
    logger.info("[Camera] Live feed started")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # detection and parking logic omitted for brevity
    cap.release()
    cv2.destroyAllWindows()
    logger.info("[Camera] Live feed stopped")

