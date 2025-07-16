import os
import shutil
import asyncio
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from threading import Thread, Lock
from bson import ObjectId
import zipfile
import numpy as np

from fastapi import (
    APIRouter, File, UploadFile, HTTPException, Query,
    BackgroundTasks, WebSocket, WebSocketDisconnect, Form, Request
)
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pymongo import MongoClient
from loguru import logger
from pydantic import BaseModel
from django.conf import settings

# Create router
router = APIRouter()

# === Resolve paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
plate_model_path = BASE_DIR / "models" / "best_car.pt"
car_model_path   = BASE_DIR / "models" / "yolo11m_car.pt"

# Define canonical output directory for all outputs
OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"All output will be saved to: {OUTPUT_DIR}")

# Initialize processors
from .processor import ANPRProcessor, ParkingProcessor

# For video/image processing
anpr_processor = ANPRProcessor(str(plate_model_path), str(car_model_path))

# For parking system (50 total slots)
parking_processor = ParkingProcessor(
    str(plate_model_path),
    str(car_model_path),
    total_slots=50
)

# ====================== Event Management System ======================
class EventManager:
    def __init__(self):
        self.connections = defaultdict(list)
        self.lock = Lock()

    def subscribe(self, websocket, event_types):
        with self.lock:
            for event_type in event_types:
                self.connections[event_type].append(websocket)

    def unsubscribe(self, websocket):
        with self.lock:
            for event_type in list(self.connections.keys()):
                if websocket in self.connections[event_type]:
                    self.connections[event_type].remove(websocket)

    async def broadcast(self, event_type, data):
        with self.lock:
            dead_connections = []
            for websocket in self.connections[event_type]:
                try:
                    await websocket.send_json({"type": event_type, "data": data})
                except Exception as e:
                    logger.warning(f"WebSocket error: {str(e)}")
                    dead_connections.append(websocket)

            # Remove dead connections
            for ws in dead_connections:
                self.unsubscribe(ws)

event_manager = EventManager()

# ====================== Zone Configuration Models ======================
class ZonePoint(BaseModel):
    x: float
    y: float

class ZoneConfig(BaseModel):
    name: str  # "entry" or "exit"
    points: list[ZonePoint]  # List of points defining the polygon

# ====================== State Management ======================
class ParkingState:
    def __init__(self):
        self.counts = defaultdict(int)
        self.last_events = {}
        self.lock = Lock()

    def update_count(self, camera_id, event_type):
        with self.lock:
            if "entry" in event_type:
                self.counts[camera_id] += 1
            elif "exit" in event_type:
                self.counts[camera_id] = max(0, self.counts[camera_id] - 1)

            # Update last event
            self.last_events[camera_id] = {
                "type": event_type,
                "timestamp": datetime.utcnow().isoformat()
            }

    def get_counts(self):
        with self.lock:
            return dict(self.counts)

    def get_last_events(self):
        with self.lock:
            return dict(self.last_events)

parking_state = ParkingState()

# ====================== Database Access Helpers ======================
def get_db(request: Request):
    """Get database instance from app state"""
    return request.app.state.mongodb_db

def get_jobs_collection(request: Request):
    """Get jobs collection from database"""
    db = get_db(request)
    return db["processing_jobs"]

def get_detections_collection(request: Request):
    """Get detections collection"""
    db = get_db(request)
    return db["detections"]

def get_parking_events_collection(request: Request):
    """Get parking events collection"""
    db = get_db(request)
    return db["parking_events"]

def get_parking_slots_collection(request: Request):
    """Get parking slots collection"""
    db = get_db(request)
    return db["parking_slots"]

def get_zone_configs_collection(request: Request):
    """Get zone configurations collection"""
    db = get_db(request)
    return db["zone_configs"]

# ====================== Setup Input Directories ======================
(Path(BASE_DIR) / "data" / "input").mkdir(parents=True, exist_ok=True)
(Path(BASE_DIR) / "data" / "images_input").mkdir(parents=True, exist_ok=True)

def create_job_record(request: Request, filename: str, job_type: str):
    """Create a job record in MongoDB"""
    jobs_collection = get_jobs_collection(request)
    job = {
        "filename": filename,
        "type": job_type,
        "status": "queued",
        "created_at": datetime.now(),
        "output_path": None,
        "completed_at": None,
        "csv_path": None,
        "xlsx_path": None
    }
    result = jobs_collection.insert_one(job)
    return str(result.inserted_id)

def update_job_status(request: Request, job_id: str, status: str, output_path: str = None,
                     csv_path: str = None, xlsx_path: str = None):
    """Update job status in MongoDB"""
    jobs_collection = get_jobs_collection(request)
    update_data = {
        "status": status,
        "completed_at": datetime.now()
    }
    if output_path:
        update_data["output_path"] = output_path
    if csv_path:
        update_data["csv_path"] = csv_path
    if xlsx_path:
        update_data["xlsx_path"] = xlsx_path

    jobs_collection.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": update_data}
    )

# ====================== Video Streaming Endpoint ======================
@router.get("/stream/{camera_id}")
async def video_stream(camera_id: int, request: Request):
    """Live video stream from camera"""
    async def generate():
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            yield b""
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw configured zones
                zones = get_zones_for_camera(request, camera_id)
                for zone in zones:
                    points = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) for p in zone.points]
                    pts = np.array(points, np.int32).reshape((-1, 1, 2))
                    color = (0, 255, 0) if zone.name == "entry" else (0, 0, 255)
                    cv2.polylines(frame, [pts], True, color, 2)
                    cv2.putText(frame, zone.name, (points[0][0], points[0][1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        finally:
            cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")

def get_zones_for_camera(request: Request, camera_id: int) -> list[ZoneConfig]:
    """Get configured zones for camera"""
    zone_configs = get_zone_configs_collection(request)
    config = zone_configs.find_one({"camera_id": camera_id})
    if config:
        return [ZoneConfig(**z) for z in config["zones"]]
    return []

# ====================== Zone Configuration Endpoints ======================
@router.post("/zones/{camera_id}")
async def configure_zones(camera_id: int, zones: list[ZoneConfig], request: Request):
    """Save zone configuration for camera"""
    zone_configs = get_zone_configs_collection(request)
    try:
        # Normalize points to relative coordinates
        for zone in zones:
            for point in zone.points:
                point.x = max(0.0, min(1.0, point.x))
                point.y = max(0.0, min(1.0, point.y))

        zone_configs.update_one(
            {"camera_id": camera_id},
            {"$set": {"zones": [z.dict() for z in zones]}},
            upsert=True
        )

        # Update processor with new zones
        if parking_processor:
            parking_processor.load_zones(camera_id)

        return {"status": "success", "message": f"Zones configured for camera {camera_id}"}
    except Exception as e:
        logger.error(f"Zone configuration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/zones/{camera_id}")
async def get_zones(camera_id: int, request: Request):
    """Get configured zones for camera"""
    zone_configs = get_zone_configs_collection(request)
    try:
        config = zone_configs.find_one({"camera_id": camera_id})
        if config:
            return [ZoneConfig(**z) for z in config["zones"]]
        return []
    except Exception as e:
        logger.error(f"Failed to get zones: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ====================== WebSocket Endpoint ======================
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        # Client sends subscription request
        subscription = await websocket.receive_json()
        event_types = subscription.get("events", [])
        event_manager.subscribe(websocket, event_types)

        # Send initial state
        if "count_update" in event_types:
            await websocket.send_json({
                "type": "count_update",
                "data": parking_state.get_counts()
            })

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        event_manager.unsubscribe(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        event_manager.unsubscribe(websocket)

# ====================== Real-time Count Endpoint ======================
@router.get("/counts")
async def get_counts():
    """Get current counts for all cameras"""
    return parking_state.get_counts()

# ====================== Core Processing Endpoints ======================
# === Video upload endpoint ===
@router.post("/video")
async def upload_video(
    request: Request,
    video: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{video.filename}"
        input_path = BASE_DIR / "data" / "input" / filename

        # Save file in chunks
        with open(input_path, "wb") as buffer:
            while chunk := await video.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)

        # Create job record
        job_id = create_job_record(request, filename, "video")
        logger.info(f"Created video job {job_id} for {filename}")

        # Add background task
        background_tasks.add_task(
            process_video_task,
            request=request,
            input_path=str(input_path),
            job_id=job_id
        )

        return {
            "status": "queued",
            "message": "Processing started",
            "job_id": job_id,
            "filename": filename
        }
    except Exception as e:
        logger.exception("Video upload failed")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video_task(request: Request, input_path: str, job_id: str):
    """Background task to process video and update job status"""
    try:
        logger.info(f"Starting video processing for job {job_id}")
        update_job_status(request, job_id, "processing")

        # Process video
        output_path, summary = anpr_processor.process_video(input_path)

        # Extract report paths from summary
        csv_path = summary.get("csv_file")
        xlsx_path = summary.get("xlsx_file")

        logger.success(f"Completed video processing for job {job_id}")

        # Update job status
        update_job_status(
            request,
            job_id,
            "completed",
            output_path=output_path,
            csv_path=csv_path,
            xlsx_path=xlsx_path
        )

        # Notify WebSocket clients
        await event_manager.broadcast("job_completed", {"job_id": job_id})

    except Exception as e:
        logger.error(f"Video processing failed for job {job_id}: {str(e)}")
        update_job_status(request, job_id, "failed")
        await event_manager.broadcast("job_failed", {"job_id": job_id, "error": str(e)})

# === Image upload & detect endpoint ===
@router.post("/image")
async def upload_image(request: Request, image: UploadFile = File(...)):
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{image.filename}"
        input_path = BASE_DIR / "data" / "images_input" / filename

        # Save file
        with open(input_path, "wb") as buffer:
            while chunk := await image.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)

        # Create job record
        job_id = create_job_record(request, filename, "image")
        logger.info(f"Created image job {job_id} for {filename}")

        # Process image
        output_image, results = anpr_processor.process_image(str(input_path))

        # Create CSV and Excel reports
        base = Path(input_path).stem
        csv_file = OUTPUT_DIR / f"annotated_{base}.csv"
        xlsx_file = OUTPUT_DIR / f"annotated_{base}.xlsx"

        if results:
            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False)
            df.to_excel(xlsx_file, index=False)
            logger.info(f"Created reports: {csv_file}, {xlsx_file}")
        else:
            csv_file = None
            xlsx_file = None

        logger.success(f"Completed image processing for job {job_id}")

        # Update job status
        update_job_status(
            request,
            job_id,
            "completed",
            output_path=output_image,
            csv_path=str(csv_file) if csv_file else None,
            xlsx_path=str(xlsx_file) if xlsx_file else None
        )

        return JSONResponse({
            "status": "completed",
            "job_id": job_id,
            "annotated_image": output_image,
            "csv_report": str(csv_file) if csv_file else None,
            "xlsx_report": str(xlsx_file) if xlsx_file else None,
            "plates": results
        })
    except Exception as e:
        logger.exception("Image upload failed")
        update_job_status(request, job_id, "failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Serve annotated images ===
@router.get("/image/{job_id}")
async def get_annotated_image(request: Request, job_id: str):
    """Get annotated image by job ID"""
    jobs_collection = get_jobs_collection(request)
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=425, detail="Processing not complete")

    output_path = job.get("output_path")
    if not output_path:
        raise HTTPException(status_code=404, detail="Output path not found")

    image_path = Path(output_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    # Determine media type
    suffix = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png"
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(image_path),
        filename=image_path.name,
        media_type=media_type
    )

# === Preview processed video ===
@router.get("/preview/{job_id}")
async def preview_video(request: Request, job_id: str):
    """Preview video by job ID"""
    jobs_collection = get_jobs_collection(request)
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=425, detail="Processing not complete")

    output_path = job.get("output_path")
    if not output_path:
        raise HTTPException(status_code=404, detail="Output path not found")

    video_path = Path(output_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path=str(video_path),
        filename=video_path.name,
        media_type="video/mp4"
    )

# === Download results ===
@router.get("/download/{job_id}")
async def download_file(request: Request, job_id: str):
    """Download results by job ID"""
    jobs_collection = get_jobs_collection(request)
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=425, detail="Processing not complete")

    output_path = job.get("output_path")
    csv_path = job.get("csv_path")
    xlsx_path = job.get("xlsx_path")

    if not any([output_path, csv_path, xlsx_path]):
        raise HTTPException(status_code=404, detail="No output files found")

    # Collect valid files
    valid_files = []
    file_types = {
        output_path: "media",
        csv_path: "csv",
        xlsx_path: "xlsx"
    }

    for path, file_type in file_types.items():
        if path and Path(path).exists():
            valid_files.append((path, file_type))

    if not valid_files:
        raise HTTPException(status_code=404, detail="No files available for download")

    # For single file, return directly
    if len(valid_files) == 1:
        path, file_type = valid_files[0]
        file_path = Path(path)

        media_types = {
            "media": "video/mp4" if file_path.suffix == ".mp4" else "image/jpeg",
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }

        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type=media_types[file_type]
        )

    # For multiple files, create a zip
    zip_filename = f"results_{job_id}.zip"
    zip_path = OUTPUT_DIR / zip_filename

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for path, _ in valid_files:
            file_path = Path(path)
            zipf.write(file_path, file_path.name)

    return FileResponse(
        path=str(zip_path),
        filename=zip_filename,
        media_type="application/zip"
    )

# === Job status endpoint ===
@router.get("/job/{job_id}")
def get_job_status(request: Request, job_id: str):
    try:
        jobs_collection = get_jobs_collection(request)
        job = jobs_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            return JSONResponse({"error": "Job not found"}, status_code=404)

        # Convert ObjectId and datetime to strings
        job["_id"] = str(job["_id"])
        job["created_at"] = job["created_at"].isoformat()
        if job.get("completed_at"):
            job["completed_at"] = job["completed_at"].isoformat()

        # Add available report paths
        reports = {}
        if job.get("csv_path"):
            reports["csv"] = job["csv_path"]
        if job.get("xlsx_path"):
            reports["xlsx"] = job["xlsx_path"]

        job["reports"] = reports

        return job
    except Exception as e:
        logger.error(f"Job status error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === Parking System Endpoints ===

@router.post("/parking/configure_cameras")
def configure_parking_cameras(
    request: Request,
    entry_cam: int = Form(0),
    exit_cam: int = Form(1)
):
    """Configure entry and exit cameras for parking system"""
    try:
        parking_processor.configure_cameras(entry_cam_id=entry_cam, exit_cam_id=exit_cam)
        return {"status": "success", "message": f"Cameras configured: Entry={entry_cam}, Exit={exit_cam}"}
    except Exception as e:
        logger.error(f"Camera configuration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/parking/start")
def start_parking_system(request: Request):
    """Start the parking system processing"""
    try:
        # Pass collections to parking processor
        parking_events_collection = get_parking_events_collection(request)
        parking_slots_collection = get_parking_slots_collection(request)
        zone_configs_collection = get_zone_configs_collection(request)

        parking_processor.start_processing(
            event_manager,
            parking_state,
            parking_events_collection,
            parking_slots_collection,
            zone_configs_collection
        )
        return {"status": "success", "message": "Parking system started"}
    except Exception as e:
        logger.error(f"Failed to start parking system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/parking/stop")
def stop_parking_system():
    """Stop the parking system processing"""
    try:
        parking_processor.stop_processing()
        return {"status": "success", "message": "Parking system stopped"}
    except Exception as e:
        logger.error(f"Failed to stop parking system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parking/status")
def get_parking_status(request: Request):
    """Get current parking status"""
    try:
        parking_slots_collection = get_parking_slots_collection(request)
        status = parking_processor.get_parking_status(parking_slots_collection)
        return status
    except Exception as e:
        logger.error(f"Failed to get parking status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parking/events")
def get_parking_events(request: Request, limit: int = Query(100, description="Number of events to retrieve")):
    """Get recent parking events"""
    try:
        parking_events_collection = get_parking_events_collection(request)
        events = list(parking_events_collection.find(
            {},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        return JSONResponse(content={"events": events})
    except Exception as e:
        logger.error(f"Failed to get parking events: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/parking/manual_assign")
def manual_assign_slot(request: Request, plate: str = Form(...)):
    """Manually assign a parking slot to a vehicle"""
    try:
        parking_slots_collection = get_parking_slots_collection(request)
        slot_id = parking_processor.assign_slot(plate, parking_slots_collection)
        if slot_id:
            # Broadcast event
            asyncio.run(event_manager.broadcast("manual_entry", {
                "plate": plate,
                "slot_id": slot_id,
                "timestamp": datetime.utcnow().isoformat()
            }))
            return {"status": "success", "message": f"Assigned slot {slot_id} to {plate}"}
        else:
            return {"status": "error", "message": "No available slots"}
    except Exception as e:
        logger.error(f"Manual slot assignment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/parking/manual_release")
def manual_release_slot(request: Request, plate: str = Form(...)):
    """Manually release a parking slot"""
    try:
        parking_slots_collection = get_parking_slots_collection(request)
        if parking_processor.release_slot(plate, parking_slots_collection):
            # Broadcast event
            asyncio.run(event_manager.broadcast("manual_exit", {
                "plate": plate,
                "timestamp": datetime.utcnow().isoformat()
            }))
            return {"status": "success", "message": f"Released slot for {plate}"}
        else:
            return {"status": "error", "message": "Vehicle not found or slot not occupied"}
    except Exception as e:
        logger.error(f"Manual slot release failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parking/slots")
def get_all_slots(request: Request):
    """Get current status of all parking slots"""
    try:
        parking_slots_collection = get_parking_slots_collection(request)
        slots = list(parking_slots_collection.find({}, {"_id": 0}))
        return JSONResponse(content={"slots": slots})
    except Exception as e:
        logger.error(f"Failed to get slot status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parking/export_logs")
def export_parking_logs(request: Request):
    """Export parking events to CSV"""
    try:
        parking_events_collection = get_parking_events_collection(request)
        logs = list(parking_events_collection.find({}, {"_id": 0}))
        df = pd.DataFrame(logs)
        file_path = OUTPUT_DIR / f"parking_log_{datetime.now():%Y%m%d%H%M%S}.csv"
        df.to_csv(file_path, index=False)
        return FileResponse(str(file_path), media_type="text/csv", filename=file_path.name)
    except Exception as e:
        logger.error(f"Failed to export parking logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export logs")
