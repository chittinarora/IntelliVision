import os
import shutil
import asyncio
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from threading import Thread
from bson import ObjectId
import zipfile

from fastapi import (
    APIRouter, File, UploadFile, HTTPException, Query,
    BackgroundTasks, WebSocket, WebSocketDisconnect, Form
)
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pymongo import MongoClient
from loguru import logger

# Create router
router = APIRouter()

# === Resolve absolute paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
plate_model_path = BASE_DIR / "models" / "best.pt"
car_model_path   = BASE_DIR / "models" / "yolo11m.pt"

# Fixed output directory
OUTPUT_DIR = Path("/home/abhishek/ANPR/backend/data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"All output will be saved to: {OUTPUT_DIR}")

# MongoDB client for jobs and results
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/anpr")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["anpr"]
jobs_collection = db["processing_jobs"]  # Track processing jobs
detections_collection = db["detections"]  # Plate detection results
parking_events = db["parking_events"]    # Parking entry/exit events
parking_slots = db["parking_slots"]      # Parking slot status

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

# WebSocket clients
websocket_clients = set()

# Create needed input directories
(Path(BASE_DIR) / "data" / "input").mkdir(parents=True, exist_ok=True)
(Path(BASE_DIR) / "data" / "images_input").mkdir(parents=True, exist_ok=True)

def create_job_record(filename: str, job_type: str):
    """Create a job record in MongoDB"""
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

def update_job_status(job_id: str, status: str, output_path: str = None, 
                     csv_path: str = None, xlsx_path: str = None):
    """Update job status in MongoDB"""
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

# === Video upload endpoint ===
@router.post("/video")
async def upload_video(
    video: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{video.filename}"
        input_path = BASE_DIR / "data" / "input" / filename
        
        # Save file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Create job record
        job_id = create_job_record(filename, "video")
        logger.info(f"Created video job {job_id} for {filename}")
        
        # Add background task with job ID
        background_tasks.add_task(process_video_task, str(input_path), job_id)
        
        return {
            "status": "queued",
            "message": "Processing started",
            "job_id": job_id,
            "filename": filename
        }
    except Exception as e:
        logger.exception("Video upload failed")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video_task(input_path: str, job_id: str):
    """Background task to process video and update job status"""
    try:
        logger.info(f"Starting video processing for job {job_id}")
        update_job_status(job_id, "processing")
        
        # Process video
        output_path, summary = anpr_processor.process_video(input_path)
        
        # Extract report paths from summary
        csv_path = summary.get("csv_file")
        xlsx_path = summary.get("xlsx_file")
        
        logger.success(f"Completed video processing for job {job_id}")
        
        # Update job status
        update_job_status(
            job_id, 
            "completed", 
            output_path=output_path,
            csv_path=csv_path,
            xlsx_path=xlsx_path
        )
        
        # Notify WebSocket clients
        for client in websocket_clients:
            await client.send_text(f"Video job {job_id} completed")
            
    except Exception as e:
        logger.error(f"Video processing failed for job {job_id}: {str(e)}")
        update_job_status(job_id, "failed")
        raise

# === Image upload & detect endpoint ===
@router.post("/image")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{image.filename}"
        input_path = BASE_DIR / "data" / "images_input" / filename
        
        # Save file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Create job record
        job_id = create_job_record(filename, "image")
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
        update_job_status(job_id, "failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Serve annotated images ===
@router.get("/image/{job_id}")
async def get_annotated_image(job_id: str):
    """Get annotated image by job ID"""
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
async def preview_video(job_id: str):
    """Preview video by job ID"""
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

# === Download results (video, csv, xlsx) ===
@router.get("/download/{job_id}")
async def download_file(job_id: str):
    """Download results by job ID"""
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

# === Get CSV report ===
@router.get("/report/csv/{job_id}")
async def get_csv_report(job_id: str):
    """Get CSV report for job"""
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=425, detail="Processing not complete")
    
    csv_path = job.get("csv_path")
    if not csv_path:
        raise HTTPException(status_code=404, detail="CSV report not available")
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    return FileResponse(
        path=str(csv_file),
        filename=csv_file.name,
        media_type="text/csv"
    )

# === WebSocket endpoint for job updates ===
@router.websocket("/ws")
async def websocket_connection(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)
    try:
        await websocket.send_text("WebSocket connected ✅")
        while True:
            data = await websocket.receive_text()
            # Keep connection alive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        websocket_clients.discard(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

# === Job status endpoint ===
@router.get("/job/{job_id}")
def get_job_status(job_id: str):
    try:
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

# === Fetch detection history ===
@router.get("/history")
def get_detection_history():
    try:
        records = list(detections_collection.find({}, {"_id": 0}))
        return JSONResponse(content={"history": records})
    except Exception as e:
        logger.exception("Failed to fetch detection history")
        raise HTTPException(status_code=500, detail=str(e))

# === Parking System Endpoints ===

@router.post("/parking/configure_cameras")
def configure_parking_cameras(
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
def start_parking_system():
    """Start the parking system processing"""
    try:
        parking_processor.start_processing()
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
def get_parking_status():
    """Get current parking status"""
    try:
        status = parking_processor.get_parking_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get parking status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parking/events")
def get_parking_events(limit: int = Query(100, description="Number of events to retrieve")):
    """Get recent parking events"""
    try:
        events = list(parking_events.find(
            {}, 
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        return JSONResponse(content={"events": events})
    except Exception as e:
        logger.error(f"Failed to get parking events: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/parking/manual_assign")
def manual_assign_slot(plate: str = Form(...)):
    """Manually assign a parking slot to a vehicle"""
    try:
        slot_id = parking_processor.assign_slot(plate)
        if slot_id:
            return {"status": "success", "message": f"Assigned slot {slot_id} to {plate}"}
        else:
            return {"status": "error", "message": "No available slots"}
    except Exception as e:
        logger.error(f"Manual slot assignment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/parking/manual_release")
def manual_release_slot(plate: str = Form(...)):
    """Manually release a parking slot"""
    try:
        if parking_processor.release_slot(plate):
            return {"status": "success", "message": f"Released slot for {plate}"}
        else:
            return {"status": "error", "message": "Vehicle not found or slot not occupied"}
    except Exception as e:
        logger.error(f"Manual slot release failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parking/slots")
def get_all_slots():
    """Get current status of all parking slots"""
    try:
        slots = list(parking_slots.find({}, {"_id": 0}))
        return JSONResponse(content={"slots": slots})
    except Exception as e:
        logger.error(f"Failed to get slot status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parking/export_logs")
def export_parking_logs():
    """Export parking events to CSV"""
    try:
        logs = list(parking_events.find({}, {"_id": 0}))
        df = pd.DataFrame(logs)
        file_path = OUTPUT_DIR / f"parking_log_{datetime.now():%Y%m%d%H%M%S}.csv"
        df.to_csv(file_path, index=False)
        return FileResponse(str(file_path), media_type="text/csv", filename=file_path.name)
    except Exception as e:
        logger.error(f"Failed to export parking logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export logs")
