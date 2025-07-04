import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import cloudinary

# Bring your project root into PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Load environment variables ===
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)

# Application settings (fallback to local if not provided)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/anpr")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from anpr.routes import router as detection_router

# Initialize FastAPI app
app = FastAPI(
    title="ANPR Backend",
    description="Automatic Number Plate Recognition API using YOLO and EasyOCR",
    version="1.0.0"
)

# Attach settings to app.state for later use
app.state.mongodb_uri = MONGODB_URI
app.state.cloudinary = {
    "cloud_name": CLOUDINARY_CLOUD_NAME,
    "api_key": CLOUDINARY_API_KEY,
    "api_secret": CLOUDINARY_API_SECRET
}

# Enable CORS for all origins (adjust as needed in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all ANPR & Parking routes under /detect
app.include_router(detection_router, prefix="/detect")

# Root health check
@app.get("/")
def read_root():
    return {"message": "✅ ANPR backend is running"}

# Debug endpoint for configuration
@app.get("/config")
def read_config():
    return {
        "mongodb_uri": app.state.mongodb_uri,
        "cloudinary": app.state.cloudinary
    }
