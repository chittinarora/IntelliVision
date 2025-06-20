import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from anpr.routes import router as detection_router  # ✅ Import the route from anpr.routes

# Initialize FastAPI app
app = FastAPI(
    title="ANPR Backend",
    description="Automatic Number Plate Recognition API using YOLO and EasyOCR",
    version="1.0.0"
)

# Enable CORS (useful for frontend integrations like React, Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all detection routes under /detect
app.include_router(detection_router, prefix="/detect")

# Root health check
@app.get("/")
def read_root():
    return {"message": "✅ ANPR backend is running"}

