import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import cloudinary
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from anpr.routes import router as detection_router
from pymongo import MongoClient

# === CRITICAL LOGGING SUPPRESSION ===
# Configure logging BEFORE any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Suppress noisy modules
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("engineio").setLevel(logging.WARNING)
logging.getLogger("socketio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("Configured logging with suppression for pymongo/urllib3")

# === PATH CONFIGURATION ===
# Bring your project root into PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logger.info("Added project root to PYTHONPATH")

# === ENVIRONMENT SETUP ===
dotenv_path = Path(__file__).resolve().parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
    logger.info("Loaded environment variables from .env file")
else:
    logger.warning("No .env file found - using system environment variables")

# Application settings (fallback to local if not provided)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/anpr")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG_VIDEO = os.getenv("DEBUG_VIDEO", "false").lower() == "true"

# Configure logging level
numeric_level = getattr(logging, LOG_LEVEL, None)
if not isinstance(numeric_level, int):
    logger.warning(f"Invalid LOG_LEVEL: {LOG_LEVEL}. Defaulting to INFO")
    numeric_level = logging.INFO

# Reconfigure logging with the specified level
root_logger = logging.getLogger()
root_logger.setLevel(numeric_level)
for handler in root_logger.handlers:
    handler.setLevel(numeric_level)
logger.info(f"Logging level set to {logging.getLevelName(numeric_level)}")
logger.info(f"Video debugging enabled: {DEBUG_VIDEO}")

# Configure Cloudinary
try:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )
    logger.info("Cloudinary configured successfully")
except Exception as e:
    logger.error(f"Cloudinary configuration failed: {str(e)}")

# Track MongoDB clients for graceful shutdown
mongo_clients = []

def get_mongodb_client():
    """Create and track MongoDB client with optimized settings"""
    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000,    # Faster timeout
        heartbeatFrequencyMS=30000,       # Reduce heartbeat frequency
        connectTimeoutMS=10000
    )
    mongo_clients.append(client)
    logger.info(f"Created MongoDB client (total: {len(mongo_clients)})")
    return client

async def close_mongo_connections():
    """Close all MongoDB connections on shutdown"""
    global mongo_clients
    logger.info(f"Closing {len(mongo_clients)} MongoDB connections")
    
    for client in mongo_clients:
        try:
            client.close()
            logger.info(f"Closed MongoDB connection to {client.address}")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {str(e)}")
    
    mongo_clients = []

# Lifespan manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 ANPR Backend starting up")
    logger.info(f"Environment: {os.getenv('ENV', 'development')}")
    logger.info(f"MongoDB URI: {MONGODB_URI}")
    logger.info(f"Cloudinary configured: {bool(CLOUDINARY_CLOUD_NAME)}")
    logger.info(f"Allowed CORS origins: *")
    logger.info(f"Debug video mode: {DEBUG_VIDEO}")
    
    # Initialize MongoDB connection
    app.state.mongodb_client = get_mongodb_client()
    app.state.mongodb_db = app.state.mongodb_client.get_database()
    logger.info("MongoDB connection established")
    
    yield
    
    # Shutdown
    logger.info("🛑 ANPR Backend shutting down")
    await close_mongo_connections()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="ANPR Backend",
    description="Automatic Number Plate Recognition API for local development",
    version="1.1.0",  # Updated version
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
    max_upload_size=1024 * 1024 * 100  # 100MB upload limit
)

# Attach settings to app.state
app.state.mongodb_uri = MONGODB_URI
app.state.cloudinary = {
    "cloud_name": CLOUDINARY_CLOUD_NAME,
    "api_key": CLOUDINARY_API_KEY,
    "api_secret": CLOUDINARY_API_SECRET
}
app.state.debug_video = DEBUG_VIDEO

# Enable CORS for all origins
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
    return {
        "message": "✅ ANPR backend is running",
        "environment": os.getenv("ENV", "development"),
        "gpu_available": torch.cuda.is_available()  # Add this if torch is imported
    }

# Debug endpoint for configuration
@app.get("/config")
def read_config():
    return {
        "mongodb_uri": app.state.mongodb_uri,
        "cloudinary": "configured" if app.state.cloudinary["cloud_name"] else "disabled",
        "log_level": logging.getLevelName(logging.getLogger().getEffectiveLevel()),
        "debug_video": app.state.debug_video,
        "max_upload_size": "100MB"
    }
