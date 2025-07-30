"""
Shared model management utilities for all video analytics modules.
Provides centralized model downloading, caching, and validation.
"""

import os
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

# ======================================
# Logger Setup
# ======================================
logger = logging.getLogger("analytics.model_manager")

# ======================================
# Constants and Paths
# ======================================
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Model configurations
MODEL_CONFIGS = {
    # YOLO Models (General Purpose) - Note: download_id is used for Ultralytics download, filename for saving
    "yolov8x": {
        "filename": "yolov8x.pt",
        "download_id": "yolov8x", 
        "loader": "ultralytics.YOLO",
        "description": "YOLOv8x object detection model"
    },
    "yolov11x": {
        "filename": "yolov11x.pt",
        "download_id": "yolov11x",
        "loader": "ultralytics.YOLO", 
        "description": "YOLOv11x object detection model"
    },
    "yolov12x": {
        "filename": "yolov12x.pt", 
        "download_id": "yolov12x",
        "loader": "ultralytics.YOLO",
        "fallback": "yolov11x",  # Use yolo11x as fallback as requested
        "description": "YOLOv12x object detection model (fallback: YOLOv11x)"
    },
    "yolo12x": {  # Alternative naming found in emergency_count.py
        "filename": "yolo12x.pt", 
        "download_id": "yolo12x",
        "loader": "ultralytics.YOLO",
        "fallback": "yolov11x",
        "description": "YOLO12x object detection model (fallback: YOLOv11x)"
    },
    
    # Specialized YOLO Models
    "yolo11m": {
        "filename": "yolo11m.pt",
        "download_id": "yolo11m",  # YOLOv11 Medium model
        "loader": "ultralytics.YOLO",
        "fallback": "yolov11x",
        "description": "YOLOv11 Medium model for general object detection"
    },
    "best_car": {
        "filename": "best_car.pt",
        "loader": "ultralytics.YOLO",
        "fallback": "yolo11m",
        "description": "Custom trained YOLO model for car/license plate detection",
        "custom_trained": True
    },
    "best_animal": {
        "filename": "best_animal.pt", 
        "loader": "ultralytics.YOLO",
        "fallback": "yolov11x",
        "description": "Custom trained YOLO model for animal/pest detection",
        "custom_trained": True
    },
    
    # RT-DETR Models
    "rtdetr-l": {
        "filename": "rtdetr-l.pt",
        "download_id": "rtdetr-l",
        "loader": "ultralytics.RTDETR", 
        "description": "RT-DETR Large detection model"
    },
    "rtdetr-x": {
        "filename": "rtdetr-x.pt",
        "download_id": "rtdetr-x",
        "loader": "ultralytics.RTDETR",
        "description": "RT-DETR Extra Large detection model"
    },
    
    # Re-ID Models
    "osnet_reid": {
        "filename": "osnet_x0_25_msmt17.pt",
        "url": "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0/osnet_x0_25_msmt17.pth",
        "cache_path": Path.home() / ".cache/torch/checkpoints/osnet_x0_25_msmt17.pt",
        "description": "OSNet Re-ID model for person tracking"
    },
    "osnet_ibn_reid": {
        "filename": "osnet_ibn_x1_0_msmt17.pth",
        "url": "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0/osnet_ibn_x1_0_msmt17.pth",
        "description": "OSNet IBN Re-ID model for enhanced person tracking"
    },
    "osnet_alternative": {
        "filename": "osnet_x0_25_msmt17.pt",
        "url": "https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0.0/osnet_x0_25_msmt17.pt",
        "description": "Alternative OSNet Re-ID model from yolo_tracking"
    },
    
    # Depth Models (torch.hub cached)
    "midas_dpt_large": {
        "hub_repo": "intel-isl/MiDaS",
        "hub_model": "DPT_Large",
        "description": "MiDaS DPT Large depth estimation model"
    },
    
    # Hugging Face Transformers Models
    "dpt_large": {
        "hf_model": "Intel/dpt-large",
        "description": "DPT Large depth estimation via Hugging Face"
    },
    "glpn_nyu": {
        "hf_model": "vinvino02/glpn-nyu",
        "description": "GLPN depth estimation model"
    }
}

# ======================================
# Core Functions
# ======================================

def ensure_models_directory(models_dir: Optional[Path] = None) -> Path:
    """
    Ensure models directory exists with proper permissions.
    
    Args:
        models_dir: Optional custom models directory path
        
    Returns:
        Path to valid models directory
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    else:
        models_dir = Path(models_dir)
    
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Models directory: {models_dir}")
        return models_dir
    except PermissionError:
        # Fallback to user cache directory
        fallback_dir = Path.home() / ".cache" / "intellivision_models"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"⚠️ Permission denied for {models_dir}. Using fallback: {fallback_dir}")
        return fallback_dir

def download_ultralytics_model(model_name: str, models_dir: Path) -> bool:
    """
    Download Ultralytics model (YOLO/RT-DETR) with fallback support.
    
    Args:
        model_name: Model identifier from MODEL_CONFIGS
        models_dir: Directory to save model
        
    Returns:
        True if successful, False otherwise
    """
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        logger.error(f"❌ Unknown model: {model_name}")
        return False
    
    filename = config["filename"]
    loader_path = config["loader"]
    filepath = models_dir / filename
    
    if filepath.exists():
        logger.info(f"✅ {filename} already exists")
        return True
    
    try:
        # Import the loader dynamically
        module_name, class_name = loader_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        loader_class = getattr(module, class_name)
        
        logger.info(f"🔄 Downloading {filename} using {loader_path}...")
        
        # Handle model loading with fallback
        download_id = config.get("download_id", model_name)  # Use download_id if available, fallback to model_name
        
        if "fallback" in config:
            try:
                model = loader_class(download_id)
            except Exception as e:
                logger.warning(f"⚠️ {download_id} not available: {e}")
                fallback_name = config["fallback"]
                fallback_config = MODEL_CONFIGS.get(fallback_name, {})
                fallback_download_id = fallback_config.get("download_id", fallback_name)
                logger.info(f"🔄 Trying fallback: {fallback_download_id}")
                model = loader_class(fallback_download_id)
                download_id = fallback_download_id
        else:
            model = loader_class(download_id)
        
        # Save the model
        if hasattr(model, 'save'):
            model.save(str(filepath))
        else:
            # For older versions, try export method
            model.export(format='pt', imgsz=640)
            # Look for exported file in common locations
            possible_paths = [
                Path(model.trainer.save_dir) / "weights" / "best.pt",
                Path.cwd() / f"{download_id}.pt"
            ]
            
            exported_path = None
            for path in possible_paths:
                if path.exists():
                    exported_path = path
                    break
            
            if exported_path:
                shutil.copy(exported_path, filepath)
            else:
                logger.error(f"❌ Could not find exported model file for {model_name}")
                return False
        
        logger.info(f"✅ Downloaded {filename}")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Required library not installed for {model_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        return False

def download_reid_model(models_dir: Path) -> bool:
    """
    Download Re-ID model for person tracking.
    
    Args:
        models_dir: Directory to save model (will use cache path)
        
    Returns:
        True if successful, False otherwise
    """
    config = MODEL_CONFIGS["osnet_reid"]
    cache_path = config["cache_path"]
    url = config["url"]
    
    if cache_path.exists():
        logger.info(f"✅ Re-ID model already exists at {cache_path}")
        return True
    
    try:
        logger.info("🔄 Downloading BotSort Re-ID model...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
        logger.info(f"✅ Re-ID model downloaded to {cache_path}")
        return True
        
    except PermissionError:
        # Try fallback location
        fallback_path = models_dir / config["filename"]
        try:
            urllib.request.urlretrieve(url, fallback_path)
            logger.warning(f"⚠️ Using fallback Re-ID location: {fallback_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Could not download Re-ID model to fallback location: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Could not download Re-ID model: {e}")
        return False

def ensure_torch_hub_model(model_name: str) -> bool:
    """
    Ensure torch.hub model is cached (like MiDaS).
    
    Args:
        model_name: Model identifier from MODEL_CONFIGS
        
    Returns:
        True if successful, False otherwise
    """
    config = MODEL_CONFIGS.get(model_name)
    if not config or "hub_repo" not in config:
        logger.error(f"❌ Invalid torch.hub model: {model_name}")
        return False
    
    try:
        logger.info(f"🔄 Ensuring {config['description']} via torch.hub...")
        
        # Load model to ensure it's cached
        torch.hub.load(config["hub_repo"], config["hub_model"], pretrained=True)
        
        # Also cache transforms if available
        if model_name == "midas_dpt_large":
            torch.hub.load(config["hub_repo"], "transforms")
        
        logger.info(f"✅ {config['description']} cached successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Could not cache {model_name}: {e}")
        return False

def ensure_huggingface_model(model_name: str) -> bool:
    """
    Ensure Hugging Face Transformers model is cached.
    
    Args:
        model_name: Model identifier from MODEL_CONFIGS
        
    Returns:
        True if successful, False otherwise
    """
    config = MODEL_CONFIGS.get(model_name)
    if not config or "hf_model" not in config:
        logger.error(f"❌ Invalid Hugging Face model: {model_name}")
        return False
    
    try:
        logger.info(f"🔄 Ensuring {config['description']} via Hugging Face...")
        
        # Import transformers dynamically
        from transformers import AutoImageProcessor, AutoModel
        
        # Cache the model and processor
        hf_model_id = config["hf_model"]
        AutoImageProcessor.from_pretrained(hf_model_id)
        AutoModel.from_pretrained(hf_model_id)
        
        logger.info(f"✅ {config['description']} cached successfully")
        return True
        
    except ImportError:
        logger.warning(f"⚠️ transformers not installed, skipping {model_name}")
        return False
    except Exception as e:
        logger.error(f"❌ Could not cache {model_name}: {e}")
        return False

def download_all_models(models_dir: Optional[Path] = None, 
                       model_list: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Download all required models for video analytics.
    
    Args:
        models_dir: Optional custom models directory
        model_list: Optional list of specific models to download
        
    Returns:
        Dictionary mapping model names to success status
    """
    models_dir = ensure_models_directory(models_dir)
    
    if model_list is None:
        # Default models for most analytics
        model_list = [
            "yolov8x", "yolov12x", "rtdetr-l", 
            "osnet_reid", "midas_dpt_large", 
            "dpt_large", "glpn_nyu"
        ]
    
    results = {}
    
    logger.info(f"🚀 Starting model downloads for: {model_list}")
    
    for model_name in model_list:
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            logger.error(f"❌ Unknown model: {model_name}")
            results[model_name] = False
            continue
        
        logger.info(f"🔄 Processing {model_name}: {config['description']}")
        
        # Route to appropriate download function
        if "loader" in config and "ultralytics" in config["loader"]:
            results[model_name] = download_ultralytics_model(model_name, models_dir)
        elif model_name == "osnet_reid":
            results[model_name] = download_reid_model(models_dir)
        elif "hub_repo" in config:
            results[model_name] = ensure_torch_hub_model(model_name)
        elif "hf_model" in config:
            results[model_name] = ensure_huggingface_model(model_name)
        else:
            logger.error(f"❌ No download handler for {model_name}")
            results[model_name] = False
    
    # Summary
    successful = sum(results.values())
    total = len(results)
    logger.info(f"📊 Model download summary: {successful}/{total} successful")
    
    if successful < total:
        failed_models = [name for name, success in results.items() if not success]
        logger.warning(f"⚠️ Failed models: {failed_models}")
    
    return results

def check_model_availability(model_list: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Check which models are currently available without downloading.
    
    Args:
        model_list: Optional list of models to check
        
    Returns:
        Dictionary mapping model names to availability status
    """
    if model_list is None:
        model_list = list(MODEL_CONFIGS.keys())
    
    availability = {}
    models_dir = MODELS_DIR
    
    for model_name in model_list:
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            availability[model_name] = False
            continue
        
        # Check based on model type
        if "filename" in config:
            # File-based model
            if "cache_path" in config:
                filepath = config["cache_path"]
            else:
                filepath = models_dir / config["filename"]
            availability[model_name] = filepath.exists()
        else:
            # Assume available for hub/HF models (they're cached separately)
            availability[model_name] = True
    
    return availability

def get_model_path(model_name: str) -> Optional[Path]:
    """
    Get the file path for a model.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Path to model file or None if not applicable
    """
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        return None
    
    if "cache_path" in config:
        return config["cache_path"]
    elif "filename" in config:
        return MODELS_DIR / config["filename"]
    else:
        # Hub/HF models don't have simple file paths
        return None

# ======================================
# Convenience Functions for Common Models
# ======================================

def ensure_yolo_models() -> bool:
    """Ensure YOLO models are available."""
    results = download_all_models(model_list=["yolov8x", "yolov12x"])
    return any(results.values())  # Success if at least one YOLO model available

def ensure_detection_models() -> bool:
    """Ensure all detection models are available."""
    results = download_all_models(model_list=["yolov8x", "yolov12x", "rtdetr-l"])
    return any(results.values())  # Success if at least one detection model available

def ensure_depth_models() -> bool:
    """Ensure depth estimation models are available."""
    results = download_all_models(model_list=["midas_dpt_large", "dpt_large", "glpn_nyu"])
    return any(results.values())  # Success if at least one depth model available

def ensure_reid_model() -> bool:
    """Ensure Re-ID model is available."""
    results = download_all_models(model_list=["osnet_reid"])
    return results.get("osnet_reid", False)

# ======================================
# Auto-initialization
# ======================================

def initialize_models(auto_download: bool = True) -> Dict[str, bool]:
    """
    Initialize model management system.
    
    Args:
        auto_download: Whether to automatically download missing models
        
    Returns:
        Model availability status
    """
    logger.info("🔧 Initializing model management system...")
    
    # Ensure models directory exists
    models_dir = ensure_models_directory()
    
    # Check current availability
    availability = check_model_availability()
    
    if auto_download:
        # Download missing critical models
        critical_models = ["yolov8x", "osnet_reid"]  # Minimum required
        missing_critical = [name for name in critical_models if not availability.get(name, False)]
        
        if missing_critical:
            logger.info(f"🔄 Downloading missing critical models: {missing_critical}")
            download_results = download_all_models(model_list=missing_critical)
            
            # Update availability
            for model_name, success in download_results.items():
                if success:
                    availability[model_name] = True
    
    logger.info("✅ Model management system initialized")
    return availability

if __name__ == "__main__":
    # Test the model manager
    print("🧪 Testing model manager...")
    results = initialize_models(auto_download=True)
    print(f"📊 Model availability: {results}")