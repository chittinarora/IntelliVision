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
from ..exception_handlers import GPUError, ModelLoadingError

# ======================================
# Logger Setup
# ======================================
logger = logging.getLogger("analytics.model_manager")

# ======================================
# Constants and Paths
# ======================================
# Use the same path as all analytics modules for consistency
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# This resolves to: /app/intellivision/apps/video_analytics/models

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
        "fallback": "yolov11x",
        "description": "YOLOv12x object detection model (fallback: YOLOv11x)"
    },

    # Specialized YOLO Models
    "yolo11m_car": {
        "filename": "yolo11m_car.pt",
        "loader": "ultralytics.YOLO",
        "fallback": "yolov11x",
        "description": "Custom trained YOLOv11m model for number plate detection and parking analysis",
        "custom_trained": True,
        "skip_download": True
    },
    "best_car": {
        "filename": "best_car.pt",
        "loader": "ultralytics.YOLO",
        "fallback": "yolov11x",
        "description": "Custom trained YOLOv11m model for number plate detection and parking analysis",
        "custom_trained": True,
        "skip_download": True
    },
    "best_animal": {
        "filename": "best_animal.pt",
        "loader": "ultralytics.YOLO",
        "fallback": "yolov11x",
        "description": "Custom trained YOLO model for animal/pest detection",
        "custom_trained": True,
        "skip_download": True
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
                # Handle special case for yolov12x fallback to yolov8x (matching people_count.py)
                if download_id == "yolov12x":
                    try:
                        model = loader_class("yolov12x")
                    except Exception as e:
                        logger.warning(f"⚠️ yolov12x not available: {e}, falling back to yolov8x")
                        model = loader_class("yolov8x")
                        download_id = "yolov8x"
                else:
                    model = loader_class(download_id)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                    raise GPUError(f"CUDA out of memory or other CUDA error during model loading: {e}", details={"model": model_name, "error_type": "CUDA_OOM"})
                else:
                    raise ModelLoadingError(f"Failed to load model {model_name}: {e}", details={"model": model_name})
            except Exception as e:
                logger.warning(f"⚠️ {download_id} not available: {e}")
                fallback_name = config["fallback"]
                fallback_config = MODEL_CONFIGS.get(fallback_name, {})
                fallback_download_id = fallback_config.get("download_id", fallback_name)
                logger.info(f"🔄 Trying fallback: {fallback_download_id}")
                try:
                    model = loader_class(fallback_download_id)
                    download_id = fallback_download_id
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                        raise GPUError(f"CUDA out of memory or other CUDA error during fallback model loading: {e}", details={"model": fallback_name, "error_type": "CUDA_OOM"})
                    else:
                        raise ModelLoadingError(f"Failed to load fallback model {fallback_name}: {e}", details={"model": fallback_name})
                except Exception as e:
                    raise ModelLoadingError(f"Failed to load fallback model {fallback_name}: {e}", details={"model": fallback_name})
        else:
            try:
                model = loader_class(download_id)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                    raise GPUError(f"CUDA out of memory or other CUDA error during model loading: {e}", details={"model": model_name, "error_type": "CUDA_OOM"})
                else:
                    raise ModelLoadingError(f"Failed to load model {model_name}: {e}", details={"model": model_name})
            except Exception as e:
                raise ModelLoadingError(f"Failed to load model {model_name}: {e}", details={"model": model_name})

        # For Ultralytics models, they're automatically cached when loaded
        # We need to find the cached model and copy it to our models directory
        model_path = None

        # Method 1: Try to get path from model object
        if hasattr(model, 'ckpt_path') and model.ckpt_path:
            model_path = Path(model.ckpt_path)
            logger.info(f"Found model path via ckpt_path: {model_path}")
        elif hasattr(model, 'model_path') and model.model_path:
            model_path = Path(model.model_path)
            logger.info(f"Found model path via model_path: {model_path}")

        # Method 2: Search common Ultralytics cache locations
        if not model_path or not model_path.exists():
            search_paths = [
                Path.home() / ".cache" / "ultralytics" / f"{download_id}.pt",
                Path.home() / ".ultralytics" / "weights" / f"{download_id}.pt"
            ]

            # Try to get official weights directory
            try:
                from ultralytics.utils import WEIGHTS_DIR
                search_paths.insert(0, WEIGHTS_DIR / f"{download_id}.pt")
            except (ImportError, AttributeError):
                pass

            for path in search_paths:
                if path.exists() and path.stat().st_size > 0:  # Check file exists and has content
                    model_path = path
                    logger.info(f"Found cached model at: {model_path}")
                    break

        # Copy model to our directory or handle failure appropriately
        if model_path and model_path.exists() and model_path.stat().st_size > 0:
            try:
                shutil.copy2(model_path, filepath)  # copy2 preserves metadata
                logger.info(f"✅ Copied model: {model_path} -> {filepath} ({filepath.stat().st_size / (1024*1024):.1f}MB)")
            except Exception as copy_error:
                logger.error(f"❌ Failed to copy model {model_name}: {copy_error}")
                return False
        else:
            logger.error(f"❌ Could not locate cached model file for {download_id}")
            logger.info(f"🔍 Searched paths: {[str(p) for p in search_paths] if 'search_paths' in locals() else 'N/A'}")
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
        models_dir: Directory to save model

    Returns:
        True if successful, False otherwise
    """
    config = MODEL_CONFIGS["osnet_reid"]
    filepath = models_dir / config["filename"]
    url = config["url"]

    if filepath.exists():
        logger.info(f"✅ Re-ID model already exists at {filepath}")
        return True

    try:
        logger.info("🔄 Downloading Re-ID model...")
        models_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, filepath)
        logger.info(f"✅ Re-ID model downloaded to {filepath}")
        return True

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

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
            raise GPUError(f"CUDA out of memory or other CUDA error during torch.hub model caching: {e}", details={"model": model_name, "error_type": "CUDA_OOM"})
        else:
            raise ModelLoadingError(f"Failed to cache torch.hub model {model_name}: {e}", details={"model": model_name})
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

        hf_model_id = config["hf_model"]

        # Use specific model classes based on model type
        if "dpt" in model_name.lower():
            try:
                from transformers import DPTImageProcessor, DPTForDepthEstimation
                DPTImageProcessor.from_pretrained(hf_model_id)
                DPTForDepthEstimation.from_pretrained(hf_model_id)
            except ImportError as e:
                logger.error(f"❌ transformers not installed or DPT classes not available: {e}")
                return False
        elif "glpn" in model_name.lower():
            try:
                from transformers import GLPNImageProcessor, GLPNForDepthEstimation
                GLPNImageProcessor.from_pretrained(hf_model_id)
                GLPNForDepthEstimation.from_pretrained(hf_model_id)
            except ImportError as e:
                logger.error(f"❌ transformers not installed or GLPN classes not available: {e}")
                return False
        else:
            # Fallback to Auto classes for other models
            try:
                from transformers import AutoImageProcessor, AutoModel
                AutoImageProcessor.from_pretrained(hf_model_id)
                AutoModel.from_pretrained(hf_model_id)
            except ImportError as e:
                logger.error(f"❌ transformers not installed or Auto classes not available: {e}")
                return False

        logger.info(f"✅ {config['description']} cached successfully")
        return True

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
            raise GPUError(f"CUDA out of memory or other CUDA error during Hugging Face model caching: {e}", details={"model": model_name, "error_type": "CUDA_OOM"})
        else:
            raise ModelLoadingError(f"Failed to cache Hugging Face model {model_name}: {e}", details={"model": model_name})
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

        # Skip custom models that require manual upload
        if config.get("skip_download", False):
            filepath = models_dir / config["filename"]
            if filepath.exists():
                logger.info(f"✅ Custom model {model_name} found at {filepath}")
                results[model_name] = True
            else:
                logger.warning(f"⚠️ Custom model {model_name} not found. Please upload manually to {filepath}")
                results[model_name] = False
            continue

        # Route to appropriate download function
        try:
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
        except (GPUError, ModelLoadingError) as e:
            logger.error(f"❌ Failed to download {model_name} due to: {e}")
            results[model_name] = False
            raise # Re-raise the specific error
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while downloading {model_name}: {e}")
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

def get_model_with_fallback(model_name: str, auto_download: bool = True) -> Path:
    """
    Get model path with automatic fallback support and optional downloading.

    Args:
        model_name: Model identifier from MODEL_CONFIGS
        auto_download: Whether to attempt downloading missing models

    Returns:
        Path to available model file

    Raises:
        FileNotFoundError: If neither primary model nor fallback are available
        ValueError: If model_name is not in MODEL_CONFIGS
    """
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")

    # Try primary model first
    primary_path = get_model_path(model_name)
    if primary_path and primary_path.exists():
        logger.info(f"✅ Using primary model: {model_name} at {primary_path}")
        return primary_path

    # If primary model is missing and auto_download is enabled, try downloading
    if auto_download and not config.get("skip_download", False):
        logger.info(f"🔄 Attempting to download missing model: {model_name}")
        models_dir = ensure_models_directory()
        try:
            download_results = download_all_models(models_dir, [model_name])
        except (GPUError, ModelLoadingError) as e:
            logger.error(f"❌ Failed to download {model_name} due to: {e}")
            raise # Re-raise the specific error

        if download_results.get(model_name, False):
            # Re-check if download was successful (refresh path)
            primary_path = get_model_path(model_name)
            if primary_path and primary_path.exists():
                logger.info(f"✅ Downloaded and using: {model_name} at {primary_path}")
                return primary_path

    # Try fallback model
    fallback_name = config.get("fallback")
    if fallback_name:
        logger.warning(f"⚠️ Primary model {model_name} not available, trying fallback: {fallback_name}")

        fallback_config = MODEL_CONFIGS.get(fallback_name)
        if not fallback_config:
            logger.error(f"❌ Fallback model {fallback_name} not found in configs")
        else:
            fallback_path = get_model_path(fallback_name)
            if fallback_path and fallback_path.exists():
                logger.info(f"✅ Using fallback model: {fallback_name} at {fallback_path}")
                return fallback_path

            # Try downloading fallback if auto_download is enabled
            if auto_download and not fallback_config.get("skip_download", False):
                logger.info(f"🔄 Attempting to download fallback model: {fallback_name}")
                models_dir = ensure_models_directory()
                try:
                    fallback_results = download_all_models(models_dir, [fallback_name])
                except (GPUError, ModelLoadingError) as e:
                    logger.error(f"❌ Failed to download fallback {fallback_name} due to: {e}")
                    raise # Re-raise the specific error

                if fallback_results.get(fallback_name, False):
                    # Re-check if fallback download was successful
                    fallback_path = get_model_path(fallback_name)
                    if fallback_path and fallback_path.exists():
                        logger.info(f"✅ Downloaded and using fallback: {fallback_name} at {fallback_path}")
                        return fallback_path

    # No models available
    error_msg = f"Neither {model_name} nor its fallback"
    if fallback_name:
        error_msg += f" ({fallback_name})"
    error_msg += " are available"

    if config.get("skip_download", False):
        error_msg += f". {model_name} requires manual upload to {primary_path}"

    logger.error(f"❌ {error_msg}")
    raise FileNotFoundError(error_msg)

def resolve_model_config(model_name: str, auto_download: bool = True) -> Dict[str, str]:
    """
    Resolve model configuration with fallback, returning the actual model info to use.

    Args:
        model_name: Model identifier from MODEL_CONFIGS
        auto_download: Whether to attempt downloading missing models

    Returns:
        Dictionary with 'name', 'path', and 'is_fallback' keys

    Raises:
        FileNotFoundError: If neither primary model nor fallback are available
        ValueError: If model_name is not in MODEL_CONFIGS
    """
    try:
        # Try to get primary model
        model_path = get_model_with_fallback(model_name, auto_download)
        primary_path = get_model_path(model_name)

        # Determine if we're using fallback
        is_fallback = primary_path != model_path
        actual_name = model_name

        if is_fallback:
            # Find which fallback model we're actually using
            fallback_name = MODEL_CONFIGS[model_name].get("fallback")
            if fallback_name and get_model_path(fallback_name) == model_path:
                actual_name = fallback_name

        return {
            "name": actual_name,
            "path": str(model_path),
            "is_fallback": is_fallback,
            "original_name": model_name
        }

    except (FileNotFoundError, ValueError, GPUError, ModelLoadingError) as e:
        logger.error(f"❌ Could not resolve model config for {model_name}: {e}")
        raise

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
    ensure_models_directory()

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
