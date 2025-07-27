# /apps/video_analytics/utils.py

"""
Utilities for video analytics app, including YOLO model loading.
"""

from ultralytics import YOLO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_yolo_model(model_path: str) -> YOLO:
    """
    Load a YOLO model, downloading from Ultralytics Hub if not present,
    except for specific models that must exist locally.

    Args:
        model_path: Path to the model file

    Returns:
        YOLO: Loaded YOLO model

    Raises:
        FileNotFoundError: If required model is missing
    """
    basename = Path(model_path).name
    must_exist = {"best_plate.pt", "yolo11m.pt", "best_animal.pt"}

    if basename in must_exist:
        if not Path(model_path).exists():
            logger.error(f"Model {model_path} must exist locally")
            raise FileNotFoundError(f"Model {model_path} must be present locally")
        logger.info(f"Loaded local model: {model_path}")
        return YOLO(str(model_path))

    if Path(model_path).exists():
        logger.info(f"Loaded local model: {model_path}")
        return YOLO(str(model_path))

    logger.info(f"Downloading model: {basename}")
    return YOLO(basename)
