from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import logging
import time
import os
import torch

# Configure logging
logger = logging.getLogger("anpr.tracker")

def get_optimal_device():
    """
    Get optimal device for tracking, avoiding import-time CUDA initialization.
    
    Returns:
        str: Device string ("0" for CUDA, "mps", or "cpu")
    """
    try:
        if torch.cuda.is_available():
            return "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except Exception as e:
        logger.warning(f"Device detection failed: {e}, defaulting to CPU")
        return "cpu"
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class VehicleTracker:
    """
    Enhanced vehicle tracker with:
    - GPU acceleration support
    - Adaptive bounding box validation
    - Performance monitoring
    - Resource-efficient configuration
    - Robust error recovery
    - Frame dimension caching
    """

    MIN_BOX_WIDTH = 20  # Minimum width for tracked boxes
    MIN_BOX_HEIGHT = 20  # Minimum height for tracked boxes
    MAX_BOX_ASPECT = 8.0  # Maximum width/height aspect ratio
    MIN_CONFIDENCE = 0.1  # Minimum detection confidence

    def __init__(self, max_age: int = 30, n_init: int = 3, device: str = "auto"):
        """
        Initialize the tracker

        Args:
            max_age: Maximum frames to keep track without detection
            n_init: Number of detections before track is confirmed
            device: "cpu", "cuda" or "auto" for automatic selection
        """
        # Auto-select device if requested
        if device == "auto":
            device = get_optimal_device()
            logger.info(f"Auto-selected device: {device}")

        try:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                nms_max_overlap=1.0,
                embedder='mobilenet',  # Efficient mobile-friendly model
                half=True              # FP16 for faster processing
            )
            logger.info(f"Tracker initialized (max_age={max_age}, n_init={n_init})")
            # Log actual device used by DeepSort embedder/model
            embedder_device = None
            if hasattr(self.tracker, "model") and hasattr(self.tracker.model, "parameters"):
                try:
                    embedder_device = next(self.tracker.model.parameters()).device
                except Exception:
                    embedder_device = "unknown"
            logger.info(f"DeepSort embedder actual device: {embedder_device}")
        except Exception as e:
            logger.error(f"Tracker initialization failed: {str(e)}")
            # Fallback to CPU mode
            try:
                self.tracker = DeepSort(
                    max_age=max_age,
                    n_init=n_init,
                    nms_max_overlap=1.0,
                    embedder='mobilenet'
                )
                logger.warning("Tracker fallback to CPU mode")
            except Exception as e2:
                logger.critical(f"Complete tracker initialization failed: {str(e2)}")
                raise RuntimeError(f"Could not initialize tracker: {str(e2)}")

        self.unique_ids = set()
        self.frame_width = 0
        self.frame_height = 0
        self.track_count = 0
        self.last_update_time = time.perf_counter()

    def validate_bbox(self, x1: float, y1: float, x2: float, y2: float):
        """Validate and normalize bounding box coordinates"""
        # Ensure coordinates are within frame boundaries
        x1 = max(0, min(x1, self.frame_width - 1))
        y1 = max(0, min(y1, self.frame_height - 1))
        x2 = max(0, min(x2, self.frame_width - 1))
        y2 = max(0, min(y2, self.frame_height - 1))

        # Calculate dimensions
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        # Check minimum size
        if w < self.MIN_BOX_WIDTH or h < self.MIN_BOX_HEIGHT:
            return None

        # Check aspect ratio
        aspect = w / max(h, 1e-5)  # Prevent division by zero
        if aspect > self.MAX_BOX_ASPECT:
            return None

        return [x1, y1, w, h]

    def update(self, detections, frame):
        """Update tracker with new detections and frame"""
        start_time = time.perf_counter()

        # Store frame dimensions for clamping
        if frame is not None:
            self.frame_height, self.frame_width = frame.shape[:2]
        elif not self.frame_width or not self.frame_height:
            logger.warning("No frame dimensions available for clamping")
            return []

        formatted = []
        for det in detections:
            # Unpack detection (bbox: [x1, y1, w, h], conf, class_id)
            bbox, conf, cls_id = det

            # Skip low-confidence detections
            if conf < self.MIN_CONFIDENCE:
                continue

            # Validate and normalize bounding box
            x1, y1, w, h = bbox
            valid_bbox = self.validate_bbox(x1, y1, x1 + w, y1 + h)
            if not valid_bbox:
                continue

            formatted.append((valid_bbox, conf, cls_id))

        try:
            tracks = self.tracker.update_tracks(formatted, frame=frame)
        except Exception as e:
            logger.error(f"Tracker update failed: {str(e)}")
            # Attempt to recover tracker
            try:
                self.reset()
                tracks = self.tracker.update_tracks(formatted, frame=frame)
                logger.warning("Tracker recovered after reset")
            except Exception as e2:
                logger.critical(f"Tracker recovery failed: {str(e2)}")
                return []

        updated = []
        current_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            current_ids.add(track_id)

            # Add to unique IDs if new
            if track_id not in self.unique_ids:
                self.unique_ids.add(track_id)

            try:
                l, t, r, b = track.to_ltrb()
            except Exception as e:
                logger.warning(f"Failed to get track coordinates: {str(e)}")
                continue

            # Validate final bounding box
            valid_bbox = self.validate_bbox(l, t, r, b)
            if not valid_bbox:
                continue

            l, t, w, h = valid_bbox
            r, b = l + w, t + h

            updated.append({
                'track_id': track_id,
                'bbox': (int(l), int(t), int(r), int(b))
            })

        # Update performance metrics
        self.track_count = len(updated)
        process_time = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Tracked {self.track_count} objects in {process_time:.1f}ms")

        return updated

    def get_total_unique_ids(self):
        """Get count of unique tracked objects"""
        return len(self.unique_ids)

    def reset(self):
        """Reset tracker state while preserving configuration"""
        logger.info("Resetting tracker state")
        try:
            # Save current configuration
            max_age = self.tracker.max_age
            n_init = self.tracker.n_init
            # device = self.tracker.device  # REMOVED: not supported

            # Reinitialize with same parameters
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                nms_max_overlap=1.0,
                embedder='mobilenet',
                half=True
            )

            # Reset state
            self.unique_ids.clear()
            self.track_count = 0
            logger.info("Tracker reset complete")
        except Exception as e:
            logger.error(f"Tracker reset failed: {str(e)}")
            # Try fallback initialization
            try:
                self.tracker = DeepSort(
                    max_age=40,
                    n_init=3,
                    nms_max_overlap=0.7
                )
                logger.warning("Tracker fallback initialization after reset failure")
            except Exception as e2:
                logger.critical(f"Complete tracker reinitialization failed: {str(e2)}")
                raise RuntimeError("Could not reinitialize tracker after reset")
