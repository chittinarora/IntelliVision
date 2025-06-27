from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import logging

# Configure logging
logger = logging.getLogger("anpr.tracker")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class VehicleTracker:
    """
    Robust vehicle tracker with:
    - Bounding box validation to prevent negative sizes
    - Coordinate clamping to frame boundaries
    - Minimum size enforcement for tracked objects
    - Error handling for tracker operations
    """
    
    MIN_BOX_WIDTH = 20  # Minimum width for tracked boxes
    MIN_BOX_HEIGHT = 20  # Minimum height for tracked boxes

    def __init__(self, max_age: int = 30, n_init: int = 3):
        try:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                nms_max_overlap=1.0,
                embedder='mobilenet',  # More efficient than original
                half=True  # Use FP16 for faster processing
            )
            logger.info(f"Tracker initialized (max_age={max_age}, n_init={n_init})")
        except Exception as e:
            logger.error(f"Tracker initialization failed: {str(e)}")
            raise RuntimeError(f"Could not initialize tracker: {str(e)}")
            
        self.unique_ids = set()
        self.frame_width = 0
        self.frame_height = 0

    def update(self, detections, frame):
        # Store frame dimensions for clamping
        if frame is not None:
            self.frame_height, self.frame_width = frame.shape[:2]
        else:
            # Use previous dimensions if available
            if not self.frame_width or not self.frame_height:
                logger.warning("No frame dimensions available for clamping")
                return []
        
        formatted = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            
            # Clamp coordinates to frame boundaries
            x1 = max(0, min(x1, self.frame_width - 1))
            y1 = max(0, min(y1, self.frame_height - 1))
            x2 = max(0, min(x2, self.frame_width - 1))
            y2 = max(0, min(y2, self.frame_height - 1))
            
            # Calculate width and height
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            
            # Skip detections that are too small
            if w < self.MIN_BOX_WIDTH or h < self.MIN_BOX_HEIGHT:
                continue
                
            formatted.append(([x1, y1, w, h], conf, cls))

        try:
            tracks = self.tracker.update_tracks(formatted, frame=frame)
        except Exception as e:
            logger.error(f"Tracker update failed: {str(e)}")
            return []
            
        updated = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            self.unique_ids.add(track_id)
            
            try:
                l, t, r, b = track.to_ltrb()
            except Exception as e:
                logger.warning(f"Failed to get track coordinates: {str(e)}")
                continue
                
            # Clamp coordinates to frame boundaries
            l = max(0, min(l, self.frame_width - 1))
            t = max(0, min(t, self.frame_height - 1))
            r = max(0, min(r, self.frame_width - 1))
            b = max(0, min(b, self.frame_height - 1))
            
            # Ensure minimum size
            r = max(l + self.MIN_BOX_WIDTH, r)
            b = max(t + self.MIN_BOX_HEIGHT, b)
            
            # Skip boxes that are still too small after clamping
            if (r - l) < self.MIN_BOX_WIDTH or (b - t) < self.MIN_BOX_HEIGHT:
                continue
                
            updated.append({
                'track_id': track_id,
                'bbox': (int(l), int(t), int(r), int(b))
            })

        return updated

    def get_total_unique_ids(self):
        return len(self.unique_ids)

    def reset(self):
        """Reset tracker state for a new video"""
        logger.info("Resetting tracker state")
        try:
            self.tracker = DeepSort(
                max_age=40, 
                n_init=3, 
                nms_max_overlap=0.7,
                embedder='mobilenet',
                half=True
            )
            self.unique_ids.clear()
        except Exception as e:
            logger.error(f"Tracker reset failed: {str(e)}")
            # Try to reinitialize with default parameters
            try:
                self.tracker = DeepSort(
                    max_age=40,
                    n_init=3,
                    nms_max_overlap=0.7
                )
            except Exception as e2:
                logger.critical(f"Complete tracker reinitialization failed: {str(e2)}")
                raise RuntimeError("Could not reinitialize tracker after reset")
