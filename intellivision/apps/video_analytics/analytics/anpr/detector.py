import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import warnings
import logging
import time
import torch

# Import CUDA error handling
try:
    from ...exception_handlers import GPUError, ModelLoadingError
except ImportError:
    # Fallback for standalone usage
    class GPUError(Exception):
        pass
    class ModelLoadingError(Exception):
        pass

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Silence PIL Exif warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

class LicensePlateDetector:
    """
    Enhanced YOLO-based license plate detector optimized for local development:
    - Automatic GPU detection
    - Simplified device selection
    - Improved resource efficiency
    - Robust error handling
    - Cleaned up Jetson-specific code
    """

    MIN_FRAME_DIM = 10  # Minimum frame dimension (width/height)
    MODEL_WARMUP_SIZE = (640, 640)  # Size for warmup inference
    VALID_ASPECT_RATIOS = (1.5, 8.0)  # Min/max width/height ratios for valid plates

    def __init__(self, model_path: str, freeze_conf: float = 0.75):
        """
        Initialize the detector with model path and configuration

        Args:
            model_path: Path to YOLO model file
            freeze_conf: Confidence threshold to lock detections
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        # Auto-select device with proper error handling
        device = self._get_optimal_device()
        logger.info(f"Selected device: {device}")

        try:
            self.model = YOLO(model_path)
            self._move_model_to_device(device)
            logger.info(f"Model loaded on {device.upper()}")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                logger.error(f"CUDA error during model initialization: {e}")
                # Try CPU fallback
                try:
                    logger.warning("Falling back to CPU due to CUDA error")
                    self.model = YOLO(model_path)
                    self._move_model_to_device("cpu")
                    device = "cpu"
                    logger.info("Model successfully loaded on CPU fallback")
                except Exception as fallback_error:
                    self._cleanup_gpu_memory()
                    raise ModelLoadingError(f"Failed to initialize model on both GPU and CPU: {fallback_error}")
            else:
                self._cleanup_gpu_memory()
                raise ModelLoadingError(f"Model initialization failed: {e}")
        except Exception as e:
            self._cleanup_gpu_memory()
            raise ModelLoadingError(f"Could not initialize model: {e}")

        self.freeze_conf = freeze_conf
        self._last_box = None  # store (x1,y1,x2,y2,score)
        self._last_frame = None
        self._last_detections = []

        # Store device for later use
        self.device = device
        
        # Warm up the model
        self._warmup_model()
    
    def _get_optimal_device(self):
        """
        Get optimal device with proper error handling.
        
        Returns:
            str: Device string ("cuda:0" for CUDA, "mps", or "cpu")
        """
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Additional CUDA health check
                try:
                    torch.cuda.get_device_properties(0)
                    return "cuda:0"
                except Exception as cuda_error:
                    logger.warning(f"CUDA device check failed: {cuda_error}, falling back")
        except Exception as e:
            logger.warning(f"CUDA availability check failed: {e}")
        
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception as e:
            logger.warning(f"MPS availability check failed: {e}")
        
        return "cpu"
    
    def _move_model_to_device(self, device):
        """
        Move model to device with proper error handling.
        
        Args:
            device: Target device string
        """
        try:
            self.model.to(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                self._cleanup_gpu_memory()
                raise GPUError(f"CUDA error when moving model to device {device}: {e}")
            else:
                raise ModelLoadingError(f"Failed to move model to device {device}: {e}")
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory to recover from errors."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleaned up")
        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")
    
    def _is_gpu_available(self):
        """
        Safely check if GPU is available and being used.
        
        Returns:
            bool: True if GPU is available and model is on GPU
        """
        try:
            return (hasattr(self, 'device') and 
                    self.device not in ["cpu"] and 
                    self.device.startswith("cuda") and
                    torch.cuda.is_available())
        except Exception as e:
            logger.warning(f"GPU availability check failed: {e}")
            return False

    def _warmup_model(self):
        """Perform initial inference to initialize model weights"""
        try:
            dummy = np.zeros((*self.MODEL_WARMUP_SIZE, 3), dtype=np.uint8)
            start = time.perf_counter()
            self.model(dummy, verbose=False)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Model warmup completed in {elapsed:.1f}ms")
        except Exception as e:
            logger.error(f"Model warmup failed: {str(e)}")

    def _valid_aspect(self, box):
        """Validate plate aspect ratio"""
        x1, y1, x2, y2, score = box
        w, h = x2 - x1, y2 - y1
        aspect = w / max(h, 1e-5)  # Prevent division by zero
        return self.VALID_ASPECT_RATIOS[0] <= aspect <= self.VALID_ASPECT_RATIOS[1]

    def _validate_frame(self, frame):
        """Ensure frame meets minimum requirements for processing"""
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame")
            return False

        # Convert grayscale to BGR
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        if h < self.MIN_FRAME_DIM or w < self.MIN_FRAME_DIM:
            logger.warning(f"Frame too small: {w}x{h} (min {self.MIN_FRAME_DIM}px)")
            return False

        return True

    def detect_plates(self, frame, conf: float = 0.4, iou: float = 0.45, classes: list = None):
        """
        Detect license plates in a frame

        Args:
            frame: Input image (numpy array)
            conf: Minimum confidence threshold
            iou: IOU threshold for NMS
            classes: List of class IDs to consider

        Returns:
            List of detections as (x1, y1, x2, y2, score)
        """
        # Frame validation
        if not self._validate_frame(frame):
            return []

        # Use cached results if frame hasn't changed
        if np.array_equal(frame, self._last_frame):
            return self._last_detections

        h, w = frame.shape[:2]

        # Perform inference with error handling
        try:
            start = time.perf_counter()

            # Only use half-precision on GPU (with safe CUDA check)
            half = self._is_gpu_available()
            results = self.model(frame, conf=conf, iou=iou, verbose=False, half=half)[0]

            infer_time = (time.perf_counter() - start) * 1000
            logger.debug(f"Inference time: {infer_time:.1f}ms")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                logger.error(f"CUDA error during inference: {e}")
                self._cleanup_gpu_memory()
                # Try inference on CPU
                try:
                    logger.warning("Retrying inference on CPU")
                    results = self.model(frame, conf=conf, iou=iou, verbose=False, half=False)[0]
                except Exception as cpu_error:
                    logger.error(f"CPU fallback inference also failed: {cpu_error}")
                    return []
            else:
                logger.error(f"YOLO inference failed: {str(e)}")
                return []
        except Exception as e:
            logger.error(f"YOLO inference failed: {str(e)}")
            return []

        candidates = []
        if hasattr(results, 'boxes') and results.boxes is not None:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy().flatten()
            clss = (results.boxes.cls.cpu().numpy().flatten()
                    if hasattr(results.boxes, 'cls') else [None] * len(confs))

            for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, clss):
                if classes is not None and cls not in classes:
                    continue

                # Convert to integers and clamp to frame boundaries
                x1, y1 = int(max(0, x1)), int(max(0, y1))
                x2, y2 = int(min(w, x2)), int(min(h, y2))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Skip tiny boxes
                if (x2 - x1) < self.MIN_FRAME_DIM or (y2 - y1) < self.MIN_FRAME_DIM:
                    continue

                candidates.append((x1, y1, x2, y2, float(score)))

        # Filter by aspect ratio
        valid_candidates = [b for b in candidates if self._valid_aspect(b)]
        use = valid_candidates if valid_candidates else candidates

        # If no candidates, try to reuse last good detection
        if not use:
            if self._last_box and self._last_box[4] >= self.freeze_conf:
                self._last_frame = frame.copy()
                self._last_detections = [self._last_box]
                return [self._last_box]
            return []

        # Pick best detection
        best = max(use, key=lambda x: x[4])

        # Update last box if high confidence
        if best[4] >= self.freeze_conf:
            self._last_box = best

        # Cache results
        self._last_frame = frame.copy()
        self._last_detections = [best]
        return [best]

    @staticmethod
    def annotate(frame, detections, text: str = None, font_scale: float = 1.2, thickness: int = 2):
        """
        Annotate detections on frame

        Args:
            frame: Image to annotate
            detections: List of detections
            text: Custom text to display
            font_scale: Text size
            thickness: Line thickness
        """
        for (x1, y1, x2, y2, score) in detections:
            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = text if text is not None else f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Calculate text position (avoid top overflow)
            y_text = max(y1 - 4, th + 4)

            # background for text
            cv2.rectangle(frame,
                         (x1, y1 - th - 6),
                         (x1 + tw, y1),
                         (0, 255, 0), -1)

            cv2.putText(frame, label,
                       (x1, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale,
                       (255, 255, 255),
                       thickness)

    def detect_image(self, image_path: str, output_path: str = None, conf: float = 0.4, iou: float = 0.45):
        """
        Detect plates in an image file and save annotated result

        Args:
            image_path: Path to input image
            output_path: Optional output path
            conf: Confidence threshold
            iou: IOU threshold

        Returns:
            (output_path, detection_results)
        """
        # load via PIL to handle Exif
        try:
            pil = Image.open(image_path).convert("RGB")
            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            return "", []

        dets = self.detect_plates(frame, conf=conf, iou=iou)
        self.annotate(frame, dets)

        in_path = Path(image_path)
        out_file = (Path(output_path) if output_path
                   else in_path.parent / f"annotated_{in_path.stem}.jpg")

        # Save via PIL
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(str(out_file), format="JPEG")
            logger.info(f"Saved annotated image: {out_file}")
        except Exception as e:
            logger.error(f"Failed to save annotated image: {str(e)}")
            return "", []

        # Format results
        results = [{
            "bbox": [x1, y1, x2, y2],
            "score": round(score, 2),
            "width": x2 - x1,
            "height": y2 - y1
        } for x1, y1, x2, y2, score in dets]

        return str(out_file), results
