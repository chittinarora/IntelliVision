import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import warnings
import logging

# Configure logging
logger = logging.getLogger("anpr.detector")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Silence PIL Exif warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

class LicensePlateDetector:
    """
    Robust YOLO-based license plate detector with:
    - Input validation to prevent resize errors
    - Model warmup for consistent performance
    - Error handling for inference failures
    - Aspect ratio filtering
    - Temporal freezing for stable detections
    """

    MIN_FRAME_DIM = 10  # Minimum frame dimension (width/height)
    MODEL_WARMUP_SIZE = (640, 640)  # Size for warmup inference

    def __init__(self, model_path: str, freeze_conf: float = 0.75):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        self.model = YOLO(model_path)
        self.freeze_conf = freeze_conf
        self._last_box = None  # store (x1,y1,x2,y2,score)

        # Warm up the model
        self._warmup_model()

    def _warmup_model(self):
        """Perform initial inference to initialize model weights"""
        try:
            dummy = np.zeros((*self.MODEL_WARMUP_SIZE, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.error(f"Model warmup failed: {str(e)}")

    @staticmethod
    def _valid_aspect(box):
        x1, y1, x2, y2, score = box
        w, h = x2 - x1, y2 - y1
        return h > 0 and (w / h) >= 2.0 and (w / h) <= 6.0

    def _validate_frame(self, frame):
        """Ensure frame meets minimum requirements for processing"""
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame")
            return False

        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        if h < self.MIN_FRAME_DIM or w < self.MIN_FRAME_DIM:
            logger.warning(f"Frame too small: {w}x{h} (min {self.MIN_FRAME_DIM}px)")
            return False

        return True

    def detect_plates(self, frame, conf: float = 0.4, iou: float = 0.45, classes: list = []):
        # Frame validation
        if not self._validate_frame(frame):
            return []

        h, w = frame.shape[:2]

        # Perform inference with error handling
        try:
            results = self.model(frame, conf=conf, iou=iou, verbose=False)[0]
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
        good = [b for b in candidates if self._valid_aspect(b)]
        use = good if good else candidates

        if not use:
            # If previously locked, reuse
            if self._last_box and self._last_box[4] >= self.freeze_conf:
                return [self._last_box]
            return []

        # Pick best detection
        best = max(use, key=lambda x: x[4])

        # Update last box if high confidence
        if best[4] >= self.freeze_conf:
            self._last_box = best
            return [best]

        # If previously locked
        if self._last_box and self._last_box[4] >= self.freeze_conf:
            return [self._last_box]

        return [best]

    @staticmethod
    def annotate(frame, detections, text: str = "", font_scale: float = 1.2, thickness: int = 2):
        for (x1, y1, x2, y2, score) in detections:
            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = text if text else f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # background for text
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    def detect_image(self, image_path: str, output_path: str = "", conf: float = 0.4, iou: float = 0.45):
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
        out_file = (Path(output_path) if output_path else in_path.parent / f"annotated_{in_path.stem}.jpg")

        # Save via PIL
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(str(out_file), format="JPEG")
        except Exception as e:
            logger.error(f"Failed to save annotated image: {str(e)}")
            return "", []

        results = [{"bbox": [x1, y1, x2, y2], "score": round(score, 2)} for x1, y1, x2, y2, score in dets]
        return str(out_file), results

if __name__ == "__main__":
    det = LicensePlateDetector("models/best.pt", freeze_conf=0.75)
    out, res = det.detect_image("data/samples/car1.jpg")
    if out:
        print("Saved annotated image to", out)
        print(res)
    else:
        print("Detection failed")
