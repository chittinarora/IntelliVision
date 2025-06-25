import os
import cv2
import torch
import torch.serialization
from ultralytics import YOLO

class LicensePlateDetector:
    """
    YOLO-based license plate detector with simple temporal freezing and robust channel checks.
    """
    def __init__(self, model_path: str, freeze_conf: float = 0.75):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        # Ensure safe deserialization for newer PyTorch
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

        # Load the YOLO detection model
        self.model = YOLO(model_path, task='detect')

        # Confidence threshold above which we "freeze" the last good detection
        self.freeze_conf = freeze_conf
        self._last_box = None   # (x1, y1, x2, y2, conf)

    def detect_plates(self, frame, conf: float = 0.4, iou: float = 0.45, classes: list = None):
        """
        Run inference on a frame and optionally freeze to the last high-confidence box.

        Args:
            frame: BGR or grayscale image
            conf: detection confidence threshold
            iou:  NMS IOU threshold
            classes: filter by YOLO class IDs
        Returns:
            List of (x1, y1, x2, y2, score)
        """
        # handle single-channel inputs by converting to BGR
        if frame is None:
            return []
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return []

        # perform detection
        try:
            results = self.model(frame, conf=conf, iou=iou, classes=classes)[0]
        except Exception:
            # on any inference error, fall back to last box if frozen
            return [self._last_box] if self._last_box is not None else []

        candidates = []
        if results and hasattr(results, 'boxes') and results.boxes is not None:
            for box in results.boxes:
                if box.xyxy is None or box.conf is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                score = float(box.conf[0])
                candidates.append((x1, y1, x2, y2, score))

        # select highest-confidence detection
        best = max(candidates, key=lambda x: x[4], default=None)
        if best:
            # if above freeze threshold, update last and return it
            if best[4] >= self.freeze_conf:
                self._last_box = best
                return [best]
            # if we have a frozen detection, keep returning it
            if self._last_box is not None and self._last_box[4] >= self.freeze_conf:
                return [self._last_box]
            # otherwise return all candidates
            return candidates
        else:
            # no new detections, fall back to freeze if set
            return [self._last_box] if self._last_box is not None else []

    @staticmethod
    def annotate(frame, detections, text=None, font_scale: float = 1.5, thickness: int = 2):
        """
        Draw bounding boxes and optional text annotation at larger font.

        Args:
            frame:     BGR image
            detections: list of (x1,y1,x2,y2,score)
            text:      override label text
            font_scale: scale factor for text
            thickness:  text thickness
        """
        for (x1, y1, x2, y2, score) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = text or f"{score:.2f}"
            # draw label background
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - h_text - 6), (x1 + w_text, y1), (0, 255, 0), -1)
            # draw label text
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

if __name__ == "__main__":
    detector = LicensePlateDetector("models/best.pt", freeze_conf=0.75)
    image = cv2.imread("data/samples/car1.jpg")
    if image is None:
        raise FileNotFoundError("Input image not found at data/samples/car1.jpg")
    dets = detector.detect_plates(image, conf=0.5, iou=0.5)
    LicensePlateDetector.annotate(image, dets)
    cv2.imshow("Detected Plates", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

