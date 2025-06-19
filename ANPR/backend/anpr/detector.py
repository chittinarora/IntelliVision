import torch
import torch.serialization
from ultralytics import YOLO
import os
import cv2

class LicensePlateDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        # Allow loading full model with class definitions (required for PyTorch 2.6+)
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

        self.model = YOLO(model_path, task='detect')

    def detect_plates(self, frame):
        results = self.model(frame)[0]
        boxes = []

        if results and results.boxes is not None:
            for box in results.boxes:
                if box.xyxy is not None and box.conf is not None:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    boxes.append((x1, y1, x2, y2, conf))

        return boxes

if __name__ == "__main__":
    detector = LicensePlateDetector("models/best.pt")
    image = cv2.imread("data/samples/car1.jpg")

    if image is None:
        raise FileNotFoundError("Input image not found or unreadable")

    detections = detector.detect_plates(image)

    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Detected Plates", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

