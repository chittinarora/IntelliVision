import os
import cv2
import torch
import torch.serialization
from ultralytics import YOLO

class LicensePlateDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        # Allow loading full model with class definitions (required for PyTorch 2.6+)
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

        # Load the YOLO model
        self.model = YOLO(model_path, task='detect')

    def detect_plates(self, frame, classes=None):
        """
        Run inference on a single frame and return bounding boxes with confidences.
        
        Args:
            frame:      OpenCV BGR image
            classes:    Optional list of COCO class IDs to keep (e.g. [2] for cars)
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples.
        """
        # Run the model, possibly filtering by classes
        results = self.model(
            frame,
            conf=0.4,
            iou=0.45,
            classes=classes    # pass [2] here to only detect cars
        )[0]

        boxes = []
        if results and results.boxes is not None:
            for box in results.boxes:
                if box.xyxy is not None and box.conf is not None:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    boxes.append((x1, y1, x2, y2, conf))

        return boxes


if __name__ == "__main__":
    # Example usage for license plates (no class filter)
    plate_detector = LicensePlateDetector("models/best.pt")
    image = cv2.imread("data/samples/car1.jpg")
    detections = plate_detector.detect_plates(image)
    print("Plate detections:", detections)

    # Example usage for cars only (class 2)
    car_detector = LicensePlateDetector("models/yolo11m.pt")
    car_boxes = car_detector.detect_plates(image, classes=[2])
    print("Car detections:", car_boxes)

