import easyocr
import cv2
import numpy as np

class PlateOCR:
    def __init__(self, use_gpu=False):
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def read_plate(self, image):
        """
        Extract text from a cropped license plate image.
        Returns the highest confidence text prediction.
        """
        if image is None or image.size == 0:
            return "", 0.0

        # Convert to RGB if needed
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        results = self.reader.readtext(image)

        if not results:
            return "", 0.0

        # Select the best result by confidence
        best_result = max(results, key=lambda r: r[2])
        text, conf = best_result[1], best_result[2]
        return text, conf

# Test usage
if __name__ == "__main__":
    ocr = PlateOCR()
    test_image = cv2.imread("data/samples/cropped_plate.jpg")
    text, conf = ocr.read_plate(test_image)
    print(f"Detected Text: {text} (Confidence: {conf:.2f})")
