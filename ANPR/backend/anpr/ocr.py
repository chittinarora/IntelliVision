import cv2
import numpy as np
import easyocr
import re

class PlateOCR:
    """
    A more robust EasyOCR wrapper for license plates:
    - grayscale → bilateral filter → Otsu threshold → Closing
    - upscale ×2 → invert if needed
    - whitelist to A–Z,0–9 only
    - return only the most confident text
    """
    def __init__(self, gpu: bool = False):
        # Initialize EasyOCR reader with English and alphanumeric only
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        # Allowed characters
        self.allowed_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        # compile a regex to strip anything else
        self._cleanup = re.compile(f'[^{self.allowed_chars}]')

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        # to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # denoise
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        # Otsu threshold
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # closing to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        # upscale for better OCR
        h, w = closed.shape
        closed = cv2.resize(closed, (w*2, h*2),
                            interpolation=cv2.INTER_CUBIC)
        # if background is light, invert
        if np.mean(closed) > 127:
            closed = cv2.bitwise_not(closed)
        return closed

    def read_plate(self, img: np.ndarray) -> (str, float):
        """
        Run OCR on a cropped plate image and return cleaned text + confidence.

        Args:
            img: BGR image of the plate region
        Returns:
            (text, confidence) or ("", 0.0) if nothing reliable detected
        """
        if img is None or img.size == 0:
            return "", 0.0

        proc = self._preprocess(img)
        # run reader, restrict to allowed chars
        try:
            results = self.reader.readtext(
                proc,
                allowlist=self.allowed_chars,
                detail=1,
                paragraph=False
            )
        except Exception:
            return "", 0.0

        if not results:
            return "", 0.0

        # collect all hypotheses
        candidates = []
        for bbox, text, conf in results:
            cleaned = self._cleanup.sub('', text.upper())
            if len(cleaned) >= 4:  # heuristic: plates are minimum length
                candidates.append((cleaned, float(conf)))

        if not candidates:
            return "", 0.0

        # pick the best confidence
        best_text, best_conf = max(candidates, key=lambda x: x[1])
        return best_text, best_conf

