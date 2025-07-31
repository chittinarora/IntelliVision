# anpr/ocr.py

import logging
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import easyocr
import re
import warnings
from collections import deque, Counter

# suppress unwanted PIL/NumPy warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class PlateOCR:
    """
    OCR engine for license plates using a hybrid Tesseract+EasyOCR pipeline.
    Features:
      - Processes only the single highest-confidence crop per frame (selection in processor).
      - Confidence-based variant selection for Tesseract.
      - Frame buffering for stable output: returns majority from last N reads.
      - EasyOCR fallback.
    """
    def __init__(self,
                 tesseract_cmd: str = None,
                 whitelist: str = None,
                 tesseract_conf_thresh: float = 0.7,
                 lang_list: list = ['en'],
                 buffer_size: int = 5,
                 gpu_enabled: bool = None):
        # configure tesseract binary
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.whitelist = whitelist or "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.tesseract_conf_thresh = tesseract_conf_thresh
        
        # Auto-detect GPU if not specified
        if gpu_enabled is None:
            gpu_enabled = self._should_use_gpu()
        
        logger.info(f"Initializing EasyOCR with GPU={'enabled' if gpu_enabled else 'disabled'}")
        try:
            self.reader = easyocr.Reader(lang_list, gpu=gpu_enabled, verbose=False)
        except Exception as e:
            if gpu_enabled:
                logger.warning(f"Failed to initialize EasyOCR with GPU: {e}, falling back to CPU")
                self.reader = easyocr.Reader(lang_list, gpu=False, verbose=False)
            else:
                raise
        # buffer for smoothing
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
    
    def _should_use_gpu(self):
        """
        Check if GPU should be used for EasyOCR.
        
        Returns:
            bool: True if GPU is available and recommended
        """
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            # torch not available, EasyOCR might still support GPU via other backends
            logger.info("PyTorch not available, EasyOCR GPU detection limited")
            return False
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return False

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype='uint8')
        return cv2.LUT(image, table)

    def _unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        return cv2.addWeighted(image, 1.5, blur, -0.5, 0)

    def _generate_variants(self, img: np.ndarray) -> list:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base = cv2.bilateralFilter(gray, 9, 75, 75)
        variants = [base]
        for g in (1.2, 1.5, 2.0):
            variants.append(self._gamma_correction(base, g))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        variants.append(clahe.apply(base))
        variants.append(self._unsharp_mask(base))
        _, otsu = cv2.threshold(variants[-1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.extend([otsu, cv2.bitwise_not(otsu)])
        h, w = base.shape
        up = cv2.resize(base, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        variants.extend([up])
        adapt = cv2.adaptiveThreshold(
            up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3
        )
        variants.extend([adapt, cv2.bitwise_not(adapt)])
        # dedupe
        seen = set(); unique = []
        for v in variants:
            key = (v.shape, int(v.sum()))
            if key not in seen:
                seen.add(key); unique.append(v)
        return unique

    def _tesseract_read(self, img: np.ndarray) -> (str, float):
        best_text, best_conf = "", 0.0
        config = f"--psm 8 --oem 3 -c tessedit_char_whitelist={self.whitelist}"
        data = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
        for i, txt in enumerate(data['text']):
            raw = txt.strip()
            if not raw: continue
            try:
                conf = float(data['conf'][i]) / 100.0
            except Exception as e:
                logger.debug(f"OCR processing failed for plate {i}: {e}")
                continue
            if conf > best_conf:
                clean = re.sub(r'[^A-Za-z0-9]', '', raw).upper()
                if clean:
                    best_conf = conf; best_text = clean
        return best_text, best_conf

    def _easyocr_read(self, img: np.ndarray) -> (str, float):
        results = self.reader.readtext(img, detail=1, paragraph=False)
        if not results: return "", 0.0
        best = max(results, key=lambda x: x[2])
        text = re.sub(r'[^A-Za-z0-9]', '', best[1]).upper()
        return text, float(best[2])

    def read_plate(self, img: np.ndarray) -> (str, float):
        if img is None or img.size == 0:
            return "", 0.0
        # Tesseract on variants
        t_best, t_conf = "", 0.0
        for var in self._generate_variants(img):
            txt, conf = self._tesseract_read(var)
            if conf > t_conf:
                t_best, t_conf = txt, conf
        # Decide final
        if t_conf >= self.tesseract_conf_thresh:
            final, fconf = t_best, t_conf
        else:
            e_best, e_conf = self._easyocr_read(img)
            final, fconf = (e_best, e_conf) if e_conf > t_conf else (t_best, t_conf)
        # Buffer smoothing
        self.buffer.append(final)
        if len(self.buffer) == self.buffer_size:
            most = Counter(self.buffer).most_common(1)[0][0]
            return most, fconf
        return final, fconf

    def read_plate_from_path(self, path: str) -> (str, float):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return self.read_plate(img)

if __name__ == '__main__':
    import sys
    ocr = PlateOCR()
    for p in sys.argv[1:]:
        txt, conf = ocr.read_plate_from_path(p)
        logger.debug(f"OCR result: {p} -> {txt} (conf={conf:.2f})")
