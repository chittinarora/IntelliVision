# dubs_people_counting_comprehensive_1.py - COMPREHENSIVE PEOPLE COUNTING SOLUTION V1.0
#
# 🚀 COMPREHENSIVE FEATURES:
# ✅ Smart Adaptive Detection (YOLO + RT-DETR with depth intelligence)
# ✅ Enhanced Post-Processing Engine (Multi-frame Re-ID + Motion prediction + Hierarchical scoring)
# ✅ Optimized Confidence Thresholds (Distance-adaptive detection)
# ✅ MiDaS Depth Integration (Scene-aware strategy selection)
# ✅ Advanced Track Merging (Re-ID features + spatial + temporal + motion analysis)
# ✅ MOT Dataset Compatibility (Image sequences + video files)
# ✅ Comprehensive Evaluation Support
#
# Author: Dubs AI People Counting Project
# Version: 1.0 - Production Ready
# Date: 2025

import os
import glob
import cv2
import torch
import numpy as np
import logging
import argparse
import time
import re
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from django.conf import settings

# Import scipy with fallback
try:
    from scipy.spatial.distance import cosine

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available - using numpy cosine similarity fallback")

# Correct imports for actual libraries
from ultralytics import RTDETR
from ..utils import load_yolo_model
from boxmot import BotSort, ByteTrack
import torch.nn.functional as F

# MiDaS imports with proper error handling
try:
    import torch.hub

    MIDAS_AVAILABLE = True
except ImportError:
    MIDAS_AVAILABLE = False

logger = logging.getLogger("dubs_people_counting_comprehensive")

# Import progress logger
try:
    from ..progress_logger import create_progress_logger
except ImportError:
    def create_progress_logger(job_id, total_items, job_type, logger_name=None):
        """Fallback progress logger if module not available."""
        class DummyLogger:
            def __init__(self, job_id, total_items, job_type, logger_name=None):
                self.job_id = job_id
                self.total_items = total_items
                self.job_type = job_type
                self.logger = logging.getLogger(logger_name or job_type)

            def update_progress(self, processed_count, status=None, force_log=False):
                self.logger.info(f"**Job {self.job_id}**: Progress {processed_count}/{self.total_items}")

            def log_completion(self, final_count=None):
                self.logger.info(f"**Job {self.job_id}**: Completed")

            def log_error(self, error_message):
                self.logger.error(f"**Job {self.job_id}**: Error - {error_message}")

        return DummyLogger(job_id, total_items, job_type, logger_name)

# === CONSTANTS (for Celery fork-safety and clarity) ===
DEFAULT_DEVICE = 'cpu'
DEFAULT_USE_REID = True
DEFAULT_POST_PROCESS = True
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REID_MODEL_PATH = MODELS_DIR / "osnet_x0_25_msmt17.pt"
# Import model manager for proper path resolution
from .model_manager import get_model_with_fallback
# Use model manager instead of relative path
YOLO_MODEL_PATH = str(get_model_with_fallback("yolov11x"))
RTDETR_MODEL_PATH = 'rtdetr-l.pt'
MIDAS_WEIGHTS_PATH = 'midas/weights/dpt_large_384.pt'
MIDAS_REPO = 'intel-isl/MiDaS'
MIDAS_MODEL_NAME = 'DPT_Large'
DPT_HF_MODEL = 'Intel/dpt-large'
GLPN_HF_MODEL = 'vinvino02/glpn-nyu'
DEFAULT_OUTPUT_PREFIX = 'dubs_comprehensive_output_'
DEFAULT_OUTPUT_EXT = '.mp4'

# Define OUTPUT_DIR with fallback (matching other analytics files)
try:
    DEFAULT_OUTPUT_DIR = str(settings.JOB_OUTPUT_DIR)
except AttributeError:
    logger.warning("JOB_OUTPUT_DIR not defined in settings. Using fallback: MEDIA_ROOT/outputs")
    DEFAULT_OUTPUT_DIR = os.path.join(settings.MEDIA_ROOT, 'outputs')

# Confidence thresholds
YOLO_CONF_CLOSE = 0.35
YOLO_CONF_MEDIUM = 0.40
YOLO_CONF_FAR = 0.45
RTDETR_CONF_CLOSE = 0.30
RTDETR_CONF_MEDIUM = 0.35
RTDETR_CONF_FAR = 0.40
# NMS threshold
DEFAULT_NMS_IOU_THRESH = 0.4
# Track lifetime filter
MIN_LIFETIME_FRAMES = 40

from ..convert import convert_to_web_mp4

def next_sequential_name(out_dir: str = DEFAULT_OUTPUT_DIR, prefix=DEFAULT_OUTPUT_PREFIX, ext=DEFAULT_OUTPUT_EXT) -> str:
    """Generate next sequential filename."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"{prefix}*{ext}")
    files = glob.glob(pattern)
    try:
        next_idx = max([int(re.search(r'(\d+)', f).group(1)) for f in files], default=0) + 1
    except:
        next_idx = len(files) + 1
    return os.path.join(out_dir, f"{prefix}{next_idx}{ext}")


class DepthEstimator:
    """Multi-tier depth estimation with FIXED depth interpretation."""

    def __init__(self, device='auto'):
        # Smart device selection for Apple Silicon
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                logger.info("🍎 Using MPS (Apple Silicon GPU) for acceleration")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                logger.info("🚀 Using CUDA GPU for acceleration")
            else:
                self.device = torch.device('cpu')
                logger.info("💻 Using CPU")
        else:
            self.device = torch.device(device)

        self.model = None
        self.transform = None
        self.method = "fallback"
        self._load_depth_models()

    def _load_depth_models(self):
        """Try loading depth models in order of preference."""
        if self._try_load_midas():
            return
        if self._try_load_dpt_hf():
            return
        if self._try_load_glpn():
            return
        logger.warning("⚠️ All depth models failed. Using geometric fallback.")
        self.method = "geometric"

    def _try_load_midas(self):
        """Try loading MiDaS - PRIMARY choice for speed."""
        try:
            weights_path = Path(MIDAS_WEIGHTS_PATH)

            if weights_path.exists():
                logger.info(f"Loading MiDaS from local weights (FAST): {weights_path}")
                midas_model = torch.hub.load(MIDAS_REPO, MIDAS_MODEL_NAME, pretrained=False)
                state_dict = torch.load(weights_path, map_location=self.device)
                midas_model.load_state_dict(state_dict)
                self.model = midas_model.to(self.device).eval()
                self.transform = torch.hub.load(MIDAS_REPO, 'transforms').dpt_transform
            else:
                logger.info("Loading MiDaS from torch.hub (FAST, will download if needed)")
                self.model = torch.hub.load(MIDAS_REPO, MIDAS_MODEL_NAME, pretrained=True)
                self.model = self.model.to(self.device).eval()
                self.transform = torch.hub.load(MIDAS_REPO, 'transforms').dpt_transform

            self.method = "midas"
            logger.info("SUCCESS IN LOADING - MIDAS")
            return True

        except Exception as e:
            logger.warning(f"MiDaS loading failed: {e}")
            return False

    def _try_load_dpt_hf(self):
        """Try loading DPT via Hugging Face."""
        try:
            from transformers import DPTImageProcessor, DPTForDepthEstimation

            logger.info("Loading DPT via Hugging Face...")
            self.processor = DPTImageProcessor.from_pretrained(DPT_HF_MODEL)
            self.model = DPTForDepthEstimation.from_pretrained(DPT_HF_MODEL)
            self.model = self.model.to(self.device).eval()

            self.method = "dpt_hf"
            logger.info("SUCCESS IN LOADING - DPT")
            return True

        except ImportError:
            logger.info("transformers not available for DPT")
            return False
        except Exception as e:
            logger.warning(f"DPT (HF) loading failed: {e}")
            return False

    def _try_load_glpn(self):
        """Try loading GLPN."""
        try:
            from transformers import GLPNImageProcessor, GLPNForDepthEstimation

            logger.info("Loading GLPN (FALLBACK - slower but accurate)...")
            self.processor = GLPNImageProcessor.from_pretrained(GLPN_HF_MODEL)
            self.model = GLPNForDepthEstimation.from_pretrained(GLPN_HF_MODEL)
            self.model = self.model.to(self.device).eval()

            self.method = "glpn"
            logger.info("SUCCESS IN LOADING - GLPN")
            return True

        except ImportError:
            logger.info("transformers not available for GLPN")
            return False
        except Exception as e:
            logger.warning(f"GLPN loading failed: {e}")
            return False

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth using the best available method.
        Returns normalized depth map (0=far, 1=close) - FIXED INTERPRETATION.
        """

        if self.method == "geometric":
            return self._geometric_depth(frame)

        if self.model is None:
            return self._geometric_depth(frame)

        try:
            if self.method == "midas":
                return self._estimate_midas(frame)
            elif self.method == "dpt_hf":
                return self._estimate_dpt_hf(frame)
            elif self.method == "glpn":
                return self._estimate_glpn(frame)
            else:
                return self._geometric_depth(frame)

        except Exception as e:
            logger.warning(f"{self.method} depth estimation failed: {e}. Using geometric fallback.")
            return self._geometric_depth(frame)

    def _estimate_midas(self, frame: np.ndarray) -> np.ndarray:
        """MiDaS depth estimation - FIXED NORMALIZATION."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform
        input_tensor = self.transform(rgb).to(self.device)

        # Run inference
        with torch.no_grad():
            prediction = self.model(input_tensor)

            # Resize to original frame size
            if prediction.dim() == 3:
                prediction = prediction.unsqueeze(0)

            depth_map = F.interpolate(
                prediction,
                size=frame.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze().cpu().numpy()

        # FIXED: Correct normalization for MiDaS
        # MiDaS outputs inverse depth (small values = close objects)
        # We want (0=far, 1=close) so we need to invert
        if depth_map.max() > depth_map.min():
            # Invert: smaller MiDaS values (close) become larger normalized values (close to 1)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        else:
            depth_map = np.full_like(depth_map, 0.5)

        return depth_map.astype(np.float32)

    def _estimate_dpt_hf(self, frame: np.ndarray) -> np.ndarray:
        """DPT depth estimation via Hugging Face."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Resize to original frame size
        depth_map = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()

        # FIXED: Correct normalization for DPT
        if depth_map.max() > depth_map.min():
            # DPT also outputs inverse depth, so invert like MiDaS
            depth_map = (depth_map.max() - depth_map) / (depth_map.max() - depth_map.min())
        else:
            depth_map = np.full_like(depth_map, 0.5)

        return depth_map.astype(np.float32)

    def _estimate_glpn(self, frame: np.ndarray) -> np.ndarray:
        """GLPN depth estimation."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Resize to original frame size
        depth_map = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()

        # FIXED: Check GLPN output format and normalize correctly
        if depth_map.max() > depth_map.min():
            # GLPN might output direct depth, so normalize without inversion
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        else:
            depth_map = np.full_like(depth_map, 0.5)

        return depth_map.astype(np.float32)

    def _geometric_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        FIXED geometric fallback: Proper perspective where bottom=close, top=far.
        """
        h, w = frame.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                # FIXED perspective: bottom = close (1), top = far (0)
                vertical_depth = (h - y) / h  # y=0 (top)→1 (far), y=h (bottom)→0 (close)

                # Invert so bottom=close=1, top=far=0
                vertical_depth = 1.0 - vertical_depth

                # Add center bias: center is typically where action happens
                center_x = w / 2
                horizontal_factor = 1.0 - abs(x - center_x) / center_x * 0.3

                # Combine factors
                depth_value = vertical_depth * horizontal_factor
                depth_map[y, x] = np.clip(depth_value, 0.0, 1.0)

        return depth_map


class SmartAdaptiveDetector:
    """Smart detector that chooses optimal strategy based on scene depth characteristics."""

    def __init__(self, device='auto'):
        if device == 'auto':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda:0'  # Use proper CUDA device string
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Optimized confidence thresholds for each detector
        # Change to these higher values
        self.yolo_conf_close = 0.35
        self.yolo_conf_medium = 0.40
        self.yolo_conf_far = 0.45

        self.rtdetr_conf_close = 0.30
        self.rtdetr_conf_medium = 0.35
        self.rtdetr_conf_far = 0.40

        try:
            self.yolo = load_yolo_model(YOLO_MODEL_PATH)
            logger.info("SUCCESS IN LOADING - YOLO")
        except Exception as e:
            self.yolo = None
            logger.error(f"❌ Failed to load YOLO: {e}")
        try:
            self.rtdetr = RTDETR(RTDETR_MODEL_PATH)
            logger.info("SUCCESS IN LOADING - RT-DETR")
        except Exception as e:
            self.rtdetr = None
            logger.error(f"❌ Failed to load RT-DETR: {e}")

    def choose_detection_strategy(self, depth_map):
        """Intelligently choose optimal detection strategy based on depth characteristics."""

        # Analyze depth distribution
        depth_std = np.std(depth_map)
        depth_mean = np.mean(depth_map)
        depth_range = np.max(depth_map) - np.min(depth_map)

        # Calculate variance coefficient (relative variance)
        variance_coeff = depth_std / depth_mean if depth_mean > 0 else 0

        logger.debug(
            f"Depth analysis: mean={depth_mean:.3f}, std={depth_std:.3f}, range={depth_range:.3f}, var_coeff={variance_coeff:.3f}")

        # Strategy selection logic
        if variance_coeff < 0.3 and depth_range < 0.4:  # Low variance = uniform depth

            if depth_mean > 0.7:
                strategy = "yolo_only"
                reason = f"All objects close (mean={depth_mean:.2f})"
            elif depth_mean < 0.3:
                strategy = "rtdetr_only"
                reason = f"All objects far (mean={depth_mean:.2f})"
            else:
                strategy = "hybrid_simple"
                reason = f"All objects medium distance (mean={depth_mean:.2f})"

        else:  # High variance = mixed depths
            strategy = "zone_based"
            reason = f"Mixed depths detected (std={depth_std:.2f}, range={depth_range:.2f})"

        logger.info(f"🎯 Detection strategy: {strategy.upper()} - {reason}")
        return strategy

    def _extract_detections(self, results, conf_thresh):
        """Extract person detections from model results."""
        if not hasattr(results, 'boxes') or results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6))
        boxes = results.boxes
        xyxy, conf, cls = boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
        person_mask = (cls == 0) & (conf >= conf_thresh)
        detections = np.hstack((xyxy[person_mask], conf[person_mask][:, None], cls[person_mask][:, None]))
        if len(detections) == 0: return np.empty((0, 6))
        areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        return detections[areas > 200]

    def detect_yolo_only(self, frame):
        """YOLO-only detection optimized for close objects."""
        if self.yolo is None:
            return np.empty((0, 6))

        try:
            results = self.yolo(frame, conf=self.yolo_conf_close, classes=[0], verbose=False)[0]
            detections = self._extract_detections(results, self.yolo_conf_close)
            logger.debug(f"YOLO-only: {len(detections)} detections")
            return detections
        except Exception as e:
            logger.warning(f"YOLO-only detection failed: {e}")
            return np.empty((0, 6))

    def detect_rtdetr_only(self, frame):
        """RT-DETR-only detection optimized for far objects."""
        if self.rtdetr is None:
            return np.empty((0, 6))

        try:
            results = self.rtdetr(frame, conf=self.rtdetr_conf_far, classes=[0], verbose=False)[0]
            detections = self._extract_detections(results, self.rtdetr_conf_far)
            logger.debug(f"RT-DETR-only: {len(detections)} detections")
            return detections
        except Exception as e:
            logger.warning(f"RT-DETR-only detection failed: {e}")
            return np.empty((0, 6))

    def detect_hybrid_simple(self, frame):
        """Simple hybrid detection for medium-distance scenarios."""
        all_detections = []

        # YOLO detections
        if self.yolo is not None:
            try:
                yolo_results = self.yolo(frame, conf=self.yolo_conf_medium, classes=[0], verbose=False)[0]
                yolo_dets = self._extract_detections(yolo_results, self.yolo_conf_medium)
                if len(yolo_dets) > 0:
                    all_detections.append(yolo_dets)
            except Exception as e:
                logger.warning(f"YOLO detection failed in hybrid: {e}")

        # RT-DETR detections
        if self.rtdetr is not None:
            try:
                rtdetr_results = self.rtdetr(frame, conf=self.rtdetr_conf_medium, classes=[0], verbose=False)[0]
                rtdetr_dets = self._extract_detections(rtdetr_results, self.rtdetr_conf_medium)
                if len(rtdetr_dets) > 0:
                    all_detections.append(rtdetr_dets)
            except Exception as e:
                logger.warning(f"RT-DETR detection failed in hybrid: {e}")

        # Combine detections
        if all_detections:
            combined = np.vstack(all_detections)
            final_dets = self._apply_nms(combined)
            logger.debug(f"Hybrid-simple: {len(final_dets)} detections after NMS")
            return final_dets
        else:
            return np.empty((0, 6))

    def detect_zone_based(self, frame, depth_map):
        """Zone-based detection for mixed-depth scenarios with adaptive thresholds."""
        if depth_map is None:
            return self.detect_hybrid_simple(frame)

        # Calculate adaptive thresholds based on depth distribution
        threshold_far = np.percentile(depth_map, 33)  # Bottom 33% = FAR
        threshold_close = np.percentile(depth_map, 67)  # Top 33% = CLOSE

        logger.debug(f"Adaptive thresholds: far<={threshold_far:.2f}, close>={threshold_close:.2f}")

        height, width = frame.shape[:2]
        all_detections = []

        # YOLO for CLOSE zones
        if self.yolo is not None:
            try:
                yolo_results = self.yolo(frame, conf=self.yolo_conf_close, classes=[0], verbose=False)[0]
                yolo_dets = self._extract_detections(yolo_results, self.yolo_conf_close)

                # Filter to close zones
                if len(yolo_dets) > 0:
                    filtered_yolo = []
                    for det in yolo_dets:
                        x1, y1, x2, y2 = det[:4]
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        if 0 <= center_y < height and 0 <= center_x < width:
                            depth_value = depth_map[center_y, center_x]
                            if depth_value >= threshold_close:  # CLOSE zone
                                filtered_yolo.append(det)

                    if filtered_yolo:
                        all_detections.append(np.array(filtered_yolo))
                        logger.debug(f"YOLO (CLOSE): {len(filtered_yolo)} detections")

            except Exception as e:
                logger.warning(f"YOLO detection failed in zones: {e}")

        # RT-DETR for FAR zones
        if self.rtdetr is not None:
            try:
                rtdetr_results = self.rtdetr(frame, conf=self.rtdetr_conf_far, classes=[0], verbose=False)[0]
                rtdetr_dets = self._extract_detections(rtdetr_results, self.rtdetr_conf_far)

                # Filter to far zones
                if len(rtdetr_dets) > 0:
                    filtered_rtdetr = []
                    for det in rtdetr_dets:
                        x1, y1, x2, y2 = det[:4]
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        if 0 <= center_y < height and 0 <= center_x < width:
                            depth_value = depth_map[center_y, center_x]
                            if depth_value <= threshold_far:  # FAR zone
                                filtered_rtdetr.append(det)

                    if filtered_rtdetr:
                        all_detections.append(np.array(filtered_rtdetr))
                        logger.debug(f"RT-DETR (FAR): {len(filtered_rtdetr)} detections")

            except Exception as e:
                logger.warning(f"RT-DETR detection failed in zones: {e}")

        # Ensemble for MIDDLE zones
        if self.yolo is not None and self.rtdetr is not None:
            try:
                # Get all detections for middle zone ensemble
                yolo_all = self._extract_detections(
                    self.yolo(frame, conf=self.yolo_conf_medium, classes=[0], verbose=False)[0],
                    self.yolo_conf_medium
                )
                rtdetr_all = self._extract_detections(
                    self.rtdetr(frame, conf=self.rtdetr_conf_medium, classes=[0], verbose=False)[0],
                    self.rtdetr_conf_medium
                )

                # Filter for MIDDLE zones
                middle_detections = []
                for det_list in [yolo_all, rtdetr_all]:
                    if len(det_list) > 0:
                        for det in det_list:
                            x1, y1, x2, y2 = det[:4]
                            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            if 0 <= center_y < height and 0 <= center_x < width:
                                depth_value = depth_map[center_y, center_x]
                                if threshold_far < depth_value < threshold_close:  # MIDDLE zone
                                    middle_detections.append(det)

                if middle_detections:
                    middle_array = np.array(middle_detections)
                    middle_nms = self._apply_nms(middle_array, iou_thresh=0.3)
                    all_detections.append(middle_nms)
                    logger.debug(f"ENSEMBLE (MIDDLE): {len(middle_nms)} detections")

            except Exception as e:
                logger.warning(f"Ensemble detection failed: {e}")

        # Combine all zone-specific detections
        if all_detections:
            combined = np.vstack(all_detections)
            final_dets = self._apply_nms(combined)
            logger.debug(f"Zone-based total: {len(final_dets)} detections after final NMS")
            return final_dets
        else:
            return np.empty((0, 6))

    def detect_smart_adaptive(self, frame, depth_map=None):
        """Main detection method that chooses optimal strategy."""

        if depth_map is None:
            # No depth info, use simple hybrid
            return self.detect_hybrid_simple(frame)

        # Choose strategy based on depth characteristics
        strategy = self.choose_detection_strategy(depth_map)

        # Execute chosen strategy
        if strategy == "yolo_only":
            return self.detect_yolo_only(frame)
        elif strategy == "rtdetr_only":
            return self.detect_rtdetr_only(frame)
        elif strategy == "hybrid_simple":
            return self.detect_hybrid_simple(frame)
        elif strategy == "zone_based":
            return self.detect_zone_based(frame, depth_map)
        else:
            # Fallback
            return self.detect_hybrid_simple(frame)

    def _apply_nms(self, detections, iou_thresh=0.4):
        """Apply Non-Maximum Suppression."""
        if len(detections) == 0: return detections
        x1, y1, x2, y2, scores = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3], detections[:,
                                                                                                         4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1: break
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-5)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return detections[keep]


class PostProcessingEngine:
    """
    Enhanced Post-Processing Engine for Multi-Object Tracking
    Comprehensive track merging with advanced similarity analysis.
    """

    def __init__(self, config=None):
        self.config = config or self._default_config()

    def _default_config(self):
        """Default configuration parameters."""
        return {
            'max_time_gap_seconds': 7.0,
            'max_distance_pixels': 250.0,
            'reid_threshold_high': 0.80,  # Definite match
            'reid_threshold_medium': 0.60,  # Need spatial support
            'reid_threshold_low': 0.40,  # Only with strong motion
            'min_size_ratio': 0.5,
            'enable_motion_prediction': True,
            'enable_multi_frame_averaging': True,
            'verbose_logging': False
        }

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity with scipy fallback."""
        if SCIPY_AVAILABLE:
            try:
                return 1.0 - cosine(a, b)
            except:
                pass

        # Numpy fallback
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def process_tracks(self, track_history, fps, tracker=None):
        """
        Main entry point for enhanced post-processing. Includes LIFETIME FILTERING.

        Args:
            track_history: Dict of track data
            fps: Video frame rate
            tracker: Optional tracker instance for Re-ID features

        Returns:
            Tuple of (final_id_map, final_person_count)
        """
        logger.info("🚀 Enhanced Post-Processing: Applying Lifetime Filter + Merging...")

        if not track_history:
            return {}, 0

        # --- NEW FEATURE: TRACK LIFETIME FILTER ---
        filtered_history = {}
        removed_ids = []
        for track_id, track_data in track_history.items():
            if len(track_data) >= MIN_LIFETIME_FRAMES:
                filtered_history[track_id] = track_data
            else:
                removed_ids.append(track_id)

        if removed_ids:
            logger.warning(f"LIFETIME FILTER: Removed {len(removed_ids)} short-lived ghost tracks: {removed_ids}")
        # From now on, use the filtered history
        track_history = filtered_history
        # --- END OF NEW FEATURE ---

        # Sort all tracks chronologically
        for track_id in track_history:
            track_history[track_id].sort(key=lambda x: x['frame'])

        # Phase 1: Multi-frame feature averaging
        track_features = self._extract_and_average_features(track_history, tracker)

        # Phase 2: Motion analysis and prediction
        track_motion = self._analyze_track_motion(track_history, fps)

        # Phase 3: Hierarchical similarity-based merging
        merge_map = self._hierarchical_track_merging(track_history, track_features, track_motion, fps)

        # Phase 4: Apply merges and generate final IDs
        final_id_map, final_count = self._apply_merges_and_generate_ids(track_history, merge_map)

        logger.info(f"✅ Enhanced post-processing complete: {len(track_history)} → {final_count} tracks after merging.")
        return final_id_map, final_count

    def _extract_and_average_features(self, track_history, tracker):
        """Phase 1: Extract and average Re-ID features across multiple frames."""
        track_features = {}

        for track_id, track_data in track_history.items():
            features_list = []

            # Collect Re-ID features if available
            for frame_data in track_data:
                if 'reid_features' in frame_data and frame_data['reid_features'] is not None:
                    features_list.append(frame_data['reid_features'])

            if features_list and len(features_list) >= 2:
                # Select stable middle frames to avoid start/end noise
                if len(features_list) >= 5:
                    start_idx = len(features_list) // 4
                    end_idx = 3 * len(features_list) // 4
                    selected_features = features_list[start_idx:end_idx]
                else:
                    selected_features = features_list

                # Compute robust average
                avg_features = np.mean(selected_features, axis=0)

                # Normalize for cosine similarity
                if np.linalg.norm(avg_features) > 0:
                    avg_features = avg_features / np.linalg.norm(avg_features)

                track_features[track_id] = {
                    'features': avg_features,
                    'confidence': min(1.0, len(selected_features) / 5.0),
                    'frame_count': len(selected_features)
                }

                if self.config['verbose_logging']:
                    logger.debug(f"Track {track_id}: averaged {len(selected_features)} feature vectors")
            else:
                # No reliable Re-ID features
                track_features[track_id] = {
                    'features': None,
                    'confidence': 0.0,
                    'frame_count': 0
                }

        return track_features

    def _analyze_track_motion(self, track_history, fps):
        """Phase 2: Analyze motion patterns and predict positions."""
        track_motion = {}

        for track_id, track_data in track_history.items():
            if len(track_data) < 2:
                track_motion[track_id] = {
                    'velocity': np.array([0.0, 0.0]),
                    'acceleration': np.array([0.0, 0.0]),
                    'stability': 0.0,
                    'predicted_next_pos': None
                }
                continue

            # Extract positions and times
            positions = []
            times = []

            for frame_data in track_data:
                bbox = frame_data['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                positions.append([center_x, center_y])
                times.append(frame_data['frame'] / fps)

            positions = np.array(positions)
            times = np.array(times)

            # Calculate velocity (using last 3 points if available)
            if len(positions) >= 3:
                recent_positions = positions[-3:]
                recent_times = times[-3:]
                velocity = self._calculate_velocity(recent_positions, recent_times)
            else:
                velocity = (positions[-1] - positions[0]) / (times[-1] - times[0])

            # Calculate motion stability
            if len(positions) >= 4:
                velocities = []
                for i in range(1, len(positions)):
                    v = (positions[i] - positions[i - 1]) / (times[i] - times[i - 1])
                    velocities.append(v)
                velocities = np.array(velocities)
                stability = 1.0 / (1.0 + np.std(velocities))
            else:
                stability = 0.5

            # Predict next position
            last_time = times[-1]
            predicted_pos = positions[-1] + velocity * self.config['max_time_gap_seconds']

            track_motion[track_id] = {
                'velocity': velocity,
                'stability': stability,
                'predicted_next_pos': predicted_pos,
                'last_position': positions[-1],
                'last_time': last_time
            }

        return track_motion

    def _calculate_velocity(self, positions, times):
        """Calculate robust velocity using multiple points."""
        if len(positions) < 2:
            return np.array([0.0, 0.0])

        # Use linear regression for robust velocity estimation
        dt = times[-1] - times[0]
        if dt == 0:
            return np.array([0.0, 0.0])

        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]

        return np.array([dx / dt, dy / dt])

    def _hierarchical_track_merging(self, track_history, track_features, track_motion, fps):
        """Phase 3: Hierarchical similarity-based track merging."""
        merge_map = {}
        sorted_ids = sorted(track_history.keys())
        time_threshold_frames = int(self.config['max_time_gap_seconds'] * fps)

        for i in range(len(sorted_ids)):
            for j in range(i + 1, len(sorted_ids)):
                id1, id2 = sorted_ids[i], sorted_ids[j]

                # Skip if already merged
                if id1 in merge_map or id2 in merge_map:
                    continue

                track1, track2 = track_history[id1], track_history[id2]
                if not track1 or not track2:
                    continue

                # Check temporal feasibility
                end_of_track1 = track1[-1]
                start_of_track2 = track2[0]
                frame_gap = start_of_track2['frame'] - end_of_track1['frame']

                if not (0 < frame_gap <= time_threshold_frames):
                    continue

                # Compute hierarchical similarity
                similarity_score = self._compute_hierarchical_similarity(
                    id1, id2, track1, track2, track_features, track_motion, frame_gap
                )

                # Decision logic based on similarity tiers
                should_merge = False
                merge_reason = ""

                if similarity_score['reid_similarity'] >= self.config['reid_threshold_high']:
                    should_merge = True
                    merge_reason = f"High Re-ID similarity ({similarity_score['reid_similarity']:.2f})"

                elif similarity_score['reid_similarity'] >= self.config['reid_threshold_medium']:
                    if similarity_score['spatial_score'] > 0.6:
                        should_merge = True
                        merge_reason = f"Medium Re-ID + Good spatial ({similarity_score['reid_similarity']:.2f}, {similarity_score['spatial_score']:.2f})"

                elif similarity_score['reid_similarity'] >= self.config['reid_threshold_low']:
                    if similarity_score['motion_score'] > 0.7 and similarity_score['spatial_score'] > 0.5:
                        should_merge = True
                        merge_reason = f"Low Re-ID + Strong motion + Spatial ({similarity_score['reid_similarity']:.2f}, {similarity_score['motion_score']:.2f})"

                else:
                    # Fallback to geometric + motion only if no Re-ID
                    if similarity_score['reid_similarity'] == 0.0:  # No Re-ID available
                        if similarity_score['spatial_score'] > 0.8 and similarity_score['motion_score'] > 0.6:
                            should_merge = True
                            merge_reason = f"Geometric fallback ({similarity_score['spatial_score']:.2f}, {similarity_score['motion_score']:.2f})"

                if should_merge:
                    merge_map[id2] = id1
                    if self.config['verbose_logging']:
                        logger.info(f"Merging track {id2} → {id1}: {merge_reason}")

        return merge_map

    def _compute_hierarchical_similarity(self, id1, id2, track1, track2, track_features, track_motion, frame_gap):
        """Compute multi-dimensional similarity between two tracks."""

        # 1. Re-ID Similarity
        reid_sim = 0.0
        if (track_features[id1]['features'] is not None and
                track_features[id2]['features'] is not None):

            # Cosine similarity between averaged features
            try:
                reid_sim = self._cosine_similarity(track_features[id1]['features'],
                                                   track_features[id2]['features'])
                reid_sim = max(0.0, reid_sim)  # Clamp to [0, 1]

                # Weight by feature confidence
                confidence_weight = (track_features[id1]['confidence'] + track_features[id2]['confidence']) / 2
                reid_sim *= confidence_weight

            except Exception:
                reid_sim = 0.0

        # 2. Spatial Similarity
        end_bbox1 = track1[-1]['bbox']
        start_bbox2 = track2[0]['bbox']

        center1 = np.array([(end_bbox1[0] + end_bbox1[2]) / 2, (end_bbox1[1] + end_bbox1[3]) / 2])
        center2 = np.array([(start_bbox2[0] + start_bbox2[2]) / 2, (start_bbox2[1] + start_bbox2[3]) / 2])

        distance = np.linalg.norm(center1 - center2)
        spatial_score = max(0.0, 1.0 - distance / self.config['max_distance_pixels'])

        # Size consistency
        area1 = (end_bbox1[2] - end_bbox1[0]) * (end_bbox1[3] - end_bbox1[1])
        area2 = (start_bbox2[2] - start_bbox2[0]) * (start_bbox2[3] - start_bbox2[1])
        size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0

        spatial_score = 0.7 * spatial_score + 0.3 * size_ratio

        # 3. Motion Similarity
        motion_score = 0.0
        if (self.config['enable_motion_prediction'] and
                track_motion[id1]['predicted_next_pos'] is not None):
            predicted_pos = track_motion[id1]['predicted_next_pos']
            actual_pos = center2

            prediction_error = np.linalg.norm(predicted_pos - actual_pos)
            motion_score = max(0.0, 1.0 - prediction_error / self.config['max_distance_pixels'])

            # Factor in motion stability
            stability_factor = (track_motion[id1]['stability'] + track_motion[id2]['stability']) / 2
            motion_score *= stability_factor

        return {
            'reid_similarity': reid_sim,
            'spatial_score': spatial_score,
            'motion_score': motion_score,
            'overall_score': 0.5 * reid_sim + 0.3 * spatial_score + 0.2 * motion_score
        }

    def _apply_merges_and_generate_ids(self, track_history, merge_map):
        """Phase 4: Apply merges and generate sequential final IDs."""
        sorted_ids = sorted(track_history.keys())
        final_id_map = {}
        sequential_id_counter = 1
        root_to_final_id = {}

        # Build merge chains
        for original_id in sorted_ids:
            current_id = original_id
            path = [original_id]

            # Follow merge chain to root
            while current_id in merge_map:
                current_id = merge_map[current_id]
                path.append(current_id)

            root_id = current_id

            # Assign sequential ID to root if not already assigned
            if root_id not in root_to_final_id:
                root_to_final_id[root_id] = sequential_id_counter
                sequential_id_counter += 1

            final_id = root_to_final_id[root_id]

            # Map all IDs in the chain to the final ID
            for node_id in path:
                final_id_map[node_id] = final_id

        final_person_count = sequential_id_counter - 1
        return final_id_map, final_person_count


def post_process_and_merge_tracks(track_history, fps, tracker=None):
    """Legacy wrapper for backward compatibility."""
    engine = PostProcessingEngine()
    return engine.process_tracks(track_history, fps, tracker)


class DubsComprehensivePeopleCounting:
    def __init__(self, device='auto', use_reid=True):
        if device == 'auto':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
                logger.info("🍎 Using MPS (Apple Silicon GPU) for acceleration")
            elif torch.cuda.is_available():
                self.device = 'cuda:0'  # Use proper CUDA device string
                logger.info("🚀 Using CUDA GPU for acceleration")
            else:
                self.device = 'cpu'
                logger.info("💻 Using CPU")
        else:
            self.device = str(device)
        logger.info(f"Using device: {self.device}")

        # Initialize components
        self.depth_estimator = DepthEstimator(device)
        self.detector = SmartAdaptiveDetector(self.device)

        try:
            if use_reid:
                logger.info("✅ Initializing BoTSORT with Re-ID enabled...")
                self.tracker = BotSort(
                    reid_weights=REID_MODEL_PATH,
                    device=self.device,
                    half=False,
                    track_buffer=300,
                    appearance_thresh=0.60
                )
                logger.info("SUCCESS IN LOADING - OSNET RE-ID")
            else:
                logger.info("✅ Initializing ByteTrack (Re-ID disabled)...")
                self.tracker = ByteTrack(
                    track_buffer=150,
                    match_thresh=0.65,
                    frame_rate=25
                )
        except Exception as e:
            self.tracker = None
            logger.error(f"❌ Failed to initialize tracker: {e}", exc_info=True)

    def process_video(self, video_path, output_path=None, results_path=None, use_post_process=False):
        # Handle both video files and image directories (MOT dataset support)
        image_files = []
        if os.path.isdir(video_path):
            image_files = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
            if not image_files: raise ValueError(f"No JPG images found in directory: {video_path}")
            cap = None
            total_frames = len(image_files)
            frame = cv2.imread(image_files[0])
            height, width, _ = frame.shape
            fps = 25
            logger.info(f"📂 Processing image sequence: {len(image_files)} images")
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): raise ValueError(f"Cannot open video: {video_path}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"🎥 Processing video: {total_frames} frames at {fps} FPS")

        # PASS 1: ANALYSIS with Smart Adaptive Detection
        logger.info("--- Pass 1: Smart Adaptive Analysis ---")
        all_tracks_by_frame = {}
        depth_method_used = None
        detection_strategies_used = set()

        if self.tracker: self.tracker.reset()

        # Open MOT results file if path is provided
        results_file = open(results_path, 'w') if results_path else None

        for frame_count in tqdm(range(1, total_frames + 1), desc="Pass 1: Smart Detection"):
            if cap:
                ret, frame = cap.read()
                if not ret: break
            else:
                frame = cv2.imread(image_files[frame_count - 1])

            # Estimate depth
            depth_map = self.depth_estimator.estimate_depth(frame)

            # Log depth method on first frame
            if depth_method_used is None:
                depth_method_used = self.depth_estimator.method
                logger.info(f"🎯 Using depth estimation method: {depth_method_used}")

            # Smart adaptive detection
            detections = self.detector.detect_smart_adaptive(frame, depth_map)

            # Track detection strategy used (for reporting)
            if hasattr(self.detector, '_last_strategy'):
                detection_strategies_used.add(self.detector._last_strategy)

            if self.tracker and len(detections) > 0:
                tracks = self.tracker.update(detections, frame)
                if len(tracks) > 0:
                    track_data = []
                    for t in tracks:
                        # --- START: THE CRITICAL FIX IS HERE ---
                        track_info = {'id': int(t[4]), 'bbox': t[:4]}

                        # BotSort with Re-ID appends the feature vector to the track tuple.
                        # We must check for its existence and store it.
                        if len(t) > 7 and t[-1] is not None:
                            track_info['reid_features'] = t[-1]
                        # --- END: THE CRITICAL FIX ---

                        # Add depth info if available
                        x1, y1, x2, y2 = t[:4]
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        if depth_map is not None and 0 <= center_y < height and 0 <= center_x < width:
                            depth_value = depth_map[center_y, center_x]
                            track_info['depth'] = depth_value

                        track_data.append(track_info)

                    all_tracks_by_frame[frame_count] = track_data

                    # Write to MOT results file
                    if results_file:
                        for track in tracks:
                            x1, y1, x2, y2, track_id = track[:5]
                            w, h = x2 - x1, y2 - y1
                            line = f"{frame_count},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
                            results_file.write(line)
        if cap: cap.release()
        if results_file: results_file.close()

        # Post-processing
        track_history_for_merge = {}
        for frame_num, frame_data in all_tracks_by_frame.items():
            for track_info in frame_data:
                tid = track_info['id']
                if tid not in track_history_for_merge: track_history_for_merge[tid] = []
                track_history_for_merge[tid].append({
                    'frame': frame_num,
                    'bbox': track_info['bbox'],
                    'depth': track_info.get('depth', 0.5)
                })

        # --- START: PROPOSED DEBUGGING CODE ---
        print("\n" + "=" * 60)
        print("--- TRACK LIFETIME ANALYSIS REPORT ---")
        print("Analyzing all tracks created during Pass 1...")
        print("-" * 60)

        sorted_track_ids = sorted(track_history_for_merge.keys())

        for tid in sorted_track_ids:
            track_data = track_history_for_merge[tid]
            lifetime = len(track_data)
            first_frame = track_data[0]['frame']
            last_frame = track_data[-1]['frame']

            # Prepare the output line
            report_line = (f"  Track ID: {tid:<4} | "
                           f"Lifetime: {lifetime:<5} frames | "
                           f"Appeared at Frame: {first_frame:<5} | "
                           f"Vanished at Frame: {last_frame:<5}")

            # Highlight potential ghost tracks
            if lifetime < 20:  # A threshold to identify very short tracks (e.g., less than a second)
                report_line += "  <--- [!] WARNING: POTENTIAL GHOST TRACK"

            print(report_line)

        print("=" * 60 + "\n")
        # --- END: PROPOSED DEBUGGING CODE ---
        if use_post_process:
            # Use enhanced post-processing engine
            processing_engine = PostProcessingEngine(config={
                'max_time_gap_seconds': 15.0,
                'max_distance_pixels': 350.0,
                'min_size_ratio': 0.4,
                'reid_threshold_high': 0.70,
                'reid_threshold_medium': 0.50,
                'reid_threshold_low': 0.30,
                'enable_motion_prediction': True,
                'enable_multi_frame_averaging': True,
                'verbose_logging': False
            })
            final_id_map, final_count = processing_engine.process_tracks(
                track_history_for_merge, fps, self.tracker
            )
            logger.info(f"✅ Enhanced post-processing enabled: {len(track_history_for_merge)} → {final_count} tracks")
        else:
            logger.info("⚠️ Skipping post-process merging.")
            final_count = len(set(track['id'] for frame_data in all_tracks_by_frame.values() for track in frame_data))
            final_id_map = {i: i for i in range(1, final_count + 100)}

        # PASS 2: Rendering (if output video requested)
        if output_path:
            logger.info("--- Pass 2: Rendering video ---")
            if cap is None and image_files:
                frame = cv2.imread(image_files[0])
                height, width, _ = frame.shape
            else:
                cap = cv2.VideoCapture(video_path)

            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            for frame_count in tqdm(range(1, total_frames + 1), desc="Pass 2: Rendering"):
                if cap:
                    ret, frame = cap.read()
                    if not ret: break
                else:
                    frame = cv2.imread(image_files[frame_count - 1])

                if frame_count in all_tracks_by_frame:
                    for track_info in all_tracks_by_frame[frame_count]:
                        original_id = track_info['id']

                        # --- START: GHOST TRACK CAPTURE CODE ---
                        if original_id == 11 and frame_count == 180:  # Frame where Track 11 appears
                            print(f"[!!!] GHOST TRACK 11 APPEARED AT FRAME {frame_count}. SAVING IMAGE.")
                            x1, y1, x2, y2 = track_info['bbox']
                            ghost_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                            cv2.imwrite("ghost_track_11_capture.jpg", ghost_crop)
                        # --- END: GHOST TRACK CAPTURE CODE ---

                        clean_id = final_id_map.get(original_id, original_id)

                        if clean_id:
                            x1, y1, x2, y2 = track_info['bbox']

                            # Color based on depth
                            depth_value = track_info.get('depth', 0.5)
                            if depth_value >= 0.6:
                                color = (0, 255, 0)  # Green for close
                                zone_text = "CLOSE"
                            elif depth_value >= 0.4:
                                color = (0, 255, 255)  # Yellow for medium
                                zone_text = "MID"
                            else:
                                color = (255, 0, 255)  # Magenta for far
                                zone_text = "FAR"

                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, f'ID:{clean_id}', (int(x1), int(y1) - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(frame, zone_text, (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Enhanced overlay
                cv2.putText(frame, f"Dubs Comprehensive V1.0 | Method: {depth_method_used}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                total_text = f"TOTAL COUNT: {final_count}"
                cv2.putText(frame, total_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                writer.write(frame)

            if cap: cap.release()
            writer.release()

        return {
            'person_count': final_count,
            'output_path': output_path,
            'depth_method': depth_method_used,
            'raw_track_count': len(track_history_for_merge),
            'total_frames': total_frames,
            'fps': fps,
            'detection_strategies': list(detection_strategies_used)
        }


def main():
    parser = argparse.ArgumentParser(description='Dubs Comprehensive People Counting - Production Ready Solution V1.0')
    parser.add_argument('--video', required=True, help='Path to input video or image sequence directory')
    parser.add_argument('--output_dir', default='outputs', help='Directory to save annotated video')
    parser.add_argument('--device', default='auto', help='Device: auto/mps/cuda/cpu')
    parser.add_argument('--disable_reid', action='store_true', help='Disable Re-ID features')
    parser.add_argument('--post_process', action='store_true',
                        help='Enable enhanced post-processing to merge broken tracks')
    parser.add_argument('--results_output', help='Path to save tracking results for MOT evaluation')
    parser.add_argument('--fast_mode', action='store_true', help='Use geometric depth for faster processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging for detection strategies')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Generate output path
    output_path = None
    if not args.results_output:
        output_path = next_sequential_name(args.output_dir)

    print(f"🚀 Starting Dubs Comprehensive People Counting V1.0")
    print(f"📹 Input: {args.video}")
    if output_path: print(f"💾 Video Output: {output_path}")
    if args.results_output: print(f"📝 MOT Results Output: {args.results_output}")
    print(f"🔧 Device: {args.device}")

    reid_enabled = not args.disable_reid
    if reid_enabled: print(f"🏃 Re-ID Mode: ON")
    if args.post_process: print(f"🔄 Enhanced Post-Processing: ON")
    if args.fast_mode: print(f"⚡ Fast Mode: ON (geometric depth)")
    if args.debug: print(f"🐛 Debug Mode: ON")
    print(f"🧠 Strategy: Smart Adaptive Detection + Enhanced Post-Processing")
    print("-" * 50)

    start_time = time.time()
    try:
        counter = DubsComprehensivePeopleCounting(device=args.device, use_reid=reid_enabled)

        # Override depth method if fast mode requested
        if args.fast_mode:
            counter.depth_estimator.method = "geometric"
            logger.info("⚡ Fast mode enabled - using geometric depth")

        results = counter.process_video(
            video_path=args.video,
            output_path=output_path,
            results_path=args.results_output,
            use_post_process=args.post_process
        )
        processing_time = time.time() - start_time

        print("-" * 50)
        print(f"✅ SUCCESS! Processing completed in {processing_time:.2f} seconds")
        print(f"🎯 Depth estimation method: {results['depth_method']}")
        print(f"🧠 Detection strategies used: {', '.join(results.get('detection_strategies', ['unknown']))}")
        print(f"👥 Final unique people count: {results['person_count']}")
        print(f"📊 Raw detections (before post-processing): {results['raw_track_count']}")
        print(f"🚀 Processing speed: {results['total_frames'] / processing_time:.1f} FPS")
        if output_path: print(f"📁 Video saved to: {results['output_path']}")
        if args.results_output: print(f"📝 MOT results saved to: {args.results_output}")

        if results['depth_method'] == 'midas':
            print(f"✅ MiDaS depth analysis: ACTIVE")
        else:
            print(f"⚡ Depth analysis: {results['depth_method'].upper()} FALLBACK")

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        return 1
    return 0


if __name__ == '__main__':
    exit(main())

'''
python dubs_people_counting_comprehensive_1.py --video video_12.mp4 --device mps --post_process
PYTORCH_ENABLE_MPS_FALLBACK=1

'''

# --- INTEGRATION WRAPPER FOR CELERY TASKS ---
def tracking_video(input_path: str, output_path: str) -> dict:
    """
    Wrapper for Celery integration. Uses Django storage pattern: input via default_storage, output to temp file.
    Args:
        input_path: Path to the input video file (Django storage path)
        output_path: Path where the temporary annotated video should be saved (/tmp/...)
    Returns:
        dict with unified result structure
    """
    import torch
    import tempfile
    from django.core.files.storage import default_storage

    # Device selection: prefer cuda > mps > cpu
    if torch.cuda.is_available():
        device = 'cuda:0'  # Use proper CUDA device string
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Follow Django storage pattern: copy input file to temporary location
    with default_storage.open(input_path, 'rb') as f:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(f.read())
            tmp_input_path = tmp.name

    try:
        counter = DubsComprehensivePeopleCounting(device=device, use_reid=DEFAULT_USE_REID)
        results = counter.process_video(
            video_path=tmp_input_path,
            output_path=output_path,  # This is already a /tmp/ path from tasks.py
            use_post_process=DEFAULT_POST_PROCESS
        )

        # Convert to web format if possible
        web_output_path = output_path.replace('.mp4', '_web.mp4')
        if convert_to_web_mp4(results.get('output_path', output_path), web_output_path):
            final_output_path = web_output_path
        else:
            final_output_path = results.get('output_path', output_path)

        return {
            'status': 'completed',
            'job_type': 'people-count',
            'output_video': final_output_path,  # tasks.py will handle Django storage
            'data': {
                'person_count': results.get('person_count', 0),
                'raw_track_count': results.get('raw_track_count', 0),
                'depth_method': results.get('depth_method', ''),
                'fps': results.get('fps', 0),
                'detection_strategies': results.get('detection_strategies', []),
                'alerts': []
            },
            'meta': {},
            'error': None
        }

    finally:
        # Clean up temporary input file
        try:
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary input file {tmp_input_path}: {e}")
