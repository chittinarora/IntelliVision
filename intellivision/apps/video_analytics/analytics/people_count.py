"""
People counting analytics using YOLO, RT-DETR, and BotSort with depth-aware detection and track merging.
Supports image, video, and MOT dataset inputs.
"""

# ======================================
# Imports and Setup
# ======================================
import os
import glob
import cv2
import torch
import time
import numpy as np
import logging
import re
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
import mimetypes
from celery import shared_task
from ultralytics import RTDETR
from boxmot import BotSort
import torch.nn.functional as F

from ..utils import load_yolo_model
from ..convert import convert_to_web_mp4

# Import scipy with fallback
try:
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("⚠️ scipy not available - using numpy cosine similarity fallback")

# MiDaS imports with proper error handling
try:
    import torch.hub
    MIDAS_AVAILABLE = True
except ImportError:
    MIDAS_AVAILABLE = False
    logger.warning("⚠️ torch.hub not available for MiDaS")

# ======================================
# Logger and Constants
# ======================================
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
                self.logger.info(f"**Job {self.job_id}**: Completed {self.job_type}")

        return DummyLogger(job_id, total_items, job_type, logger_name)
VALID_EXTENSIONS = {'.mp4', '.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REID_MODEL_PATH = MODELS_DIR / "osnet_x0_25_msmt17.pt"
YOLO_MODEL_PATH = MODELS_DIR / "yolo12x.pt"
RTDETR_MODEL_PATH = MODELS_DIR / "rtdetr-l.pt"
MIDAS_WEIGHTS_PATH = MODELS_DIR / "midas/weights/dpt_large_384.pt"
MIDAS_REPO = 'intel-isl/MiDaS'
MIDAS_MODEL_NAME = 'DPT_Large'
DPT_HF_MODEL = 'Intel/dpt-large'
GLPN_HF_MODEL = 'vinvino02/glpn-nyu'
DEFAULT_OUTPUT_PREFIX = 'dubs_comprehensive_output_'
DEFAULT_OUTPUT_EXT = '.mp4'
DEFAULT_USE_REID = True
DEFAULT_POST_PROCESS = True
YOLO_CONF_CLOSE = 0.35
YOLO_CONF_MEDIUM = 0.40
YOLO_CONF_FAR = 0.45
RTDETR_CONF_CLOSE = 0.30
RTDETR_CONF_MEDIUM = 0.35
RTDETR_CONF_FAR = 0.40
DEFAULT_NMS_IOU_THRESH = 0.4
MIN_LIFETIME_FRAMES = 40

# Define OUTPUT_DIR with fallback
try:
    OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
except AttributeError:
    logger.warning("JOB_OUTPUT_DIR not defined in settings. Using fallback: MEDIA_ROOT/outputs")
    OUTPUT_DIR = Path(settings.MEDIA_ROOT) / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check model existence
MODEL_FILES = ["yolo12x.pt", "rtdetr-l.pt", "osnet_x0_25_msmt17.pt"]
for model_file in MODEL_FILES:
    if not (MODELS_DIR / model_file).exists():
        logger.error(f"Model file missing: {model_file}")
        raise FileNotFoundError(f"Model file {model_file} not found in {MODELS_DIR}")

# ======================================
# Depth Estimator
# ======================================

class DepthEstimator:
    """Multi-tier depth estimation with fixed depth interpretation."""
    def __init__(self, device='auto'):
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                logger.info("🍎 Using MPS (Apple Silicon GPU) for acceleration")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
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
        """Try loading MiDaS - primary choice for speed."""
        if not MIDAS_AVAILABLE:
            return False
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
            logger.info("✅ MiDaS loaded successfully (PRIMARY - optimized for speed)")
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
            logger.info("✅ DPT (Hugging Face) loaded successfully")
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
            logger.info("✅ GLPN loaded successfully (WARNING: This will be slow ~0.3 FPS)")
            return True
        except ImportError:
            logger.info("transformers not available for GLPN")
            return False
        except Exception as e:
            logger.warning(f"GLPN loading failed: {e}")
            return False

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using the best available method (0=far, 1=close)."""
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
        """MiDaS depth estimation."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor)
            if prediction.dim() == 3:
                prediction = prediction.unsqueeze(0)
            depth_map = F.interpolate(
                prediction,
                size=frame.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze().cpu().numpy()
        if depth_map.max() > depth_map.min():
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        else:
            depth_map = np.full_like(depth_map, 0.5)
        return depth_map.astype(np.float32)

    def _estimate_dpt_hf(self, frame: np.ndarray) -> np.ndarray:
        """DPT depth estimation via Hugging Face."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        depth_map = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()
        if depth_map.max() > depth_map.min():
            depth_map = (depth_map.max() - depth_map) / (depth_map.max() - depth_map.min())
        else:
            depth_map = np.full_like(depth_map, 0.5)
        return depth_map.astype(np.float32)

    def _estimate_glpn(self, frame: np.ndarray) -> np.ndarray:
        """GLPN depth estimation."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        depth_map = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()
        if depth_map.max() > depth_map.min():
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        else:
            depth_map = np.full_like(depth_map, 0.5)
        return depth_map.astype(np.float32)

    def _geometric_depth(self, frame: np.ndarray) -> np.ndarray:
        """Geometric fallback: bottom=close, top=far."""
        h, w = frame.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                vertical_depth = (h - y) / h
                vertical_depth = 1.0 - vertical_depth
                center_x = w / 2
                horizontal_factor = 1.0 - abs(x - center_x) / center_x * 0.3
                depth_value = vertical_depth * horizontal_factor
                depth_map[y, x] = np.clip(depth_value, 0.0, 1.0)
        return depth_map

# ======================================
# Smart Adaptive Detector
# ======================================

class SmartAdaptiveDetector:
    """Smart detector with depth-aware strategies."""
    def __init__(self, device='auto'):
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        self.yolo_conf_close = YOLO_CONF_CLOSE
        self.yolo_conf_medium = YOLO_CONF_MEDIUM
        self.yolo_conf_far = YOLO_CONF_FAR
        self.rtdetr_conf_close = RTDETR_CONF_CLOSE
        self.rtdetr_conf_medium = RTDETR_CONF_MEDIUM
        self.rtdetr_conf_far = RTDETR_CONF_FAR
        try:
            self.yolo = load_yolo_model(YOLO_MODEL_PATH)
            logger.info("✅ YOLO loaded")
        except Exception as e:
            self.yolo = None
            logger.error(f"❌ Failed to load YOLO: {e}")
        try:
            self.rtdetr = RTDETR(RTDETR_MODEL_PATH)
            logger.info("✅ RT-DETR loaded")
        except Exception as e:
            self.rtdetr = None
            logger.error(f"❌ Failed to load RT-DETR: {e}")
        self._last_strategy = None

    def choose_detection_strategy(self, depth_map):
        """Choose optimal detection strategy based on depth characteristics."""
        depth_std = np.std(depth_map)
        depth_mean = np.mean(depth_map)
        depth_range = np.max(depth_map) - np.min(depth_map)
        variance_coeff = depth_std / depth_mean if depth_mean > 0 else 0
        logger.debug(f"Depth analysis: mean={depth_mean:.3f}, std={depth_std:.3f}, range={depth_range:.3f}, var_coeff={variance_coeff:.3f}")
        if variance_coeff < 0.3 and depth_range < 0.4:
            if depth_mean > 0.7:
                self._last_strategy = "yolo_only"
                reason = f"All objects close (mean={depth_mean:.2f})"
            elif depth_mean < 0.3:
                self._last_strategy = "rtdetr_only"
                reason = f"All objects far (mean={depth_mean:.2f})"
            else:
                self._last_strategy = "hybrid_simple"
                reason = f"All objects medium distance (mean={depth_mean:.2f})"
        else:
            self._last_strategy = "zone_based"
            reason = f"Mixed depths detected (std={depth_std:.2f}, range={depth_range:.2f})"
        logger.info(f"🎯 Detection strategy: {self._last_strategy.upper()} - {reason}")
        return self._last_strategy

    def _extract_detections(self, results, conf_thresh):
        """Extract person detections from model results."""
        if not hasattr(results, 'boxes') or results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6))
        boxes = results.boxes
        xyxy, conf, cls = boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
        person_mask = (cls == 0) & (conf >= conf_thresh)
        detections = np.hstack((xyxy[person_mask], conf[person_mask][:, None], cls[person_mask][:, None]))
        if len(detections) == 0:
            return np.empty((0, 6))
        areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        return detections[areas > 200]

    def detect_yolo_only(self, frame):
        """YOLO-only detection for close objects."""
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
        """RT-DETR-only detection for far objects."""
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
        if self.yolo is not None:
            try:
                yolo_results = self.yolo(frame, conf=self.yolo_conf_medium, classes=[0], verbose=False)[0]
                yolo_dets = self._extract_detections(yolo_results, self.yolo_conf_medium)
                if len(yolo_dets) > 0:
                    all_detections.append(yolo_dets)
            except Exception as e:
                logger.warning(f"YOLO detection failed in hybrid: {e}")
        if self.rtdetr is not None:
            try:
                rtdetr_results = self.rtdetr(frame, conf=self.rtdetr_conf_medium, classes=[0], verbose=False)[0]
                rtdetr_dets = self._extract_detections(rtdetr_results, self.rtdetr_conf_medium)
                if len(rtdetr_dets) > 0:
                    all_detections.append(rtdetr_dets)
            except Exception as e:
                logger.warning(f"RT-DETR detection failed in hybrid: {e}")
        if all_detections:
            combined = np.vstack(all_detections)
            final_dets = self._apply_nms(combined)
            logger.debug(f"Hybrid-simple: {len(final_dets)} detections after NMS")
            return final_dets
        return np.empty((0, 6))

    def detect_zone_based(self, frame, depth_map):
        """Zone-based detection for mixed-depth scenarios."""
        if depth_map is None:
            return self.detect_hybrid_simple(frame)
        threshold_far = np.percentile(depth_map, 33)
        threshold_close = np.percentile(depth_map, 67)
        logger.debug(f"Adaptive thresholds: far<={threshold_far:.2f}, close>={threshold_close:.2f}")
        height, width = frame.shape[:2]
        all_detections = []
        if self.yolo is not None:
            try:
                yolo_results = self.yolo(frame, conf=self.yolo_conf_close, classes=[0], verbose=False)[0]
                yolo_dets = self._extract_detections(yolo_results, self.yolo_conf_close)
                if len(yolo_dets) > 0:
                    filtered_yolo = []
                    for det in yolo_dets:
                        x1, y1, x2, y2 = det[:4]
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        if 0 <= center_y < height and 0 <= center_x < width:
                            depth_value = depth_map[center_y, center_x]
                            if depth_value >= threshold_close:
                                filtered_yolo.append(det)
                    if filtered_yolo:
                        all_detections.append(np.array(filtered_yolo))
                        logger.debug(f"YOLO (CLOSE): {len(filtered_yolo)} detections")
            except Exception as e:
                logger.warning(f"YOLO detection failed in zones: {e}")
        if self.rtdetr is not None:
            try:
                rtdetr_results = self.rtdetr(frame, conf=self.rtdetr_conf_far, classes=[0], verbose=False)[0]
                rtdetr_dets = self._extract_detections(rtdetr_results, self.rtdetr_conf_far)
                if len(rtdetr_dets) > 0:
                    filtered_rtdetr = []
                    for det in rtdetr_dets:
                        x1, y1, x2, y2 = det[:4]
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        if 0 <= center_y < height and 0 <= center_x < width:
                            depth_value = depth_map[center_y, center_x]
                            if depth_value <= threshold_far:
                                filtered_rtdetr.append(det)
                    if filtered_rtdetr:
                        all_detections.append(np.array(filtered_rtdetr))
                        logger.debug(f"RT-DETR (FAR): {len(filtered_rtdetr)} detections")
            except Exception as e:
                logger.warning(f"RT-DETR detection failed in zones: {e}")
        if self.yolo is not None and self.rtdetr is not None:
            try:
                yolo_all = self._extract_detections(
                    self.yolo(frame, conf=self.yolo_conf_medium, classes=[0], verbose=False)[0],
                    self.yolo_conf_medium
                )
                rtdetr_all = self._extract_detections(
                    self.rtdetr(frame, conf=self.rtdetr_conf_medium, classes=[0], verbose=False)[0],
                    self.rtdetr_conf_medium
                )
                middle_detections = []
                for det_list in [yolo_all, rtdetr_all]:
                    if len(det_list) > 0:
                        for det in det_list:
                            x1, y1, x2, y2 = det[:4]
                            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            if 0 <= center_y < height and 0 <= center_x < width:
                                depth_value = depth_map[center_y, center_x]
                                if threshold_far < depth_value < threshold_close:
                                    middle_detections.append(det)
                if middle_detections:
                    middle_array = np.array(middle_detections)
                    middle_nms = self._apply_nms(middle_array, iou_thresh=0.3)
                    all_detections.append(middle_nms)
                    logger.debug(f"ENSEMBLE (MIDDLE): {len(middle_nms)} detections")
            except Exception as e:
                logger.warning(f"Ensemble detection failed: {e}")
        if all_detections:
            combined = np.vstack(all_detections)
            final_dets = self._apply_nms(combined)
            logger.debug(f"Zone-based total: {len(final_dets)} detections after final NMS")
            return final_dets
        return np.empty((0, 6))

    def detect_smart_adaptive(self, frame, depth_map=None):
        """Main detection method with optimal strategy."""
        if depth_map is None:
            return self.detect_hybrid_simple(frame)
        strategy = self.choose_detection_strategy(depth_map)
        if strategy == "yolo_only":
            return self.detect_yolo_only(frame)
        elif strategy == "rtdetr_only":
            return self.detect_rtdetr_only(frame)
        elif strategy == "hybrid_simple":
            return self.detect_hybrid_simple(frame)
        elif strategy == "zone_based":
            return self.detect_zone_based(frame, depth_map)
        return self.detect_hybrid_simple(frame)

    def _apply_nms(self, detections, iou_thresh=DEFAULT_NMS_IOU_THRESH):
        """Apply Non-Maximum Suppression."""
        if len(detections) == 0:
            return detections
        x1, y1, x2, y2, scores = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3], detections[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-5)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return detections[keep]

# ======================================
# Post-Processing Engine
# ======================================

class PostProcessingEngine:
    """Enhanced post-processing for track merging."""
    def __init__(self, config=None):
        self.config = config or self._default_config()

    def _default_config(self):
        """Default configuration parameters."""
        return {
            'max_time_gap_seconds': 15.0,
            'max_distance_pixels': 350.0,
            'min_size_ratio': 0.4,
            'reid_threshold_high': 0.70,
            'reid_threshold_medium': 0.50,
            'reid_threshold_low': 0.30,
            'enable_motion_prediction': True,
            'enable_multi_frame_averaging': True,
            'verbose_logging': False
        }

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity with scipy fallback."""
        if SCIPY_AVAILABLE:
            try:
                return 1.0 - cosine(a, b)
            except Exception as e:
                logger.warning(f"Failed to compute cosine similarity: {e}")
                # Fall back to manual calculation
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def process_tracks(self, track_history, fps, tracker=None):
        """Main entry point for track merging with lifetime filtering."""
        logger.info("🚀 Enhanced Post-Processing: Applying Lifetime Filter + Merging...")
        if not track_history:
            return {}, 0
        filtered_history = {}
        removed_ids = []
        for track_id, track_data in track_history.items():
            if len(track_data) >= MIN_LIFETIME_FRAMES:
                filtered_history[track_id] = track_data
            else:
                removed_ids.append(track_id)
        if removed_ids:
            logger.warning(f"LIFETIME FILTER: Removed {len(removed_ids)} short-lived ghost tracks: {removed_ids}")
        track_history = filtered_history
        for track_id in track_history:
            track_history[track_id].sort(key=lambda x: x['frame'])
        track_features = self._extract_and_average_features(track_history, tracker)
        track_motion = self._analyze_track_motion(track_history, fps)
        merge_map = self._hierarchical_track_merging(track_history, track_features, track_motion, fps)
        final_id_map, final_count = self._apply_merges_and_generate_ids(track_history, merge_map)
        logger.info(f"✅ Enhanced post-processing complete: {len(track_history)} → {final_count} tracks after merging.")
        return final_id_map, final_count

    def _extract_and_average_features(self, track_history, tracker):
        """Extract and average Re-ID features."""
        track_features = {}
        for track_id, track_data in track_history.items():
            features_list = []
            for frame_data in track_data:
                if 'reid_features' in frame_data and frame_data['reid_features'] is not None:
                    features_list.append(frame_data['reid_features'])
            if features_list and len(features_list) >= 2:
                if len(features_list) >= 5:
                    start_idx = len(features_list) // 4
                    end_idx = 3 * len(features_list) // 4
                    selected_features = features_list[start_idx:end_idx]
                else:
                    selected_features = features_list
                avg_features = np.mean(selected_features, axis=0)
                if np.linalg.norm(avg_features) > 0:
                    avg_features = avg_features / np.linalg.norm(avg_features)
                track_features[track_id] = {
                    'features': avg_features,
                    'confidence': min(1.0, len(selected_features) / 5.0),
                    'frame_count': len(selected_features)
                }
            else:
                track_features[track_id] = {
                    'features': None,
                    'confidence': 0.0,
                    'frame_count': 0
                }
        return track_features

    def _analyze_track_motion(self, track_history, fps):
        """Analyze motion patterns and predict positions."""
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
            if len(positions) >= 3:
                recent_positions = positions[-3:]
                recent_times = times[-3:]
                velocity = self._calculate_velocity(recent_positions, recent_times)
            else:
                velocity = (positions[-1] - positions[0]) / (times[-1] - times[0]) if times[-1] != times[0] else np.array([0.0, 0.0])
            if len(positions) >= 4:
                velocities = []
                for i in range(1, len(positions)):
                    v = (positions[i] - positions[i - 1]) / (times[i] - times[i - 1]) if times[i] != times[i - 1] else np.array([0.0, 0.0])
                    velocities.append(v)
                velocities = np.array(velocities)
                stability = 1.0 / (1.0 + np.std(velocities)) if len(velocities) > 0 else 0.5
            else:
                stability = 0.5
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
        """Calculate robust velocity."""
        if len(positions) < 2:
            return np.array([0.0, 0.0])
        dt = times[-1] - times[0]
        if dt == 0:
            return np.array([0.0, 0.0])
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        return np.array([dx / dt, dy / dt])

    def _hierarchical_track_merging(self, track_history, track_features, track_motion, fps):
        """Hierarchical similarity-based track merging."""
        merge_map = {}
        sorted_ids = sorted(track_history.keys())
        time_threshold_frames = int(self.config['max_time_gap_seconds'] * fps)
        for i in range(len(sorted_ids)):
            for j in range(i + 1, len(sorted_ids)):
                id1, id2 = sorted_ids[i], sorted_ids[j]
                if id1 in merge_map or id2 in merge_map:
                    continue
                track1, track2 = track_history[id1], track_history[id2]
                if not track1 or not track2:
                    continue
                end_of_track1 = track1[-1]
                start_of_track2 = track2[0]
                frame_gap = start_of_track2['frame'] - end_of_track1['frame']
                if not (0 < frame_gap <= time_threshold_frames):
                    continue
                similarity_score = self._compute_hierarchical_similarity(id1, id2, track1, track2, track_features, track_motion, frame_gap)
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
                    if similarity_score['reid_similarity'] == 0.0:
                        if similarity_score['spatial_score'] > 0.8 and similarity_score['motion_score'] > 0.6:
                            should_merge = True
                            merge_reason = f"Geometric fallback ({similarity_score['spatial_score']:.2f}, {similarity_score['motion_score']:.2f})"
                if should_merge:
                    merge_map[id2] = id1
                    if self.config['verbose_logging']:
                        logger.info(f"Merging track {id2} → {id1}: {merge_reason}")
        return merge_map

    def _compute_hierarchical_similarity(self, id1, id2, track1, track2, track_features, track_motion, frame_gap):
        """Compute multi-dimensional similarity between tracks."""
        reid_sim = 0.0
        if (track_features[id1]['features'] is not None and track_features[id2]['features'] is not None):
            try:
                reid_sim = self._cosine_similarity(track_features[id1]['features'], track_features[id2]['features'])
                reid_sim = max(0.0, reid_sim)
                confidence_weight = (track_features[id1]['confidence'] + track_features[id2]['confidence']) / 2
                reid_sim *= confidence_weight
            except Exception:
                reid_sim = 0.0
        end_bbox1 = track1[-1]['bbox']
        start_bbox2 = track2[0]['bbox']
        center1 = np.array([(end_bbox1[0] + end_bbox1[2]) / 2, (end_bbox1[1] + end_bbox1[3]) / 2])
        center2 = np.array([(start_bbox2[0] + start_bbox2[2]) / 2, (start_bbox2[1] + start_bbox2[3]) / 2])
        distance = np.linalg.norm(center1 - center2)
        spatial_score = max(0.0, 1.0 - distance / self.config['max_distance_pixels'])
        area1 = (end_bbox1[2] - end_bbox1[0]) * (end_bbox1[3] - end_bbox1[1])
        area2 = (start_bbox2[2] - start_bbox2[0]) * (start_bbox2[3] - start_bbox2[1])
        size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        spatial_score = 0.7 * spatial_score + 0.3 * size_ratio
        motion_score = 0.0
        if (self.config['enable_motion_prediction'] and track_motion[id1]['predicted_next_pos'] is not None):
            predicted_pos = track_motion[id1]['predicted_next_pos']
            actual_pos = center2
            prediction_error = np.linalg.norm(predicted_pos - actual_pos)
            motion_score = max(0.0, 1.0 - prediction_error / self.config['max_distance_pixels'])
            stability_factor = (track_motion[id1]['stability'] + track_motion[id2]['stability']) / 2
            motion_score *= stability_factor
        return {
            'reid_similarity': reid_sim,
            'spatial_score': spatial_score,
            'motion_score': motion_score,
            'overall_score': 0.5 * reid_sim + 0.3 * spatial_score + 0.2 * motion_score
        }

    def _apply_merges_and_generate_ids(self, track_history, merge_map):
        """Apply merges and generate sequential IDs."""
        sorted_ids = sorted(track_history.keys())
        final_id_map = {}
        sequential_id_counter = 1
        root_to_final_id = {}
        for original_id in sorted_ids:
            current_id = original_id
            path = [original_id]
            while current_id in merge_map:
                current_id = merge_map[current_id]
                path.append(current_id)
            root_id = current_id
            if root_id not in root_to_final_id:
                root_to_final_id[root_id] = sequential_id_counter
                sequential_id_counter += 1
            final_id = root_to_final_id[root_id]
            for node_id in path:
                final_id_map[node_id] = final_id
        final_person_count = sequential_id_counter - 1
        return final_id_map, final_person_count

# ======================================
# Main Processing Class
# ======================================

class DubsComprehensivePeopleCounting:
    """Comprehensive people counting with adaptive detection and post-processing."""
    def __init__(self, device='auto', use_reid=DEFAULT_USE_REID):
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
                logger.info("🍎 Using MPS (Apple Silicon GPU) for acceleration")
            elif torch.cuda.is_available():
                self.device = 0  # Use first CUDA device
                logger.info(f"🚀 Using CUDA GPU (device 0) for acceleration")
            else:
                self.device = 'cpu'
                logger.info("💻 Using CPU")
        elif device == 'cuda':
            if torch.cuda.is_available():
                self.device = 0  # Use first CUDA device
                logger.info(f"🚀 Using CUDA GPU (device 0) for acceleration")
            else:
                self.device = 'cpu'
                logger.warning("CUDA requested but not available, falling back to CPU")
        else:
            self.device = str(device)

        logger.info(f"Using device: {self.device}")
        self.depth_estimator = DepthEstimator(self.device)
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
            else:
                logger.info("✅ Initializing ByteTrack (Re-ID disabled)...")
                self.tracker = BotSort( # Changed from ByteTrack to BotSort
                    reid_weights=REID_MODEL_PATH, # Assuming REID_MODEL_PATH is available for BotSort
                    device=self.device,
                    half=False,
                    track_buffer=150, # Changed from 300 to 150
                    match_thresh=0.65, # Changed from 0.65 to 0.65
                    frame_rate=25 # Added frame_rate
                )
        except Exception as e:
            self.tracker = None
            logger.error(f"❌ Failed to initialize tracker: {e}")
            logger.error(f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}")
            logger.error(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            logger.error(f"os.environ['CUDA_VISIBLE_DEVICES']: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}\n")

    def process_image(self, image_path: str) -> Dict:
        """Process a single image for people counting."""
        start_time = time.time()
        # Validate with full path, but get filename for storage operations
        is_valid, error_msg = validate_input_file(image_path)
        if not is_valid:
            logger.error(f"Invalid input: {error_msg}")
            return {
                'status': 'failed',
                'job_type': 'people_count',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': error_msg},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
            }

        # Get just the filename for storage operations
        image_filename = Path(image_path).name

        try:
            with default_storage.open(image_path, 'rb') as f:
                frame = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            depth_map = self.depth_estimator.estimate_depth(frame)
            detections = self.detector.detect_smart_adaptive(frame, depth_map)
            person_count = len(detections)
            alerts = []
            if person_count > 10:
                alerts.append({
                    "message": f"High crowd density detected: {person_count} people",
                    "timestamp": timezone.now().isoformat()
                })
            if self.tracker:
                self.tracker.reset()
                tracks = self.tracker.update(detections, frame) if len(detections) > 0 else np.empty((0, 7))
                person_count = len(tracks) if len(tracks) > 0 else person_count
            output_filename = f"outputs/dubs_comprehensive_output_{image_filename}"
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                if len(detections) > 0:
                    results = self.detector.yolo(frame, conf=YOLO_CONF_CLOSE, classes=[0], verbose=False)[0] if self.detector.yolo else \
                              self.detector.rtdetr(frame, conf=RTDETR_CONF_CLOSE, classes=[0], verbose=False)[0]
                    annotated_frame = results.plot()
                else:
                    annotated_frame = frame
                cv2.imwrite(tmp.name, annotated_frame)
                with open(tmp.name, 'rb') as f:
                    default_storage.save(output_filename, f)
            output_url = default_storage.url(output_filename)
            processing_time = time.time() - start_time
            return {
                'status': 'completed',
                'job_type': 'people_count',
                'output_image': output_url,
                'output_video': None,
                'data': {
                    'person_count': person_count,
                    'raw_track_count': len(detections),
                    'depth_method': self.depth_estimator.method,
                    'fps': None,
                    'detection_strategies': [self.detector._last_strategy] if self.detector._last_strategy else ['unknown'],
                    'alerts': alerts
                },
                'meta': {
                    'timestamp': timezone.now().isoformat(),
                    'processing_time_seconds': processing_time,
                    'fps': None,
                    'frame_count': 1
                },
                'error': None
            }
        except Exception as e:
            logger.exception(f"Image processing failed: {str(e)}")
            return {
                'status': 'failed',
                'job_type': 'people_count',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': str(e)},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
            }
        finally:
            if 'tmp' in locals() and os.path.exists(tmp.name):
                os.remove(tmp.name)

    def process_video(self, video_path: str, output_path: str) -> Dict:
        """Process a video file for people counting."""
        start_time = time.time()  # Initialize start_time for error handling
        try:
            # Validate input file
            if not default_storage.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Get the physical file path
            file_path = default_storage.path(video_path)

            # Check if it's a directory using os.path
            if os.path.isdir(file_path):
                raise IsADirectoryError(f"Expected a file but got a directory: {video_path}")

            # Initialize video capture
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")

            # Get just the filename for storage operations
            video_filename = Path(video_path).name

            # Get just the filename for storage operations
            image_files = []
            if default_storage.exists(video_path) and os.path.isdir(file_path):
                image_files = sorted([f for f in default_storage.listdir(video_path)[1] if f.endswith(('.jpg', '.jpeg', '.png'))])
                if not image_files:
                    raise ValueError(f"No JPG images found in directory: {video_path}")
                with default_storage.open(os.path.join(video_path, image_files[0]), 'rb') as f:
                    frame = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                height, width = frame.shape[:2]
                fps = 25
                total_frames = len(image_files)
                logger.info(f"📂 Processing image sequence: {total_frames} images")
                cap = None
            else:
                with default_storage.open(video_path, 'rb') as f:
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video: {video_filename}")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                logger.info(f"🎥 Processing video: {total_frames} frames at {fps} FPS")

            job_id = re.search(r'(\d+)', video_filename)
            job_id = job_id.group(1) if job_id else str(int(time.time()))
            output_filename = f"outputs/dubs_comprehensive_output_{job_id}.mp4"
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
                writer = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                all_tracks_by_frame = {}
                depth_method_used = None
                detection_strategies_used = set()
                alerts = []
                if self.tracker:
                    self.tracker.reset()
                frame_count = 0
                last_log_time = start_time

                for frame_num in tqdm(range(1, total_frames + 1), desc="Pass 1: Smart Detection"):
                    frame_count += 1
                    if cap:
                        ret, frame = cap.read()
                        if not ret:
                            break
                    else:
                        with default_storage.open(os.path.join(video_path, image_files[frame_num - 1]), 'rb') as f:
                            frame = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                    depth_map = self.depth_estimator.estimate_depth(frame)
                    if depth_method_used is None:
                        depth_method_used = self.depth_estimator.method
                        logger.info(f"🎯 Using depth estimation method: {depth_method_used}")
                    detections = self.detector.detect_smart_adaptive(frame, depth_map)
                    if self.detector._last_strategy:
                        detection_strategies_used.add(self.detector._last_strategy)
                    if len(detections) > 0 and self.tracker:
                        tracks = self.tracker.update(detections, frame)
                        track_data = []
                        if len(tracks) > 0:
                            for t in tracks:
                                track_info = {'id': int(t[4]), 'bbox': t[:4]}
                                if len(t) > 7 and t[-1] is not None:
                                    track_info['reid_features'] = t[-1]
                                x1, y1, x2, y2 = t[:4]
                                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                                if depth_map is not None and 0 <= center_y < height and 0 <= center_x < width:
                                    depth_value = depth_map[center_y, center_x]
                                    track_info['depth'] = depth_value
                                track_data.append(track_info)
                            all_tracks_by_frame[frame_num] = track_data
                        if len(track_data) > 10:
                            alerts.append({
                                "message": f"High crowd density detected: {len(track_data)} people at frame {frame_num}",
                                "timestamp": timezone.now().isoformat()
                            })
                    # Periodic logging
                    current_time = time.time()
                    if current_time - last_log_time >= 5 or frame_num == total_frames:
                        progress = (frame_num / total_frames) * 100
                        elapsed_time = current_time - start_time
                        time_remaining = (elapsed_time / frame_num) * (total_frames - frame_num) if frame_num > 0 else 0
                        avg_fps = frame_num / elapsed_time if elapsed_time > 0 else 0
                        logger.info(f"**Job {job_id}**: Progress **{progress:.1f}%** ({frame_num}/{total_frames}), Status: Processing...")
                        logger.info(f"[{'#' * int(progress // 10)}{'-' * (10 - int(progress // 10))}] Done: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d} | Left: {int(time_remaining // 60):02d}:{int(time_remaining % 60):02d} | Avg FPS: {avg_fps:.1f}")
                        last_log_time = current_time
                if cap:
                    cap.release()
                track_history_for_merge = {}
                for frame_num, frame_data in all_tracks_by_frame.items():
                    for track_info in frame_data:
                        tid = track_info['id']
                        if tid not in track_history_for_merge:
                            track_history_for_merge[tid] = []
                        track_history_for_merge[tid].append({
                            'frame': frame_num,
                            'bbox': track_info['bbox'],
                            'depth': track_info.get('depth', 0.5),
                            'reid_features': track_info.get('reid_features')
                        })
                if track_history_for_merge:
                    processing_engine = PostProcessingEngine()
                    final_id_map, final_count = processing_engine.process_tracks(track_history_for_merge, fps, self.tracker)
                    logger.info(f"✅ Enhanced post-processing enabled: {len(track_history_for_merge)} → {final_count} tracks")
                else:
                    final_count = 0
                    final_id_map = {}
                if cap is None and image_files:
                    frame = cv2.imread(image_files[0])
                    height, width = frame.shape[:2]
                else:
                    cap = cv2.VideoCapture(tmp_path)
                for frame_num in tqdm(range(1, total_frames + 1), desc="Pass 2: Rendering"):
                    if cap:
                        ret, frame = cap.read()
                        if not ret:
                            break
                    else:
                        with default_storage.open(os.path.join(video_path, image_files[frame_num - 1]), 'rb') as f:
                            frame = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                    if frame_num in all_tracks_by_frame:
                        for track_info in all_tracks_by_frame[frame_num]:
                            original_id = track_info['id']
                            clean_id = final_id_map.get(original_id, original_id)
                            if clean_id:
                                x1, y1, x2, y2 = track_info['bbox']
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
                    cv2.putText(frame, f"Dubs Comprehensive V1.0 | Method: {depth_method_used}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    total_text = f"TOTAL COUNT: {final_count}"
                    cv2.putText(frame, total_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    writer.write(frame)
                if cap:
                    cap.release()
                writer.release()
                with open(tmp_out.name, 'rb') as f:
                    default_storage.save(output_filename, f)
                output_url = default_storage.url(output_filename)
                web_output_filename = output_filename.replace('.mp4', '_web.mp4')
                if convert_to_web_mp4(tmp_out.name, web_output_filename):
                    output_url = default_storage.url(web_output_filename)
                    os.remove(tmp_out.name)
                processing_time = time.time() - start_time
                return {
                    'status': 'completed',
                    'job_type': 'people_count',
                    'output_image': None,
                    'output_video': output_url,
                    'data': {
                        'person_count': final_count,
                        'raw_track_count': len(track_history_for_merge),
                        'depth_method': depth_method_used,
                        'fps': fps,
                        'detection_strategies': list(detection_strategies_used),
                        'alerts': alerts
                    },
                    'meta': {
                        'timestamp': timezone.now().isoformat(),
                        'processing_time_seconds': processing_time,
                        'fps': fps,
                        'frame_count': frame_count
                    },
                    'error': None
                }
        except Exception as e:
            logger.exception(f"Video processing failed: {str(e)}")
            return {
                'status': 'failed',
                'job_type': 'people_count',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': str(e)},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
            }
        finally:
            # Ensure all resources are properly cleaned up
            try:
                if 'cap' in locals() and cap:
                    cap.release()
            except Exception as e:
                logger.warning(f"Failed to release video capture: {e}")

            try:
                if 'writer' in locals() and writer:
                    writer.release()
            except Exception as e:
                logger.warning(f"Failed to release video writer: {e}")

            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {tmp_path}: {e}")

            try:
                if 'tmp_out' in locals() and os.path.exists(tmp_out.name):
                    os.remove(tmp_out.name)
            except Exception as e:
                logger.warning(f"Failed to remove temporary output file {tmp_out.name}: {e}")

# ======================================
# Celery Integration
# ======================================

def tracking_video(input_path: str, output_path: str, job_id: str = None) -> Dict:
    """
    Analytics function for people counting.

    Args:
        input_path: Path to input video or image sequence
        output_path: Path to save output video
        job_id: VideoJob ID

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    logger.info(f"🚀 Starting people count job {job_id}")

    ext = os.path.splitext(input_path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png']

    if ext in image_exts:
        # Initialize progress logger for image processing
        progress_logger = create_progress_logger(
            job_id=str(job_id) if job_id else "unknown",
            total_items=1,  # Single image
            job_type="people_count"
        )

        progress_logger.update_progress(0, status="Processing image...", force_log=True)
        result = DubsComprehensivePeopleCounting(device='auto').process_image(input_path)
        progress_logger.update_progress(1, status="Analysis completed", force_log=True)
        progress_logger.log_completion(1)
    else:
        # For video processing, we'll need to modify the process_video method to accept job_id
        # For now, we'll use a simple progress logger
        progress_logger = create_progress_logger(
            job_id=str(job_id) if job_id else "unknown",
            total_items=100,  # Estimate for video frames
            job_type="people_count"
        )

        progress_logger.update_progress(0, status="Starting video processing...", force_log=True)
        result = DubsComprehensivePeopleCounting(device='auto').process_video(input_path, output_path)
        progress_logger.update_progress(100, status="Video processing completed", force_log=True)
        progress_logger.log_completion(100)

    processing_time = time.time() - start_time
    result['meta']['processing_time_seconds'] = processing_time
    result['meta']['timestamp'] = timezone.now().isoformat()

    return result

# ======================================
# Helper Functions
# ======================================

def validate_input_file(file_path: str) -> tuple[bool, str]:
    """Validate file type and size."""
    if not default_storage.exists(file_path):
        return False, f"File not found: {file_path}"
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        return False, f"Invalid file type: {ext}. Allowed: {', '.join(VALID_EXTENSIONS)}"
    size = default_storage.size(file_path)
    if size > MAX_FILE_SIZE:
        return False, f"File size {size / (1024*1024):.2f}MB exceeds 500MB limit"
    return True, ""
