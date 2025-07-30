"""
Enhanced Hotel room readiness analytics using Azure OpenAI GPT-4o-2 API.
Supports image and video inputs with improved room type detection and deduplication.
"""

# ======================================
# Imports and Setup
# ======================================
import os
import json
import re
import logging
import tempfile

import requests
import numpy as np
from typing import List, Dict, Union
import cv2
import base64
import time
from pathlib import Path
from datetime import datetime, timedelta
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone
import mimetypes
from celery import shared_task

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    raise ImportError("python-dotenv is required. Install it with: pip install python-dotenv")

# ======================================
# Logger and Constants
# ======================================
logger = logging.getLogger(__name__)

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
AZURE_OPENAI_ENDPOINT = "https://ai-labadministrator7921ai913285980324.openai.azure.com/openai/deployments/gpt-4o-2/chat/completions?api-version=2025-01-01-preview"
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("AZURE_OPENAI_API_KEY not set in environment. Please add it to your .env file.")
HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}
MIN_FRAMES = 5
MAX_FRAMES = 10
MAX_BLUR_THRESHOLD = 40
API_TIMEOUT_CONNECT = 30
API_TIMEOUT_READ = 300
MAX_IMAGE_SIZE = 3 * 1024 * 1024
MAX_PAYLOAD_SIZE = 8 * 1024 * 1024

# Define OUTPUT_DIR with fallback
try:
    OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
except AttributeError:
    logger.warning("JOB_OUTPUT_DIR not defined in settings. Using fallback: MEDIA_ROOT/outputs")
    OUTPUT_DIR = Path(settings.MEDIA_ROOT) / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROOM_PARAMETERS = {
    "bedroom": [
        "Personal Items & Clutter", "Minibar/Fridge Condition", "Bedside Areas",
        "Bed Making Quality", "Pillow Count & Placement", "Comforter/Duvet Condition",
        "Bed Covers Cleanliness", "Floor Condition", "General Room Cleanliness"
    ],
    "bathroom": [
        "Personal Items & Clutter", "Toilet Condition", "Bathtub/Shower Cleanliness",
        "Floor Dryness & Cleanliness", "Mirror Cleanliness", "Towel Status",
        "Sink/Counter Areas", "Toiletries & Amenities", "Ventilation", "Safety Features"
    ],
    "living_area": [
        "Personal Items & Clutter", "Seating Cleanliness", "Coffee Table Condition",
        "Remote Controls", "Entertainment Center", "General Area Cleanliness"
    ],
    "entryway": [
        "Personal Items & Clutter", "General Cleanliness", "Organization", "Floor Condition"
    ],
    "closet": [
        "Personal Items & Clutter", "Organization", "General Cleanliness", "Floor Condition"
    ],
    "vanity_area": [
        "Personal Items & Clutter", "Mirror Cleanliness", "General Cleanliness", "Organization"
    ]
}

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

def parse_json_from_response(text: str) -> Dict:
    """Parse a JSON object from the model's response text."""
    try:
        start_brace = text.find('{')
        end_brace = text.rfind('}')
        if start_brace == -1 or end_brace == -1:
            return {"error": "No JSON object found in the response", "raw_response": text}
        json_text = text[start_brace:end_brace + 1].strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)
    except Exception as e:
        return {"error": f"Parsing error: {str(e)}", "raw_response": text}

def convert_status_to_frontend(status: str) -> str:
    """Convert AI response status to frontend format."""
    status_mapping = {
        "present": "Ready",
        "missing": "Issue Found",
        "defective": "Issue Found",
        "needs_attention": "Needs Attention"
    }
    return status_mapping.get(status.lower(), "Needs Attention")

def process_ai_checklist_directly(ai_checklist: List[Dict]) -> List[Dict]:
    """Process AI checklist with corrected logic."""
    processed_checklist = []
    for item in ai_checklist:
        ai_status = item.get('status', 'present').lower()
        notes = item.get('notes', '').lower()
        positive_indicators = [
            'no visible stains detected', 'no personal items or clutter visible', 'clean',
            'properly positioned', 'adequately stocked', 'present and symmetrically arranged',
            'working and clean', 'neatly positioned', 'functional and placed properly'
        ]
        negative_indicators = [
            'wrinkled', 'scattered', 'missing', 'dirty', 'unclean', 'disorganized',
            'clutter', 'personal items detected', 'has items', 'misplaced', 'not visible',
            'not properly'
        ]
        has_positive_content = any(indicator in notes for indicator in positive_indicators)
        has_negative_content = any(indicator in notes for indicator in negative_indicators)
        if ai_status == 'present':
            frontend_status = "Ready"
        elif ai_status in ['missing', 'defective']:
            if has_negative_content and not has_positive_content:
                frontend_status = "Issue Found"
            elif has_positive_content:
                frontend_status = "Ready"
            else:
                frontend_status = "Issue Found"
        else:
            frontend_status = "Ready"
        processed_item = {
            "parameter": item.get('parameter', 'Unknown'),
            "status": frontend_status,
            "box": item.get('box'),
            "notes": item.get('notes', ''),
            "ai_suggested_fix": item.get('ai_suggested_fix', '')
        }
        processed_checklist.append(processed_item)
    return processed_checklist

def clean_issue_text(text: str) -> str:
    """Clean up issue text to remove incomplete sentences and artifacts."""
    if not text:
        return text
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip(' ,.')
    if text.endswith(' and'):
        text = text[:-4].strip()
    if text.endswith(' and.'):
        text = text[:-5].strip()
    if text.endswith(' visible and'):
        text = text[:-12].strip() + ' visible'
    if text.endswith(' visible and.'):
        text = text[:-13].strip() + ' visible'
    return text.strip()

def deduplicate_issues_by_location(room_issues: List[Dict]) -> List[Dict]:
    """Smart deduplication to ensure each location is mentioned once."""
    logger.debug(f"Starting deduplication with {len(room_issues)} issues")
    key_locations = [
        'minibar', 'fridge', 'nightstand', 'bedside', 'coffee table',
        'seating', 'sofa', 'chair', 'bed', 'toilet', 'bathtub', 'shower',
        'mirror', 'sink', 'counter', 'closet', 'entryway', 'vanity'
    ]
    location_claims = {}
    deduplicated_issues = []
    parameter_priority = {
        'minibar/fridge': 1,
        'bedside': 2,
        'coffee table': 3,
        'seating': 4,
        'bed making': 5,
        'personal items': 10
    }

    def get_priority(issue):
        param = issue.get('parameter', '').lower()
        for key, priority in parameter_priority.items():
            if key in param:
                return priority
        return 5

    sorted_issues = sorted(room_issues, key=get_priority)
    for issue in sorted_issues:
        issue_text = issue.get('issue', '').lower()
        parameter = issue.get('parameter', '').lower()
        mentioned_locations = [loc for loc in key_locations if loc in issue_text]
        conflict_found = False
        conflicting_locations = []
        for location in mentioned_locations:
            if location in location_claims:
                conflict_found = True
                conflicting_locations.append(location)
        if conflict_found:
            modified_issue_text = issue.get('issue', '')
            for location in conflicting_locations:
                location_phrases = [
                    f"{location} surface", f"on {location}", f"on the {location}",
                    f"{location} area", f"and {location}", location
                ]
                for phrase in location_phrases:
                    modified_issue_text = re.sub(r'\b' + re.escape(phrase) + r'\b', '', modified_issue_text, flags=re.IGNORECASE)
            modified_issue_text = clean_issue_text(modified_issue_text)
            if (len(modified_issue_text) < 15 or
                not any(word in modified_issue_text.lower() for word in
                        ['clutter', 'items', 'clean', 'arrange', 'organize', 'personal', 'visible'])):
                param_name = parameter.replace('&', 'and').lower()
                modified_issue_text = f"General organization and cleanliness issues in {param_name}"
            if len(modified_issue_text) > 8:
                modified_issue = issue.copy()
                modified_issue['issue'] = modified_issue_text
                modified_issue['fix'] = f"Address remaining {parameter.lower()} issues"
                deduplicated_issues.append(modified_issue)
                logger.debug(f"MODIFIED: {parameter} -> '{modified_issue_text}'")
            else:
                logger.debug(f"DROPPED: {parameter} (no meaningful content after deduplication)")
        else:
            for location in mentioned_locations:
                location_claims[location] = parameter
            deduplicated_issues.append(issue)
            logger.debug(f"KEPT: {parameter} (claimed locations: {mentioned_locations})")
    logger.debug(f"Deduplication complete: {len(room_issues)} -> {len(deduplicated_issues)} issues")
    return deduplicated_issues

def calculate_room_score_improved(ai_score: int, actual_issues: List[Dict]) -> int:
    """Calculate room score with improved penalty system."""
    real_issues = [issue for issue in actual_issues if 'no action required' not in issue.get('fix', '').lower()]
    if len(real_issues) == 0:
        return min(ai_score, 100)
    total_penalty = 0
    for issue in real_issues:
        severity = issue.get('severity', 'medium')
        parameter = issue.get('parameter', '').lower()
        if 'personal items' in parameter or 'clutter' in parameter:
            penalty = 12 if severity == 'high' else 8
        elif 'bed' in parameter or 'seating' in parameter:
            penalty = 10 if severity == 'high' else 7
        elif 'organization' in parameter or 'arrangement' in parameter:
            penalty = 8 if severity == 'high' else 5
        else:
            penalty = 10 if severity == 'high' else 6
        total_penalty += penalty
    final_score = max(ai_score - total_penalty, 55)
    return final_score

def consolidate_reports_by_room_type_fixed(reports: List[Dict]) -> Dict:
    """Consolidate individual frame reports with room type preservation."""
    if not reports or not isinstance(reports, list):
        return {
            "room_analysis": {},
            "overall_status": "Error",
            "rooms_analyzed": 0,
            "rooms_ready": 0,
            "total_issues": 0,
            "all_issues": [],
            "individual_reports": []
        }
    logger.debug(f"Processing {len(reports)} frame reports")
    room_groups = {}
    for i, report in enumerate(reports):
        if not isinstance(report, dict) or 'zone' not in report:
            continue
        zone = report.get('zone', 'unknown').lower()
        if zone in ['other', 'unknown', '']:
            checklist = report.get('checklist', [])
            bathroom_indicators = ['toilet', 'bathtub', 'bathroom', 'sink', 'shower', 'towel', 'mirror']
            bedroom_indicators = ['bed', 'pillow', 'comforter', 'duvet', 'minibar', 'nightstand']
            living_indicators = ['seating', 'coffee', 'sofa', 'remote', 'entertainment']
            closet_indicators = ['closet', 'hanging', 'wardrobe', 'organization', 'clothes']
            entryway_indicators = ['entryway', 'entrance', 'foyer', 'entry']
            vanity_indicators = ['vanity', 'dressing', 'makeup']
            param_text = ' '.join([item.get('parameter', '') + ' ' + item.get('notes', '') for item in checklist]).lower()
            if any(indicator in param_text for indicator in bathroom_indicators):
                zone = 'bathroom'
            elif any(indicator in param_text for indicator in bedroom_indicators):
                zone = 'bedroom'
            elif any(indicator in param_text for indicator in living_indicators):
                zone = 'living_area'
            elif any(indicator in param_text for indicator in closet_indicators):
                zone = 'closet'
            elif any(indicator in param_text for indicator in entryway_indicators):
                zone = 'entryway'
            elif any(indicator in param_text for indicator in vanity_indicators):
                zone = 'vanity_area'
            else:
                zone = f'other_area_{i}' if i == 0 else 'entryway' if i == len(reports) - 1 else 'closet'
        room_key = zone
        if zone in room_groups:
            existing_score = room_groups[zone][0].get('readiness_score', 0)
            current_score = report.get('readiness_score', 0)
            if abs(existing_score - current_score) > 20:
                room_key = f"{zone}_{len([k for k in room_groups.keys() if k.startswith(zone)])}"
        if room_key not in room_groups:
            room_groups[room_key] = []
        room_groups[room_key].append(report)
    logger.debug(f"Room groups after improved detection: {list(room_groups.keys())}")
    room_analysis = {}
    all_unique_issues = []
    total_rooms_analyzed = 0
    total_rooms_ready = 0
    for room_type, room_reports in room_groups.items():
        if not room_reports:
            continue
        best_report = max(room_reports, key=lambda x: x.get('readiness_score', 0))
        ai_score = best_report.get('readiness_score', 0)
        ai_checklist = best_report.get('checklist', [])
        processed_checklist = process_ai_checklist_directly(ai_checklist)
        room_issues = extract_issues_with_ai_fixes(ai_checklist, room_type)
        room_issues = deduplicate_issues_by_location(room_issues)
        logger.debug(f"{room_type.upper()}: {len(room_issues)} issues after deduplication")
        room_score = calculate_room_score_improved(ai_score, room_issues)
        for issue in room_issues:
            all_unique_issues.append(issue)
        processed_checklist = sync_checklist_with_final_issues(processed_checklist, room_issues)
        checklist_summary = calculate_checklist_summary(processed_checklist)
        alerts = [issue['issue'] for issue in room_issues]
        is_guest_ready = room_score >= 90 and len(room_issues) == 0
        room_analysis[room_type] = {
            "score": room_score,
            "status": "Guest Ready" if is_guest_ready else "Not Guest Ready",
            "checklist": processed_checklist,
            "alerts": alerts,
            "issues": room_issues,
            "parameters_checked": len(processed_checklist),
            "frames_analyzed": len(room_reports),
            "checklist_summary": checklist_summary
        }
        total_rooms_analyzed += 1
        if is_guest_ready:
            total_rooms_ready += 1
    logger.debug(f"Total unique issues after deduplication: {len(all_unique_issues)}")
    overall_ready = total_rooms_ready == total_rooms_analyzed if total_rooms_analyzed > 0 else False
    return {
        "room_analysis": room_analysis,
        "overall_status": "All Rooms Guest Ready" if overall_ready else "Rooms Need Attention",
        "rooms_analyzed": total_rooms_analyzed,
        "rooms_ready": total_rooms_ready,
        "total_issues": len(all_unique_issues),
        "all_issues": all_unique_issues,
        "individual_reports": reports
    }

def sync_checklist_with_final_issues(processed_checklist: List[Dict], final_issues: List[Dict]) -> List[Dict]:
    """Sync checklist status with final issues."""
    issue_parameters = {issue.get('parameter', '') for issue in final_issues}
    synced_checklist = []
    for item in processed_checklist:
        new_item = item.copy()
        parameter = new_item.get('parameter', '')
        new_item['status'] = "Issue Found" if parameter in issue_parameters else "Ready"
        synced_checklist.append(new_item)
    return synced_checklist

def calculate_checklist_summary(checklist: List[Dict]) -> Dict:
    """Calculate checklist summary with proper status counting."""
    summary = {"present": 0, "missing": 0, "defective": 0}
    for item in checklist:
        status = item.get('status', '').lower()
        notes = item.get('notes', '').lower()
        if status == 'ready':
            summary['present'] += 1
        elif status == 'issue found':
            if ('missing' in notes or 'not found' in notes or 'absent' in notes or
                'not visible' in notes or 'not detected' in notes):
                summary['missing'] += 1
            else:
                summary['defective'] += 1
        elif status == 'needs attention':
            summary['defective'] += 1
        else:
            summary['present'] += 1
    return summary

def extract_issues_with_ai_fixes(ai_checklist: List[Dict], room_type: str) -> List[Dict]:
    """Extract issues with proper filtering."""
    room_issues = []
    for item in ai_checklist:
        status = item.get('status', '')
        notes = item.get('notes', '')
        ai_fix = item.get('ai_suggested_fix', '') or f"Address {item.get('parameter', 'issue')}"
        parameter = item.get('parameter', 'Unknown')
        if "personal items" in parameter.lower() and "clutter" in parameter.lower():
            if status == 'present':
                continue
            elif status == 'missing':
                if "no " in notes.lower() and ("visible" in notes.lower() or "found" in notes.lower()):
                    continue
        if status in ['missing', 'defective']:
            positive_phrases = [
                "no personal items visible", "no clutter visible",
                "no action required", "area is clean"
            ]
            if any(phrase in notes.lower() for phrase in positive_phrases) and "personal items" in parameter.lower():
                continue
            issue_fix_pair = {
                "zone": room_type,
                "issue": notes,
                "fix": ai_fix,
                "severity": "high" if status == 'missing' else "medium",
                "parameter": parameter,
                "status": status,
                "ai_generated": True
            }
            room_issues.append(issue_fix_pair)
    return room_issues

def assess_frame_quality(image: np.ndarray) -> Dict[str, float]:
    """Assess frame quality for blurriness."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {'blur_score': blur_score, 'is_blurry': blur_score < MAX_BLUR_THRESHOLD}
    except Exception:
        return {'blur_score': 0, 'is_blurry': True}

def extract_key_bedroom_frames(video_path: str, output_dir: Path = OUTPUT_DIR) -> List[str]:
    """Extract visually distinct frames by sampling one frame per second."""
    start_time = time.time()
    logger.debug("Starting intelligent frame extraction (1 frame/sec sampling)...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)
    is_valid, error_msg = validate_input_file(video_path.name)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return []
    try:
        with default_storage.open(video_path.name, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
        vidcap = cv2.VideoCapture(tmp_path)
        if not vidcap.isOpened():
            logger.error(f"Failed to open video: {video_path.name}")
            return []
        fps = vidcap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = int(fps)
        logger.debug(f"Video FPS: {fps:.2f}, processing one frame every {frame_interval} frames.")
        selected_frame_paths = []
        selected_histograms = []
        frame_idx = 0
        job_id = re.search(r'(\d+)', video_path.name)
        job_id = job_id.group(1) if job_id else str(int(time.time()))
        while len(selected_frame_paths) < MAX_FRAMES:
            success, frame = vidcap.read()
            if not success:
                break
            if frame_idx % frame_interval == 0:
                quality = assess_frame_quality(frame)
                if quality['is_blurry']:
                    frame_idx += 1
                    continue
                try:
                    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    if not selected_frame_paths:
                        logger.debug(f"Selecting first frame (index {frame_idx}).")
                        path = output_dir / f"frame_{job_id}_{frame_idx:05}.jpg"
                        cv2.imwrite(str(path), frame)
                        with open(str(path), 'rb') as f:
                            default_storage.save(f"outputs/frame_{job_id}_{frame_idx:05}.jpg", f)
                        selected_frame_paths.append(f"outputs/frame_{job_id}_{frame_idx:05}.jpg")
                        selected_histograms.append(hist)
                    else:
                        is_too_similar = any(cv2.compareHist(hist, h, cv2.HISTCMP_CORREL) > 0.95 for h in selected_histograms)
                        if not is_too_similar:
                            logger.debug(f"Selecting distinct frame (index {frame_idx}).")
                            path = output_dir / f"frame_{job_id}_{frame_idx:05}.jpg"
                            cv2.imwrite(str(path), frame)
                            with open(str(path), 'rb') as f:
                                default_storage.save(f"outputs/frame_{job_id}_{frame_idx:05}.jpg", f)
                            selected_frame_paths.append(f"outputs/frame_{job_id}_{frame_idx:05}.jpg")
                            selected_histograms.append(hist)
                except Exception as e:
                    logger.error(f"Could not process frame {frame_idx}: {e}")
            frame_idx += 1
        vidcap.release()
        logger.debug(f"Finished extraction. Selected {len(selected_frame_paths)} diverse frames.")
        return selected_frame_paths
    except Exception as e:
        logger.error(f"Frame extraction failed: {str(e)}")
        return []
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

def optimize_image_for_api(image_path: str, max_size: int = MAX_IMAGE_SIZE) -> str:
    """Optimize image size and return base64 encoded string."""
    try:
        with default_storage.open(image_path, 'rb') as f:
            image_data = f.read()
        if len(image_data) <= max_size:
            return base64.b64encode(image_data).decode('utf-8')
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise Exception("Could not load image")
        for quality in [70, 50, 30]:
            _, encoded_img = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            compressed_data = encoded_img.tobytes()
            if len(compressed_data) <= max_size:
                return base64.b64encode(compressed_data).decode('utf-8')
        height, width = image.shape[:2]
        for scale in [0.8, 0.6, 0.4]:
            new_height, new_width = int(height * scale), int(width * scale)
            resized_image = cv2.resize(image, (new_width, new_height))
            _, encoded_img = cv2.imencode('.jpg', resized_image, [cv2.IMWRITE_JPEG_QUALITY, 40])
            compressed_data = encoded_img.tobytes()
            if len(compressed_data) <= max_size:
                return base64.b64encode(compressed_data).decode('utf-8')
        final_image = cv2.resize(image, (640, 480))
        _, encoded_img = cv2.imencode('.jpg', final_image, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return base64.b64encode(encoded_img.tobytes()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error optimizing image {image_path}: {e}")
        raise

# ======================================
# Main Analysis Functions
# ======================================

def analyze_room_image(image_path: Union[str, List[str]], output_path: str = None, job_id: str = None) -> Dict:
    """
    Analyze room image(s) for readiness using Azure OpenAI API.

    Args:
        image_path: Path to input image or list of images
        output_path: Path to save output image (for tasks.py integration)
        job_id: VideoJob ID for progress tracking

    Returns:
        Standardized response dictionary with filesystem paths
    """
    start_time = time.time()

    # Add job_id logging for progress tracking
    if job_id:
        logger.info(f"🚀 Starting room readiness image job {job_id}")

    if isinstance(image_path, str):
        image_paths, is_multi_frame = [image_path], False
    elif isinstance(image_path, list) and image_path:
        image_paths, is_multi_frame = image_path, len(image_path) > 1
    else:
        return {
            'status': 'failed',
            'job_type': 'room_readiness',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': 'Invalid image_path parameter'},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': 'Invalid image_path parameter', 'code': 'INVALID_INPUT'}
        }

    for img_path in image_paths:
        is_valid, error_msg = validate_input_file(img_path)
        if not is_valid:
            logger.error(f"Invalid input: {error_msg}")
            return {
                'status': 'failed',
                'job_type': 'room_readiness',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': error_msg},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
            }

    multi_frame_prompt = """You are an expert AI assistant specializing in hotel room readiness analysis. You will be given multiple images from a single hotel room inspection.

CRITICAL INSTRUCTIONS:
1. Analyze EACH image INDIVIDUALLY for ALL parameters listed below.
2. For each image, produce a complete JSON report object.
3. Return one final JSON object with key "frame_reports" containing the list of individual reports.
4. IMPORTANT: You must analyze ALL images provided - do not skip any images.
5. MANDATORY: Assess EVERY parameter for each room type - do not skip any parameters.
6. For each parameter with issues, provide a specific, actionable AI fix instruction.

**CRITICAL ROOM TYPE IDENTIFICATION:**
You MUST identify the correct room type from these specific options:
- **bedroom**: Bed, pillows, bedding, nightstands, minibar, dresser
- **bathroom**: Toilet, bathtub, shower, bathroom fixtures, tiles, bathroom sink
- **living_area**: Sofa, coffee table, seating, entertainment center, TV area
- **entryway**: Entrance door, foyer, entry hallway, welcome area, coat area
- **closet**: Hanging clothes, wardrobe, storage area, hangers, closet doors
- **vanity_area**: Vanity mirror with lighting, dressing table, makeup area, vanity sink

DO NOT use "other" or "unknown" - always pick the most appropriate specific room type.

**MANDATORY PARAMETER ASSESSMENT:**
{room_parameters}

**CRITICAL STATUS ASSIGNMENT RULES:**
- **Personal Items & Clutter**:
  - STATUS "present" = NO personal items/clutter visible (GOOD condition)
  - STATUS "defective" = Personal items/clutter IS visible (BAD condition)
  - STATUS "missing" = Only if this parameter cannot be assessed
- **All Other Parameters**:
  - STATUS "present" = Parameter meets requirements (GOOD condition)
  - STATUS "missing" = Required item is absent (BAD condition)
  - STATUS "defective" = Item present but poor condition (BAD condition)

**MANDATORY ASSESSMENT RULE:**
You MUST provide a checklist entry for EVERY parameter listed above for each zone type. If you cannot see a specific area clearly, state "not clearly visible" in notes but still provide the parameter entry.

**AI FIX GENERATION:**
For each parameter with issues (status "missing" or "defective"), provide specific fixes:
- Personal Items & Clutter: "Remove items from floor and general surfaces" (only if status is "defective")
- Minibar/Fridge: "Clear minibar surface of bottles and cups"
- Bedside Areas: "Clear nightstand surface and organize items"
- Bed Making: "Smooth bed covers and remove wrinkles"

**OUTPUT STRUCTURE - CRITICAL:**
```json
{
  "frame_reports": [
    {
      "zone": "bedroom",
      "checklist": [
        {
          "parameter": "Personal Items & Clutter",
          "status": "present",
          "box": null,
          "notes": "No personal items or clutter visible on surfaces",
          "ai_suggested_fix": "No action required"
        },
        {
          "parameter": "Minibar/Fridge Condition",
          "status": "defective",
          "box": [x1, y1, x2, y2],
          "notes": "Bottles and cups visible on minibar surface",
          "ai_suggested_fix": "Clear bottles and cups from minibar surface"
        }
      ],
      "readiness_score": 85,
      "critical_triggers": [],
      "instructions": [],
      "status": "Guest Ready/Not Guest Ready",
      "image_width": 1280,
      "image_height": 853
    }
  ]
}
```""".format(room_parameters='\n'.join([f"**For {key.upper()} zones - MUST assess ALL {len(params)} parameters separately:**\n" + '\n'.join([f"{i+1}. **{p}**" for i, p in enumerate(params)]) for key, params in ROOM_PARAMETERS.items()]))

    single_frame_prompt = """You are an expert AI assistant specializing in hotel room readiness analysis.

**CRITICAL ROOM TYPE IDENTIFICATION:**
You MUST identify the correct room type from these specific options. Look carefully at the visual elements:
- **bedroom**: If you see a bed, pillows, bedding, bedroom furniture, nightstands, minibar, dresser
- **bathroom**: If you see toilet, bathtub, shower, bathroom sink, bathroom tiles, bathroom fixtures, towel racks
- **living_area**: If you see sofa, coffee table, seating area, entertainment center, living room furniture, TV area
- **entryway**: If you see entrance door, foyer, entry hallway, welcome area, coat hooks, shoe area
- **closet**: If you see hanging clothes, wardrobe, storage area, hangers, closet doors, clothing storage
- **vanity_area**: If you see vanity mirror with lighting, dressing table, makeup area, vanity sink (not bathroom sink)

CRITICAL: DO NOT use "other" or "unknown" - always select the most appropriate specific room type from the list above.

**MANDATORY PARAMETER ASSESSMENT:**
{room_parameters}

**CRITICAL STATUS RULES:**
- Personal Items & Clutter: "present" = NO clutter (GOOD), "defective" = clutter visible (BAD)
- All others: "present" = meets requirements, "missing" = absent, "defective" = poor condition

**MANDATORY COMPLETENESS:**
You MUST provide checklist entries for ALL required parameters. Do not skip any parameters.

**RESPONSE FORMAT:**
```json
{
  "zone": "bedroom",
  "checklist": [
    {
      "parameter": "Personal Items & Clutter",
      "status": "present/missing/defective",
      "box": [x1, y1, x2, y2] or null,
      "notes": "Specific to this parameter only",
      "ai_suggested_fix": "Action specific to this parameter only or 'No action required'"
    }
  ],
  "readiness_score": 85,
  "critical_triggers": [],
  "instructions": [],
  "status": "Guest Ready/Not Guest Ready",
  "image_width": 1280,
  "image_height": 853
}
```""".format(room_parameters='\n'.join([f"**FOR {key.upper()} - MUST assess ALL {len(params)} parameters:**\n" + '\n'.join([f"{i+1}. {p}" for i, p in enumerate(params)]) for key, params in ROOM_PARAMETERS.items()]))

    prompt = multi_frame_prompt if is_multi_frame else single_frame_prompt
    encoded_images, valid_paths = [], []
    for img_path in image_paths:
        try:
            encoded_images.append(optimize_image_for_api(img_path))
            valid_paths.append(img_path)
        except Exception as e:
            logger.error(f"Failed to encode image {img_path}: {e}")
            continue
    if not encoded_images:
        return {
            'status': 'failed',
            'job_type': 'room_readiness',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': 'No valid images to analyze'},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': 'No valid images to analyze', 'code': 'INVALID_INPUT'}
        }
    content = [{"type": "text", "text": prompt}]
    for img_b64 in encoded_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
    try:
        response = requests.post(
            AZURE_OPENAI_ENDPOINT, headers=HEADERS, data=json.dumps({
                "messages": [
                    {"role": "system", "content": "You are a precise hotel room readiness AI. Analyze ALL images provided and assess EVERY required parameter separately. Return JSON with one report per image containing ALL mandatory parameters. NEVER use 'other' or 'unknown' for zone - always pick specific room type."},
                    {"role": "user", "content": content}
                ],
                "max_tokens": 4096,
                "temperature": 0.2
            }),
            timeout=(API_TIMEOUT_CONNECT, API_TIMEOUT_READ)
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        parsed = parse_json_from_response(text)
        if "error" in parsed:
            return {
                'status': 'failed',
                'job_type': 'room_readiness',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': f"AI response parsing failed: {parsed['error']}"},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': f"AI response parsing failed: {parsed['error']}", 'code': 'API_PARSING_ERROR'}
            }
        if is_multi_frame:
            frame_reports = parsed.get("frame_reports", [])
            consolidated_data = consolidate_reports_by_room_type_fixed(frame_reports)
        else:
            consolidated_data = consolidate_reports_by_room_type_fixed([parsed])
        individual_reports = consolidated_data.get('individual_reports', [])
        room_analysis = consolidated_data.get('room_analysis', {})
        all_issues = consolidated_data.get('all_issues', [])
        updated_individual_reports = []
        for report in individual_reports:
            zone = report.get('zone', 'unknown')
            updated_report = report.copy()
            if zone in room_analysis:
                final_score = room_analysis[zone].get('score', report.get('readiness_score', 0))
                updated_report['readiness_score'] = final_score
                final_status = room_analysis[zone].get('status', 'Unknown')
                updated_report['status'] = final_status
            updated_individual_reports.append(updated_report)
        overall_score = sum(room['score'] for room in room_analysis.values()) // len(room_analysis) if room_analysis else 0
        overall_status = consolidated_data.get('overall_status', 'Unknown')
        master_checklist = []
        for report in updated_individual_reports:
            checklist_items = report.get('checklist', [])
            for item in checklist_items:
                zone_prefix = report.get('zone', 'unknown')
                item_copy = item.copy()
                item_copy['item'] = f"{zone_prefix.title()}: {item_copy.get('parameter', 'Unknown')}"
                master_checklist.append(item_copy)
        rooms_data = {}
        for room_type, room_info in room_analysis.items():
            rooms_data[room_type] = {
                "score": room_info.get('score', 0),
                "status": room_info.get('status', 'Unknown'),
                "checklist": room_info.get('checklist', []),
                "alerts": room_info.get('alerts', []),
                "issues": room_info.get('issues', []),
                "parameters_checked": room_info.get('parameters_checked', 0),
                "frames_analyzed": room_info.get('frames_analyzed', 0),
                "checklist_summary": room_info.get('checklist_summary', {"present": 0, "missing": 0, "defective": 0})
            }
        unified_data = {
            "readiness_score": overall_score,
            "status": "Guest Ready" if overall_status == "All Rooms Guest Ready" else "Not Guest Ready",
            "zone": "general",
            "checklist": master_checklist,
            "fail_reasons": [item['issue'] for item in all_issues],
            "instructions": [item['fix'] for item in all_issues],
            "alerts": [{"type": "critical", "message": item['issue'], "severity": item['severity']} for item in all_issues],
            "image_width": updated_individual_reports[0].get('image_width', 0) if updated_individual_reports else 0,
            "image_height": updated_individual_reports[0].get('image_height', 0) if updated_individual_reports else 0,
            "overall_status": overall_status,
            "rooms_analyzed": consolidated_data.get('rooms_analyzed', 0),
            "rooms_ready": consolidated_data.get('rooms_ready', 0),
            "total_issues": consolidated_data.get('total_issues', 0),
            "rooms": rooms_data,
            "all_issues": all_issues,
            "individual_reports": updated_individual_reports,
            "deduplication_stats": {
                "raw_issues": sum(len(r.get('checklist', [])) for r in updated_individual_reports),
                "unique_issues": len(all_issues),
                "efficiency": f"{len(all_issues) / max(1, sum(len(r.get('checklist', [])) for r in updated_individual_reports)) * 100:.1f}%"
            }
        }
        # Create temporary output file for tasks.py integration
        import tempfile
        if isinstance(image_path, str):
            input_image_path = image_path
        else:
            input_image_path = valid_paths[0] if valid_paths else None

        if input_image_path:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as output_tmp:
                # Copy original image to temporary file for output
                with default_storage.open(input_image_path, 'rb') as f:
                    output_tmp.write(f.read())
                final_output_path = output_tmp.name
        else:
            final_output_path = None

        processing_time = time.time() - start_time
        logger.info(f"✅ Room readiness image analysis completed, output saved to {final_output_path}")

        return {
            'status': 'completed',
            'job_type': 'room_readiness',
            'output_image': final_output_path,
            'output_video': None,
            'data': unified_data,
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': processing_time, 'frames_analyzed': len(valid_paths)},
            'error': None
        }
    except requests.exceptions.Timeout as e:
        return {
            'status': 'failed',
            'job_type': 'room_readiness',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': f"API request timed out: {str(e)}"},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': f"API request timed out: {str(e)}", 'code': 'API_TIMEOUT'}
        }
    except requests.exceptions.RequestException as e:
        return {
            'status': 'failed',
            'job_type': 'room_readiness',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': f"API request failed: {str(e)}"},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': f"API request failed: {str(e)}", 'code': 'API_REQUEST_ERROR'}
        }
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {
            'status': 'failed',
            'job_type': 'room_readiness',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': f"Failed to parse API response: {str(e)}"},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': f"Failed to parse API response: {str(e)}", 'code': 'API_PARSING_ERROR'}
        }

def analyze_room_video_multi_zone_only(video_path: str, output_path: str = None, job_id: str = None, output_dir: Path = OUTPUT_DIR) -> Dict:
    """
    Analyze video for room readiness by extracting frames and using Azure OpenAI API.

    Args:
        video_path: Path to input video
        output_path: Path to save output video (for tasks.py integration)
        job_id: VideoJob ID for progress tracking
        output_dir: Directory to save extracted frames

    Returns:
        Standardized response dictionary with filesystem paths
    """
    start_time = time.time()

    # Add job_id logging for progress tracking
    if job_id:
        logger.info(f"🚀 Starting room readiness video job {job_id}")

    video_path = Path(video_path)
    is_valid, error_msg = validate_input_file(video_path.name)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'room_readiness',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }
    frame_paths = extract_key_bedroom_frames(video_path.name, output_dir)
    if not frame_paths:
        return {
            'status': 'failed',
            'job_type': 'room_readiness',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': 'No frames could be extracted from video'},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': 'No frames could be extracted from video', 'code': 'FRAME_EXTRACTION_ERROR'}
        }
    result = analyze_room_image(frame_paths, output_path, job_id)
    processing_time = time.time() - start_time
    if result.get('status') == 'completed':
        return {
            'status': 'completed',
            'job_type': 'room_readiness',
            'output_image': None,
            'output_video': None,
            'data': result.get('data', {}),
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'frames_analyzed': result.get('meta', {}).get('frames_analyzed', 0)
            },
            'error': None
        }
    return {
        'status': 'failed',
        'job_type': 'room_readiness',
        'output_image': None,
        'output_video': None,
        'data': result.get('data', {'alerts': [], 'error': result.get('error', 'Unknown error')}),
        'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': processing_time},
        'error': result.get('error', {'message': 'Unknown error', 'code': 'UNKNOWN'})
    }

# ======================================
# Utility Functions
# ======================================

def test_api_connection() -> Dict:
    """Test the API connection with a simple request."""
    logger.debug("Testing API connection...")
    test_payload = {
        "messages": [{"role": "user", "content": "Hello, respond with just 'API_TEST_SUCCESS'"}],
        "max_tokens": 10, "temperature": 0
    }
    try:
        response = requests.post(
            AZURE_OPENAI_ENDPOINT, headers=HEADERS, data=json.dumps(test_payload), timeout=(10, 30)
        )
        if response.status_code == 200:
            return {
                'success': True, 'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(), 'message': 'API connection successful'
            }
        return {
            'success': False, 'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(), 'error': response.text[:500],
            'message': 'API returned error status'
        }
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': str(e), 'message': 'API connection failed'}

def get_current_config() -> Dict:
    """Get current configuration settings."""
    return {
        'min_frames': MIN_FRAMES,
        'max_frames': MAX_FRAMES,
        'max_blur_threshold': MAX_BLUR_THRESHOLD,
        'api_timeout_connect': API_TIMEOUT_CONNECT,
        'api_timeout_read': API_TIMEOUT_READ,
        'max_image_size': MAX_IMAGE_SIZE,
        'max_payload_size': MAX_PAYLOAD_SIZE
    }

def get_room_analysis_summary(data: Dict) -> Dict:
    """Extract summary statistics from analysis data."""
    return {
        "overall_score": data.get('readiness_score', 0),
        "overall_status": data.get('overall_status', 'Unknown'),
        "rooms_analyzed": data.get('rooms_analyzed', 0),
        "rooms_ready": data.get('rooms_ready', 0),
        "total_issues": data.get('total_issues', 0),
        "issue_fix_pairs": len(data.get('all_issues', [])),
        "ai_generated_fixes": sum(1 for issue in data.get('all_issues', []) if issue.get('ai_generated', False)),
        "deduplication_efficiency": data.get('deduplication_stats', {}).get('efficiency', '0%')
    }

def get_room_specific_data(data: Dict, room_type: str) -> Dict:
    """Get detailed data for a specific room type."""
    rooms = data.get('rooms', {})
    room_data = rooms.get(room_type.lower(), {})
    return {
        "score": room_data.get('score', 0),
        "status": room_data.get('status', 'Unknown'),
        "checklist": room_data.get('checklist', []),
        "alerts": room_data.get('alerts', []),
        "issues": room_data.get('issues', []),
        "parameters_checked": room_data.get('parameters_checked', 0),
        "frames_analyzed": room_data.get('frames_analyzed', 0),
        "ai_fixes_provided": sum(1 for issue in room_data.get('issues', []) if issue.get('ai_generated', False))
    }

def get_actionable_fixes(data: Dict) -> List[Dict]:
    """Get all actionable fixes with priority ranking."""
    all_issues = data.get('all_issues', [])
    prioritized_fixes = sorted(all_issues, key=lambda x: 0 if x.get('severity') == 'high' else 1)
    return [
        {
            "zone": issue.get('zone', 'unknown'),
            "issue": issue.get('issue', ''),
            "fix": issue.get('fix', ''),
            "priority": "HIGH" if issue.get('severity') == 'high' else "MEDIUM",
            "parameter": issue.get('parameter', ''),
            "ai_generated": issue.get('ai_generated', False)
        }
        for issue in prioritized_fixes
    ]

# ======================================
# Celery Integration
# ======================================

def tracking_video(video_path: str, output_path: str = None, job_id: str = None, output_dir: str = str(OUTPUT_DIR)) -> Dict:
    """
    Celery task for room readiness analysis.

    Args:
        video_path: Path to input video
        output_path: Path to save output video (for tasks.py integration)
        job_id: VideoJob ID for progress tracking
        output_dir: Directory to save output frames

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    # Extract job ID from video path if not provided as parameter
    extracted_job_id = re.search(r'(\d+)', video_path)
    file_job_id = extracted_job_id.group(1) if extracted_job_id else str(int(time.time()))

    # Use provided job_id or fallback to extracted/generated one
    effective_job_id = job_id or file_job_id
    logger.info(f"🚀 Starting room readiness job {effective_job_id}")

    # Initialize progress logger for video processing
    progress_logger = create_progress_logger(
        job_id=str(effective_job_id),
        total_items=100,  # Estimate for video frames
        job_type="room_readiness"
    )

    progress_logger.update_progress(0, status="Starting room readiness analysis...", force_log=True)
    result = analyze_room_video_multi_zone_only(video_path, output_path, job_id, Path(output_dir))
    progress_logger.update_progress(100, status="Room readiness analysis completed", force_log=True)
    progress_logger.log_completion(100)

    processing_time = time.time() - start_time
    result['meta']['processing_time_seconds'] = processing_time
    result['meta']['timestamp'] = timezone.now().isoformat()

    return result

# Export all functions for compatibility
__all__ = [
    'analyze_room_video_multi_zone_only',
    'analyze_room_image',
    'extract_key_bedroom_frames',
    'test_api_connection',
    'get_current_config',
    'get_room_analysis_summary',
    'get_room_specific_data',
    'get_actionable_fixes',
    'parse_json_from_response',
    'convert_status_to_frontend',
    'process_ai_checklist_directly',
    'extract_issues_with_ai_fixes',
    'calculate_checklist_summary'
]
