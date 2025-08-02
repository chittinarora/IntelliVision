"""
Food waste estimation analytics using Azure OpenAI GPT-4o-2 API.
Analyzes food images to identify items, estimate portions, calories, and waste.
"""

import json
import logging
# ======================================
# Imports and Setup
# ======================================
import os
import re
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Union

import cv2
import numpy as np
import requests
from PIL import Image
from celery import shared_task
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.mp4'}  # Added .mp4 for video support
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Video processing constants
MIN_FRAMES = 3
MAX_FRAMES = 8  # Fewer frames needed for food analysis
MAX_BLUR_THRESHOLD = 40

# Azure OpenAI API Setup
AZURE_OPENAI_ENDPOINT = "https://ai-labadministrator7921ai913285980324.openai.azure.com/openai/deployments/gpt-4o-2/chat/completions?api-version=2025-01-01-preview"
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if not API_KEY:
    logger.warning("AZURE_OPENAI_API_KEY not set. Using mock response.")

HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY or "mock-key",
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
        return False, f"File size {size / (1024 * 1024):.2f}MB exceeds 500MB limit"

    return True, ""


def parse_json_from_response(text: str) -> Dict:
    """Parse JSON from model response."""
    try:
        json_match = re.search(r"```json\n(\{.*?\})\n```", text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            json_text_match = re.search(r"(\{.*?\})", text, re.DOTALL)
            if json_text_match:
                json_text = json_text_match.group(1)
            else:
                return {"error": "No JSON object found in the response"}
        return json.loads(json_text)
    except Exception as e:
        return {"error": f"Parsing error: {str(e)}", "raw_response": text}


def assess_frame_quality(image: np.ndarray) -> Dict[str, float]:
    """Assess frame quality for blurriness."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {'blur_score': blur_score, 'is_blurry': blur_score < MAX_BLUR_THRESHOLD}
    except Exception:
        return {'blur_score': 0, 'is_blurry': True}


def extract_key_food_frames(video_path: str, output_dir: Path = OUTPUT_DIR) -> List[str]:
    """Extract visually distinct frames from food video for analysis."""
    start_time = time.time()
    logger.debug("Starting food video frame extraction...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate the original video file
    original_video_path = video_path
    video_path = Path(video_path).name
    is_valid, error_msg = validate_input_file(video_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return []

    try:
        # Open video from Django storage
        with default_storage.open(video_path, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

        vidcap = cv2.VideoCapture(tmp_path)
        if not vidcap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        fps = vidcap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = int(fps)  # Sample 1 frame per second
        logger.debug(f"Video FPS: {fps:.2f}, processing one frame every {frame_interval} frames.")

        selected_frame_paths = []
        selected_histograms = []
        frame_idx = 0
        job_id = re.search(r'(\d+)', video_path)
        job_id = job_id.group(1) if job_id else str(int(time.time()))

        while len(selected_frame_paths) < MAX_FRAMES:
            success, frame = vidcap.read()
            if not success:
                break

            if frame_idx % frame_interval == 0:
                # Quality assessment
                quality = assess_frame_quality(frame)
                if quality['is_blurry']:
                    frame_idx += 1
                    continue

                try:
                    # Histogram for diversity check
                    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                    if not selected_frame_paths:
                        # Select first good frame
                        logger.debug(f"Selecting first frame (index {frame_idx}).")
                        path = output_dir / f"food_frame_{job_id}_{frame_idx:05}.jpg"
                        cv2.imwrite(str(path), frame)

                        # Save to storage and add to selected paths
                        frame_filename = f"food_frame_{job_id}_{frame_idx:05}.jpg"
                        with open(str(path), 'rb') as f:
                            default_storage.save(frame_filename, f)
                        selected_frame_paths.append(frame_filename)
                        selected_histograms.append(hist)
                    else:
                        # Check for visual diversity
                        is_too_similar = any(cv2.compareHist(hist, h, cv2.HISTCMP_CORREL) > 0.90
                                             for h in selected_histograms)  # Slightly less strict for food
                        if not is_too_similar:
                            logger.debug(f"Selecting distinct frame (index {frame_idx}).")
                            path = output_dir / f"food_frame_{job_id}_{frame_idx:05}.jpg"
                            cv2.imwrite(str(path), frame)

                            # Save to storage and add to selected paths
                            frame_filename = f"food_frame_{job_id}_{frame_idx:05}.jpg"
                            with open(str(path), 'rb') as f:
                                default_storage.save(frame_filename, f)
                            selected_frame_paths.append(frame_filename)
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


def consolidate_food_analysis_reports(reports: List[Dict]) -> Dict:
    """Consolidate multiple frame analyses into unified food waste report."""
    if not reports:
        return {"error": "No reports to consolidate"}

    # Aggregate food items across all frames
    all_items = {}
    total_frames = len(reports)

    # Collect all unique food items
    for report in reports:
        items = report.get('data', {}).get('items', [])
        for item in items:
            name = item.get('name', 'Unknown')
            if name not in all_items:
                all_items[name] = {
                    'name': name,
                    'portion_estimates': [],
                    'calorie_estimates': [],
                    'waste_percentages': [],
                    'confidence_scores': [],
                    'all_tags': []
                }

            all_items[name]['portion_estimates'].append(item.get('estimated_portion_grams', 0))
            all_items[name]['calorie_estimates'].append(item.get('estimated_calories', 0))
            all_items[name]['waste_percentages'].append(item.get('percent_uneaten', 0))
            all_items[name]['confidence_scores'].append(item.get('confidence_score', 0.5))
            all_items[name]['all_tags'].extend(item.get('tags', []))

    # Calculate consolidated values for each item
    consolidated_items = []
    for name, data in all_items.items():
        # Use median for more robust estimates
        consolidated_item = {
            'name': name,
            'estimated_portion_grams': int(np.median(data['portion_estimates'])),
            'estimated_calories': int(np.median(data['calorie_estimates'])),
            'percent_uneaten': int(np.median(data['waste_percentages'])),
            'confidence_score': round(np.mean(data['confidence_scores']), 2),
            'tags': list(set(data['all_tags']))  # Unique tags
        }
        consolidated_items.append(consolidated_item)

    # Calculate summary statistics
    total_calories_served = sum(item['estimated_calories'] for item in consolidated_items)
    total_calories_wasted = sum(
        item['estimated_calories'] * (item['percent_uneaten'] / 100)
        for item in consolidated_items
    )
    overall_waste_percentage = int((total_calories_wasted / total_calories_served * 100)
                                   if total_calories_served > 0 else 0)

    return {
        'items': consolidated_items,
        'total_calories_served': total_calories_served,
        'total_calories_wasted': int(total_calories_wasted),
        'overall_waste_percentage': overall_waste_percentage,
        'frames_analyzed': total_frames,
        'analysis_method': 'multi_frame_consolidation'
    }


# ======================================
# Mock Response for API Failure
# ======================================

def get_mock_response() -> Dict:
    """Return mock response for Azure OpenAI API failure."""
    logger.warning("Using mock response due to API failure")
    return {
        'status': 'failed',
        'job_type': 'food_waste_estimation',
        'output_image': None,
        'output_video': None,
        'data': {'alerts': [], 'error': 'API unavailable'},
        'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': 0.0},
        'error': {'message': 'Azure OpenAI API unavailable', 'code': 'API_UNAVAILABLE'}
    }


# ======================================
# Main Analysis Functions
# ======================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    reraise=True
)
def analyze_food_image(image_path: Union[str, List[str]], output_path: str = None, job_id: str = None) -> Dict:
    """
    Analyze food image(s) to identify items, estimate portions, calories, and waste.
    NOW SUPPORTS BOTH SINGLE IMAGES AND MULTIPLE IMAGES (from video frames).

    Args:
        image_path: Path to input image OR list of image paths
        output_path: Path to save output image (for tasks.py integration)
        job_id: VideoJob ID for progress tracking

    Returns:
        Standardized response dictionary with filesystem paths
    """
    start_time = time.time()

    # Add job_id logging for progress tracking
    if job_id:
        logger.info(f"🚀 Starting food waste estimation job {job_id}")

    # Handle both single image and multiple images
    if isinstance(image_path, str):
        image_paths, is_multi_frame = [image_path], False
    elif isinstance(image_path, list) and image_path:
        image_paths, is_multi_frame = image_path, len(image_path) > 1
    else:
        return {
            'status': 'failed',
            'job_type': 'food_waste_estimation',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': 'Invalid image_path parameter'},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': 'Invalid image_path parameter', 'code': 'INVALID_INPUT'}
        }

    # Validate all images
    for img_path in image_paths:
        is_valid, error_msg = validate_input_file(img_path)
        if not is_valid:
            logger.error(f"Invalid input: {error_msg}")
            return {
                'status': 'failed',
                'job_type': 'food_waste_estimation',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': error_msg},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
            }

    # Create appropriate prompts
    if is_multi_frame:
        prompt = """You are an expert nutritionist specializing in food waste analysis.
You will be given multiple images of the SAME MEAL taken at different times (e.g., before eating, during eating, after eating).

CRITICAL INSTRUCTIONS:
1. Analyze EACH image INDIVIDUALLY to track how the meal changes over time.
2. For each image, produce a complete JSON report with all visible food items.
3. Return one final JSON object with key "frame_reports" containing the list of individual reports.
4. IMPORTANT: You must analyze ALL images provided - do not skip any images.

For every food item in each frame, provide these fields:
- name: Name of the food item (string).
- estimated_portion_grams: Total portion size visible in THIS frame (integer).
- estimated_calories: Calories for the portion visible in THIS frame (integer).
- percent_uneaten: Percentage of the original item left uneaten in THIS frame (integer, 0-100).
- confidence_score: Confidence in this estimation (float, 0.0-1.0).
- tags: List of relevant food descriptors (e.g., ["fried", "spicy", "vegan"]).

OUTPUT STRUCTURE:
```json
{
  "frame_reports": [
    {
      "items": [
        {"name": "White rice", "estimated_portion_grams": 150, "estimated_calories": 200, "percent_uneaten": 0, "confidence_score": 0.9, "tags": ["plain"]},
        {"name": "Chicken curry", "estimated_portion_grams": 120, "estimated_calories": 250, "percent_uneaten": 0, "confidence_score": 0.85, "tags": ["spicy"]}
      ],
      "total_calories_served": 450,
      "total_calories_wasted": 0,
      "overall_waste_percentage": 0
    },
    {
      "items": [
        {"name": "White rice", "estimated_portion_grams": 75, "estimated_calories": 100, "percent_uneaten": 50, "confidence_score": 0.9, "tags": ["plain"]},
        {"name": "Chicken curry", "estimated_portion_grams": 108, "estimated_calories": 225, "percent_uneaten": 10, "confidence_score": 0.85, "tags": ["spicy"]}
      ],
      "total_calories_served": 450,
      "total_calories_wasted": 125,
      "overall_waste_percentage": 28
    }
  ]
}
```"""
    else:
        prompt = (
            "You are an expert nutritionist specializing in food waste analysis.\n"
            "Analyze the provided image of a meal. Identify all visible food items and estimate waste for each.\n\n"
            "For every food item, provide these fields:\n"
            "- name: Name of the food item (string).\n"
            "- estimated_portion_grams: Total served portion size in grams (integer).\n"
            "- estimated_calories: Calories for the served portion (integer).\n"
            "- percent_uneaten: Percentage of the item left uneaten (integer, 0-100).\n"
            "- confidence_score: Confidence in this estimation (float, 0.0-1.0).\n"
            "- tags: List of relevant food descriptors (e.g., [\"fried\", \"spicy\", \"vegan\"]). Leave empty if no tags apply.\n\n"
            "Also, in the main JSON object, include these summary fields:\n"
            "- total_calories_served: The sum of all calories served.\n"
            "- total_calories_wasted: The total estimated calories that were wasted.\n"
            "- overall_waste_percentage: The overall percentage of the meal that was wasted.\n\n"
            "Respond ONLY with a strictly valid JSON object inside a markdown code block labeled 'json'. Do not include any explanations, apologies, or comments outside the code block.\n"
            "Here is the format to use:\n"
            "```json\n"
            "{\n"
            "  \"items\": [\n"
            "    {\"name\": \"White rice\", \"estimated_portion_grams\": 150, \"estimated_calories\": 200, \"percent_uneaten\": 50, \"confidence_score\": 0.9, \"tags\": [\"plain\"]},\n"
            "    {\"name\": \"Chicken curry\", \"estimated_portion_grams\": 120, \"estimated_calories\": 250, \"percent_uneaten\": 10, \"confidence_score\": 0.85, \"tags\": [\"spicy\", \"fried\"]}\n"
            "  ],\n"
            "  \"total_calories_served\": 450,\n"
            "  \"total_calories_wasted\": 125,\n"
            "  \"overall_waste_percentage\": 28\n"
            "}\n"
            "```\n"
        )

    try:
        # Process all images
        encoded_images, valid_paths = [], []
        for img_path in image_paths:
            try:
                with default_storage.open(img_path, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                    img.thumbnail((512, 512))
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG")
                    image_data = buffer.getvalue()

                import base64
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                encoded_images.append(image_b64)
                valid_paths.append(img_path)
            except Exception as e:
                logger.error(f"Failed to encode image {img_path}: {e}")
                continue

        if not encoded_images:
            return {
                'status': 'failed',
                'job_type': 'food_waste_estimation',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': 'No valid images to analyze'},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': 'No valid images to analyze', 'code': 'INVALID_INPUT'}
            }

        if not API_KEY:
            return get_mock_response()

        # Build API request content
        content = [{"type": "text", "text": prompt}]
        for img_b64 in encoded_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

        payload = {
            "messages": [
                {"role": "system",
                 "content": "You are an expert nutritionist that analyzes food images and returns structured JSON results."},
                {"role": "user", "content": content}
            ],
            "max_tokens": 2048 if is_multi_frame else 1024,
            "temperature": 0.2
        }

        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        parsed = parse_json_from_response(text)

        if "error" in parsed:
            logger.error(f"API response parsing failed: {parsed['error']}")
            return {
                'status': 'failed',
                'job_type': 'food_waste_estimation',
                'output_image': None,
                'output_video': None,
                'data': {'alerts': [], 'error': parsed['error']},
                'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
                'error': {'message': parsed['error'], 'code': 'API_PARSING_ERROR'}
            }

        # Handle multi-frame vs single-frame results
        if is_multi_frame:
            frame_reports = parsed.get("frame_reports", [])
            # Convert frame reports to the format expected by consolidation
            formatted_reports = []
            for report in frame_reports:
                formatted_reports.append({'data': report})

            consolidated_data = consolidate_food_analysis_reports(formatted_reports)
        else:
            consolidated_data = parsed

        # Save output image (first image for multi-frame)
        output_image = valid_paths[0] if valid_paths else None
        if output_image:
            # Create temporary output file for tasks.py integration
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as output_tmp:
                # Copy original image to temporary file for output
                with default_storage.open(output_image, 'rb') as f:
                    output_tmp.write(f.read())
                final_output_path = output_tmp.name
        else:
            final_output_path = None

        logger.info(f"✅ Food waste analysis completed, output saved to {final_output_path}")

        processing_time = time.time() - start_time
        return {
            'status': 'completed',
            'job_type': 'food_waste_estimation',
            'output_image': final_output_path,
            'output_video': None,
            'data': {**consolidated_data, 'alerts': []},
            'meta': {
                'timestamp': timezone.now().isoformat(),
                'processing_time_seconds': processing_time,
                'frames_analyzed': len(valid_paths)
            },
            'error': None
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Azure OpenAI API failed: {str(e)}")
        return get_mock_response()
    except Exception as e:
        logger.exception(f"Food waste analysis failed: {str(e)}")
        return {
            'status': 'failed',
            'job_type': 'food_waste_estimation',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': str(e)},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': str(e), 'code': 'PROCESSING_ERROR'}
        }


def tracking_image(input_path: str, output_path: str = None, job_id: str = None) -> Dict:
    """
    Celery task for food waste estimation.

    Args:
        self: Celery task instance
        input_path: Path to input image
        job_id: VideoJob ID

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    logger.info(f"🚀 Starting food waste estimation job {job_id}")

    # Initialize progress logger for image processing
    progress_logger = create_progress_logger(
        job_id=str(job_id),
        total_items=1,  # Single image
        job_type="food-waste-estimation"
    )

    # Update progress to show processing started
    progress_logger.update_progress(0, status="Initializing image analysis...", force_log=True)

    result = analyze_food_image(input_path, None, job_id)

    # Update progress to show completion
    progress_logger.update_progress(1, status="Analysis completed", force_log=True)
    progress_logger.log_completion(1)

    # Update Celery task state

    processing_time = time.time() - start_time
    result['meta']['processing_time_seconds'] = processing_time
    result['meta']['timestamp'] = timezone.now().isoformat()

    return result


def analyze_food_video(video_path: str, output_dir: Path = OUTPUT_DIR, job_id: str = None) -> Dict:
    """
    Analyze video for food waste by extracting frames and using image analysis.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        job_id: VideoJob ID for progress tracking

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    video_path = Path(video_path).name
    is_valid, error_msg = validate_input_file(video_path)
    if not is_valid:
        logger.error(f"Invalid input: {error_msg}")
        return {
            'status': 'failed',
            'job_type': 'food_waste_estimation',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': error_msg},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': error_msg, 'code': 'INVALID_INPUT'}
        }

    # Extract frames from video
    frame_paths = extract_key_food_frames(video_path, output_dir)
    if not frame_paths:
        return {
            'status': 'failed',
            'job_type': 'food_waste_estimation',
            'output_image': None,
            'output_video': None,
            'data': {'alerts': [], 'error': 'No frames could be extracted from video'},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
            'error': {'message': 'No frames could be extracted from video', 'code': 'FRAME_EXTRACTION_ERROR'}
        }

    # Analyze extracted frames using existing image analysis
    result = analyze_food_image(frame_paths, None, job_id)

    processing_time = time.time() - start_time
    if result.get('status') == 'completed':
        return {
            'status': 'completed',
            'job_type': 'food_waste_estimation',
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
        'job_type': 'food_waste_estimation',
        'output_image': None,
        'output_video': None,
        'data': result.get('data', {'alerts': [], 'error': result.get('error', 'Unknown error')}),
        'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': processing_time},
        'error': result.get('error', {'message': 'Unknown error', 'code': 'UNKNOWN'})
    }


def tracking_video(input_path: str, job_id: str, output_dir: str = str(OUTPUT_DIR)) -> Dict:
    """
    Celery task for food waste estimation (VIDEOS).

    Args:
        input_path: Path to input video
        job_id: VideoJob ID
        output_dir: Directory to save output frames

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    logger.info(f"🚀 Starting food waste video analysis job {job_id}")

    # Initialize progress logger for video processing
    progress_logger = create_progress_logger(
        job_id=str(job_id),
        total_items=100,  # Use percentage for video
        job_type="food-waste-estimation"
    )

    # Update progress to show processing started
    progress_logger.update_progress(0, status="Starting video analysis for food waste...", force_log=True)

    result = analyze_food_video(input_path, Path(output_dir), job_id)

    # Update progress to show completion
    progress_logger.update_progress(100, status="Video analysis completed", force_log=True)
    progress_logger.log_completion(100)

    processing_time = time.time() - start_time
    result['meta']['processing_time_seconds'] = processing_time
    result['meta']['timestamp'] = timezone.now().isoformat()

    logger.info(
        f"**Job {job_id}**: Progress **100.0%** ({result['meta'].get('frames_analyzed', 1)}/{result['meta'].get('frames_analyzed', 1)}), Status: {result['status']}...")
    logger.info(
        f"[##########] Done: {int(processing_time // 60):02d}:{int(processing_time % 60):02d} | Left: 00:00 | Avg FPS: N/A")

    return result


def analyze_multiple_food_images(image_paths: List[str], job_id: str = None) -> Union[List[Dict], Dict]:
    """
    Analyze multiple food images.

    Args:
        image_paths: List of image paths
        job_id: Job identifier for progress logging

    Returns:
        Single dict for one image, list of dicts for multiple
    """
    start_time = time.time()
    total_images = len(image_paths)
    results = []

    # Initialize progress logger for batch processing
    if job_id:
        progress_logger = create_progress_logger(
            job_id=str(job_id),
            total_items=total_images,
            job_type="food-waste-estimation"
        )
    else:
        progress_logger = None

    for idx, image_path in enumerate(image_paths, 1):
        result = analyze_food_image(image_path)
        results.append(result)

        # Update progress if logger is available
        if progress_logger:
            progress_logger.update_progress(
                idx,
                status="Awaiting API response..."
            )

    # Log completion if logger is available
    if progress_logger:
        progress_logger.log_completion(total_images)

    return results[0] if len(image_paths) == 1 else results
