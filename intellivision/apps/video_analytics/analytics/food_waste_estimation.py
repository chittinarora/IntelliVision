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
import time
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Union

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
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
OUTPUT_DIR = Path(settings.JOB_OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    file_path = Path(file_path).name
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
def analyze_food_image(image_path: str) -> Dict:
    """
    Analyze a food image to identify items, estimate portions, calories, and waste.

    Args:
        image_path: Path to input image

    Returns:
        Standardized response dictionary
    """
    start_time = time.time()
    image_path = Path(image_path).name
    is_valid, error_msg = validate_input_file(image_path)
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

    try:
        # Open image with default_storage
        with default_storage.open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            img.thumbnail((512, 512))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_data = buffer.getvalue()

        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')

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

        if not API_KEY:
            return get_mock_response()

        payload = {
            "messages": [
                {"role": "system",
                 "content": "You are an expert nutritionist that analyzes food images and returns structured JSON results."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
            "max_tokens": 1024,
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

        # Save output image
        output_filename = f"outputs/food_waste_{image_path}"
        with default_storage.open(image_path, 'rb') as f:
            default_storage.save(output_filename, f)
        output_url = default_storage.url(output_filename)

        return {
            'status': 'completed',
            'job_type': 'food_waste_estimation',
            'output_image': output_url,
            'output_video': None,
            'data': {**parsed, 'alerts': []},
            'meta': {'timestamp': timezone.now().isoformat(), 'processing_time_seconds': time.time() - start_time},
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


@shared_task(bind=True)
def tracking_image(self, input_path: str, job_id: str) -> Dict:
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
        job_type="food_waste_estimation"
    )

    # Update progress to show processing started
    progress_logger.update_progress(0, status="Initializing image analysis...", force_log=True)

    result = analyze_food_image(input_path)

    # Update progress to show completion
    progress_logger.update_progress(1, status="Analysis completed", force_log=True)
    progress_logger.log_completion(1)

    # Update Celery task state
    self.update_state(
        state='PROGRESS',
        meta={
            'progress': 100.0,
            'time_remaining': 0,
            'frame': 1,
            'total_frames': 1,
            'status': result['status'],
            'job_id': job_id
        }
    )

    processing_time = time.time() - start_time
    result['meta']['processing_time_seconds'] = processing_time
    result['meta']['timestamp'] = timezone.now().isoformat()

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
            job_type="food_waste_estimation"
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
