"""
Food waste estimation analytics using Azure OpenAI GPT-4o-2 API.
Provides functions to analyze food images and estimate calories and waste.

Requires the environment variable AZURE_OPENAI_API_KEY to be set in a .env file.

Dependencies:
- requests
- python-dotenv
Install with: pip install requests python-dotenv
"""

# === Standard Library Imports ===
import os
import json
import re
import logging
from typing import List, Dict
from io import BytesIO

import requests
from PIL import Image
from django.conf import settings

# Canonical models directory for all analytics jobs
from pathlib import Path
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# --- Set up logger for this module ---
logger = logging.getLogger(__name__)

# --- Environment Setup ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    raise ImportError("python-dotenv is required. Install it with: pip install python-dotenv")

# --- Azure OpenAI API Setup ---
AZURE_OPENAI_ENDPOINT = "https://ai-labadministrator7921ai913285980324.openai.azure.com/openai/deployments/gpt-4o-2/chat/completions?api-version=2025-01-01-preview"
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("AZURE_OPENAI_API_KEY not set in environment. Please add it to your .env file.")

HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# --- JSON Parsing Helper ---
def parse_json_from_response(text: str) -> Dict:
    """
    Parse a JSON object from the model's response text.
    Handles markdown-wrapped and plain JSON.
    """
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

# --- Main Analysis ---
def analyze_food_image(image_path: str) -> Dict:
    """
    Analyze a food image to identify items, estimate portions and calories, and tag properties.
    Returns a unified result structure.
    """
    try:
        # Resize image to 512px width while maintaining aspect ratio
        with Image.open(image_path) as img:
            img.thumbnail((512, 512))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_data = buffer.getvalue()

        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')


        prompt = (
            "Identify all visible food items in this photo.\n"
                "For each item, return:\n"
                "- name\n"
                "- estimated portion in grams as an integer (e.g., 150, 50)\n"
                "- estimated calories as an integer (e.g., 120)\n"
                "- tags like spicy, oily, fried, etc., if applicable\n\n"
                "Output strictly in JSON, inside a markdown block, without any explanation or extra text.\n"
                "Example format:\n"
                "```json\n"
                "{\n"
                "  \"items\": [\n"
                "    {\"name\": \"White rice\", \"estimated_portion\": 150, \"estimated_calories\": 200, \"tags\": [\"plain\"]},\n"
                "    {\"name\": \"Cooked vegetables\", \"estimated_portion\": 50, \"estimated_calories\": 50, \"tags\": [\"stir-fried\"]}\n"
                "  ],\n"
                "  \"total_calories\": 250\n"
                "}\n"
                "```"
        )


        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that analyzes food images and returns structured JSON results."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
            "max_tokens": 1024,
            "temperature": 0.2
        }

        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=HEADERS, data=json.dumps(payload))
        if response.status_code != 200:
            return {
                'status': 'failed',
                'job_type': 'food_waste_estimation',
                'output_image': None,
                'results': {'alerts': [], 'error': f"OpenAI API error: {response.status_code} {response.text}"},
                'meta': {},
                'error': f"OpenAI API error: {response.status_code} {response.text}"
            }
        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception as e:
            return {
                'status': 'failed',
                'job_type': 'food_waste_estimation',
                'output_image': None,
                'results': {'alerts': [], 'error': f"Unexpected API response: {str(e)}"},
                'meta': {},
                'error': f"Unexpected API response: {str(e)}"
            }
        parsed = parse_json_from_response(text)
        # Convert image_path to a URL for output_image
        rel_path = os.path.relpath(image_path, settings.MEDIA_ROOT)
        output_image_url = settings.MEDIA_URL + rel_path.replace(os.sep, '/')
        if not output_image_url.startswith('/api/media/'):
            output_image_url = '/api/media/' + rel_path.replace(os.sep, '/')
        return {
            'status': 'completed',
            'job_type': 'food_waste_estimation',
            'output_image': output_image_url,
            'data': {**parsed, 'alerts': []},
            'meta': {},
            'error': None
        }
    except Exception as e:
        logging.exception("OpenAI image analysis failed")
        return {
            'status': 'failed',
            'job_type': 'food_waste_estimation',
            'output_image': None,
            'results': {'alerts': [], 'error': str(e)},
            'meta': {},
            'error': str(e)
        }

# --- Batch Function ---
def analyze_multiple_food_images(image_paths: List[str]) -> List[Dict] | Dict:
    """
    Analyze multiple food images and return:
    - A single dict if only one image is provided
    - A list of dicts if multiple images are provided
    """
    results = [analyze_food_image(path) for path in image_paths]
    if len(image_paths) == 1:
        return results[0]
    return results


OUTPUT_DIR = settings.JOB_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
