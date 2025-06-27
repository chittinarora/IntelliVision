"""
Food waste estimation analytics using Azure OpenAI GPT-4o-2 API.
Provides functions to analyze food images and estimate calories and waste.

Requires the environment variable AZURE_OPENAI_API_KEY to be set in a .env file.

Dependencies:
- requests
- python-dotenv
Install with: pip install requests python-dotenv
"""

import os
import json
import re
import logging
import requests
from typing import List, Dict

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

# --- Main Food Waste Estimation ---
def analyze_food_image(image_path: str) -> Dict:
    """
    Analyze a food image to identify items, estimate portions and calories, and tag properties.
    Returns a dictionary with items and total calories.
    """
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        prompt = (
            "Identify all visible food items in this photo.\n"
            "For each item, return:\n"
            "- name\n"
            "- estimated portion (g or ml)\n"
            "- estimated calories\n"
            "- tags like spicy/oily/fried if applicable\n\n"
            "Format the output strictly in JSON, inside a markdown block:\n"
            "```json\n"
            "{\n  \"items\": [...],\n  \"total_calories\": ...\n}\n"
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
            return {"error": f"OpenAI API error: {response.status_code} {response.text}"}
        data = response.json()
        # Extract the model's reply
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception as e:
            return {"error": f"Unexpected API response: {str(e)}", "raw_response": data}
        return parse_json_from_response(text)

    except Exception as e:
        logging.exception("OpenAI image analysis failed")
        return {"error": str(e), "image": image_path}

# --- Batch Food Waste Estimation ---
def analyze_multiple_food_images(image_paths: List[str]) -> List[Dict]:
    """
    Analyze multiple food images and return a list of results for each image.
    """
    results = []
    for path in image_paths:
        result = analyze_food_image(path)
        result["image"] = str(path)
        results.append(result)
    return results
