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
from utils.json_parser import parse_json_from_response

# --- Hotel Room Readiness Analysis ---
def analyze_room_image(image_path: str) -> Dict:
    """
    Analyze a hotel room image for readiness using OpenAI Vision.
    Returns a dict with checklist, bounding box info, readiness score, and status.
    """
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        prompt = """
You are an expert AI assistant specializing in hotel room readiness analysis. Your primary goal is to function as an automated housekeeping inspector.

Carefully analyze the provided photo of a hotel room and perform the following tasks:

1.  Zone Identification:
    Identify the primary zone shown in the image. You must choose one from the following list:

  - bedroom
  - bathroom
  - living_area
  - entryway

2.  Detailed Checklist Evaluation:
    Based on the identified zone, evaluate the room against the corresponding detailed checklist below.

  - Bedroom Criteria:

      - Bed made: Covers are smooth, taut, and unwrinkled.
      - Pillows arranged: Pillows are plumped and neatly arranged.
      - Bed runner present: A decorative runner is placed at the foot of the bed.
      - Bedside table clean: The surface is free of dust, stains, or items left by previous guests.
      - No clutter: The room is free of trash, misplaced items, or general untidiness.
      - Extra pillow/blanket present: An extra set is visibly available or stored correctly.

  - Bathroom Criteria:

      - Towels present: A full set of clean towels is neatly folded and hung.
      - Toiletries arranged: Amenities are full and arranged tidily.
      - Sink/counter clean: Surfaces are wiped down, dry, and free of residue.
      - Toilet seat down: The toilet lid and seat are down and clean.
      - No trash: Wastebaskets are empty or lined, and there is no loose trash.
      - Shower area clean: Glass is streak-free, and surfaces are clean and dry.

3.  Scoring and Instructions:

  - Calculate a readiness_score out of 100 based on the evaluation. Deduct points for any missing, defective, or untidy items.
  - Provide a list of clear, actionable instructions for housekeeping staff to resolve the identified issues.

You must format your entire response as a single JSON object inside a markdown code block. The JSON object must adhere to the following schema and rules:

1.  zone (string): The identified zone (e.g., "bedroom").
2.  checklist (array): An array of objects. This array must contain an object for every standard item listed in the criteria for the identified zone. Do not use "N/A". Each object must contain:
      - item (string): The name of the checklist item (e.g., "bed made").
      - status (string): Its status: present, missing, or defective.
      - box (array or null): The bounding box as [x_min, y_min, x_max, y_max] if the status is present. If status is missing or defective, or for abstract items like "no clutter," this value must be null.
      - notes (string): A brief explanation, required only if the status is missing or defective.
3.  readiness_score (integer): A score from 0 to 100.
4.  fail_reasons (array of strings): A list of all reasons points were deducted. If the score is 100, this should be an empty array [].
5.  instructions (array of strings): A list of actionable steps for staff.
6.  status (string): The final status: "Guest Ready" or "Not Guest Ready".
7.  image_width (integer): The width of the original image in pixels.
8.  image_height (integer): The height of the original image in pixels.

Example Output:

json
{
  "zone": "bedroom",
  "checklist": [
    {
      "item": "bed made",
      "status": "defective",
      "box": [145, 450, 1275, 850],
      "notes": "The quilt and blanket are wrinkled and not pulled taut."
    },
    {
      "item": "pillows arranged",
      "status": "present",
      "box": [455, 395, 745, 550]
    },
    {
      "item": "bed runner present",
      "status": "missing",
      "box": null,
      "notes": "No decorative bed runner was found at the foot of the bed."
    },
    {
      "item": "bedside table clean",
      "status": "present",
      "box": [1040, 610, 1250, 820]
    },
    {
      "item": "no clutter",
      "status": "present",
      "box": null
    },
    {
      "item": "extra pillow/blanket present",
      "status": "missing",
      "box": null,
      "notes": "An extra pillow or blanket is not visibly present in the room."
    }
  ],
  "readiness_score": 75,
  "fail_reasons": [
    "Bed is not neatly arranged.",
    "Bed runner is missing.",
    "Extra pillow or blanket not visible."
  ],
  "instructions": [
    "Smooth out wrinkles on the quilt and ensure it is pulled taut.",
    "Place a standard bed runner at the foot of the bed.",
    "Ensure an extra pillow and blanket are placed in the wardrobe."
  ],
  "status": "Not Guest Ready",
  "image_width": 1280,
  "image_height": 853
}

"""

        payload = {
            "messages": [
                {"role": "system", "content": "You are a precise hotel room readiness AI. Return JSON only."},
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
        logging.exception("OpenAI room image analysis failed")
        return {"error": str(e), "image": image_path}