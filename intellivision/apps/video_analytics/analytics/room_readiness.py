"""
Enhanced Hotel room readiness analytics using Azure OpenAI GPT-4o-2 API.
FIXED VERSION - Addresses room type detection, deduplication, and consolidation issues.
100% Drop-in replacement with improved room detection and smart deduplication.

Fixes:
1. Improved room type detection in AI prompts
2. Smart location-based deduplication
3. Better room consolidation logic
4. Preserved individual room identities
5. Fixed text cleanup issues

Requires the environment variable AZURE_OPENAI_API_KEY to be set in a .env file.
"""

import os
import json
import re
import logging
import requests
import numpy as np
from typing import List, Dict, Union
import cv2
import base64
import time

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

# --- Configuration ---
MIN_FRAMES = 5
MAX_FRAMES = 10
MAX_BLUR_THRESHOLD = 40

# API Configuration
API_TIMEOUT_CONNECT = 30
API_TIMEOUT_READ = 300
MAX_IMAGE_SIZE = 3 * 1024 * 1024
MAX_PAYLOAD_SIZE = 8 * 1024 * 1024

# --- Output Directory Setup ---
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'media/outputs'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Room Parameter Definitions ---
ROOM_PARAMETERS = {
    "bedroom": [
        "Pillow Count & Placement",
        "Comforter/Duvet/Bed Runner Presence",
        "Bed Covers Cleanliness",
        "Minibar/Fridge Status",
        "Bed Making Quality",
        "Room Lighting",
        "Curtains/Blinds",
        "General Cleanliness",
        "Floor Condition",
        "Electronics"
    ],
    "bathroom": [
        "Bathtub Cleanliness",
        "Toilet Flush Lid Position",
        "Bathroom Floor Dryness",
        "Toiletries & Towel Placement",
        "Mirror Cleanliness",
        "Sink/Counter Cleanliness",
        "Shower Area Clean",
        "Ventilation",
        "Safety Features",
        "Amenities Present"
    ],
    "living_area": [
        "Remote Controls Present",
        "Seating Arrangement",
        "Coffee Table Cleanliness",
        "Entertainment Center",
        "General Cleanliness",
        "Temperature Control"
    ],
    "entryway": [
        "Floor Condition",
        "General Cleanliness",
        "Organization",
        "Lighting"
    ],
    "closet": [
        "Organization",
        "General Cleanliness",
        "Floor Condition",
        "Lighting"
    ],
    "vanity_area": [
        "Mirror Cleanliness",
        "General Cleanliness",
        "Organization",
        "Lighting"
    ]
}


# --- Helper Functions ---
def parse_json_from_response(text: str) -> Dict:
    """More robustly parse a JSON object from the model's response text."""
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
    """Convert AI response status to frontend format"""
    status_mapping = {
        "present": "Ready",
        "missing": "Issue Found",
        "defective": "Issue Found",
        "needs_attention": "Needs Attention"
    }
    return status_mapping.get(status.lower(), "Needs Attention")


def process_ai_checklist_directly(ai_checklist: List[Dict]) -> List[Dict]:
    """Process AI checklist with corrected logic - FIXED VERSION"""
    processed_checklist = []

    for item in ai_checklist:
        ai_status = item.get('status', 'present').lower()
        notes = item.get('notes', '').lower()

        # CRITICAL FIX: Determine frontend status based on actual content AND AI status
        positive_indicators = [
            'no visible stains detected',
            'no personal items or clutter visible',
            'clean',
            'properly positioned',
            'adequately stocked',
            'present and symmetrically arranged',
            'working and clean',
            'neatly positioned',
            'functional and placed properly'
        ]

        negative_indicators = [
            'wrinkled',
            'scattered',
            'missing',
            'dirty',
            'unclean',
            'disorganized',
            'clutter',
            'personal items detected',
            'has items',
            'misplaced',
            'not visible',
            'not properly'
        ]

        # FIXED LOGIC: Combine AI status with content analysis
        has_positive_content = any(indicator in notes for indicator in positive_indicators)
        has_negative_content = any(indicator in notes for indicator in negative_indicators)

        if ai_status == 'present':
            # AI says parameter is satisfied
            frontend_status = "Ready"
        elif ai_status in ['missing', 'defective']:
            # AI detected an issue - verify with content
            if has_negative_content and not has_positive_content:
                frontend_status = "Issue Found"
            elif has_positive_content:
                # Positive content overrides AI status (AI made an error)
                frontend_status = "Ready"
            else:
                # Unclear content, trust AI but be conservative
                frontend_status = "Issue Found"
        else:
            # Default case
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
    """Clean up issue text to remove incomplete sentences and artifacts"""
    if not text:
        return text

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove leading/trailing commas and spaces
    text = text.strip(' ,.')

    # Fix common incomplete patterns
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
    """
    FIXED: Smart deduplication that ensures each physical location is mentioned only once.
    Prevents minibar, nightstand, coffee table from appearing in multiple issues.
    """
    print(f"[DEBUG DEDUP] Starting deduplication with {len(room_issues)} issues")

    # Define key locations that should appear only once per room
    key_locations = [
        'minibar', 'fridge', 'nightstand', 'bedside', 'coffee table',
        'seating', 'sofa', 'chair', 'bed', 'toilet', 'bathtub', 'shower',
        'mirror', 'sink', 'counter', 'closet', 'entryway', 'vanity'
    ]

    # Track which locations have been claimed by which parameters
    location_claims = {}
    deduplicated_issues = []

    # Sort issues by parameter priority (most specific first)
    parameter_priority = {
        'minibar/fridge': 1,
        'bedside': 2,
        'coffee table': 3,
        'seating': 4,
        'bed making': 5,
        'personal items': 10  # Least specific, lowest priority
    }

    def get_priority(issue):
        param = issue.get('parameter', '').lower()
        for key, priority in parameter_priority.items():
            if key in param:
                return priority
        return 5  # Default priority

    sorted_issues = sorted(room_issues, key=get_priority)

    for issue in sorted_issues:
        issue_text = issue.get('issue', '').lower()
        parameter = issue.get('parameter', '').lower()

        # Find which locations this issue mentions
        mentioned_locations = []
        for location in key_locations:
            if location in issue_text:
                mentioned_locations.append(location)

        print(f"[DEBUG DEDUP] Issue '{parameter}' mentions locations: {mentioned_locations}")

        # Check if any mentioned location is already claimed
        conflict_found = False
        conflicting_locations = []

        for location in mentioned_locations:
            if location in location_claims:
                conflict_found = True
                conflicting_locations.append(location)
                print(f"[DEBUG DEDUP] CONFLICT: {location} already claimed by {location_claims[location]}")

        if conflict_found:
            # This issue conflicts with a higher priority issue
            # Modify the issue text to remove conflicting locations
            modified_issue_text = issue.get('issue', '')

            for location in conflicting_locations:
                # Remove mentions of conflicting locations
                location_phrases = [
                    f"{location} surface",
                    f"on {location}",
                    f"on the {location}",
                    f"{location} area",
                    f"and {location}",
                    location
                ]

                for phrase in location_phrases:
                    modified_issue_text = re.sub(r'\b' + re.escape(phrase) + r'\b', '', modified_issue_text,
                                                 flags=re.IGNORECASE)

            # Clean up the modified text
            modified_issue_text = clean_issue_text(modified_issue_text)

            # If text becomes too short or meaningless, create a generic description
            if (len(modified_issue_text) < 15 or
                    not any(word in modified_issue_text.lower() for word in
                            ['clutter', 'items', 'clean', 'arrange', 'organize', 'personal', 'visible'])):
                param_name = parameter.replace('&', 'and').lower()
                modified_issue_text = f"General organization and cleanliness issues in {param_name}"

            # Only keep this issue if there's meaningful content
            if len(modified_issue_text) > 8:
                modified_issue = issue.copy()
                modified_issue['issue'] = modified_issue_text
                modified_issue['fix'] = f"Address remaining {parameter.lower()} issues"
                deduplicated_issues.append(modified_issue)
                print(f"[DEBUG DEDUP] MODIFIED: {parameter} -> '{modified_issue_text}'")
            else:
                print(f"[DEBUG DEDUP] DROPPED: {parameter} (no meaningful content after deduplication)")
        else:
            # No conflict, keep this issue and claim its locations
            for location in mentioned_locations:
                location_claims[location] = parameter
            deduplicated_issues.append(issue)
            print(f"[DEBUG DEDUP] KEPT: {parameter} (claimed locations: {mentioned_locations})")

    print(f"[DEBUG DEDUP] Deduplication complete: {len(room_issues)} -> {len(deduplicated_issues)} issues")
    print(f"[DEBUG DEDUP] Final location claims: {location_claims}")

    return deduplicated_issues


def calculate_room_score_improved(ai_score: int, actual_issues: List[Dict]) -> int:
    """Calculate room score with improved penalty system - FIXED VERSION"""

    # Only count real issues for scoring
    real_issues = [issue for issue in actual_issues
                   if 'no action required' not in issue.get('fix', '').lower()]

    if len(real_issues) == 0:
        # No real issues found, use AI score but cap reasonably
        return min(ai_score, 100)

    # IMPROVED: More reasonable penalty system
    total_penalty = 0
    for issue in real_issues:
        severity = issue.get('severity', 'medium')
        parameter = issue.get('parameter', '').lower()

        # More reasonable penalties
        if 'personal items' in parameter or 'clutter' in parameter:
            # Medium penalty for guest comfort issues
            penalty = 12 if severity == 'high' else 8
        elif 'bed' in parameter or 'seating' in parameter:
            # Medium penalty for comfort issues
            penalty = 10 if severity == 'high' else 7
        elif 'organization' in parameter or 'arrangement' in parameter:
            # Lower penalty for minor organization issues
            penalty = 8 if severity == 'high' else 5
        else:
            # Standard penalty
            penalty = 10 if severity == 'high' else 6

        total_penalty += penalty

    # Calculate final score with reasonable minimum threshold
    final_score = max(ai_score - total_penalty, 55)  # Minimum 55 instead of 40

    return final_score


def consolidate_reports_by_room_type_fixed(reports: List[Dict]) -> Dict:
    """
    FIXED: Consolidates individual frame reports with improved room type preservation.
    Prevents over-consolidation and maintains individual room identities.
    """
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

    print(f"[DEBUG FIXED] Processing {len(reports)} frame reports")

    # FIXED: Don't group by zone type, preserve individual room identities
    room_groups = {}

    for i, report in enumerate(reports):
        if not isinstance(report, dict) or 'zone' not in report:
            continue

        zone = report.get('zone', 'unknown').lower()

        # FIXED: Better room type detection and preservation
        if zone in ['other', 'unknown', '']:
            # Try to determine room type from parameters instead of defaulting to general_area
            checklist = report.get('checklist', [])

            # More specific room type detection
            bathroom_indicators = ['toilet', 'bathtub', 'bathroom', 'sink', 'shower', 'towel', 'mirror']
            bedroom_indicators = ['bed', 'pillow', 'comforter', 'duvet', 'minibar', 'nightstand']
            living_indicators = ['seating', 'coffee', 'sofa', 'remote', 'entertainment']
            closet_indicators = ['closet', 'hanging', 'wardrobe', 'organization', 'clothes']
            entryway_indicators = ['entryway', 'entrance', 'foyer', 'entry']
            vanity_indicators = ['vanity', 'dressing', 'makeup']

            # Check parameters for room type clues
            param_text = ' '.join([item.get('parameter', '') + ' ' + item.get('notes', '')
                                   for item in checklist]).lower()

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
                # Use frame position to guess room type
                if i == 0:
                    zone = 'entryway'  # First frame often entryway
                elif i == len(reports) - 1:
                    zone = 'closet'  # Last frame often closet
                else:
                    zone = f'other_area_{i}'  # Preserve as separate room

        # FIXED: Don't consolidate different room instances
        # Each frame gets its own room entry unless it's clearly the same room type
        room_key = zone
        if zone in room_groups:
            # Only merge if it's clearly the same room (same zone + similar score)
            existing_score = room_groups[zone][0].get('readiness_score', 0)
            current_score = report.get('readiness_score', 0)

            # If scores are very different, treat as separate rooms
            if abs(existing_score - current_score) > 20:
                room_key = f"{zone}_{len([k for k in room_groups.keys() if k.startswith(zone)])}"

        if room_key not in room_groups:
            room_groups[room_key] = []
        room_groups[room_key].append(report)

    print(f"[DEBUG FIXED] Room groups after improved detection: {list(room_groups.keys())}")

    # Analyze each room type with FIXED deduplication
    room_analysis = {}
    all_unique_issues = []
    total_rooms_analyzed = 0
    total_rooms_ready = 0

    for room_type, room_reports in room_groups.items():
        if not room_reports:
            continue

        # Get the best (highest scoring) report for this room type
        best_report = max(room_reports, key=lambda x: x.get('readiness_score', 0))
        ai_score = best_report.get('readiness_score', 0)

        # Use AI checklist directly and extract issues with AI fixes
        ai_checklist = best_report.get('checklist', [])
        processed_checklist = process_ai_checklist_directly(ai_checklist)

        # FIXED: Extract issues with proper parameter preservation
        room_issues = extract_issues_with_ai_fixes(ai_checklist, room_type)

        # FIXED: Apply smart location-based deduplication
        room_issues = deduplicate_issues_by_location(room_issues)

        print(f"[DEBUG FIXED] {room_type.upper()}: {len(room_issues)} issues after FIXED deduplication")

        # IMPROVED: Use new scoring system with more reasonable penalties
        room_score = calculate_room_score_improved(ai_score, room_issues)

        # Add to global unique issues list - NO cross-room deduplication
        for issue in room_issues:
            all_unique_issues.append(issue)

        # Calculate checklist summary - After room_issues is created and deduplicated
        processed_checklist = sync_checklist_with_final_issues(processed_checklist, room_issues)
        checklist_summary = calculate_checklist_summary(processed_checklist)

        # Generate alerts from issues
        alerts = [issue['issue'] for issue in room_issues]

        # Determine room readiness with corrected logic
        is_guest_ready = room_score >= 90 and len(room_issues) == 0  # More reasonable threshold

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

    print(f"[DEBUG FIXED] Total unique issues after deduplication: {len(all_unique_issues)}")

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


def extract_issues_with_ai_fixes(ai_checklist: List[Dict], room_type: str) -> List[Dict]:
    """FIXED: Extract issues with corrected personal items logic and proper filtering."""
    room_issues = []

    for item in ai_checklist:
        status = item.get('status', '')
        notes = item.get('notes', '')
        ai_fix = item.get('ai_suggested_fix', '') or f"Address {item.get('parameter', 'issue')}"
        parameter = item.get('parameter', 'Unknown')

        # CRITICAL FIX: Handle Personal Items & Clutter logic correctly
        if "personal items" in parameter.lower() and "clutter" in parameter.lower():
            if status == 'present':
                # No clutter found - this is GOOD, skip it
                continue
            elif status == 'defective':
                # Clutter found - this is BAD, keep it as issue
                pass
            elif status == 'missing':
                # Can't assess - only keep if there's a real issue described
                if "no " in notes.lower() and ("visible" in notes.lower() or "found" in notes.lower()):
                    continue

        # For all parameters: only create issues for problematic statuses
        if status in ['missing', 'defective']:
            # Additional check: don't treat positive findings as issues
            positive_phrases = [
                "no personal items visible",
                "no clutter visible",
                "no action required",
                "area is clean"
            ]

            # If notes contain positive phrases and it's about personal items, skip
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


def sync_checklist_with_final_issues(processed_checklist: List[Dict], final_issues: List[Dict]) -> List[Dict]:
    """Sync checklist status with final issues to ensure they match - FIXED VERSION"""
    # Get parameters that have final issues
    issue_parameters = {issue.get('parameter', '') for issue in final_issues}

    # Update checklist to match final issues
    synced_checklist = []
    for item in processed_checklist:
        # CREATE A NEW ITEM (don't modify original)
        new_item = item.copy()
        parameter = new_item.get('parameter', '')

        if parameter in issue_parameters:
            # This parameter has a final issue
            new_item['status'] = "Issue Found"
        else:
            # This parameter doesn't have a final issue
            new_item['status'] = "Ready"

        synced_checklist.append(new_item)

    return synced_checklist


def calculate_checklist_summary(checklist: List[Dict]) -> Dict:
    """FIXED: Calculate checklist summary with proper status counting."""
    summary = {"present": 0, "missing": 0, "defective": 0}

    for item in checklist:
        status = item.get('status', '').lower()
        notes = item.get('notes', '').lower()

        if status == 'ready':
            summary['present'] += 1
        elif status == 'issue found':
            # Determine if missing or defective based on original AI status and notes
            if ('missing' in notes or 'not found' in notes or 'absent' in notes or
                    'not visible' in notes or 'not detected' in notes):
                summary['missing'] += 1
            else:
                summary['defective'] += 1
        elif status == 'needs attention':
            summary['defective'] += 1
        else:
            # Default to present for unclear cases
            summary['present'] += 1

    return summary


def assess_frame_quality(image: np.ndarray) -> Dict[str, float]:
    """Assess frame quality for blurriness."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {'blur_score': blur_score, 'is_blurry': blur_score < MAX_BLUR_THRESHOLD}
    except Exception:
        return {'blur_score': 0, 'is_blurry': True}


def extract_key_bedroom_frames(video_path: str, output_dir: str) -> List[str]:
    """Extracts visually distinct frames by sampling one frame per second."""
    print("[DEBUG] Starting intelligent frame extraction (1 frame/sec sampling)...")
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)

    if not vidcap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return []

    fps = vidcap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps)
    print(f"[DEBUG] Video FPS: {fps:.2f}, processing one frame every {frame_interval} frames.")

    selected_frame_paths = []
    selected_histograms = []
    frame_idx = 0

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
                    print(f"[DEBUG] Selecting first frame (index {frame_idx}).")
                    path = os.path.join(output_dir, f"frame_{frame_idx:05}.jpg")
                    cv2.imwrite(path, frame)
                    selected_frame_paths.append(path)
                    selected_histograms.append(hist)
                else:
                    is_too_similar = any(
                        cv2.compareHist(hist, h, cv2.HISTCMP_CORREL) > 0.95 for h in selected_histograms)
                    if not is_too_similar:
                        print(f"[DEBUG] Selecting distinct frame (index {frame_idx}).")
                        path = os.path.join(output_dir, f"frame_{frame_idx:05}.jpg")
                        cv2.imwrite(path, frame)
                        selected_frame_paths.append(path)
                        selected_histograms.append(hist)
            except Exception as e:
                print(f"[ERROR] Could not process frame {frame_idx}: {e}")
        frame_idx += 1

    vidcap.release()
    print(f"[DEBUG] Finished extraction. Selected {len(selected_frame_paths)} diverse frames.")
    return selected_frame_paths


def optimize_image_for_api(image_path: str, max_size: int = MAX_IMAGE_SIZE) -> str:
    """Optimize image size and returns base64 encoded string."""
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        if len(image_data) <= max_size:
            return base64.b64encode(image_data).decode('utf-8')

        image = cv2.imread(image_path)
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
        print(f"[DEBUG] Error optimizing image {image_path}: {e}")
        raise


def analyze_room_image(image_path: Union[str, List[str]]) -> Dict:
    """
    FIXED: Enhanced room analysis with improved room type detection and deduplication.
    """
    if isinstance(image_path, str):
        image_paths, is_multi_frame = [image_path], False
    elif isinstance(image_path, list) and image_path:
        image_paths, is_multi_frame = image_path, len(image_path) > 1
    else:
        return {
            'status': 'failed',
            'job_type': 'room-readiness',
            'output_image': None,
            'results': {'alerts': [], 'error': 'Invalid image_path parameter'},
            'meta': {},
            'error': 'Invalid image_path parameter'
        }

    # FIXED: Improved prompts with better room type detection
    multi_frame_prompt = """You are an expert AI assistant specializing in hotel room readiness analysis. You will be given multiple images from a single hotel room inspection.

CRITICAL INSTRUCTIONS:
1. Analyze EACH image INDIVIDUALLY for ALL parameters listed below.
2. For each image, produce a complete JSON report object.
3. Return one final JSON object with key "frame_reports" containing the list of individual reports.
4. IMPORTANT: You must analyze ALL images provided - do not skip any images.
5. MANDATORY: Assess EVERY parameter for each room type - do not skip any parameters.
6. For each parameter found to have issues, provide a specific, actionable AI fix instruction.

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

**For BEDROOM zones - MUST assess ALL 9 parameters separately:**
1. **Personal Items & Clutter** - General room clutter on floors/surfaces (STATUS: "present" if NO clutter, "defective" if clutter exists)
2. **Minibar/Fridge Condition** - MANDATORY: Always assess minibar surface and fridge interior even if not clearly visible
3. **Bedside Areas** - Nightstand surfaces and organization
4. **Bed Making Quality** - Bed covers smoothness, wrinkles, arrangement
5. **Pillow Count & Placement** - Pillow presence and arrangement
6. **Comforter/Duvet Condition** - Comforter presence and positioning
7. **Bed Covers Cleanliness** - Stains on bed linens
8. **Floor Condition** - Floor cleanliness
9. **General Room Cleanliness** - Overall room excluding other specific parameters

**For BATHROOM zones - MUST assess ALL 10 parameters separately:**
1. **Personal Items & Clutter** - General bathroom clutter (STATUS: "present" if NO clutter, "defective" if clutter exists)
2. **Toilet Condition** - Toilet lid, bowl, flush area
3. **Bathtub/Shower Cleanliness** - Tub/shower specific issues
4. **Floor Dryness & Cleanliness** - Bathroom floor condition
5. **Mirror Cleanliness** - Mirror surfaces
6. **Towel Status** - Towel placement and condition
7. **Sink/Counter Areas** - Sink and counter surfaces
8. **Toiletries & Amenities** - Toiletry placement and stock
9. **Ventilation** - Air quality
10. **Safety Features** - Safety equipment

**For LIVING AREA zones - MUST assess ALL 6 parameters separately:**
1. **Personal Items & Clutter** - General living area clutter (STATUS: "present" if NO clutter, "defective" if clutter exists)
2. **Seating Cleanliness** - Sofa/chair condition
3. **Coffee Table Condition** - Coffee table surface
4. **Remote Controls** - Remote presence and placement
5. **Entertainment Center** - TV and equipment
6. **General Area Cleanliness** - Overall area excluding specific items

**For ENTRYWAY zones - MUST assess ALL 4 parameters:**
1. **Personal Items & Clutter** - Any belongings or items (STATUS: "present" if NO clutter, "defective" if clutter exists)
2. **General Cleanliness** - Surfaces clean, floors swept/vacuumed
3. **Organization** - Items properly arranged
4. **Floor Condition** - Floor cleanliness and condition

**For CLOSET zones - MUST assess ALL 4 parameters:**
1. **Personal Items & Clutter** - Clothing or items not properly stored (STATUS: "present" if NO clutter, "defective" if clutter exists)
2. **Organization** - Clothes hanging properly, items arranged
3. **General Cleanliness** - Closet surfaces and shelves clean
4. **Floor Condition** - Closet floor condition

**For VANITY_AREA zones - MUST assess ALL 4 parameters:**
1. **Personal Items & Clutter** - Personal toiletries or items scattered (STATUS: "present" if NO clutter, "defective" if clutter exists)
2. **Mirror Cleanliness** - Mirror surface condition
3. **General Cleanliness** - Vanity surface and area clean
4. **Organization** - Items properly arranged

**CRITICAL STATUS ASSIGNMENT RULES:**
- **Personal Items & Clutter Parameter**:
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
```

MANDATORY: Return exactly one report per image with ALL required parameters assessed. If you receive 10 images, return 10 reports in frame_reports array."""

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

**FOR BEDROOM - MUST assess ALL 9 parameters:**
1. Personal Items & Clutter (STATUS: "present" if NO clutter, "defective" if clutter visible)
2. Minibar/Fridge Condition (ALWAYS assess even if not clearly visible)
3. Bedside Areas, Bed Making Quality, Pillow Count & Placement
4. Comforter/Duvet Condition, Bed Covers Cleanliness, Floor Condition, General Room Cleanliness

**FOR BATHROOM - MUST assess ALL 10 parameters:**
1. Personal Items & Clutter (STATUS: "present" if NO clutter, "defective" if clutter visible)
2. Toilet Condition, Bathtub/Shower Cleanliness, Floor Dryness & Cleanliness
3. Mirror Cleanliness, Towel Status, Sink/Counter Areas, Toiletries & Amenities, Ventilation, Safety Features

**FOR LIVING AREA - MUST assess ALL 6 parameters:**
1. Personal Items & Clutter (STATUS: "present" if NO clutter, "defective" if clutter visible)
2. Seating Cleanliness, Coffee Table Condition, Remote Controls, Entertainment Center, General Area Cleanliness

**FOR ENTRYWAY - MUST assess ALL 4 parameters:**
1. Personal Items & Clutter (STATUS: "present" if NO clutter, "defective" if clutter visible)
2. General Cleanliness, Organization, Floor Condition

**FOR CLOSET - MUST assess ALL 4 parameters:**
1. Personal Items & Clutter (STATUS: "present" if NO clutter, "defective" if clutter visible)
2. Organization, General Cleanliness, Floor Condition

**FOR VANITY_AREA - MUST assess ALL 4 parameters:**
1. Personal Items & Clutter (STATUS: "present" if NO clutter, "defective" if clutter visible)
2. Mirror Cleanliness, General Cleanliness, Organization

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
```"""

    prompt = multi_frame_prompt if is_multi_frame else single_frame_prompt

    # Rest of the function remains the same
    encoded_images, valid_paths = [], []
    for img_path in image_paths:
        try:
            encoded_images.append(optimize_image_for_api(img_path))
            valid_paths.append(img_path)
        except Exception as e:
            print(f"[DEBUG] Failed to encode image {img_path}: {e}")
            continue

    if not encoded_images:
        return {
            'status': 'failed',
            'job_type': 'room-readiness',
            'output_image': None,
            'results': {'alerts': [], 'error': 'No valid images to analyze'},
            'meta': {},
            'error': 'No valid images to analyze'
        }

    content = [{"type": "text", "text": prompt}]
    for img_b64 in encoded_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

    payload = {
        "messages": [
            {"role": "system",
             "content": "You are a precise hotel room readiness AI. Analyze ALL images provided and assess EVERY required parameter separately. Return JSON with one report per image containing ALL mandatory parameters. NEVER use 'other' or 'unknown' for zone - always pick specific room type."},
            {"role": "user", "content": content}
        ],
        "max_tokens": 4096,
        "temperature": 0.2
    }

    try:
        response = requests.post(
            AZURE_OPENAI_ENDPOINT, headers=HEADERS, data=json.dumps(payload),
            timeout=(API_TIMEOUT_CONNECT, API_TIMEOUT_READ)
        )
        response.raise_for_status()

        data = response.json()
        text = data["choices"][0]["message"]["content"]
        parsed = parse_json_from_response(text)

        if "error" in parsed:
            return {
                'status': 'failed',
                'job_type': 'room-readiness',
                'output_image': None,
                'results': {'alerts': [], 'error': f"AI response parsing failed: {parsed['error']}"},
                'meta': {},
                'error': f"AI response parsing failed: {parsed['error']}"
            }

        # Process results with FIXED consolidation
        if is_multi_frame:
            frame_reports = parsed.get("frame_reports", [])
            consolidated_data = consolidate_reports_by_room_type_fixed(frame_reports)
        else:
            consolidated_data = consolidate_reports_by_room_type_fixed([parsed])

        # Build response with FIXED score consistency
        individual_reports = consolidated_data.get('individual_reports', [])
        room_analysis = consolidated_data.get('room_analysis', {})
        all_issues = consolidated_data.get('all_issues', [])

        # CRITICAL FIX: Update individual_reports with final processed scores
        updated_individual_reports = []
        for report in individual_reports:
            zone = report.get('zone', 'unknown')
            updated_report = report.copy()

            # Use the final processed score from room_analysis instead of AI raw score
            if zone in room_analysis:
                final_score = room_analysis[zone].get('score', report.get('readiness_score', 0))
                updated_report['readiness_score'] = final_score

                # Also update status to match final analysis
                final_status = room_analysis[zone].get('status', 'Unknown')
                if final_status == "Guest Ready":
                    updated_report['status'] = "Guest Ready"
                else:
                    updated_report['status'] = "Not Guest Ready"

            updated_individual_reports.append(updated_report)

        overall_score = sum(room['score'] for room in room_analysis.values()) // len(
            room_analysis) if room_analysis else 0
        overall_status = consolidated_data.get('overall_status', 'Unknown')

        # Create master checklist for legacy compatibility
        master_checklist = []
        for report in updated_individual_reports:  # Use updated reports
            checklist_items = report.get('checklist', [])
            for item in checklist_items:
                zone_prefix = report.get('zone', 'unknown')
                item_copy = item.copy()
                item_copy['item'] = f"{zone_prefix.title()}: {item_copy.get('parameter', 'Unknown')}"
                master_checklist.append(item_copy)

        # Build frontend-compatible rooms data structure
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

        # Build unified data structure
        unified_data = {
            # Legacy compatibility fields
            "readiness_score": overall_score,
            "status": "Guest Ready" if overall_status == "All Rooms Guest Ready" else "Not Guest Ready",
            "zone": "general",
            "checklist": master_checklist,
            "fail_reasons": [item['issue'] for item in all_issues],
            "instructions": [item['fix'] for item in all_issues],
            "alerts": [{"type": "critical", "message": item['issue'], "severity": item['severity']} for item in
                       all_issues],
            "image_width": updated_individual_reports[0].get('image_width', 0) if updated_individual_reports else 0,
            "image_height": updated_individual_reports[0].get('image_height', 0) if updated_individual_reports else 0,

            # Enhanced features
            "overall_status": overall_status,
            "rooms_analyzed": consolidated_data.get('rooms_analyzed', 0),
            "rooms_ready": consolidated_data.get('rooms_ready', 0),
            "total_issues": consolidated_data.get('total_issues', 0),
            "rooms": rooms_data,
            "all_issues": all_issues,
            "individual_reports": updated_individual_reports,  # Use consistent scores
            "deduplication_stats": {
                "raw_issues": sum(len(r.get('checklist', [])) for r in updated_individual_reports),
                "unique_issues": len(all_issues),
                "efficiency": f"{len(all_issues) / max(1, sum(len(r.get('checklist', [])) for r in updated_individual_reports)) * 100:.1f}%"
            }
        }

        return {
            'status': 'completed',
            'job_type': 'room-readiness',
            'output_image': image_path if isinstance(image_path, str) else valid_paths[0] if valid_paths else None,
            'data': unified_data,
            'meta': {'frames_analyzed': len(valid_paths), 'request_time': response.elapsed.total_seconds()},
            'error': None
        }

    except requests.exceptions.Timeout as e:
        return {
            'status': 'failed',
            'job_type': 'room-readiness',
            'output_image': None,
            'results': {'alerts': [], 'error': f"API request timed out: {str(e)}"},
            'meta': {},
            'error': f"API request timed out: {str(e)}"
        }
    except requests.exceptions.RequestException as e:
        return {
            'status': 'failed',
            'job_type': 'room-readiness',
            'output_image': None,
            'results': {'alerts': [], 'error': f"API request failed: {str(e)}"},
            'meta': {},
            'error': f"API request failed: {str(e)}"
        }
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {
            'status': 'failed',
            'job_type': 'room-readiness',
            'output_image': None,
            'results': {'alerts': [], 'error': f"Failed to parse API response: {str(e)}"},
            'meta': {},
            'error': f"Failed to parse API response: {str(e)}"
        }


def analyze_room_video_multi_zone_only(video_path: str, output_dir: str = OUTPUT_DIR) -> Dict:
    """
    Main video analysis function with AI-generated fixes and frontend compatibility.
    100% drop-in replacement that provides both legacy API and AI-powered fix suggestions.
    """
    frame_paths = extract_key_bedroom_frames(video_path, output_dir)
    if not frame_paths:
        return {
            'status': 'failed',
            'job_type': 'room-readiness',
            'output_image': None,
            'results': {'alerts': [], 'error': 'No frames could be extracted from video'},
            'meta': {},
            'error': 'No frames could be extracted from video'
        }

    result = analyze_room_image(frame_paths)

    if result.get('status') == 'completed':
        return {
            'status': 'completed',
            'job_type': 'room-readiness',
            'output_image': None,
            'data': result.get('data', {}),
            'meta': result.get('meta', {}),
            'error': None
        }
    else:
        return result


def test_api_connection() -> Dict:
    """Test the API connection with a simple request."""
    print("[DEBUG] Testing API connection...")
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
        else:
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


# Legacy wrapper functions for 100% compatibility
def analyze_room_video_legacy_format(video_path: str, output_dir: str = OUTPUT_DIR) -> Dict:
    """Legacy wrapper that returns the old backend format for perfect compatibility."""
    return analyze_room_video_multi_zone_only(video_path, output_dir)


def extract_key_bedroom_frames_legacy(video_path: str, output_dir: str) -> List[str]:
    """Legacy wrapper for frame extraction that matches old behavior"""
    return extract_key_bedroom_frames(video_path, output_dir)


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
    """Get all actionable fixes with priority ranking and AI generation info."""
    all_issues = data.get('all_issues', [])

    # Sort by severity (high priority first)
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


# Export all functions for compatibility
__all__ = [
    # Main analysis functions
    'analyze_room_video_multi_zone_only',
    'analyze_room_image',

    # Legacy compatibility functions
    'analyze_room_video_legacy_format',
    'extract_key_bedroom_frames_legacy',
    'extract_key_bedroom_frames',

    # Utility functions
    'test_api_connection',
    'get_current_config',

    # Enhanced feature functions
    'get_room_analysis_summary',
    'get_room_specific_data',
    'get_actionable_fixes',

    # Helper functions
    'parse_json_from_response',
    'convert_status_to_frontend',
    'process_ai_checklist_directly',
    'extract_issues_with_ai_fixes',
    'calculate_checklist_summary'
]
