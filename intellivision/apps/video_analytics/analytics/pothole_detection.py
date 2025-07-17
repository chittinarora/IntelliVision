import cv2
import os
import base64
import requests
from typing import List, Dict, Any
from apps.video_analytics.convert import convert_to_web_mp4
from django.conf import settings
from pathlib import Path

"""
Pothole detection analytics using Roboflow HTTP API.
Supports both video and image input.
"""

# Canonical models directory for all analytics jobs
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ROBOFLOW_API_URL = "https://detect.roboflow.com/pothole-voxrl/1"
ROBOFLOW_API_KEY = "vfMnQeFixryhPw18Thmz"

MAX_SIZE = 1024

def resize_frame_if_needed(frame):
    h, w = frame.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame

def send_frame_to_roboflow(frame) -> List[Dict[str, Any]]:
    _, img_encoded = cv2.imencode('.jpg', frame)
    base64_encoded_image = base64.b64encode(img_encoded).decode('utf-8')
    response = requests.post(
        ROBOFLOW_API_URL,
        params={"api_key": ROBOFLOW_API_KEY},
        data=base64_encoded_image,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    response.raise_for_status()
    result = response.json()
    return result.get("predictions", [])

def tracking_video(input_path: str, output_path: str = None) -> Dict[str, Any]:
    OUTPUT_DIR = settings.JOB_OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"output_pothole_{base_name}.mp4")
    result = run_pothole_detection(input_path, output_path)
    return result

def run_pothole_detection(input_path: str, output_path: str) -> Dict[str, Any]:
    print("🎥 Opening video:", input_path)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ Failed to open video!")
        return {"error": "Failed to open video."}

    print("✅ Video opened successfully")

    orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    print(f"🎥 Original FPS: {orig_fps}, Width: {width}, Height: {height}")

    FRAME_SKIP = 5
    output_fps = max(1, orig_fps // FRAME_SKIP)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    MAX_FRAMES = 200
    frame_idx = 0
    processed_frames = 0
    total_potholes = 0
    frame_details: List[Dict[str, Any]] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or processed_frames >= MAX_FRAMES:
            print("✅ Done processing frames or reached max frame limit.")
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        print(f"➡️ Processing frame {frame_idx}")

        try:
            resized_frame = resize_frame_if_needed(frame)
            predictions = send_frame_to_roboflow(resized_frame)
            print("🧠 Prediction received:", predictions)
            potholes_in_frame = [
                {
                    "x": float(pred['x']),
                    "y": float(pred['y']),
                    "width": float(pred['width']),
                    "height": float(pred['height']),
                    "confidence": float(pred.get('confidence', 0)),
                    "class": pred.get('class', 'pothole')
                }
                for pred in predictions if pred.get('confidence', 0) >= 0.1
            ]
            total_potholes += len(potholes_in_frame)
            frame_details.append({
                "frame_index": frame_idx,
                "potholes": potholes_in_frame
            })

            for pred in potholes_in_frame:
                x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                top_left = (x - w // 2, y - h // 2)
                bottom_right = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)
                cv2.putText(frame, pred['class'], (top_left[0], top_left[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.putText(frame, f"Total Potholes: {total_potholes}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            out.write(frame)

        except Exception as e:
            print("❌ Prediction error:", e)
            break

        processed_frames += 1
        frame_idx += 1

    cap.release()
    out.release()
    print("✅ All done, video saved.")

    # Convert to web-friendly MP4
    web_output_path = output_path.replace('.mp4', '_web.mp4')
    if convert_to_web_mp4(output_path, web_output_path):
        final_output_path = web_output_path
        if os.path.exists(output_path):
            os.remove(output_path)
    else:
        final_output_path = output_path

    return {
        'status': 'completed',
        'job_type': 'pothole-detection',
        'output_video': final_output_path,
        'results': {
            'total_potholes': total_potholes,
            'frames': frame_details,
            'alerts': []
        },
        'meta': {},
        'error': None
    }

def run_pothole_image_detection(input_path: str, output_path: str) -> Dict[str, Any]:
    print("🖼️ Running image detection on:", input_path)
    frame = cv2.imread(input_path)
    if frame is None:
        print("❌ Failed to read image")
        return {"error": "Failed to read image."}

    try:
        resized_frame = resize_frame_if_needed(frame)
        predictions = send_frame_to_roboflow(resized_frame)
        print("🧠 Prediction received:", predictions)
        potholes = [
            {
                "x": float(pred['x']),
                "y": float(pred['y']),
                "width": float(pred['width']),
                "height": float(pred['height']),
                "confidence": float(pred.get('confidence', 0)),
                "class": pred.get('class', 'pothole')
            }
            for pred in predictions if pred.get('confidence', 0) >= 0.1
        ]

        for pred in potholes:
            x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (x + w // 2, y + h // 2)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)
            cv2.putText(frame, pred['class'], (top_left[0], top_left[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(frame, f"Total Potholes: {len(potholes)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(output_path, frame)
        print("✅ Image saved with predictions.")

        # Check if output file was created
        if not os.path.exists(output_path):
            print(f"❌ Output file not created: {output_path}")
            return {"error": "Detection failed, output file not found."}

        return {
            "total_potholes": len(potholes),
            "potholes": potholes,
            "output_path": output_path
        }

    except Exception as e:
        print("❌ Prediction error (image):", e)
        return {"error": str(e)}
