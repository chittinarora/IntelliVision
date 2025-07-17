import cv2
import os
import requests
import base64

ROBOFLOW_API_URL = "https://detect.roboflow.com/pothole-voxrl/1"
ROBOFLOW_API_KEY = "vfMnQeFixryhPw18Thmz"

def run_pothole_detection(input_path, output_path):
    print("📼 Opening video:", input_path)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ Failed to open video!")
        return

    print("✅ Video opened successfully")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    print(f"🎥 FPS: {fps}, Width: {width}, Height: {height}")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    MAX_FRAMES = 200
    FRAME_SKIP = 5
    frame_idx = 0
    processed_frames = 0
    total_potholes = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or processed_frames >= MAX_FRAMES:
            print("✅ Finished reading video or reached max frame limit.")
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        print(f"➡️ Processing frame {frame_idx}")
        
        _, img_encoded = cv2.imencode('.jpg', frame)
        base64_encoded_image = base64.b64encode(img_encoded).decode('utf-8')

        try:
            response = requests.post(
                ROBOFLOW_API_URL,
                params={"api_key": ROBOFLOW_API_KEY},
                data=base64_encoded_image,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status() 
            result = response.json()
            
            print("🧠 Prediction received:", result)
            predictions = result.get("predictions", [])
            total_potholes += len(predictions)

            for pred in predictions:
                if pred.get('confidence', 0) < 0.1:
                    continue

                x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                top_left = (x - w // 2, y - h // 2)
                bottom_right = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)
                cv2.putText(frame, pred['class'], (top_left[0], top_left[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.putText(frame, f"Total Potholes: {total_potholes}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            out.write(frame)

        except requests.exceptions.RequestException as e:
            print(f"❌ HTTP request error: {e}")
            break
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            break

        processed_frames += 1
        frame_idx += 1

    cap.release()
    out.release()
    print("✅ All done, video saved.")

def run_pothole_image_detection(input_path, output_path):
    print("🖼️ Running image detection on:", input_path)
    frame = cv2.imread(input_path)
    if frame is None:
        print("❌ Failed to read image")
        return

    _, img_encoded = cv2.imencode('.jpg', frame)
    base64_encoded_image = base64.b64encode(img_encoded).decode('utf-8')

    try:
        response = requests.post(
            ROBOFLOW_API_URL,
            params={"api_key": ROBOFLOW_API_KEY},
            data=base64_encoded_image,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        result = response.json()

        print("🧠 Prediction received:", result)
        predictions = result.get("predictions", [])

        for pred in predictions:
            x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (x + w // 2, y + h // 2)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)
            cv2.putText(frame, pred['class'], (top_left[0], top_left[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(frame, f"Total Potholes: {len(predictions)}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(output_path, frame)
        print("✅ Image saved with predictions.")

    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP request error (image): {e}")
    except Exception as e:
        print(f"❌ Prediction error (image): {e}")