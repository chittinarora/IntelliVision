import os
import cv2
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://infer.roboflow.com",  # Roboflow inference endpoint
    api_key="vfMnQeFixryhPw18Thmz"         # Replace with your actual API key
)

# -----------------------------
# Video Pothole Detection
# -----------------------------
def run_pothole_detection(input_path, output_path):
    print("📼 Opening video:", input_path)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ Failed to open video!")
        return

    print("✅ Video opened successfully")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    MAX_FRAMES = 200  # Optional limit
    frame_idx = 0
    processed_frames = 0
    total_potholes = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or processed_frames >= MAX_FRAMES:
            break

        print(f"➡️ Processing frame {frame_idx}")
        temp_input = "temp_frame.jpg"
        cv2.imwrite(temp_input, frame)

        try:
            result = CLIENT.infer(temp_input, model_id="pothole-voxrl/1")
            predictions = result.get("predictions", [])
            total_potholes += len(predictions)

            for pred in predictions:
                if pred.get('confidence', 0) < 0.1:
                    continue
                x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                top_left = (x - w // 2, y - h // 2)
                bottom_right = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, pred['class'], (top_left[0], top_left[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(frame, f"Total Potholes: {total_potholes}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            out.write(frame)

        except Exception as e:
            print("❌ Prediction error:", e)
            break

        processed_frames += 1
        frame_idx += 1

    cap.release()
    out.release()
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")
    print("✅ Video saved:", output_path)


# -----------------------------
# Image Pothole Detection
# -----------------------------
def run_pothole_image_detection(input_path, output_path):
    print("🖼️ Running image detection on:", input_path)

    if not os.path.exists(input_path):
        print(f"❌ Input file does not exist: {input_path}")
        return False

    frame = cv2.imread(input_path)
    if frame is None:
        print("❌ Failed to read image (cv2 returned None)")
        return False

    try:
        print("📤 Sending image path to Roboflow...")
        result = CLIENT.infer(input_path, model_id="pothole-voxrl/1")
        print("✅ Inference result received")

        predictions = result.get("predictions", [])
        print(f"🕳️ Total predictions: {len(predictions)}")

        for pred in predictions:
            if pred.get('confidence', 0) < 0.1:
                continue
            x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (x + w // 2, y + h // 2)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, pred['class'], (top_left[0], top_left[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f"Total Potholes: {len(predictions)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        success = cv2.imwrite(output_path, frame)
        if success:
            print(f"✅ Image saved at: {output_path}")
            return True
        else:
            print(f"❌ Failed to save image at: {output_path}")
            return False

    except Exception as e:
        print("❌ Prediction error (image):", e)
        return False
