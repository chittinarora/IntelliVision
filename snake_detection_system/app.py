import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from pymongo import MongoClient

# ---------------------- MongoDB Connection ----------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["snake_db"]
collection = db["detections"]

# ---------------------- YOLO + File Settings ---------------------
MODEL_PATH = r"C:\Users\Hp\runs\detect\snakes_v7_finetune\weights\best.pt"
SAVE_DIR = Path("F:/snake_detections")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

# ---------------------- Streamlit UI Setup ----------------------
st.set_page_config(page_title="Snake Detection", layout="centered")
st.title("🐍 Snake Detection App")

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ---------------------- Sidebar Selection ----------------------
st.sidebar.header("🎛️ Options")
input_type = st.sidebar.radio("Choose Input Type", ["Image", "Video"])

# ---------------------- Image Detection -------------------------
if input_type == "Image":
    st.header("📷 Upload Image")
    img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="img")

    if img_file:
        img = Image.open(img_file).convert("RGB")
        img_array = np.array(img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)

        results = model(img_array)
        plotted = results[0].plot()
        num_snakes = len(results[0].boxes)

        with col2:
            st.image(plotted, caption="Detected Image", use_container_width=True)

        # Save to disk
        timestamp = get_timestamp()
        save_path = SAVE_DIR / f"detection_img_{timestamp}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR))

        # Save to MongoDB
        collection.insert_one({
            "type": "image",
            "file_name": img_file.name,
            "detected_snakes": num_snakes,
            "timestamp": datetime.now()
        })

        st.success(f"✅ Image saved to: {save_path}")

# ---------------------- Video Detection -------------------------
elif input_type == "Video":
    st.header("🎞️ Upload Video")
    vid_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"], key="vid")

    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())

        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        timestamp = get_timestamp()
        save_path = SAVE_DIR / f"detection_vid_{timestamp}.mp4"
        out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        stframe = st.empty()
        total_detected = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            total_detected += len(results[0].boxes)
            plotted = results[0].plot()
            out.write(plotted)
            stframe.image(plotted, channels="BGR", use_container_width=True)

        cap.release()
        out.release()

        # Save to MongoDB
        collection.insert_one({
            "type": "video",
            "file_name": vid_file.name,
            "detected_snakes_total": total_detected,
            "timestamp": datetime.now()
        })

        st.success(f"✅ Video saved to: {save_path}")
