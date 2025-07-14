import streamlit as st
from ultralytics import YOLO
from pymongo import MongoClient
from datetime import datetime
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
import time

# ─── OUTPUT FOLDER ─────────────────────────────────────
output_dir = r"F:\animal-result"
os.makedirs(output_dir, exist_ok=True)

# ─── LOAD MODEL ────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = r"C:\Users\Hp\runs\detect\animal_yolov11n_data_1\weights\best.pt"
    return YOLO(model_path)

model = load_model()

# ─── MONGODB CONNECTION ────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db     = client["new_animal_db"]   # You can change the DB name
col    = db["detections"]          # Collection to store detection metadata

# ─── STREAMLIT UI ───────────────────────────────────────
st.title("🦁 Animal Detection using YOLOv11n")
st.markdown("Upload an image or video to detect animals")

mode = st.sidebar.radio("Choose input type:", ['Image', 'Video'])

# ───────────────────────────────────────────────────────
# 🖼️ IMAGE DETECTION
# ───────────────────────────────────────────────────────
if mode == 'Image':
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Run detection
        results = model.predict(np.array(img), imgsz=640, conf=0.4)
        annotated_img = results[0].plot()

        # Save result image
        save_path = os.path.join(output_dir, "detected_image.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

        # Display result
        st.image(annotated_img, caption="Detection Result", use_container_width=True)
        st.success(f"✅ Image saved to: {save_path}")

        # Save to MongoDB
        classes = [model.names[int(c)] for c in results[0].boxes.cls]
        confs   = [float(c) for c in results[0].boxes.conf]
        col.insert_one({
            "type": "image",
            "image_name": uploaded_img.name,
            "detected_animals": classes,
            "confidences": confs,
            "timestamp": datetime.now()
        })
        st.info("📥 Detection info stored in MongoDB")

# ───────────────────────────────────────────────────────
# 🎥 VIDEO DETECTION
# ───────────────────────────────────────────────────────
elif mode == 'Video':
    uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_vid is not None:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        tfile.close()
        vid_path = tfile.name

        cap = cv2.VideoCapture(vid_path)
        stframe = st.empty()

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        # Output video path
        out_path = os.path.join(output_dir, "detected_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = model.predict(frame, imgsz=640, conf=0.4)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            # Display frame
            stframe.image(annotated_frame, channels="BGR")
            time.sleep(0.03)  # simulate video speed

            # Save metadata to MongoDB
            classes = [model.names[int(c)] for c in results[0].boxes.cls]
            confs   = [float(c) for c in results[0].boxes.conf]
            col.insert_one({
                "type": "video",
                "video_name": uploaded_vid.name,
                "frame_id": frame_id,
                "detected_animals": classes,
                "confidences": confs,
                "timestamp": datetime.now()
            })
            frame_id += 1

        cap.release()
        out.release()

        # Clean up
        try:
            os.remove(vid_path)
        except PermissionError:
            st.warning("⚠️ Temp file in use. Delete manually later.")

        st.success(f"🎥 Video saved to: {out_path}")
        st.info("📥 Detection metadata stored in MongoDB")
