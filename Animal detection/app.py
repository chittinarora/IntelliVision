import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
import time

# Ensure output folder exists
output_dir = r"F:\animal-result"
os.makedirs(output_dir, exist_ok=True)

# Load model once
@st.cache_resource
def load_model():
    model_path = r"C:\Users\Hp\runs\detect\animal_yolov11n_data_1\weights\best.pt"
    return YOLO(model_path)

model = load_model()

st.title("🦁 Animal Detection using YOLOv11n")
st.markdown("Upload an image or video to detect animals")

mode = st.sidebar.radio("Choose input type:", ['Image', 'Video'])

# ======================================
# IMAGE DETECTION
# ======================================
if mode == 'Image':
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Run YOLOv11n detection
        results = model.predict(np.array(img), imgsz=640, conf=0.4)
        annotated_img = results[0].plot()

        # Save result
        save_path = os.path.join(output_dir, "detected_image.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

        # Show result
        st.image(annotated_img, caption="Detection Result", use_container_width=True)
        st.success(f"✅ Image saved to: {save_path}")

# ======================================
# VIDEO DETECTION
# ======================================
elif mode == 'Video':
    uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_vid is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        tfile.close()
        vid_path = tfile.name

        cap = cv2.VideoCapture(vid_path)
        stframe = st.empty()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = os.path.join(output_dir, "detected_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict
            results = model.predict(frame, imgsz=640, conf=0.4)
            annotated_frame = results[0].plot()

            # Save and show
            out.write(annotated_frame)
            stframe.image(annotated_frame, channels="BGR")

            time.sleep(0.03)  # simulate video speed

        cap.release()
        out.release()

        try:
            os.remove(vid_path)
        except PermissionError:
            st.warning("⚠️ Temp file in use. Delete manually later.")

        st.success(f"🎥 Video saved to: {out_path}")
