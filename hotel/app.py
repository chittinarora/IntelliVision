import streamlit as st
from analyze.image_analysis import analyze_room_image
from analyze.video_processor import extract_key_bedroom_frames
import os
import shutil

st.set_page_config(page_title="Hotel Room Inspector", layout="centered")

st.title("🏨 Hotel Room Readiness Inspector")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1]
    file_path = os.path.join("temp_upload", uploaded_file.name)
    os.makedirs("temp_upload", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    if file_ext in ["jpg", "jpeg", "png"]:
        st.image(file_path, caption="Uploaded Image", use_column_width=True)
        result = analyze_room_image(file_path)
        st.json(result)

    elif file_ext in ["mp4", "mov", "avi"]:
        st.video(file_path)
        st.info("Extracting key frames...")
        frames = extract_key_bedroom_frames(file_path, "temp_frames")
        for idx, frame in enumerate(frames):
            st.image(frame, caption=f"Frame {idx+1}")
            result = analyze_room_image(frame)
            st.subheader(f"📝 Analysis for Frame {idx+1}")
            with st.expander("📦 Room Readiness Report (click to expand)", expanded=True):
                st.json(result)

    # Cleanup
    shutil.rmtree("temp_upload")
