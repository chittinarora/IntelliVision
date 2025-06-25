import streamlit as st
import os
from video_processor import run_pothole_detection, run_pothole_image_detection

st.set_page_config(page_title="Pothole Detector", layout="centered")

st.title("🕳️ Pothole Detection App")

file = st.file_uploader("Upload a road video or image", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])

if file is not None:
    file_extension = file.name.split(".")[-1].lower()
    input_path = os.path.join("test_files", file.name)
    with open(input_path, "wb") as f:
        f.write(file.read())

    st.success("File uploaded successfully!")

    if st.button("🚀 Run Pothole Detection"):
        output_path = os.path.join("output_files", f"output_{file.name}")
        with st.spinner("Detecting potholes..."):
            if file_extension in ["mp4", "mov", "avi"]:
                run_pothole_detection(input_path, output_path)
                st.success("Detection complete!")
                st.video(output_path)
            elif file_extension in ["jpg", "jpeg", "png"]:
                run_pothole_image_detection(input_path, output_path)
                st.success("Detection complete!")
                st.image(output_path)