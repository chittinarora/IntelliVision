import streamlit as st
import os
from video_processor import run_pothole_detection, run_pothole_image_detection

# App config
st.set_page_config(page_title="Pothole Detector", layout="centered")
st.title("🕳️ Pothole Detection App")

# Create necessary directories
os.makedirs("test_files", exist_ok=True)
os.makedirs("output_files", exist_ok=True)

# File uploader
file = st.file_uploader("Upload a road video or image", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])

if file is not None:
    file_extension = file.name.split(".")[-1].lower()
    input_path = os.path.join("test_files", file.name)

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(file.read())

    st.success("✅ File uploaded successfully!")

    if st.button("🚀 Run Pothole Detection"):
        output_path = os.path.join("output_files", f"output_{file.name}")

        with st.spinner("🔍 Detecting potholes..."):
            try:
                if file_extension in ["mp4", "mov", "avi"]:
                    run_pothole_detection(input_path, output_path)

                    if os.path.exists(output_path):
                        st.success("✅ Detection complete!")
                        st.video(output_path)
                    else:
                        st.error("❌ Failed to generate output video. Check logs for details.")
                        st.text(f"Input Path: {input_path}")
                        st.text(f"Expected Output Path: {output_path}")

                elif file_extension in ["jpg", "jpeg", "png"]:
                    success = run_pothole_image_detection(input_path, output_path)

                    if success and os.path.exists(output_path):
                        st.success("✅ Detection complete!")
                        st.image(output_path, caption="Detected Potholes", use_column_width=True)
                    else:
                        st.error("❌ Failed to process image. Check logs below.")
                        st.text(f"Input Path: {input_path}")
                        st.text(f"Expected Output Path: {output_path}")
                else:
                    st.warning("⚠️ Unsupported file type.")

            except Exception as e:
                st.error(f"❌ An error occurred during detection:\n\n{str(e)}")
