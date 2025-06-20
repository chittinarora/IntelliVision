# frontend/app.py

import streamlit as st
import requests
import pandas as pd
from io import BytesIO
from PIL import Image

# Base URL for the FastAPI backend
BASE_URL = "http://localhost:8000/detect"

st.set_page_config(page_title="ANPR & Parking Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["ANPR Video Processing", "Parking System"])

# -------------------------
# ANPR Video Processing UI
# -------------------------
if page == "ANPR Video Processing":
    st.title("ANPR Video Processing")

    # 1. Upload video
    video_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    if video_file:
        if st.button("Start Processing"):
            files = {"video": (video_file.name, video_file, video_file.type)}
            with st.spinner("Uploading and starting processing..."):
                resp = requests.post(f"{BASE_URL}/video", files=files)
            if resp.status_code == 200:
                data = resp.json()
                filename = data["filename"]
                st.success(f"Processing started: {filename}")
                st.session_state["anpr_filename"] = filename
            else:
                st.error(f"Upload failed: {resp.status_code} {resp.text}")

    # 2. Preview & Download annotated video
    filename = st.session_state.get("anpr_filename", None)
    if filename:
        annotated_name = f"annotated_{filename}"
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Preview Annotated Video"):
                video_url = f"{BASE_URL}/preview/{annotated_name}"
                st.video(video_url)
        with col2:
            download_url = f"{BASE_URL}/download/{annotated_name}"
            st.markdown(f"[Download Annotated Video]({download_url})")

    # 3. Detection history table
    if st.button("Refresh Detection History"):
        resp = requests.get(f"{BASE_URL}/history")
        if resp.status_code == 200:
            records = resp.json().get("history", [])
            if records:
                st.subheader("Finalized Plate Detections")
                df = pd.DataFrame(records)
                # Display cropped images
                for _, row in df.iterrows():
                    if row.get("image_data"):
                        img = Image.open(BytesIO(bytes.fromhex(row["image_data"])))
                        st.image(img, caption=f"Plate: {row['plate']}", width=200)
                # Drop large hex column for table view
                df_display = df.drop(columns=["image_data"], errors="ignore")
                st.dataframe(df_display)
            else:
                st.info("No detections yet.")
        else:
            st.error("Failed to fetch history.")

# ---------------------
# Parking System UI
# ---------------------
else:
    st.title("Parking Lot System")

    # 1. Set entry/exit lines
    st.subheader("Configure Entry/Exit Lines")
    entry_y = st.number_input("Entry Line Y-coordinate", value=200, step=1)
    exit_y = st.number_input("Exit Line Y-coordinate", value=400, step=1)
    if st.button("Update Lines"):
        resp = requests.post(f"{BASE_URL}/set_lines", data={"entry_y": entry_y, "exit_y": exit_y})
        if resp.status_code == 200:
            st.success("Entry/Exit lines updated.")
        else:
            st.error(f"Failed to set lines: {resp.text}")

    # 2. Set capacity
    st.subheader("Set Total Parking Slots")
    slots = st.number_input("Total Slots", min_value=1, value=50, step=1)
    if st.button("Update Capacity"):
        resp = requests.post(f"{BASE_URL}/set_capacity", data={"slots": slots})
        if resp.status_code == 200:
            st.success(f"Capacity set to {slots}.")
        else:
            st.error(f"Failed to set capacity: {resp.text}")

    # 3. Start live camera
    st.subheader("Live Camera Tracking")
    if st.button("Start Camera Feed"):
        resp = requests.post(f"{BASE_URL}/start_camera")
        if resp.status_code == 200:
            st.success("Camera feed started.")
        else:
            st.error(f"Failed to start camera: {resp.text}")

    # 4. Real-time stats
    st.subheader("Parking Statistics")
    resp = requests.get(f"{BASE_URL}/stats")
    if resp.status_code == 200:
        stats = resp.json()
        st.metric("Total Slots", stats["total_slots"])
        st.metric("Occupied", stats["occupied"])
        st.metric("Available", stats["available"])
        st.write(f"Last Updated: {stats['last_updated']}")
    else:
        st.error("Failed to fetch stats.")

    # 5. Event dashboard & export
    st.subheader("Entry/Exit Event Log")
    if st.button("Refresh Log"):
        resp = requests.get(f"{BASE_URL}/dashboard")
        if resp.status_code == 200:
            logs = resp.json().get("dashboard", [])
            if logs:
                st.dataframe(pd.DataFrame(logs))
            else:
                st.info("No events logged yet.")
        else:
            st.error("Failed to fetch dashboard.")

    if st.button("Download Log CSV"):
        export_url = f"{BASE_URL}/export_logs"
        st.markdown(f"[Download CSV]({export_url})")

