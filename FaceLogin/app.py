import streamlit as st
st.set_page_config(page_title="Face Login", layout="centered")

import cv2
import face_recognition
import numpy as np
from PIL import Image
import tempfile
from face_auth import register_user, login_user, face_exists
import time

message_placeholder = st.empty()  # For success/warning messages

st.title("🔐 Face Login System")

# Setup session state
for key in ["mode", "register_done", "login_done", "encoding", "frame", "name", "confirm_clicked", "login_failed"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "name" else False

# Timestamp to limit error notifications
if "last_error_time" not in st.session_state:
    st.session_state.last_error_time = 0

# Select mode
mode = st.sidebar.radio("Choose Option", ["Login", "Register"])
previous_mode = st.session_state.mode
st.session_state.mode = mode.lower()

# Reset flags on mode switch
if previous_mode != st.session_state.mode:
    if st.session_state.mode == "login":
        st.session_state.register_done = False
        st.session_state.confirm_clicked = False
    elif st.session_state.mode == "register":
        st.session_state.login_done = False
        st.session_state.login_failed = False

    message_placeholder.empty()


# Camera setup (once)
if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

cap = st.session_state.cap
if not cap.isOpened():
    st.error("❌ Could not open camera.")
    st.stop()

# Name input for register
if st.session_state.mode == "register":
    st.session_state.name = st.text_input("Enter your name", value=st.session_state.name or "")

    if st.button("📸 Confirm Registration"):
        if not st.session_state.name or st.session_state.name.strip() == "":
            message_placeholder.warning("⚠️ Please enter your name before confirming.")
            time.sleep(3)
            message_placeholder.empty()
        else:
            st.session_state.confirm_clicked = True

# Retry button after login fail
if st.session_state.mode == "login" and st.session_state.login_failed:
    if st.button("🔄 Retry Login"):
        st.session_state.login_failed = False
        st.session_state.login_done = False

# Live video feed
frame_placeholder = st.empty()
error_placeholder = st.empty()  # for dynamic error updates

def process_frame():
    ret, frame = cap.read()
    if not ret:
        return None, None, None

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(rgb, channels="RGB")
    faces = face_recognition.face_locations(rgb)
    return frame, rgb, faces

# Continuous loop for processing
while True:
    frame, rgb, faces = process_frame()
    if frame is None:
        st.error("😓 Failed to read from camera.")
        break

    # Handle registration
    if st.session_state.mode == "register" and st.session_state.confirm_clicked and not st.session_state.register_done:
        if faces:
            encoding = face_recognition.face_encodings(rgb, faces)[0]

            # Check if face is already registered
            if face_exists(encoding, tolerance=0.5):
                message_placeholder.warning("🚫 This face is already registered. Try logging in instead.")
                time.sleep(3)
                message_placeholder.empty()
                st.session_state.confirm_clicked = False
                continue

            image = Image.fromarray(rgb)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                image.save(tmp_file.name)
                image_path = tmp_file.name

            result = register_user(st.session_state.name, encoding, image_path)
            message_placeholder.success(result)
            time.sleep(3)
            message_placeholder.empty()
            st.session_state.register_done = True
            st.session_state.confirm_clicked = False
        else:
            message_placeholder.warning("🙈 No face detected. Try again.")
            time.sleep(3)
            message_placeholder.empty()
            st.session_state.confirm_clicked = False

    # Handle login
    elif st.session_state.mode == "login" and not st.session_state.login_done:
        if faces:
            encoding = face_recognition.face_encodings(rgb, faces)[0]
            result = login_user(encoding)

            if result.startswith("🎉 Welcome"):
                name = result.split(",")[1].split("!")[0].strip()
                image_url = result.split("Image: ")[1]
                message_placeholder.success(f"🎉 Welcome back, {name}!")
                st.image(image_url, caption=f"{name}'s face", width=250)
                st.session_state.login_done = True
                st.session_state.last_error_time = 0
                error_placeholder.empty()  # clear error on success
            else:
                now = time.time()
                if now - st.session_state.last_error_time > 5:
                    error_placeholder.error(result)  # show/replace error
                    st.session_state.last_error_time = now
                else:
                    error_placeholder.empty()  # hide while waiting to re-show

    time.sleep(0.1)  # Chill the CPU

