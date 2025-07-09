import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

from face_auth import register_user, login_user, face_exists
from embedding import get_embedding_from_image

st.set_page_config(page_title="Face Login", layout="centered")
st.title("🔐 Face Login System")

# Setup session state
for key in ["mode", "register_done", "login_done", "encoding", "frame", "name", "confirm_clicked", "login_failed"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "name" else False

if "last_error_time" not in st.session_state:
    st.session_state.last_error_time = 0

mode = st.sidebar.radio("Choose Option", ["Login", "Register"])
previous_mode = st.session_state.mode
st.session_state.mode = mode.lower()

if previous_mode != st.session_state.mode:
    st.session_state.register_done = False
    st.session_state.login_done = False
    st.session_state.login_failed = False
    st.session_state.confirm_clicked = False

message_placeholder = st.empty()
frame_placeholder = st.empty()
error_placeholder = st.empty()

# Camera setup
if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

cap = st.session_state.cap
if not cap.isOpened():
    st.error("❌ Could not open camera.")
    st.stop()

# Register input
if st.session_state.mode == "register":
    st.session_state.name = st.text_input("Enter your name", value=st.session_state.name or "")
    if st.button("📸 Confirm Registration"):
        if not st.session_state.name.strip():
            message_placeholder.warning("⚠️ Please enter your name before confirming.")
        else:
            st.session_state.confirm_clicked = True

# Retry login
if st.session_state.mode == "login" and st.session_state.login_failed:
    if st.button("🔄 Retry Login"):
        st.session_state.login_failed = False
        st.session_state.login_done = False

# Frame processing
def process_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(rgb, channels="RGB")
    return rgb

# Main loop
while True:
    rgb = process_frame()
    if rgb is None:
        st.error("😓 Failed to read from camera.")
        break

    image = Image.fromarray(rgb)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(tmp_file.name)
    image_path = tmp_file.name

    # Registration logic
    if st.session_state.mode == "register" and st.session_state.confirm_clicked and not st.session_state.register_done:
        encoding = get_embedding_from_image(image_path)
        if encoding is not None:
            if face_exists(encoding, tolerance=0.93):
                message_placeholder.warning("🚫 This face is already registered. Try logging in instead.")
            else:
                result = register_user(st.session_state.name, encoding, image_path)
                message_placeholder.success(result)
                st.session_state.register_done = True
        else:
            message_placeholder.warning("🙈 No face detected. Try again.")

        st.session_state.confirm_clicked = False
        time.sleep(3)
        message_placeholder.empty()

    # Login logic
    elif st.session_state.mode == "login" and not st.session_state.login_done:
        encoding = get_embedding_from_image(image_path)
        if encoding is not None:
            result = login_user(encoding)
            if result.startswith("🎉 Welcome"):
                name = result.split(",")[1].split("!")[0].strip()
                image_url = result.split("Image: ")[1]
                message_placeholder.success(f"🎉 Welcome back, {name}!")
                st.image(image_url, caption=f"{name}'s face", width=250)
                st.session_state.login_done = True
                st.session_state.last_error_time = 0
                error_placeholder.empty()
            else:
                now = time.time()
                if now - st.session_state.last_error_time > 5:
                    error_placeholder.error(result)
                    st.session_state.last_error_time = now
        else:
            st.session_state.login_failed = True
            error_placeholder.error("🙈 No face detected. Try again.")

    time.sleep(0.2)