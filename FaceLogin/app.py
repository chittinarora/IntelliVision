import streamlit as st
import cv2
import face_recognition
import numpy as np
from face_auth import register_user, login_user

st.set_page_config(page_title="Face Login", layout="centered")
st.title("🔐 Face Login System")

# Session state to hold captured data
if "frame" not in st.session_state:
    st.session_state.frame = None
if "encoding" not in st.session_state:
    st.session_state.encoding = None
if "mode" not in st.session_state:
    st.session_state.mode = None
if "name" not in st.session_state:
    st.session_state.name = ""

def capture_face_once():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb)
    if faces:
        encoding = face_recognition.face_encodings(rgb, faces)[0]
        top, right, bottom, left = faces[0]
        cv2.rectangle(rgb, (left, top), (right, bottom), (0, 255, 0), 2)
        return encoding, rgb
    return None, rgb

# Sidebar selection
option = st.sidebar.radio("Choose Option", ("Login", "Register"))

if option == "Register":
    st.session_state.mode = "register"
    st.session_state.name = st.text_input("Enter your name")

elif option == "Login":
    st.session_state.mode = "login"

# Start camera and capture face
if st.button("Start Camera"):
    encoding, frame = capture_face_once()
    st.session_state.encoding = encoding
    st.session_state.frame = frame

# Show captured face
if st.session_state.frame is not None:
    st.image(st.session_state.frame, channels="RGB", caption="Captured Face")
    if st.button("Confirm"):
        if st.session_state.encoding is not None:
            if st.session_state.mode == "register":
                result = register_user(st.session_state.name, st.session_state.encoding)
                st.success(result)
            elif st.session_state.mode == "login":
                result = login_user(st.session_state.encoding)
                if result.startswith("Welcome"):
                    name = result.split(",")[1].split("!")[0].strip()
                    image_url = result.split("Image: ")[1]
                    st.success(f"Welcome back, {name}!")
                    st.image(image_url, caption=f"{name}'s face", width=250)
                else:
                    st.error(result)
        else:
            st.error("No face encoding found.")

        # Clear state after confirming
        st.session_state.frame = None
        st.session_state.encoding = None
