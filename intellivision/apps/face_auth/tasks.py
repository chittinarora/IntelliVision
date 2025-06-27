# tasks.py - Celery tasks for face authentication
# Handles background registration and login using face encodings.

import os
import face_recognition
from celery import shared_task
from .face_auth import register_user, login_user
from rest_framework_simplejwt.tokens import RefreshToken


@shared_task
def register_face_user_task(username, image_path):
    """Background task to register a user with a face image."""
    # Load image and compute face encoding
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        # Cleanup temp file
        if os.path.exists(image_path):
            os.remove(image_path)
        return "❌ No face detected. Try again."
    encoding = encodings[0]
    return register_user(username, encoding, image_path)


@shared_task
def login_face_user_task(username, image_path):
    """Background task to authenticate a user with a face image."""
    # Load image and compute face encoding
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        if os.path.exists(image_path):
            os.remove(image_path)
        return {"success": False, "message": "❌ No face detected. Try again."}
    encoding = encodings[0]
    return login_user(encoding)
