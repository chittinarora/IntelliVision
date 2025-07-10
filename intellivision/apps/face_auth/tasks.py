# tasks.py - Celery tasks for face authentication
# Handles background registration and login using face encodings.

import os
import face_recognition
from celery import shared_task
from .face_auth import register_user, login_user
from rest_framework_simplejwt.tokens import RefreshToken
from .embedding import get_embedding_from_image


@shared_task
def register_face_user_task(username, image_path):
    """Background task to register a user with a face image."""
    encoding = get_embedding_from_image(image_path)
    if encoding is None:
        if os.path.exists(image_path):
            os.remove(image_path)
        return {"success": False, "message": "❌ No face detected. Try again."}
    return register_user(username, encoding, image_path)


@shared_task
def login_face_user_task(username, image_path):
    """Background task to authenticate a user with a face image."""
    encoding = get_embedding_from_image(image_path)
    if encoding is None:
        if os.path.exists(image_path):
            os.remove(image_path)
        return {"success": False, "message": "❌ No face detected. Try again."}
    return login_user(encoding)
