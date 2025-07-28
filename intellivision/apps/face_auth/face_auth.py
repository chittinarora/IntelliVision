# face_auth.py - Core logic for face registration and authentication
# Handles user registration, login, and face matching using Qdrant and MongoDB.

import logging
from gettext import dngettext
import os

logger = logging.getLogger(__name__)
import uuid
import tempfile
import numpy as np
from django.contrib.auth.models import User
from .jwt_utils import get_tokens_for_user
from .db import db, qdrant, COLLECTION_NAME
from .cloudinary_utils import upload_face_image as upload_image
from .utils import match_face, normalize


def face_exists(new_encoding, tolerance=0.5):
    """Check if a similar face encoding exists in Qdrant."""
    encoding = normalize(new_encoding)
    response = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=new_encoding.tolist(),
        limit=1,
        score_threshold=tolerance
    )
    return len(response) > 0


def register_user(name, encoding, image_path):
    """Register a new user with a face encoding and image."""
    if encoding is None:
        return {"success": False, "message": "❌ No face detected. Try again."}

    # Create a UUID for Qdrant point_id
    point_id = str(uuid.uuid4())

    # Check if user already exists
    if db.users.find_one({"name": name}):
        return {"success": False, "message": f"⚠️ User {name} is already registered."}

    # Upload image to Cloudinary
    image_url = upload_image(image_path)

    # Create Django user if it doesn't exist (for auth/JWT)
    django_user, _ = User.objects.get_or_create(username=name)
    access_token = get_tokens_for_user(django_user)

    # Normalize encoding
    encoding = normalize(encoding)

    # Add user to Qdrant
    try:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[{
                "id": point_id,
                "vector": encoding.tolist(),
                "payload": {"name": name, "image": image_url}
            }]
        )
    except Exception as e:
        import traceback
        logger.error(f"Qdrant upsert error: {e}")
        traceback.print_exc()
        return {"success": False, "message": f"Qdrant upsert error: {str(e)}"}

    db.users.insert_one({
        "_id": point_id,
        "name": name,
        "image": image_url
    })

    if os.path.exists(image_path):
        os.remove(image_path)

    return {
        "success": True,
        "name": name,
        "image": image_url,
        "token": access_token,
        "message": f"✅ Registered {name} successfully!"
    }


def login_user(encoding):
    """Authenticate a user by matching their face encoding."""
    if encoding is None:
        return {"success": False, "message": "❌ No face detected. Try again."}

    # Normalize encoding
    encoding = normalize(encoding)

    # Match face
    try:
        match = match_face(qdrant, encoding)
    except Exception as e:
        import traceback
        logger.error(f"Qdrant search error: {e}")
        traceback.print_exc()
        return {"success": False, "message": f"Qdrant search error: {str(e)}"}

    if not match:
        return {"success": False, "message": "No match found."}
    if match.score < 0.4:
        return {"success": False, "message": "Match too weak. Try again."}

    user_id = match.id
    mongo_user = db.users.find_one({"_id": user_id})
    if not mongo_user:
        return {"success": False, "message": "❌ User data not found."}

    django_user, _ = User.objects.get_or_create(username=mongo_user["name"])

    access_token = get_tokens_for_user(django_user)

    return {
        "success": True,
        "name": django_user.username,
        "image": getattr(mongo_user, "image", None),
        "message": f"🎉 Welcome back, {django_user.username}!",
        "token": access_token
    }
