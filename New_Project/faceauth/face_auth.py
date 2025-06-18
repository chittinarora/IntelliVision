import os
import uuid
import tempfile
import numpy as np
from django.contrib.auth.models import User
from .jwt_utils import get_tokens_for_user

from .db import db, qdrant, COLLECTION_NAME
from .cloudinary_utils import upload_face_image as upload_image
from .utils import match_face


def face_exists(new_encoding, tolerance=0.5):
    # Checks if a similar face encoding exists in Qdrant
    response = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=new_encoding.tolist(),
        limit=1,
        score_threshold=tolerance
    )
    return len(response) > 0


def register_user(name, encoding, image_path):
    if encoding is None:
        return {"success": False, "message": "❌ No face detected. Try again."}

    if db.users.find_one({"name": name}):
        return {"success": False, "message": f"⚠️ User {name} is already registered."}

    point_id = str(uuid.uuid4())
    image_url = upload_image(image_path)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": point_id,
            "vector": encoding.tolist(),
            "payload": {"name": name, "image": image_url}
        }]
    )

    db.users.insert_one({
        "_id": point_id,
        "name": name,
        "image": image_url
    })

    user, _ = User.objects.get_or_create(username=name)
    token = get_tokens_for_user(user)

    if os.path.exists(image_path):
        os.remove(image_path)

    return {
        "success": True,
        "name": name,
        "image": image_url,
        "token": token,
        "message": f"✅ Registered {name} successfully!"
    }


def login_user(encoding):
    if encoding is None:
        return {"success": False, "message": "❌ No face detected. Try again."}

    match = match_face(qdrant, encoding)
    if not match:
        return {"success": False, "message": "😔 No match found."}

    user_id = match.id
    user = db.users.find_one({"_id": user_id})
    if not user:
        return {"success": False, "message": "❌ User data not found."}

    user, _ = User.objects.get_or_create(username=user["name"])

    access_token = get_tokens_for_user(user)

    return {
        "success": True,
        "name": user.username,
        "image": getattr(user, "image", None),
        "message": f"🎉 Welcome back, {user.username}!",
        "token": access_token
    }
