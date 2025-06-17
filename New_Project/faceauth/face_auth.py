import os
import uuid
import tempfile
import numpy as np

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
        return "❌ No face detected. Try again."

    point_id = str(uuid.uuid4())

    # Check if user already exists
    if db.users.find_one({"name": name}):
        return f"⚠️ User {name} is already registered."

    image_url = upload_image(image_path)

    # Upsert into Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": point_id,
            "vector": encoding.tolist(),
            "payload": {"name": name, "image": image_url}
        }]
    )

    # Insert into MongoDB
    db.users.insert_one({
        "_id": point_id,
        "name": name,
        "image": image_url
    })

    # Cleanup temp image
    if os.path.exists(image_path):
        os.remove(image_path)

    return f"✅ Registered {name} successfully!"


def login_user(encoding):
    if encoding is None:
        return "❌ No face detected. Try again."

    match = match_face(qdrant, encoding)
    if not match:
        return "😔 No match found."

    user_id = match.id
    user = db.users.find_one({"_id": user_id})
    if not user:
        return "❌ User data not found."

    return f"🎉 Welcome back, {user['name']}!\n🖼️ Image: {user['image']}"
