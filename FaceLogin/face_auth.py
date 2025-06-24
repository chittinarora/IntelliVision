import os
import uuid
import numpy as np
from db import db, qdrant, COLLECTION_NAME
from cloudinary_utils import upload_face_image as upload_image
from utils import match_face, normalize

def face_exists(new_encoding, tolerance=0.5):
    encoding = normalize(new_encoding)
    response = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=encoding.tolist(),
        limit=1,
        score_threshold=tolerance
    )
    return len(response) > 0

def register_user(name, encoding, image_path):
    if encoding is None:
        return "❌ No face detected. Try again."

    # Create a UUID for Qdrant point_id
    point_id = str(uuid.uuid4())

    # Check if user already exists in Mongo by name
    if db.users.find_one({"name": name}):
        return f"⚠️ User {name} is already registered."

    image_url = upload_image(image_path)

    encoding = normalize(encoding)

    # Add to Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": point_id,
            "vector": encoding.tolist(),
            "payload": {"name": name, "image": image_url}
        }]
    )

    # Add to MongoDB
    db.users.insert_one({
        "_id": point_id,
        "name": name,
        "image": image_url
    })

    # cleanup
    if os.path.exists(image_path):
        os.remove(image_path)

    return f"✅ Registered {name} successfully!"



def login_user(encoding):
    if encoding is None:
        return "❌ No face detected. Try again."
    
    encoding = normalize(encoding)

    match = match_face(qdrant, encoding)

    if not match:
        return "😔 No match found."
    
    if match.score >= 0.6:
        return "😔 Match too weak. Try again."

    user_id = match.id
    user = db.users.find_one({"_id": user_id})
    if not user:
        return "❌ User data not found."

    return f"🎉 Welcome back, {user['name']}!\n🖼️ Image: {user['image']}"
