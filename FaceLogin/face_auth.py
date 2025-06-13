import uuid
from db import db, qdrant
from cloudinary_utils import upload_face_image as upload_image
from utils import capture_face, match_face

COLLECTION_NAME = "face_encodings"

def register_user(name, encoding):
    encoding, image_path = capture_face()
    if encoding is None:
        return "❌ No face detected. Try again."

    # Create a UUID for Qdrant point_id
    point_id = str(uuid.uuid4())

    # Check if user already exists in Mongo by name
    if db.users.find_one({"name": name}):
        return f"⚠️ User {name} is already registered."

    image_url = upload_image(image_path)

    # ➕ Add to Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": point_id,
            "vector": encoding.tolist(),
            "payload": {"name": name, "image": image_url}
        }]
    )

    # ➕ Add to MongoDB
    db.users.insert_one({
        "_id": point_id,  # Same as Qdrant point ID
        "name": name,
        "image": image_url
    })

    return f"✅ Registered {name} successfully!"

def login_user(encoding):
    encoding, _ = capture_face()
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