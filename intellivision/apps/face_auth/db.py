"""
db.py - Face Auth App
MongoDB and Qdrant setup for face authentication data storage.
"""

import os
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# --- MongoDB Setup ---
mongo = MongoClient(os.environ.get("MONGO_URI"))
db = mongo.face_auth
users_collection = db.users

# --- Qdrant Setup ---
qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY")
)

# Ensure collection exists in Qdrant
COLLECTION_NAME = "face_encodings"

# Vector size of face_recognition encodings is 512
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.EUCLID)
    )
