"""
db.py - Face Auth App
MongoDB and Qdrant setup for face authentication data storage.
"""

import os
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# --- MongoDB Setup ---
# Load the URI from environment variables without a hardcoded fallback
mongo_uri = os.environ.get("MONGO_URI")
if not mongo_uri:
    raise ValueError("Error: MONGO_URI environment variable not set")

# Connect to MongoDB
try:
    mongo = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = mongo.face_auth
    users_collection = db.users
except Exception as e:
    raise ConnectionError(f"Could not connect to MongoDB. Error: {e}")


# --- Qdrant Setup ---
# Load Qdrant credentials from environment variables
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")

if not qdrant_url or not qdrant_api_key:
    raise ValueError("Error: QDRANT_URL or/and QDRANT_API_KEY environment variables not set.")

# Connect to Qdrant
try:
    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )

    # Ensure collection exists in Qdrant
    COLLECTION_NAME = "face_encodings"
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.EUCLID)
        )
except Exception as e:
    raise ConnectionError(f"Could not connect to Qdrant or configure collection. Error: {e}")