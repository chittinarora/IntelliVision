import os
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# --- MongoDB Setup ---
mongo = MongoClient(os.getenv("MONGO_URI"))
db = mongo.face_auth
users_collection = db.users

# --- Qdrant Setup ---
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Ensure collection exists in Qdrant
COLLECTION_NAME = "face_encodings"

# Vector size of face_recognition encodings is 128
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=128, distance=Distance.COSINE)
    )