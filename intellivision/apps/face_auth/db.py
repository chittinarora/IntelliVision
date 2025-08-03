"""
db.py - Face Auth App
MongoDB and Qdrant setup for face authentication data storage.
"""

import os
import sys
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Global variables for lazy initialization
mongo = None
db = None
users_collection = None
qdrant = None
COLLECTION_NAME = "face_encodings"

def get_mongo_connection():
    """Lazy initialize MongoDB connection only when needed"""
    global mongo, db, users_collection
    
    # Skip during Django management commands to prevent memory corruption
    if any(cmd in sys.argv for cmd in ['migrate', 'collectstatic', 'showmigrations', 'makemigrations']):
        return None, None, None
    
    if mongo is None:
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
    
    return mongo, db, users_collection

def get_qdrant_connection():
    """Lazy initialize Qdrant connection only when needed"""
    global qdrant
    
    # Skip during Django management commands to prevent memory corruption
    if any(cmd in sys.argv for cmd in ['migrate', 'collectstatic', 'showmigrations', 'makemigrations']):
        return None
    
    if qdrant is None:
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
            collections = qdrant.get_collections().collections
            if COLLECTION_NAME not in [c.name for c in collections]:
                qdrant.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=512, distance=Distance.EUCLID)
                )
        except Exception as e:
            raise ConnectionError(f"Could not connect to Qdrant or configure collection. Error: {e}")
    
    return qdrant

# All database connections are now lazy-loaded through functions
# Legacy compatibility - provide None values to prevent import errors
db = None
users_collection = None  
qdrant = None

# Use get_mongo_connection() and get_qdrant_connection() instead of these globals