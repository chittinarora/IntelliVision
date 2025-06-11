from db import qdrant
from qdrant_client.models import Distance, VectorParams

def create_face_collection():
    collection_name = "face_encodings"
    
    # Define the collection schema
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=128,              # Set to 512 if using FaceNet
            distance=Distance.COSINE  # COSINE is best for face similarity
        )
    )
    print(f"✅ Collection '{collection_name}' created successfully.")

if __name__ == "__main__":
    create_face_collection()
