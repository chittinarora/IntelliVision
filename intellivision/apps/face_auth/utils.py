# utils.py - Utility functions for face encoding and matching
# Provides helpers for capturing, normalizing, and matching face encodings.

import cv2
import face_recognition
import numpy as np
import tempfile
import numpy as np
from .embedding import get_embedding_from_image


def capture_face():
    """Capture a face from the webcam and return its encoding and image path."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(tmp.name, frame)

    embedding = get_embedding_from_image(tmp.name)
    return embedding, tmp.name


def match_face(qdrant, encoding):
    """Find the closest match for a face encoding in Qdrant."""
    encoding = normalize(encoding)
    results = qdrant.search(
        collection_name="face_encodings",
        query_vector=encoding.tolist(),
        limit=1,
        score_threshold=0.93
    )
    return results[0] if results else None


def normalize(vec):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm
