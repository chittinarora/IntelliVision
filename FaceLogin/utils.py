import cv2
import tempfile
import numpy as np
from embedding import get_embedding_from_image

def capture_face():
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
    encoding = normalize(encoding)
    results = qdrant.search(
        collection_name="face_encodings",
        query_vector=encoding.tolist(),
        limit=1,
        score_threshold=0.93  # ⬅️ much stricter for ArcFace
    )
    return results[0] if results else None

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm