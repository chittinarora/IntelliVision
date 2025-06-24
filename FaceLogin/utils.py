import cv2
import face_recognition
import numpy as np
import tempfile
from . import normalize

def capture_face():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="cnn")
    encodings = face_recognition.face_encodings(rgb, locs)
    if not encodings:
        return None, None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(tmp.name, frame)
    return encodings[0], tmp.name

def match_face(qdrant, encoding):
    encoding = normalize(encoding)
    results = qdrant.search(
        collection_name="face_encodings",
        query_vector=encoding.tolist(),
        limit=1,
        score_threshold=0.6
    )
    return results[0] if results else None

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm