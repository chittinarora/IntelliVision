# embedding.py

import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # use CUDAExecutionProvider for GPU
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding_from_image(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        return None
    return faces[0].embedding  # 512D vector
