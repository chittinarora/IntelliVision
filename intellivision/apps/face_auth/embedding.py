# embedding.py

import cv2
import numpy as np
import sys
import os
from insightface.app import FaceAnalysis

app = None

def get_face_analysis_app():
    """Lazy initialize face analysis app only when needed"""
    global app
    if app is None:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def preload_face_models():
    """Preload face analysis models during startup (Celery only)"""
    # Only preload in Celery workers, not during Django migrations
    if any(cmd in sys.argv for cmd in ['migrate', 'collectstatic', 'showmigrations', 'makemigrations']):
        return
    
    # Only preload if we're in Celery worker environment
    if os.environ.get('SERVICE_TYPE') == 'celery':
        print("🔄 Preloading buffalo_l face models...")
        try:
            get_face_analysis_app()
            print("✅ Face models preloaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to preload face models: {e}")
            # Don't fail startup, just log the warning

def enhance_image_lighting(img):
    """Apply multiple lighting enhancement techniques to improve face detection."""
    # Convert to LAB color space for better lighting adjustment
    # LAB separates lightness (L) from color information (A, B), allowing us to adjust brightness
    # without affecting color balance, which is crucial for face detection
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    # CLAHE improves local contrast by working on small regions (tiles) of the image
    # This helps reveal facial features in shadowed areas or uneven lighting
    # clipLimit=3.0 prevents over-amplification of noise\
    # tileGridSize=(8,8) creates 64 regions for localized enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Additional gamma correction for very dark images
    # Gamma < 1 brightens dark regions more than bright regions
    # This helps improve visibility of faces in low-light conditions
    gamma = 1.2
    enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255.0
    enhanced = enhanced.astype(np.uint8)

    return enhanced

def preprocess_for_detection(img):
    """Preprocess image to improve face detection in various lighting conditions."""
    # Enhance lighting using LAB color space and CLAHE
    enhanced = enhance_image_lighting(img)

    # Apply bilateral filter to reduce noise while preserving edges
    # This is crucial for face detection as it smooths textures while keeping facial features sharp
    # d=9: diameter of pixel neighborhood, sigmaColor=75, sigmaSpace=75 for good edge preservation
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Sharpen the image slightly to enhance facial features
    # This kernel emphasizes edges and fine details that are important for face recognition
    # The center value of 9 amplifies the current pixel, while surrounding -1 values create contrast
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)

    return sharpened

def get_embedding_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Get face analysis app (lazy loaded)
    app = get_face_analysis_app()

    # Strategy 1: Try with original image first (maintains performance for good lighting)
    faces = app.get(img)

    # Strategy 2: If no faces detected, try with enhanced image using advanced preprocessing
    # This handles uneven lighting, shadows, and low contrast scenarios
    if not faces:
        enhanced_img = preprocess_for_detection(img)
        faces = app.get(enhanced_img)

    # Strategy 3: If still no faces, try with simple brightness adjustments
    # This covers cases where the image is just too bright or too dark overall
    if not faces:
        # Try brighter version for underexposed images
        # alpha=1.3 increases contrast, beta=30 adds brightness
        bright_img = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
        faces = app.get(bright_img)

        # Strategy 4: Try darker version for overexposed images
        # alpha=0.8 reduces contrast, beta=-20 reduces brightness
        if not faces:
            dark_img = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
            faces = app.get(dark_img)

    # If no faces found after all attempts, return None
    if not faces:
        return None

    # Return the face embedding (512-dimensional vector) for the best detected face
    return faces[0].embedding  # 512D vector
