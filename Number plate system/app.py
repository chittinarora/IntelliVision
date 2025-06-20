# app.py  ─ run with:  streamlit run app.py
import streamlit as st
from ultralytics import YOLO
import pytesseract, cv2, tempfile, shutil, re
from pathlib import Path
from datetime import datetime
from pymongo import MongoClient
import numpy as np
import tempfile

# ─── Config ─────────────────────────────────────────────────────
MODEL_PATH = r"C:/Users/Hp/runs/detect/train2/weights/best.pt"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# MongoDB (local)
client = MongoClient("mongodb://yourUsername:yourPassword@localhost:27017/")
db = client["vehicle_db"]
collection = db["detected_plates"]

TMP_ROOT = Path(tempfile.gettempdir()) / "plate_app"
TMP_ROOT.mkdir(exist_ok=True)

model = YOLO(MODEL_PATH)

# ─── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="Plate Detector", layout="centered")
st.title("🔍 License‑Plate Detection + OCR (YOLOv8)")
file = st.file_uploader(
    "📂 Upload image (.jpg/.png) or video (.mp4/.avi)",
    type=["jpg", "jpeg", "png", "mp4", "avi"]
)

# ─── Image branch ───────────────────────────────────────────────
if file and file.type.startswith("image"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file.read())
        img_path = tmp.name

    results = model(img_path)
    st.image(results[0].plot(), channels="BGR", caption="Detected plates")

    st.subheader("🔠 OCR")
    found = False
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = results[0].orig_img[y1:y2, x1:x2]

        # OCR preprocessing
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.bilateralFilter(g, 11, 17, 17)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(
            th,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ).strip().upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        match = re.search(r"[A-Z]{2}\d{2}[A-Z]{1,3}\d{3,4}", text)
        if match:
            plate_text = match.group()
            st.success(f"📍 `{plate_text}`")

            # Insert into MongoDB
            collection.insert_one({
                "plate_number": plate_text,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source": file.name,
                "confidence": float(box.conf)
            })
            found = True
    if not found:
        st.warning("No clear plate text detected.")

# ─── Video branch ───────────────────────────────────────────────
elif file and file.type.startswith("video"):
    suffix = Path(file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        vid_in = tmp.name

    st.video(vid_in)
    st.info("Running YOLO — please wait…")

    run_dir = TMP_ROOT / "runs"
    shutil.rmtree(run_dir, ignore_errors=True)

    results = model.predict(
        source=vid_in,
        save=True,
        imgsz=960,
        conf=0.25,
        stream=False,
        project=str(run_dir),
        name="predict",
        exist_ok=True,
    )

    # Optional debug: first 3 frames
    for i, r in enumerate(results[:3]):
        st.write(f"Frame {i}: {len(r.boxes)} detections")

    # Optional: log each detected plate
    for i, frame in enumerate(results):
        for b in frame.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            crop = frame.orig_img[y1:y2, x1:x2]
            txt = pytesseract.image_to_string(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                config="--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            ).strip().upper()
            txt = re.sub(r"[^A-Z0-9]", "", txt)
            m = re.search(r"[A-Z]{2}\d{2}[A-Z]{1,3}\d{3,4}", txt)
            if m:
                collection.insert_one({
                    "plate_number": m.group(),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "source": file.name,
                    "frame": i,
                    "confidence": float(b.conf)
                })

    pred_dir = run_dir / "predict"
    vids = sorted(pred_dir.glob("*.*"))
    if vids:
        st.video(str(vids[0]))
    else:
        st.error("⚠️ YOLO did not save an annotated video. "
                 "Re‑encode your clip to H.264 or install ffmpeg.")

else:
    st.info("👆 Upload an image or video to begin.")
