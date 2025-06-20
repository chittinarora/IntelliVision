
import streamlit as st
from ultralytics import YOLO
import cv2, tempfile, os
from pathlib import Path

# ---------- CONFIG ---------------------------------------------------
weights   = r"C:\Users\Hp\runs\detect\snakes_v7\weights\best.pt"
save_root = Path(r"F:\results")
save_root.mkdir(parents=True, exist_ok=True)
model = YOLO(weights)

st.title("🐍 Snake Detection Demo")
uploaded = st.file_uploader(
    "Upload an image (.jpg/.png) or video (.mp4/.mov)",
    type=["jpg", "jpeg", "png", "mp4", "mov"]
)

# ---------- HANDLE UPLOAD -------------------------------------------
if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp_path = tmp.name

    # ----- IMAGE -----------------------------------------------------
    if suffix in [".jpg", ".jpeg", ".png"]:
        res = model(tmp_path, imgsz=640, conf=0.25)[0]
        bgr = res.plot()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        out_img = save_root / "annotated_upload.jpg"
        cv2.imwrite(str(out_img), bgr)

        st.image(rgb, caption="Detection Result", use_container_width=True)
        snakes = (res.boxes.cls == 0).sum().item()
        st.success(f"✅ Saved to {out_img}")
        

    # ----- VIDEO -----------------------------------------------------
    elif suffix in [".mp4", ".mov"]:
        pred_root = save_root / "pred"
        pred_root.mkdir(exist_ok=True)

        model.predict(
            source=tmp_path,
            imgsz=640,
            conf=0.25,
            save=True,
            project=pred_root,    # F:\results\pred
            name="predict",       # → F:\results\pred\predict\…
            exist_ok=True
        )

        mp4_files = list(pred_root.rglob("*.mp4"))
        if mp4_files:
            out_vid = mp4_files[0]
            st.video(str(out_vid))          # use path string
            st.success(f"🎥 Saved to: {out_vid}")
        else:
            st.error(
                "Prediction finished, but no output video was written.\n"
                "Ensure FFmpeg is installed and try a smaller MP4."
            )



