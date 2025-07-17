from ultralytics import YOLO
from pathlib import Path

def load_yolo_model(model_path: str):
    """
    Loads a YOLO model, downloading from Ultralytics Hub if not present,
    except for best_plate.pt, yolo11m.pt, best_animal.pt which must exist locally.
    """
    basename = Path(model_path).name
    must_exist = {"best_plate.pt", "yolo11m.pt", "best_animal.pt"}
    if basename in must_exist:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} must be present locally and will not be auto-downloaded.")
        return YOLO(str(model_path))
    else:
        # If the file exists locally, load it; otherwise, auto-download using the basename
        if Path(model_path).exists():
            return YOLO(str(model_path))
        else:
            return YOLO(basename)
