import warnings
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import subprocess


def convert_to_web_mp4(input_file, output_file):
    """
    Converts input_file to a browser-friendly mp4 (H.264, yuv420p, faststart)
    """
    command = [
        "ffmpeg",
        "-y",  # Overwrite output if exists
        "-i", input_file,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Converted {input_file} → {output_file} (web compatible)")
        return True
    except subprocess.CalledProcessError as e:
        print("ffmpeg error:", e.stderr.decode())
        return False


def tracking_video(input_path, output_path):
    print("=== process_video CALLED ===")
    print("Input path:", input_path)
    print("Output path:", output_path)

    warnings.filterwarnings("ignore", category=UserWarning, module="deep_sort_realtime.embedder.embedder_pytorch")

    # Get the path to the project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "yolo12m.pt")

    # Load model
    model = YOLO(MODEL_PATH)
    capture = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width, frame_height = 1280, 720
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 25

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    tracker = DeepSort(
        max_age=60,
        n_init=6,
        max_iou_distance=0.05,
        nn_budget=1000,
        embedder="torchreid",
        embedder_model_name="osnet_x1_0",
        embedder_wts="/Users/adidubbs/.cache/torch/checkpoints/osnet_x1_0_msmt17.pth",
    )

    track_positions = {}
    compact_id_map = {}
    next_compact_id = 1

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))

        results = model(frame, conf=0.4, iou=0.7)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if model.names[cls] == 'person':
                detections.append(([x1, y1, w, h], conf, model.names[cls]))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id

            if track_id not in compact_id_map:
                compact_id_map[track_id] = next_compact_id
                next_compact_id += 1
            display_id = compact_id_map[track_id]

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, y2

            prev_pos = track_positions.get(track_id, (cx, cy))
            prev_cx, prev_cy = prev_pos
            track_positions[track_id] = (cx, cy)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {display_id}', (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    out.release()
    capture.release()
    cv2.destroyAllWindows()

    fixed_output_path = output_path.replace(".mp4", "_fixed.mp4")
    conversion_success = convert_to_web_mp4(output_path, fixed_output_path)

    # Optionally, replace the original with the fixed one (careful if you want to keep the original)
    if conversion_success:
        os.replace(fixed_output_path, output_path)
