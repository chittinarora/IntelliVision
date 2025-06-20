import warnings
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

warnings.filterwarnings("ignore", category=UserWarning, module="deep_sort_realtime.embedder.embedder_pytorch")

model = YOLO("yolo12l.pt")
capture = cv2.VideoCapture("../videos/crowd4.mp4")

output_path = "../output/output_in_out.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width, frame_height = 1280, 720
fps = capture.get(cv2.CAP_PROP_FPS)
if not fps or fps < 1:
    fps = 25

out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

tracker = DeepSort(
    max_age=60,
    n_init=4,
    max_iou_distance=0.2,
    nn_budget=1000,
    embedder="torchreid",
    embedder_model_name="osnet_x1_0",
    embedder_wts="/Users/adidubbs/.cache/torch/checkpoints/osnet_x1_0_msmt17.pth",
)

left_roi = ((400, 200), (1000, 600))
left_line_y = left_roi[1][1]

track_positions = {}
track_fully_in = {}

counts = {"in": 0, "out": 0}

compact_id_map = {}
next_compact_id = 1

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))

    height, width, _ = frame.shape

    cv2.rectangle(frame, left_roi[0], left_roi[1], (255, 0, 0), 2)
    cv2.line(frame, (left_roi[0][0], left_line_y),
             (left_roi[1][0], left_line_y), (255, 255, 0), 1)

    results = model(frame, conf=0.4, iou=0.4)[0]
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

        roi_x1, roi_y1 = left_roi[0]
        roi_x2, roi_y2 = left_roi[1]

        prev_in = (roi_x1 <= prev_cx <= roi_x2) and (roi_y1 <= prev_cy <= roi_y2)
        current_in = (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2)

        if not prev_in and current_in:
            counts["in"] += 1
        if prev_in and not current_in:
            counts["out"] += 1

        if current_in:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {display_id}', (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    counter_bg_color = (255, 0, 0)
    counter_text_color = (255, 255, 255)

    rect_x1, rect_y1 = 10, 10
    rect_x2, rect_y2 = 300, 70
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), counter_bg_color, -1)

    cv2.putText(frame, f"IN: {counts['in']}", (rect_x1 + 15, rect_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, counter_text_color, 2)
    cv2.putText(frame, f"OUT: {counts['out']}", (rect_x1 + 15, rect_y1 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, counter_text_color, 2)

    out.write(frame)

    cv2.imshow('People Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()