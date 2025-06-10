import warnings
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

warnings.filterwarnings("ignore", category=UserWarning, module="deep_sort_realtime.embedder.embedder_pytorch")

model = YOLO("yolo12m.pt")
capture = cv2.VideoCapture("../videos/crowd2.mp4")

output_path = "output.mp4"
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
    nn_budget=1500,
    embedder="torchreid",
    embedder_model_name="osnet_x1_0",
    embedder_wts="osnet_x1_0_msmt17.pt",
)

left_roi = ((400, 200), (1000, 600))  # (x1, y1), (x2, y2)
left_line_y = left_roi[1][1]

track_positions = {}

counts = {"in": 0, "out": 0}

compact_id_map = {}
next_compact_id = 1

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (1280, 720))  # Resize for better performance
    if not ret:
        break

    height, width, _ = frame.shape

    cv2.rectangle(frame, left_roi[0], left_roi[1], (255, 0, 0), 2)
    cv2.line(frame, (left_roi[0][0], left_line_y),
             (left_roi[1][0], left_line_y), (255, 255, 0), 1)

    results = model(frame, conf=0.6, iou=0.25)[0]
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, y2

        in_roi = left_roi[0][0] <= cx <= left_roi[1][0] and left_roi[0][1] <= cy <= left_roi[1][1]

        if model.names[cls] == 'person':
            detections.append(([x1, y1, w, h], conf, model.names[cls]))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id

        # Map internal tracker ID to a compact display ID (1, 2, 3, ...)
        if track_id not in compact_id_map:
            compact_id_map[track_id] = next_compact_id
            next_compact_id += 1
        display_id = compact_id_map[track_id]

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx, cy = (x1 + x2) // 2, y2

        prev_pos = track_positions.get(track_id, (cx, cy))
        prev_cx, prev_cy = prev_pos
        track_positions[track_id] = (cx, cy)

        movement_vector_y = cy - prev_cy

        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        cv2.arrowedLine(frame, (prev_cx, prev_cy), (cx, cy), (255, 0, 255), 2)

        roi_x1, roi_y1 = left_roi[0]
        roi_x2, roi_y2 = left_roi[1]

        ''' 
        prev_in = (roi_x1 <= prev_cx <= roi_x2) and (roi_y1 <= prev_cy <= roi_y2)
        current_in = (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2)

        if prev_in and not current_in:
            if cx < roi_x1:
                counts["out"] += 1
            elif cx > roi_x2:
                counts["in"] += 1
            elif cy < roi_y1:
                counts["in"] += 1
            elif cy > roi_y2:
                counts["out"] += 1
        '''
        prev_in = (roi_x1 <= prev_cx <= roi_x2) and (roi_y1 <= prev_cy <= roi_y2)
        current_in = (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2)

        # If track enters ROI: count as "IN"
        if not prev_in and current_in:
            counts["in"] += 1
        # If track exits ROI: count as "OUT"
        elif prev_in and not current_in:
            counts["out"] += 1

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {display_id}', (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display counts
    cv2.putText(frame, f"IN: {counts['in']}  OUT: {counts['out']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    out.write(frame)  # Save frame

    cv2.imshow('People Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()