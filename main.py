import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the YOLO model with the specified weights file
model = YOLO("yolo12m.pt")

# Open the video file for processing
# capture = cv2.VideoCapture("videos/crowd2.mp4")
capture = cv2.VideoCapture("videos/soccer.mp4")

# Initialize the DeepSort tracker with a maximum age for tracks
tracker = DeepSort(
    max_age=30,
    n_init=4,
    max_iou_distance=0.4,
    nn_budget=400,
    embedder="mobilenet")

# Initialize frame counter
frame_num = 0

# Check if the video capture is opened successfully, and read frame from the video
# Break the loop if no frame is returned (end of video)
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Perform object detection using the YOLO model
    results = model(frame, conf=0.4, iou=0.3)[0]

    # List to store detections for tracking
    detections = []

    # Process each detected bounding box, and iterate through the bounding boxes
    # Save the class index of the detected object, confidence score, and bounding box coordinates
    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        w = x2 - x1
        h = y2 - y1

        # Filter detections for 'person' class
        if model.names[cls] == 'person':
            detections.append(([x1, y1, w, h], conf, model.names[cls]))

    # Update tracker with the current frame and detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Iterate through each track returned by the tracker
    for track in tracks:

        # Check if the track is confirmed before processing
        if not track.is_confirmed():
            continue

        # Save the track unique id, bounding box coordinates, and map the coordinates to integers
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Draw bounding box and label for the tracked object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f'Person ID: {track_id}'
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Display the processed frame in a window
    cv2.namedWindow("People Detection", cv2.WINDOW_AUTOSIZE)
    cv2.imshow('People Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()
