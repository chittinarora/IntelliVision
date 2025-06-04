import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the YOLO model with the specified weights file
model = YOLO("yolo11m.pt")

# Open the video file for processing
capture = cv2.VideoCapture("images/crowd3.mp4")

# Initialize the DeepSort tracker with a maximum age for tracks
tracker = DeepSort(max_age=30)

# Initialize frame counter
frame_num = 0

# Check if the video capture is opened successfully, and read frame from the video
# Break the loop if no frame is returned (end of video)
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Perform object detection using the YOLO model
    results = model(frame)[0]
    # List to store detections for tracking
    detections = []

    # Process each detected bounding box, and iterate through the bounding boxes
    # Save the class index of the detected object, confidence score, and bounding box coordinates
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Filter detections for 'person' class with confidence > 0.5
        if model.names[cls] == 'person' and conf > 0.5:
            detections.append(([x1, y1, x2, y2], conf, model.names[cls]))

    # Update tracker with the current frame and detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Iterate through each track returned by the tracker
    for track in tracks:

        # Check if the track is confirmed before processing
        if not track.is_confirmed():
            continue

        # Save the track unique id, bounding box coordinates, and map the coordinates to integers
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box and label for the tracked object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f'Person ID: {track_id}'
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the processed frame in a window
    cv2.imshow('People Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()