import cv2
import datetime
import os

def draw_label(frame, text, x, y, color=(0, 255, 0)):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_box(frame, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_frame(frame, output_path, filename_prefix="frame"):
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.jpg"
    full_path = os.path.join(output_path, filename)
    cv2.imwrite(full_path, frame)
    return full_path

