import cv2
import os
from typing import List

def extract_key_bedroom_frames(video_path: str, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    frame_paths = []

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame_idx = True, 0

    while success:
        success, frame = vidcap.read()
        if not success:
            break

        frame_path = os.path.join(output_dir, f"frame_{frame_idx:05}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_idx += 1

    vidcap.release()

    if len(frame_paths) < 3:
        return frame_paths
    return [frame_paths[0], frame_paths[len(frame_paths) // 2], frame_paths[-1]]
