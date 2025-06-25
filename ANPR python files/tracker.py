from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
        self.unique_ids = set()

    def update(self, detections, frame):
        formatted = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            w = x2 - x1
            h = y2 - y1
            formatted.append(([x1, y1, w, h], conf, cls))

        tracks = self.tracker.update_tracks(formatted, frame=frame)
        updated = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            self.unique_ids.add(track_id)
            l, t, r, b = track.to_ltrb()

            updated.append({
                'track_id': track_id,
                'bbox': (int(l), int(t), int(r), int(b))
            })

        return updated

    def get_total_unique_ids(self):
        return len(self.unique_ids)

    def reset(self):
        """Reset tracker state for a new video."""
        self.tracker = DeepSort(max_age=40, n_init=3, nms_max_overlap=0.7)
        self.unique_ids.clear()

