import cv2
from pathlib import Path


class ThumbnailManager:

    def __init__(
        self,
        save_dir="output/thumbnails",
        history_dir="output/global_thumbnails",
        max_items=100
    ):
        self.thumbnails = {}
        self.save_dir = Path(save_dir)
        self.history_dir = Path(history_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.max_items = max_items

    def update(self, frame, detections):
        current_ids = set()

        for det in detections:
            track_id = det.get("track_id")

            if track_id is None:
                continue

            current_ids.add(track_id)

            x1, y1, x2, y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            thumb = cv2.resize(crop, (120, 80))

            self.thumbnails[track_id] = {
                "track_id": track_id,
                "image": thumb,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "global_id": det.get("global_id"),
                "vehicle_info": det.get("vehicle_info")
            }

            global_id = det.get("global_id")

            if global_id is not None:
                self.save_global_thumbnail(global_id, thumb)

        self.thumbnails = {
            track_id: data
            for track_id, data in self.thumbnails.items()
            if track_id in current_ids
        }

        self.thumbnails = dict(
            list(self.thumbnails.items())[-self.max_items:]
        )

        self.save_current_thumbnails()

    def save_current_thumbnails(self):
        for old_file in self.save_dir.glob("track_*.jpg"):
            old_file.unlink()

        for track_id, data in self.thumbnails.items():
            save_path = self.save_dir / f"track_{track_id}.jpg"
            cv2.imwrite(str(save_path), data["image"])

    def save_global_thumbnail(self, global_id, thumb):
        save_path = self.history_dir / f"car_{global_id}.jpg"
        cv2.imwrite(str(save_path), thumb)

    def get_thumbnail(self, track_id):
        return self.thumbnails.get(track_id)

    def get_all(self):
        return self.thumbnails