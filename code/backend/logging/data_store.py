import json
from pathlib import Path


class DataStore:
    def __init__(
        self,
        output_path="output/tracking_log.json",
        event_path="output/global_id_events.json",
        feature_path="output/vehicle_features.json",
        global_feature_path="output/global_vehicle_features.json",
    ):
        self.output_path = Path(output_path)
        self.event_path = Path(event_path)
        self.feature_path = Path(feature_path)
        self.global_feature_path = Path(global_feature_path)
    

        self.output_path.parent.mkdir(exist_ok=True)
        self.event_path.parent.mkdir(exist_ok=True)
        self.feature_path.parent.mkdir(exist_ok=True)
        self.global_feature_path.parent.mkdir(exist_ok=True)

        self.logs = []
        self.events = self.load_events()
        self.features = {}
        self.global_features = self.load_global_vehicle_features()
        self.enable_frame_log = False

        # 啟動時先讀取既有事件，避免重啟後覆蓋
        self.events = self.load_events()

    def load_events(self):
        if not self.event_path.exists():
            return []

        try:
            with open(self.event_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data

            return []

        except Exception:
            return []

    def add_frame_log(self, frame_data):
        if self.enable_frame_log:
            self.logs.append(frame_data)

    def add_event(self, event_data):
        self.events.append(event_data)
        self.save_events()

    def save(self):
        if self.enable_frame_log:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.logs, f, ensure_ascii=False, indent=2)

        self.save_events()
        self.save_vehicle_features()
        self.save_global_vehicle_features()


    def load_global_vehicle_features(self):
        if not self.global_feature_path.exists():
            return {}

        try:
            with open(self.global_feature_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                return data

            return {}

        except Exception:
            return {}


    def add_global_vehicle_feature(self, global_id, feature):
        if global_id is None or feature is None:
            return
        area_ratio = feature.get("area_ratio", 0)

        # if area_ratio < 0.002:
        #     return

        key = str(global_id)

        if key not in self.global_features:
            self.global_features[key] = {
                "global_id": global_id,
                "samples": []
            }

        samples = self.global_features[key]["samples"]

        match_score = feature.get("match_score")

        # 如果這台 Car 已經有樣本，只有高信心才允許加入新樣本
        if len(samples) > 0 and (match_score is None or match_score < 0.95):
            return

        current_track_id = feature.get("track_id")
        current_frame = feature.get("last_seen_frame", 0)

        for sample in samples:
            same_track = sample.get("track_id") == current_track_id
            old_frame = sample.get("last_seen_frame", 0)

            if same_track and current_frame - old_frame < 30:
                return

        if len(samples) >= 10:
            samples.pop(0)

        samples.append(feature)

        self.save_global_vehicle_features()

        samples.append(feature)

        self.save_global_vehicle_features()


    def save_global_vehicle_features(self):
        with open(self.global_feature_path, "w", encoding="utf-8") as f:
            json.dump(self.global_features, f, ensure_ascii=False, indent=2)

    def save_events(self):
        with open(self.event_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, ensure_ascii=False, indent=2)

    def add_vehicle_feature(self, feature):
        if feature is None:
            return

        track_id = feature.get("track_id")

        if track_id is None:
            return

        self.features[str(track_id)] = feature


    def save_vehicle_features(self):
        with open(self.feature_path, "w", encoding="utf-8") as f:
            json.dump(self.features, f, ensure_ascii=False, indent=2)

    def load_global_vehicle_features(self):
        if not self.global_feature_path.exists():
            return {}

        try:
            with open(self.global_feature_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                return data

            return {}

        except Exception:
            return {}


    def save_global_vehicle_features(self):
        with open(self.global_feature_path, "w", encoding="utf-8") as f:
            json.dump(
                self.global_features,
                f,
                ensure_ascii=False,
                indent=2
            )