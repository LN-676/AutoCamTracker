import json
from pathlib import Path


class DataStore:
    def __init__(
        self,
        output_path="output/tracking_log.json",
        event_path="output/global_id_events.json"
    ):
        self.output_path = Path(output_path)
        self.event_path = Path(event_path)

        self.output_path.parent.mkdir(exist_ok=True)
        self.event_path.parent.mkdir(exist_ok=True)

        self.logs = []

        # 開發階段關閉逐幀 log，避免產生超大 JSON
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

    def save_events(self):
        with open(self.event_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, ensure_ascii=False, indent=2)