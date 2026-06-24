class TargetSelector:
    def __init__(self):
        self.selected_target_id = None

    def select_target(self, track_id):
        self.selected_target_id = track_id

    def unlock_target(self):
        self.selected_target_id = None

    def get_selected_target(self):
        return self.selected_target_id

    def find_selected_detection(self, detections):
        if self.selected_target_id is None:
            return None

        for det in detections:
            if det["track_id"] == self.selected_target_id:
                return det

        return None