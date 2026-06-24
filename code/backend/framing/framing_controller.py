class FramingController:
    def __init__(self, target_x=0.5, target_y=0.5, dead_zone=0.05):
        self.target_x = target_x
        self.target_y = target_y
        self.dead_zone = dead_zone

    def calculate_error(self, detection):
        error_x = detection["center_x"] - self.target_x
        error_y = detection["center_y"] - self.target_y

        if abs(error_x) < self.dead_zone:
            error_x = 0

        if abs(error_y) < self.dead_zone:
            error_y = 0

        return {
            "track_id": detection["track_id"],
            "error_x": round(error_x, 4),
            "error_y": round(error_y, 4)
        }