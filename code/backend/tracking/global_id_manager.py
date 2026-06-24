import json
from pathlib import Path


class GlobalIDManager:
    def __init__(self, profile_path="data/vehicle_profiles.json"):
        self.profile_path = Path(profile_path)
        self.vehicles = self.load_vehicle_profiles()
        self.track_to_global = {}

    def load_vehicle_profiles(self):
        if not self.profile_path.exists():
            return {}

        with open(self.profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vehicles = {}

        for vehicle in data.get("vehicles", []):
            global_id = vehicle["global_id"]
            vehicles[global_id] = vehicle

        return vehicles

    def save_vehicle_profiles(self):
        data = {
            "vehicles": list(self.vehicles.values())
        }

        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_vehicle_if_missing(self, global_id):
        if global_id in self.vehicles:
            return self.vehicles[global_id]

        vehicle = {
            "global_id": global_id,
            "display_name": f"Car {global_id}",
            "car_number": str(global_id),
            "color": "unknown",
            "model": "Unknown",
            "notes": "自動新增車輛",
            "feature_ready": False
        }

        self.vehicles[global_id] = vehicle
        self.save_vehicle_profiles()

        return vehicle


    def create_next_vehicle(self):
        if len(self.vehicles) == 0:
            next_id = 1
        else:
            next_id = max(self.vehicles.keys()) + 1

        self.create_vehicle_if_missing(next_id)
        return next_id

    def bind_track_to_global(self, local_track_id, global_id):
        self.create_vehicle_if_missing(global_id)
        self.track_to_global[local_track_id] = global_id

    def unbind_track(self, local_track_id):
        if local_track_id in self.track_to_global:
            del self.track_to_global[local_track_id]

    def get_global_id(self, local_track_id):
        return self.track_to_global.get(local_track_id)

    def get_vehicle_info(self, global_id):
        if global_id is None:
            return None

        return self.vehicles.get(global_id)

    def enrich_detection(self, detection):
        local_track_id = detection.get("track_id")
        global_id = self.get_global_id(local_track_id)
        vehicle_info = self.get_vehicle_info(global_id)

        detection["global_id"] = global_id
        detection["vehicle_info"] = vehicle_info

        return detection