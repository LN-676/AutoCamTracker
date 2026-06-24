import math


class VehicleMatcher:
    def __init__(self, match_threshold=0.75):
        self.match_threshold = match_threshold
        self.track_to_global = {}
        self.global_features = {}
        self.track_stable_count = {}
        self.recent_global_frame = {}
        self.global_cooldown_frames = 15

    def set_global_features(self, global_features):
        self.global_features = global_features

    def match_or_create(self, feature, global_id_manager, data_store, used_global_ids=None):
        if feature is None:
            return None
        if used_global_ids is None:
            used_global_ids = set()

        track_id = feature.get("track_id")
        if track_id is None:
            return None

        if track_id in self.track_to_global:
            current_global_id = self.track_to_global[track_id]
            feature["global_id"] = current_global_id
            feature["match_score"] = None
            feature["match_type"] = "track_lock"
            return current_global_id

        best_global_id = None
        best_score = 0
        current_frame = feature.get("last_seen_frame", 0)

        for global_id, vehicle_data in self.global_features.items():
            global_id_int = int(global_id)

            if global_id_int in used_global_ids:
                continue

            last_used_frame = self.recent_global_frame.get(global_id_int)

            if last_used_frame is not None and current_frame - last_used_frame < self.global_cooldown_frames:
                continue

            samples = vehicle_data.get("samples", [])

            for old_feature in samples:

                score = self.calculate_similarity(
                    feature,
                    old_feature
                )

                if score > best_score:
                    best_score = score
                    best_global_id = global_id_int

        if best_score >= self.match_threshold:
            global_id = best_global_id
        else:
            global_id = global_id_manager.create_next_vehicle()

        self.track_to_global[track_id] = global_id
        global_id_manager.bind_track_to_global(track_id, global_id)
        self.track_stable_count[track_id] = 1
        self.recent_global_frame[global_id] = current_frame

        feature["global_id"] = global_id
        feature["match_score"] = round(best_score, 4)
        feature["match_type"] = "reid_match"

        print(
            f"[REID] Track {track_id} -> Car {global_id}, "
            f"best_id={best_global_id}, score={best_score:.4f}, "
            f"H={feature.get('debug_hist_score')}, "
            f"C={feature.get('debug_color_score')}, "
            f"A={feature.get('debug_aspect_score')}, "
            f"R={feature.get('debug_area_score')}, "
            f"threshold={self.match_threshold:.2f}, "
            f"known_cars={len(self.global_features)}"
        )

        return global_id

    def calculate_similarity(self, a, b):
        hist_score = self.histogram_similarity(
            a.get("color_histogram"),
            b.get("color_histogram")
        )
        color_score = self.color_similarity(
            a.get("average_color_bgr"),
            b.get("average_color_bgr")
        )

        aspect_score = self.number_similarity(
            a.get("aspect_ratio"),
            b.get("aspect_ratio"),
            tolerance=0.7
        )

        area_score = self.number_similarity(
            a.get("area_ratio"),
            b.get("area_ratio"),
            tolerance=0.03
        )

        a["debug_hist_score"] = round(hist_score, 4)
        a["debug_color_score"] = round(color_score, 4)
        a["debug_aspect_score"] = round(aspect_score, 4)
        a["debug_area_score"] = round(area_score, 4)


        return (
            hist_score * 0.35 +
            color_score * 0.35 +
            aspect_score * 0.20 +
            area_score * 0.10
        )
            
    def histogram_similarity(self, h1, h2):
        if h1 is None or h2 is None:
            return 0

        if len(h1) != len(h2):
            return 0

        score = 0

        for a, b in zip(h1, h2):
            score += min(a, b)

        return score


    def color_similarity(self, c1, c2):
        if c1 is None or c2 is None:
            return 0

        dist = math.sqrt(
            (c1[0] - c2[0]) ** 2 +
            (c1[1] - c2[1]) ** 2 +
            (c1[2] - c2[2]) ** 2
        )

        score = 1 - min(dist / 255, 1)
        return score

    def number_similarity(self, n1, n2, tolerance):
        if n1 is None or n2 is None:
            return 0

        diff = abs(n1 - n2)
        score = 1 - min(diff / tolerance, 1)
        return score