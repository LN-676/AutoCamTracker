import cv2
import numpy as np


class VehicleFeatureBuilder:
    def build_feature(self, frame, detection, frame_id):
        x1, y1, x2, y2 = detection["bbox"]
        frame_h, frame_w = frame.shape[:2]

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        avg_color = self.get_average_color(crop)
        color_name = self.get_simple_color_name(avg_color)
        color_histogram = self.get_color_histogram(crop)
        dominant_colors = {}

        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        frame_area = frame_w * frame_h

        aspect_ratio = box_w / box_h if box_h > 0 else 0
        area_ratio = box_area / frame_area if frame_area > 0 else 0

        center_x = ((x1 + x2) / 2) / frame_w
        center_y = ((y1 + y2) / 2) / frame_h

        feature = {
            "track_id": detection.get("track_id"),
            "global_id": detection.get("global_id"),
            "color_name": color_name,
            "average_color_bgr": [
                int(avg_color[0]),
                int(avg_color[1]),
                int(avg_color[2])
            ],
            "color_histogram": color_histogram,
            "dominant_colors": dominant_colors,
            "aspect_ratio": round(aspect_ratio, 4),
            "area_ratio": round(area_ratio, 6),
            "last_center_x": round(center_x, 4),
            "last_center_y": round(center_y, 4),
            "last_seen_frame": frame_id
        }

        return feature

    def get_average_color(self, crop):
        # 避免背景太多，取中間區域
        h, w = crop.shape[:2]

        cx1 = int(w * 0.25)
        cx2 = int(w * 0.75)
        cy1 = int(h * 0.25)
        cy2 = int(h * 0.75)

        center_crop = crop[cy1:cy2, cx1:cx2]

        if center_crop.size == 0:
            center_crop = crop

        avg_color = np.mean(center_crop, axis=(0, 1))
        return avg_color

    def get_simple_color_name(self, avg_bgr):
        b, g, r = avg_bgr

        if r > 180 and g > 180 and b > 180:
            return "white"

        if r < 80 and g < 80 and b < 80:
            return "black"

        if r > g + 40 and r > b + 40:
            return "red"

        if b > r + 40 and b > g + 40:
            return "blue"

        if g > r + 40 and g > b + 40:
            return "green"

        if r > 150 and g > 120 and b < 100:
            return "yellow"

        return "unknown"
    
    def get_color_histogram(self, crop):
        h, w = crop.shape[:2]

        cx1 = int(w * 0.15)
        cx2 = int(w * 0.85)
        cy1 = int(h * 0.15)
        cy2 = int(h * 0.85)

        center_crop = crop[cy1:cy2, cx1:cx2]

        if center_crop.size == 0:
            center_crop = crop

        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [8, 8],
            [0, 180, 0, 256]
        )

        hist_sum = hist.sum()

        if hist_sum > 0:
            hist = hist / hist_sum

        return hist.flatten().round(6).tolist()
    

        def get_dominant_colors(self, crop):
            h, w = crop.shape[:2]

            cx1 = int(w * 0.15)
            cx2 = int(w * 0.85)
            cy1 = int(h * 0.15)
            cy2 = int(h * 0.85)

            center_crop = crop[cy1:cy2, cx1:cx2]

            if center_crop.size == 0:
                center_crop = crop

            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            hsv = cv2.resize(hsv, (32, 32))

            color_counts = {
                "red": 0,
                "yellow": 0,
                "green": 0,
                "blue": 0,
                "white": 0,
                "black": 0,
                "gray": 0,
                "other": 0
            }

            total = 0

            for row in hsv:
                for pixel in row:
                    h_val, s_val, v_val = pixel
                    total += 1

                    if v_val < 50:
                        color_counts["black"] += 1
                    elif s_val < 40 and v_val > 180:
                        color_counts["white"] += 1
                    elif s_val < 40:
                        color_counts["gray"] += 1
                    elif h_val < 10 or h_val > 170:
                        color_counts["red"] += 1
                    elif 20 <= h_val <= 35:
                        color_counts["yellow"] += 1
                    elif 35 < h_val <= 85:
                        color_counts["green"] += 1
                    elif 85 < h_val <= 130:
                        color_counts["blue"] += 1
                    else:
                        color_counts["other"] += 1

            if total == 0:
                return color_counts

            for key in color_counts:
                color_counts[key] = round(color_counts[key] / total, 4)

            return color_counts