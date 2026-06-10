from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path, conf=0.25, imgsz=960):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect_and_track(self, frame):
        results = self.model.track(
            frame,
            classes=[2],              # COCO class 2 = car
            conf=self.conf,
            imgsz=self.imgsz,
            tracker="configs/bytetrack_race.yaml",
            persist=True,
            verbose=False
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        frame_h, frame_w = frame.shape[:2]

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            bbox = [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ]

            # 第一步：過濾不合理的框
            if not self.is_valid_box(bbox, frame_w, frame_h):
                continue

            center_x = ((x1 + x2) / 2) / frame_w
            center_y = ((y1 + y2) / 2) / frame_h

            detections.append({
                "track_id": track_id,
                "label": "car",
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "center_x": round(center_x, 4),
                "center_y": round(center_y, 4)
            })

        # 第二步：移除重複框
        detections = self.remove_duplicate_boxes(detections, iou_threshold=0.5)
        return detections

        if not results or results[0].boxes is None:
            return detections

        frame_h, frame_w = frame.shape[:2]

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            center_x = ((x1 + x2) / 2) / frame_w
            center_y = ((y1 + y2) / 2) / frame_h

            detections.append({
                "track_id": track_id,
                "label": "car",
                "confidence": round(confidence, 4),
                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ],
                "center_x": round(center_x, 4),
                "center_y": round(center_y, 4)
            })

        return detections
    
    def is_valid_box(self, bbox, frame_w, frame_h):
        x1, y1, x2, y2 = bbox

        box_w = x2 - x1
        box_h = y2 - y1

        area = box_w * box_h
        frame_area = frame_w * frame_h

        # 太小的框不要
        if area < frame_area * 0.0015:
            return False

        # 太扁或太窄的框不要
        if box_w < 35 or box_h < 25:
            return False

        # 左側排行榜區域先排除
        if x2 < frame_w * 0.28:
            return False

        return True

    def iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        inter_area = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        union_area = area_a + area_b - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def remove_duplicate_boxes(self, detections, iou_threshold=0.5):
        detections = sorted(
            detections,
            key=lambda x: x["confidence"],
            reverse=True
        )

        filtered = []

        for det in detections:
            keep = True

            for kept in filtered:
                if self.iou(det["bbox"], kept["bbox"]) > iou_threshold:
                    keep = False
                    break

            if keep:
                filtered.append(det)

        return filtered