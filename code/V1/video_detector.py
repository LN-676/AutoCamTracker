"""Input Video + YOLO Detection module for AutoCamTracker V1.

Responsibilities:
- Open webcam, local video file, or screen-region sources.
- Load an Ultralytics YOLO model.
- Run YOLO tracking with BoT-SORT or Deep OC-SORT.
- Return raw frames plus tracked detections.

This module should not manage target selection, UI layout, reframing, or
recording file output.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Iterable, Literal


SourceType = Literal["webcam", "video_file", "screen_region"]
TrackerName = Literal["botsort", "deepocsort"]


TRACKER_CONFIGS: dict[TrackerName, str] = {
    "botsort": "botsort.yaml",
    "deepocsort": "deepocsort.yaml",
}

VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class InputConfig:
    source_type: SourceType = "webcam"
    camera_index: int = 0
    video_path: str | None = None
    screen_region: tuple[int, int, int, int] | None = None
    model_path: str = "yolo11n.pt"
    tracker_name: TrackerName = "botsort"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7
    vehicle_classes_only: bool = True


@dataclass
class TrackedDetection:
    track_id: int | None
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    center: tuple[float, float]
    frame_index: int
    timestamp: float
    tracker_name: TrackerName


class VideoDetector:
    """Reads frames and runs Ultralytics YOLO track mode."""

    def __init__(self, config: InputConfig) -> None:
        self.config = config
        self.model: Any | None = None
        self.capture: Any | None = None
        self.screen_capture: Any | None = None
        self.frame_index = 0
        self._cv2 = None

    def load_model(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self.config.model_path)

    def open_source(self) -> None:
        if self.config.source_type in {"webcam", "video_file"}:
            import cv2

            self._cv2 = cv2
            source: int | str
            if self.config.source_type == "webcam":
                source = self.config.camera_index
            else:
                if not self.config.video_path:
                    raise ValueError("video_path is required for video_file input")
                source = self.config.video_path

            self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened():
                raise RuntimeError(f"Unable to open input source: {source}")

        elif self.config.source_type == "screen_region":
            if self.config.screen_region is None:
                raise ValueError("screen_region is required for screen_region input")
            import mss

            self.screen_capture = mss.mss()

        else:
            raise ValueError(f"Unsupported source_type: {self.config.source_type}")

    def read_frame(self) -> Any | None:
        if self.config.source_type in {"webcam", "video_file"}:
            if self.capture is None:
                raise RuntimeError("Input source is not open")
            ok, frame = self.capture.read()
            if not ok:
                return None
            self.frame_index += 1
            return frame

        if self.config.source_type == "screen_region":
            if self.screen_capture is None:
                raise RuntimeError("Screen capture source is not open")
            import cv2
            import numpy as np

            x, y, width, height = self.config.screen_region or (0, 0, 0, 0)
            image = self.screen_capture.grab(
                {"left": x, "top": y, "width": width, "height": height}
            )
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            self.frame_index += 1
            return frame

        return None

    def track_frame(self, frame: Any) -> list[TrackedDetection]:
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded")

        tracker_config = TRACKER_CONFIGS[self.config.tracker_name]
        results = self.model.track(
            frame,
            persist=True,
            tracker=tracker_config,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )
        return self._parse_results(results)

    def read_and_track(self) -> tuple[Any | None, list[TrackedDetection]]:
        frame = self.read_frame()
        if frame is None:
            return None, []
        detections = self.track_frame(frame)
        return frame, detections

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        if self.screen_capture is not None:
            self.screen_capture.close()

    def _parse_results(self, results: Iterable[Any]) -> list[TrackedDetection]:
        parsed: list[TrackedDetection] = []
        timestamp = time()

        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy = self._to_list(getattr(boxes, "xyxy", []))
            cls_values = self._to_list(getattr(boxes, "cls", []))
            conf_values = self._to_list(getattr(boxes, "conf", []))
            id_values = self._to_list(getattr(boxes, "id", []))

            for index, bbox_values in enumerate(xyxy):
                class_id = int(cls_values[index]) if index < len(cls_values) else -1
                class_name = str(names.get(class_id, class_id))
                confidence = float(conf_values[index]) if index < len(conf_values) else 0.0

                if self.config.vehicle_classes_only and class_name not in VEHICLE_CLASS_NAMES:
                    continue
                if confidence < self.config.confidence_threshold:
                    continue

                x1, y1, x2, y2 = [float(value) for value in bbox_values]
                track_id = int(id_values[index]) if index < len(id_values) else None
                parsed.append(
                    TrackedDetection(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        frame_index=self.frame_index,
                        timestamp=timestamp,
                        tracker_name=self.config.tracker_name,
                    )
                )

        return parsed

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)
