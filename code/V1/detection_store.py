"""YOLO Data module for AutoCamTracker V1.

Responsibilities:
- Store current tracked detections.
- Maintain vehicle tracks keyed by track_id.
- Keep detection history for debug and simple reacquire logic.
- Rank vehicle candidates for auto-select.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Literal

try:
    from video_detector import TrackedDetection
except ImportError:  # pragma: no cover
    from .video_detector import TrackedDetection


RankStrategy = Literal["largest", "center", "confidence"]


@dataclass
class VehicleCandidate:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float
    center: tuple[float, float]
    area: float
    distance_to_frame_center: float
    frame_index: int


@dataclass
class VehicleTrack:
    track_id: int
    class_name: str
    first_seen_frame: int
    last_seen_frame: int
    latest_bbox: tuple[float, float, float, float]
    latest_confidence: float
    latest_center: tuple[float, float]
    center_history: list[tuple[float, float]] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)
    lost_frame_count: int = 0
    tracker_name: str = "botsort"

    def update(self, detection: TrackedDetection) -> None:
        self.last_seen_frame = detection.frame_index
        self.latest_bbox = detection.bbox
        self.latest_confidence = detection.confidence
        self.latest_center = detection.center
        self.center_history.append(detection.center)
        self.confidence_history.append(detection.confidence)
        self.lost_frame_count = 0
        self.tracker_name = detection.tracker_name


class DetectionStore:
    """Stores YOLO tracking output and exposes ranked vehicle candidates."""

    def __init__(self, history_size: int = 90) -> None:
        self.history_size = history_size
        self.current_detections: list[TrackedDetection] = []
        self.vehicle_tracks: dict[int, VehicleTrack] = {}
        self.detection_history: deque[list[TrackedDetection]] = deque(maxlen=history_size)
        self.current_frame_index = 0

    def update(
        self,
        detections: list[TrackedDetection],
        frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    ) -> list[VehicleCandidate]:
        self.current_detections = [d for d in detections if d.track_id is not None]
        self.detection_history.append(self.current_detections)

        if detections:
            self.current_frame_index = max(d.frame_index for d in detections)
        else:
            self.current_frame_index += 1

        seen_track_ids: set[int] = set()
        for detection in self.current_detections:
            assert detection.track_id is not None
            seen_track_ids.add(detection.track_id)
            if detection.track_id not in self.vehicle_tracks:
                self.vehicle_tracks[detection.track_id] = VehicleTrack(
                    track_id=detection.track_id,
                    class_name=detection.class_name,
                    first_seen_frame=detection.frame_index,
                    last_seen_frame=detection.frame_index,
                    latest_bbox=detection.bbox,
                    latest_confidence=detection.confidence,
                    latest_center=detection.center,
                    center_history=[detection.center],
                    confidence_history=[detection.confidence],
                    tracker_name=detection.tracker_name,
                )
            else:
                self.vehicle_tracks[detection.track_id].update(detection)

        for track_id, track in self.vehicle_tracks.items():
            if track_id not in seen_track_ids:
                track.lost_frame_count += 1

        return self.get_candidates(frame_shape)

    def get_candidates(
        self,
        frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    ) -> list[VehicleCandidate]:
        frame_center = self._frame_center(frame_shape)
        candidates: list[VehicleCandidate] = []

        for detection in self.current_detections:
            assert detection.track_id is not None
            x1, y1, x2, y2 = detection.bbox
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            distance = hypot(
                detection.center[0] - frame_center[0],
                detection.center[1] - frame_center[1],
            )
            candidates.append(
                VehicleCandidate(
                    track_id=detection.track_id,
                    bbox=detection.bbox,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    center=detection.center,
                    area=area,
                    distance_to_frame_center=distance,
                    frame_index=detection.frame_index,
                )
            )
        return candidates

    def rank_candidates(
        self,
        frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
        strategy: RankStrategy = "largest",
    ) -> list[VehicleCandidate]:
        candidates = self.get_candidates(frame_shape)
        if strategy == "largest":
            return sorted(candidates, key=lambda item: item.area, reverse=True)
        if strategy == "center":
            return sorted(candidates, key=lambda item: item.distance_to_frame_center)
        if strategy == "confidence":
            return sorted(candidates, key=lambda item: item.confidence, reverse=True)
        raise ValueError(f"Unsupported rank strategy: {strategy}")

    def get_track(self, track_id: int) -> VehicleTrack | None:
        return self.vehicle_tracks.get(track_id)

    @staticmethod
    def _frame_center(
        frame_shape: tuple[int, int, int] | tuple[int, int] | None,
    ) -> tuple[float, float]:
        if frame_shape is None:
            return (0.0, 0.0)
        height, width = frame_shape[:2]
        return (width / 2.0, height / 2.0)
