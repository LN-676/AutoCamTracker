"""Automatic Master feature sampling for selected vehicle GIDs."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from detection_store import DetectionStore
    from feature_gallery import FeatureAddResult, FeatureGallery
    from video_detector import TrackedDetection
except ImportError:  # pragma: no cover
    from .detection_store import DetectionStore
    from .feature_gallery import FeatureAddResult, FeatureGallery
    from .video_detector import TrackedDetection


@dataclass
class AutoFeatureSamplerConfig:
    min_quality_score: float = 0.65
    min_detection_confidence: float = 0.55
    min_area_ratio: float = 0.012
    edge_margin_ratio: float = 0.015
    min_interval_frames: int = 12
    min_track_age_frames: int = 3
    min_center_shift_ratio: float = 0.10
    min_area_change_ratio: float = 0.22
    min_brightness_delta: float = 18.0


@dataclass
class AutoFeatureSampleResult:
    attempted: bool
    accepted: bool
    vehicle_id: int | None
    reason: str
    feature_id: int | None = None
    quality_score: float = 0.0
    duplicate_score: float | None = None


@dataclass
class _AcceptedSignature:
    center: tuple[float, float]
    area: float
    brightness: float
    frame_index: int


@dataclass
class _VehicleSamplingState:
    last_attempt_frame: int = -10_000
    accepted: list[_AcceptedSignature] = field(default_factory=list)


class AutoFeatureSampler:
    """Captures diverse, high-quality Master features for one active GID."""

    def __init__(
        self,
        feature_gallery: FeatureGallery,
        config: AutoFeatureSamplerConfig | None = None,
    ) -> None:
        self.feature_gallery = feature_gallery
        self.config = config or AutoFeatureSamplerConfig()
        self.active_vehicle_id: int | None = None
        self._states: dict[int, _VehicleSamplingState] = {}

    def start(
        self,
        vehicle_id: int,
        detection: TrackedDetection,
        frame,
        store: DetectionStore,
    ) -> AutoFeatureSampleResult:
        self.active_vehicle_id = vehicle_id
        return self.sample(vehicle_id, detection, frame, store, force=True)

    def stop(self) -> None:
        self.active_vehicle_id = None

    def set_quality_threshold(self, threshold: float) -> None:
        self.config.min_quality_score = max(0.0, min(1.0, float(threshold)))

    def update(
        self,
        detection: TrackedDetection | None,
        frame,
        store: DetectionStore,
    ) -> AutoFeatureSampleResult:
        if self.active_vehicle_id is None or detection is None:
            return AutoFeatureSampleResult(False, False, self.active_vehicle_id, "no active visible GID")
        return self.sample(self.active_vehicle_id, detection, frame, store, force=False)

    def sample(
        self,
        vehicle_id: int,
        detection: TrackedDetection,
        frame,
        store: DetectionStore,
        force: bool = False,
    ) -> AutoFeatureSampleResult:
        state = self._states.setdefault(vehicle_id, _VehicleSamplingState())
        if not force and detection.frame_index - state.last_attempt_frame < self.config.min_interval_frames:
            return AutoFeatureSampleResult(False, False, vehicle_id, "waiting for sample interval")

        state.last_attempt_frame = detection.frame_index
        reason = self._gate_reason(detection, frame, store, force=force)
        if reason is not None:
            return AutoFeatureSampleResult(True, False, vehicle_id, reason)

        quality = self.feature_gallery.assess_crop_quality(frame, detection.bbox)
        if not quality.accepted:
            return AutoFeatureSampleResult(True, False, vehicle_id, quality.reason, quality_score=quality.score)
        if quality.score < self.config.min_quality_score:
            return AutoFeatureSampleResult(
                True,
                False,
                vehicle_id,
                f"quality {quality.score:.2f} below auto threshold {self.config.min_quality_score:.2f}",
                quality_score=quality.score,
            )

        signature = _AcceptedSignature(
            center=detection.center,
            area=self._area(detection.bbox),
            brightness=quality.brightness,
            frame_index=detection.frame_index,
        )
        if not force and not self._is_diverse_enough(state, signature, frame.shape):
            return AutoFeatureSampleResult(
                True,
                False,
                vehicle_id,
                "viewpoint/light change is not large enough",
                quality_score=quality.score,
            )

        result = self.feature_gallery.add_master_feature(vehicle_id, detection, frame)
        if result.accepted:
            state.accepted.append(signature)
            state.accepted = state.accepted[-80:]
        return self._from_feature_result(result)

    def _gate_reason(
        self,
        detection: TrackedDetection,
        frame,
        store: DetectionStore,
        force: bool,
    ) -> str | None:
        if detection.confidence < self.config.min_detection_confidence:
            return f"detection confidence {detection.confidence:.2f} is too low"

        frame_h, frame_w = frame.shape[:2]
        area_ratio = self._area(detection.bbox) / max(1.0, float(frame_w * frame_h))
        if area_ratio < self.config.min_area_ratio:
            return f"bbox area ratio {area_ratio:.3f} is too small"
        if self._near_frame_edge(detection.bbox, frame_w, frame_h):
            return "bbox is too close to frame edge"

        if force:
            return None

        if detection.track_id is None:
            return "detection has no local track id"
        track = store.get_track(detection.track_id)
        if track is None:
            return "track is not stable yet"
        age = max(1, detection.frame_index - track.first_seen_frame + 1)
        if age < self.config.min_track_age_frames:
            return f"track age {age} is below {self.config.min_track_age_frames}"
        return None

    def _is_diverse_enough(
        self,
        state: _VehicleSamplingState,
        signature: _AcceptedSignature,
        frame_shape,
    ) -> bool:
        if not state.accepted:
            return True

        frame_h, frame_w = frame_shape[:2]
        diagonal = max(1.0, float((frame_w**2 + frame_h**2) ** 0.5))
        for previous in reversed(state.accepted[-12:]):
            center_shift = (
                (signature.center[0] - previous.center[0]) ** 2
                + (signature.center[1] - previous.center[1]) ** 2
            ) ** 0.5 / diagonal
            area_change = abs(signature.area - previous.area) / max(signature.area, previous.area, 1.0)
            brightness_delta = abs(signature.brightness - previous.brightness)
            if (
                center_shift >= self.config.min_center_shift_ratio
                or area_change >= self.config.min_area_change_ratio
                or brightness_delta >= self.config.min_brightness_delta
            ):
                return True
        return False

    def _near_frame_edge(
        self,
        bbox: tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        x1, y1, x2, y2 = bbox
        margin = max(2.0, min(frame_w, frame_h) * self.config.edge_margin_ratio)
        return x1 <= margin or y1 <= margin or x2 >= frame_w - margin or y2 >= frame_h - margin

    @staticmethod
    def _area(bbox: tuple[float, float, float, float]) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    @staticmethod
    def _from_feature_result(result: FeatureAddResult) -> AutoFeatureSampleResult:
        return AutoFeatureSampleResult(
            attempted=True,
            accepted=result.accepted,
            vehicle_id=result.vehicle_id,
            reason=result.reason,
            feature_id=result.feature_id,
            quality_score=result.quality.score,
            duplicate_score=result.duplicate_score,
        )
