"""Manual vehicle ReID feature gallery for AutoCamTracker V1.21."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from time import time
from typing import Any, Literal, Protocol

try:
    from reid_embedding import ReIDEmbeddingExtractor
    from video_detector import TrackedDetection
except ImportError:  # pragma: no cover
    from .reid_embedding import ReIDEmbeddingExtractor
    from .video_detector import TrackedDetection


GalleryType = Literal["master", "pending", "candidate"]


@dataclass
class CropQuality:
    accepted: bool
    score: float
    reason: str
    width: int
    height: int
    sharpness: float
    brightness: float


@dataclass
class FeatureAddResult:
    accepted: bool
    vehicle_id: int
    gallery_type: GalleryType
    feature_id: int | None
    quality: CropQuality
    duplicate_score: float | None = None
    reason: str = ""


@dataclass
class FeatureMatch:
    feature_id: int
    vehicle_id: int
    gallery_type: GalleryType
    score: float
    quality_score: float
    frame_index: int


@dataclass
class DetectionFeatureMatch:
    detection: TrackedDetection
    score: float
    matches: list[FeatureMatch]


class FeatureIndexBackend(Protocol):
    """Reserved vector-index interface for future FAISS, Qdrant, or Milvus use."""

    name: str

    def top_k(
        self,
        query_embedding: list[float],
        gallery_type: GalleryType,
        top_k: int,
        vehicle_id: int | None = None,
    ) -> list[FeatureMatch]:
        ...


class FaissFeatureIndex:
    name = "faiss"

    def top_k(
        self,
        query_embedding: list[float],
        gallery_type: GalleryType,
        top_k: int,
        vehicle_id: int | None = None,
    ) -> list[FeatureMatch]:
        raise NotImplementedError("FAISS backend is reserved for a future release")


class QdrantFeatureIndex:
    name = "qdrant"

    def top_k(
        self,
        query_embedding: list[float],
        gallery_type: GalleryType,
        top_k: int,
        vehicle_id: int | None = None,
    ) -> list[FeatureMatch]:
        raise NotImplementedError("Qdrant backend is reserved for a future release")


class MilvusFeatureIndex:
    name = "milvus"

    def top_k(
        self,
        query_embedding: list[float],
        gallery_type: GalleryType,
        top_k: int,
        vehicle_id: int | None = None,
    ) -> list[FeatureMatch]:
        raise NotImplementedError("Milvus backend is reserved for a future release")


class FeatureGallery:
    """SQLite default gallery. Master features are written only by Add Feature."""

    MASTER_FEATURE_LIMIT = 30

    def __init__(
        self,
        db_path: Path | str,
        duplicate_threshold: float = 0.985,
        min_match_score: float = 0.72,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.duplicate_threshold = duplicate_threshold
        self.min_match_score = min_match_score
        self.embedding_extractor: ReIDEmbeddingExtractor | None = None
        self._ensure_schema()

    def close(self) -> None:
        self.connection.close()

    def add_master_feature(
        self,
        vehicle_id: int,
        detection: TrackedDetection,
        frame,
    ) -> FeatureAddResult:
        quality = self.assess_crop_quality(frame, detection.bbox)
        if not quality.accepted:
            return FeatureAddResult(
                accepted=False,
                vehicle_id=vehicle_id,
                gallery_type="master",
                feature_id=None,
                quality=quality,
                reason=quality.reason,
            )

        embedding = self._extract_embedding(frame, detection)
        if embedding is None:
            return FeatureAddResult(
                accepted=False,
                vehicle_id=vehicle_id,
                gallery_type="master",
                feature_id=None,
                quality=quality,
                reason="ReID model is unavailable or failed to extract a feature",
            )

        duplicate_score = self._max_similarity_for_vehicle(vehicle_id, embedding, "master")
        if duplicate_score is not None and duplicate_score >= self.duplicate_threshold:
            return FeatureAddResult(
                accepted=False,
                vehicle_id=vehicle_id,
                gallery_type="master",
                feature_id=None,
                quality=quality,
                duplicate_score=duplicate_score,
                reason=f"duplicate master feature ({duplicate_score:.3f})",
            )

        feature_id = self._insert_feature(
            vehicle_id=vehicle_id,
            gallery_type="master",
            detection=detection,
            frame=frame,
            quality=quality,
            embedding=embedding,
            duplicate_score=duplicate_score,
        )
        self._prune_master_features(vehicle_id)
        return FeatureAddResult(
            accepted=True,
            vehicle_id=vehicle_id,
            gallery_type="master",
            feature_id=feature_id,
            quality=quality,
            duplicate_score=duplicate_score,
            reason="added to master gallery",
        )

    def add_pending_feature(
        self,
        vehicle_id: int,
        detection: TrackedDetection,
        frame,
    ) -> FeatureAddResult:
        return self._add_non_master_feature(vehicle_id, detection, frame, "pending")

    def add_candidate_feature(
        self,
        vehicle_id: int,
        detection: TrackedDetection,
        frame,
    ) -> FeatureAddResult:
        return self._add_non_master_feature(vehicle_id, detection, frame, "candidate")

    def match_top_k(
        self,
        query_embedding: list[float],
        gallery_type: GalleryType = "master",
        top_k: int = 5,
        vehicle_id: int | None = None,
    ) -> list[FeatureMatch]:
        rows = self.connection.execute(
            """
            SELECT id, vehicle_id, gallery_type, embedding_json, quality_score, frame_index
            FROM vehicle_features
            WHERE gallery_type = ?
              AND embedding_json IS NOT NULL
              AND (? IS NULL OR vehicle_id = ?)
            """,
            (gallery_type, vehicle_id, vehicle_id),
        ).fetchall()
        matches: list[FeatureMatch] = []
        for row in rows:
            embedding = self._vector_from_json(row["embedding_json"])
            score = self.cosine_similarity(query_embedding, embedding)
            if score <= 0.0:
                continue
            matches.append(
                FeatureMatch(
                    feature_id=int(row["id"]),
                    vehicle_id=int(row["vehicle_id"]),
                    gallery_type=str(row["gallery_type"]),  # type: ignore[arg-type]
                    score=score,
                    quality_score=float(row["quality_score"]),
                    frame_index=int(row["frame_index"]),
                )
            )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[: max(1, top_k)]

    def rank_detections_for_vehicle(
        self,
        vehicle_id: int,
        detections: list[TrackedDetection],
        frame,
        top_k: int = 5,
    ) -> list[DetectionFeatureMatch]:
        if not self.has_master_features(vehicle_id):
            return []

        ranked: list[DetectionFeatureMatch] = []
        for detection in detections:
            quality = self.assess_crop_quality(frame, detection.bbox)
            if not quality.accepted:
                continue
            embedding = self._extract_embedding(frame, detection)
            if embedding is None:
                continue
            matches = self.match_top_k(
                embedding,
                gallery_type="master",
                top_k=top_k,
                vehicle_id=vehicle_id,
            )
            if not matches:
                continue
            best_score = matches[0].score
            ranked.append(DetectionFeatureMatch(detection=detection, score=best_score, matches=matches))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def has_master_features(self, vehicle_id: int) -> bool:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS feature_count
            FROM vehicle_features
            WHERE vehicle_id = ? AND gallery_type = 'master'
            """,
            (vehicle_id,),
        ).fetchone()
        return bool(row and int(row["feature_count"]) > 0)

    def summary_by_vehicle(self) -> dict[int, dict[str, int]]:
        rows = self.connection.execute(
            """
            SELECT vehicle_id, gallery_type, COUNT(*) AS feature_count
            FROM vehicle_features
            GROUP BY vehicle_id, gallery_type
            """
        ).fetchall()
        summary: dict[int, dict[str, int]] = {}
        for row in rows:
            counts = summary.setdefault(int(row["vehicle_id"]), {})
            counts[str(row["gallery_type"])] = int(row["feature_count"])
        return summary

    def delete_vehicle_features(self, vehicle_id: int) -> int:
        cursor = self.connection.execute("DELETE FROM vehicle_features WHERE vehicle_id = ?", (vehicle_id,))
        self.connection.commit()
        return int(cursor.rowcount or 0)

    def assess_crop_quality(self, frame, bbox: tuple[float, float, float, float]) -> CropQuality:
        import cv2

        crop = self._crop(frame, bbox)
        if crop is None:
            return CropQuality(False, 0.0, "bbox is outside the frame", 0, 0, 0.0, 0.0)

        height, width = crop.shape[:2]
        area = width * height
        if width < 32 or height < 32 or area < 1600:
            return CropQuality(False, 0.0, "crop is too small", width, height, 0.0, 0.0)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        if brightness < 20.0 or brightness > 240.0:
            return CropQuality(False, 0.0, "crop brightness is outside usable range", width, height, sharpness, brightness)
        if sharpness < 5.0:
            return CropQuality(False, 0.0, "crop is too blurry", width, height, sharpness, brightness)

        area_score = min(1.0, area / 12000.0)
        sharpness_score = min(1.0, sharpness / 120.0)
        brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        score = 0.45 * area_score + 0.40 * sharpness_score + 0.15 * brightness_score
        return CropQuality(True, float(score), "ok", width, height, sharpness, brightness)

    def _add_non_master_feature(
        self,
        vehicle_id: int,
        detection: TrackedDetection,
        frame,
        gallery_type: Literal["pending", "candidate"],
    ) -> FeatureAddResult:
        quality = self.assess_crop_quality(frame, detection.bbox)
        if not quality.accepted:
            return FeatureAddResult(False, vehicle_id, gallery_type, None, quality, reason=quality.reason)
        embedding = self._extract_embedding(frame, detection)
        if embedding is None:
            return FeatureAddResult(
                False,
                vehicle_id,
                gallery_type,
                None,
                quality,
                reason="ReID model is unavailable or failed to extract a feature",
            )
        duplicate_score = self._max_similarity_for_vehicle(vehicle_id, embedding, gallery_type)
        feature_id = self._insert_feature(
            vehicle_id=vehicle_id,
            gallery_type=gallery_type,
            detection=detection,
            frame=frame,
            quality=quality,
            embedding=embedding,
            duplicate_score=duplicate_score,
        )
        return FeatureAddResult(True, vehicle_id, gallery_type, feature_id, quality, duplicate_score, "added")

    def _insert_feature(
        self,
        vehicle_id: int,
        gallery_type: GalleryType,
        detection: TrackedDetection,
        frame,
        quality: CropQuality,
        embedding: list[float],
        duplicate_score: float | None,
    ) -> int:
        crop_jpeg = self._encode_crop_jpeg(frame, detection.bbox)
        cursor = self.connection.execute(
            """
            INSERT INTO vehicle_features (
                vehicle_id,
                gallery_type,
                created_at,
                frame_index,
                track_id,
                bbox_json,
                quality_score,
                duplicate_score,
                embedding_json,
                crop_jpeg,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                vehicle_id,
                gallery_type,
                time(),
                detection.frame_index,
                detection.track_id,
                json.dumps(list(detection.bbox)),
                quality.score,
                duplicate_score,
                self._vector_json(embedding),
                crop_jpeg,
                json.dumps(
                    {
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "quality_reason": quality.reason,
                        "crop_width": quality.width,
                        "crop_height": quality.height,
                        "sharpness": quality.sharpness,
                        "brightness": quality.brightness,
                    },
                    sort_keys=True,
                ),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def _prune_master_features(self, vehicle_id: int) -> None:
        rows = self.connection.execute(
            """
            SELECT id
            FROM vehicle_features
            WHERE vehicle_id = ? AND gallery_type = 'master'
            ORDER BY quality_score ASC, created_at ASC
            """,
            (vehicle_id,),
        ).fetchall()
        overflow = len(rows) - self.MASTER_FEATURE_LIMIT
        if overflow <= 0:
            return
        ids = [int(row["id"]) for row in rows[:overflow]]
        placeholders = ",".join("?" for _ in ids)
        self.connection.execute(f"DELETE FROM vehicle_features WHERE id IN ({placeholders})", ids)
        self.connection.commit()

    def _max_similarity_for_vehicle(
        self,
        vehicle_id: int,
        embedding: list[float],
        gallery_type: GalleryType,
    ) -> float | None:
        matches = self.match_top_k(embedding, gallery_type=gallery_type, top_k=1, vehicle_id=vehicle_id)
        return matches[0].score if matches else None

    def _extract_embedding(self, frame, detection: TrackedDetection) -> list[float] | None:
        if self.embedding_extractor is None:
            self.embedding_extractor = ReIDEmbeddingExtractor()
        return self.embedding_extractor.extract(frame, detection.bbox)

    def _ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicle_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER NOT NULL,
                gallery_type TEXT NOT NULL CHECK(gallery_type IN ('master', 'pending', 'candidate')),
                created_at REAL NOT NULL,
                frame_index INTEGER NOT NULL,
                track_id INTEGER,
                bbox_json TEXT NOT NULL,
                quality_score REAL NOT NULL,
                duplicate_score REAL,
                embedding_json TEXT NOT NULL,
                crop_jpeg BLOB,
                metadata_json TEXT
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_vehicle_features_vehicle_gallery ON vehicle_features(vehicle_id, gallery_type)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_vehicle_features_gallery ON vehicle_features(gallery_type)"
        )
        self.connection.commit()

    @staticmethod
    def _crop(frame, bbox: tuple[float, float, float, float]):
        if frame is None:
            return None
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        left = max(0, min(frame_w - 1, int(round(x1))))
        top = max(0, min(frame_h - 1, int(round(y1))))
        right = max(left + 1, min(frame_w, int(round(x2))))
        bottom = max(top + 1, min(frame_h, int(round(y2))))
        if right <= left or bottom <= top:
            return None
        return frame[top:bottom, left:right]

    @classmethod
    def _encode_crop_jpeg(cls, frame, bbox: tuple[float, float, float, float]) -> bytes | None:
        import cv2

        crop = cls._crop(frame, bbox)
        if crop is None:
            return None
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return None
        return bytes(encoded)

    @staticmethod
    def _vector_json(vector: list[float]) -> str:
        return json.dumps([float(value) for value in vector])

    @staticmethod
    def _vector_from_json(value: str | None) -> list[float] | None:
        if not value:
            return None
        try:
            return [float(item) for item in json.loads(value)]
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def cosine_similarity(first: Any | None, second: Any | None) -> float:
        if first is None or second is None:
            return 0.0
        if hasattr(first, "tolist"):
            first = first.tolist()
        if hasattr(second, "tolist"):
            second = second.tolist()
        first_values = [float(value) for value in first]
        second_values = [float(value) for value in second]
        if len(first_values) != len(second_values) or not first_values:
            return 0.0
        numerator = sum(a * b for a, b in zip(first_values, second_values))
        first_norm = sum(a * a for a in first_values) ** 0.5
        second_norm = sum(b * b for b in second_values) ** 0.5
        if first_norm <= 1e-12 or second_norm <= 1e-12:
            return 0.0
        return max(0.0, min(1.0, numerator / (first_norm * second_norm)))
