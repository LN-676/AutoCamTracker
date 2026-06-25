from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import tempfile
from time import monotonic, sleep
import unittest


V1_DIR = Path(__file__).resolve().parent
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from autocamtracker.core.desktop_state import IdentitySessionLinks
from autocamtracker.tracking.feature_gallery import FeatureGallery
from autocamtracker.tracking.identity_manager import GlobalIdentityManager
from autocamtracker.core.pipeline_worker import DetectionWorker
from autocamtracker.tracking.vehicle_identity_store import VehicleIdentityStore
from autocamtracker.vision.detector import TrackedDetection


def detection(track_id: int = 12, frame_index: int = 1) -> TrackedDetection:
    return TrackedDetection(
        track_id=track_id,
        bbox=(10.0, 20.0, 90.0, 80.0),
        class_id=2,
        class_name="car",
        confidence=0.88,
        center=(50.0, 50.0),
        frame_index=frame_index,
        timestamp=float(frame_index),
        tracker_name="botsort",
    )


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def read_and_track(self):
        sleep(0.02)
        self.calls += 1
        return f"frame-{self.calls}", [self.calls]


class DetectionWorkerTests(unittest.TestCase):
    def test_requested_detection_returns_from_background_worker(self) -> None:
        detector = FakeDetector()
        worker = DetectionWorker(detector)
        try:
            self.assertTrue(worker.request_frame())
            self.assertFalse(worker.request_frame())
            deadline = monotonic() + 1.0
            result = None
            while result is None and monotonic() < deadline:
                result = worker.poll()
                sleep(0.005)
            self.assertIsNotNone(result)
            self.assertIsNone(result.error)
            self.assertEqual(result.frame, "frame-1")
            self.assertEqual(result.detections, [1])
            self.assertGreaterEqual(result.inference_time_ms, 0.0)
        finally:
            worker.close()


class IdentitySessionLinksTests(unittest.TestCase):
    def test_links_can_be_replaced_removed_and_cleared(self) -> None:
        links = IdentitySessionLinks()
        links.link(7, 101)
        self.assertEqual(links.vehicle_for_track(7), 101)
        links.link(7, 202)
        self.assertEqual(links.vehicle_for_track(7), 202)
        links.link(8, 202)
        links.unlink_vehicle(202)
        self.assertIsNone(links.vehicle_for_track(7))
        self.assertIsNone(links.vehicle_for_track(8))
        links.link(9, 303)
        links.clear()
        self.assertIsNone(links.vehicle_for_track(9))


class VehicleIdentityStoreBatchingTests(unittest.TestCase):
    def test_frame_updates_are_flushed_in_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "identity.sqlite3"
            store = VehicleIdentityStore(db_path, commit_interval_seconds=60.0)
            vehicle_id = store.create_vehicle(detection(frame_index=1))
            store.update_vehicle(vehicle_id, detection(frame_index=2))

            observer = sqlite3.connect(db_path)
            try:
                before_flush = observer.execute(
                    "SELECT last_frame_index FROM vehicles WHERE id = ?", (vehicle_id,)
                ).fetchone()[0]
                self.assertEqual(before_flush, 1)

                store.flush()
                after_flush = observer.execute(
                    "SELECT last_frame_index FROM vehicles WHERE id = ?", (vehicle_id,)
                ).fetchone()[0]
                self.assertEqual(after_flush, 2)
            finally:
                observer.close()
                store.close()


class ReIDRuntimeOptimizationTests(unittest.TestCase):
    def test_embedding_is_reused_for_stable_local_track(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((180, 260, 3), dtype=np.uint8)
        rng = np.random.default_rng(17)
        frame[30:130, 50:170] = rng.integers(35, 225, size=(100, 120, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 30), (170, 130), (255, 255, 255), 2)

        class CountingExtractor:
            def __init__(self) -> None:
                self.calls = 0

            def extract(self, _frame, bbox):
                self.calls += 1
                vector = np.asarray([bbox[0] + bbox[2], bbox[1] + bbox[3], 1.0], dtype=np.float32)
                vector /= np.linalg.norm(vector)
                return vector.tolist()

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "identity.sqlite3"
            store = VehicleIdentityStore(db_path)
            first = detection(track_id=7, frame_index=1)
            first.bbox = (50.0, 30.0, 170.0, 130.0)
            first.center = (110.0, 80.0)
            vehicle_id = store.create_vehicle(first)
            gallery = FeatureGallery(db_path)
            extractor = CountingExtractor()
            gallery.embedding_extractor = extractor
            self.assertTrue(gallery.add_master_feature(vehicle_id, first, frame).accepted)

            for frame_index in (2, 3, 4):
                candidate = detection(track_id=11, frame_index=frame_index)
                candidate.bbox = (51.0, 30.0, 171.0, 130.0)
                candidate.center = (111.0, 80.0)
                self.assertTrue(gallery.rank_detections_for_vehicle(vehicle_id, [candidate], frame))

            self.assertEqual(extractor.calls, 2)
            gallery.close()
            store.close()

    def test_reid_search_prioritizes_predicted_track_corridor(self) -> None:
        import numpy as np

        class RecordingGallery:
            def __init__(self) -> None:
                self.seen_track_ids: list[int | None] = []

            def has_master_features(self, _vehicle_id: int) -> bool:
                return True

            def rank_detections_for_vehicle(self, _vehicle_id, detections, _frame):
                self.seen_track_ids = [item.track_id for item in detections]
                return []

        frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        gallery = RecordingGallery()
        manager = GlobalIdentityManager(feature_gallery=gallery)  # type: ignore[arg-type]
        initial = detection(track_id=1, frame_index=1)
        initial.center = (100.0, 180.0)
        initial.bbox = (60.0, 140.0, 140.0, 220.0)
        manager.select_detection(initial, frame, persist=True)
        moved = detection(track_id=1, frame_index=2)
        moved.center = (120.0, 180.0)
        moved.bbox = (80.0, 140.0, 160.0, 220.0)
        manager.update([moved], frame)

        nearby = detection(track_id=2, frame_index=3)
        nearby.center = (145.0, 180.0)
        far = detection(track_id=3, frame_index=3)
        far.center = (560.0, 50.0)
        manager.update([far, nearby], frame)

        self.assertEqual(gallery.seen_track_ids, [2])
        self.assertEqual(manager.selected_global_vehicle_id, 1)
        self.assertEqual(manager.selected_local_track_id, 1)


if __name__ == "__main__":
    unittest.main()
