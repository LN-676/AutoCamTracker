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

from desktop_state import IdentitySessionLinks
from pipeline_worker import DetectionWorker
from vehicle_identity_store import VehicleIdentityStore
from video_detector import TrackedDetection


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


if __name__ == "__main__":
    unittest.main()
