import asyncio
import json
import socket
from types import SimpleNamespace
from time import monotonic, sleep
import unittest

try:
    from tracking_server import (
        TrackingServerConfig,
        TrackingWebSocketServer,
        frame_tracking_message,
        tracking_message,
    )
except ImportError:  # pragma: no cover
    from .tracking_server import (
        TrackingServerConfig,
        TrackingWebSocketServer,
        frame_tracking_message,
        tracking_message,
    )


class TrackingMessageTests(unittest.TestCase):
    def test_normalizes_pixel_error(self) -> None:
        frame_data = SimpleNamespace(
            selected_targets=[SimpleNamespace(confidence=0.91, status="tracking", lost_frame_count=0)],
            tracking_status="tracking",
            framing_status=SimpleNamespace(error_x=160.0, error_y=-90.0),
            selected_global_vehicle_id=12,
            selected_local_track_id=7,
        )

        message = frame_tracking_message(frame_data, (360, 640, 3), sequence=42)

        self.assertTrue(message["target_locked"])
        self.assertEqual(message["target_id"], 12)
        self.assertAlmostEqual(message["error_x"], 0.5)
        self.assertAlmostEqual(message["error_y"], -0.5)
        self.assertEqual(message["sequence"], 42)

    def test_lost_target_emits_stop(self) -> None:
        frame_data = SimpleNamespace(
            selected_targets=[],
            tracking_status="lost",
        )

        message = frame_tracking_message(frame_data, (360, 640, 3))

        self.assertFalse(message["target_locked"])
        self.assertEqual(message["error_x"], 0.0)
        self.assertEqual(message["error_y"], 0.0)

    def test_stale_selected_bbox_emits_stop(self) -> None:
        frame_data = SimpleNamespace(
            selected_targets=[SimpleNamespace(confidence=0.91, status="lost", lost_frame_count=1)],
            tracking_status="tracking",
        )

        message = frame_tracking_message(frame_data, (360, 640, 3), sequence=43)

        self.assertFalse(message["target_locked"])
        self.assertEqual(message["sequence"], 43)

    def test_wire_values_are_clamped(self) -> None:
        message = tracking_message(
            target_locked=True,
            error_x=8.0,
            error_y=-4.0,
            confidence=3.0,
        )

        self.assertEqual(message["error_x"], 1.0)
        self.assertEqual(message["error_y"], -1.0)
        self.assertEqual(message["confidence"], 1.0)
        self.assertEqual(message["source_version"], "1.6")

    def test_server_round_trip(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        server = TrackingWebSocketServer(TrackingServerConfig(host="127.0.0.1", port=port))
        server.start()
        deadline = monotonic() + 5.0
        while not server.is_running and monotonic() < deadline:
            sleep(0.01)
        self.assertTrue(server.is_running)
        try:
            initial, pulse, camera_frame = asyncio.run(self._receive_server_messages(server))
        finally:
            server.stop()

        self.assertFalse(initial["target_locked"])
        self.assertTrue(pulse["target_locked"])
        self.assertAlmostEqual(pulse["error_x"], 0.12)
        self.assertEqual(camera_frame.shape, (24, 32, 3))

    async def _receive_server_messages(self, server: TrackingWebSocketServer):
        from websockets.asyncio.client import connect
        import cv2
        import numpy as np

        async with connect(f"ws://127.0.0.1:{server.config.port}/ws/tracking") as websocket:
            initial = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2.0))
            server.publish_test_pulse()
            pulse = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2.0))
            ok, jpeg = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))
            self.assertTrue(ok)
            await websocket.send(jpeg.tobytes())
            camera_frame = None
            for _ in range(50):
                await asyncio.sleep(0.01)
                camera_frame = server.read_latest_frame()
                if camera_frame is not None:
                    break
            self.assertIsNotNone(camera_frame)
            return initial, pulse, camera_frame


if __name__ == "__main__":
    unittest.main()
