"""WebSocket bridge from AutoCamTracker V1.6 to the DockKit iOS app."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import ipaddress
import re
import socket
import subprocess
import threading
from time import monotonic, time
from typing import Any, Callable


@dataclass(frozen=True)
class TrackingServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    path: str = "/ws/tracking"
    publish_hz: float = 20.0


def tracking_message(
    *,
    target_locked: bool,
    error_x: float = 0.0,
    error_y: float = 0.0,
    confidence: float = 0.0,
    target_id: int | None = None,
    sequence: int = 0,
) -> dict[str, Any]:
    """Build the versioned wire message consumed by TrackingCommand.swift."""

    return {
        "type": "tracking",
        "version": "1.0",
        "source_version": "1.6",
        "sequence": sequence,
        "target_locked": bool(target_locked),
        "target_id": target_id,
        "error_x": max(-1.0, min(1.0, float(error_x))),
        "error_y": max(-1.0, min(1.0, float(error_y))),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "timestamp_ms": int(time() * 1000),
    }


def frame_tracking_message(frame_data, frame_shape, sequence: int = 0) -> dict[str, Any]:
    """Convert V1.6 pixel-space framing status into normalized gimbal error."""

    frame_h, frame_w = frame_shape[:2]
    targets = frame_data.selected_targets
    fresh_target = next(
        (
            target
            for target in targets
            if target.status == "tracking" and target.lost_frame_count == 0
        ),
        None,
    )
    if fresh_target is None or frame_data.tracking_status != "tracking":
        return tracking_message(target_locked=False, sequence=sequence)

    status = frame_data.framing_status
    target_id = frame_data.selected_global_vehicle_id
    if target_id is None:
        target_id = frame_data.selected_local_track_id
    return tracking_message(
        target_locked=True,
        target_id=target_id,
        error_x=status.error_x / max(1.0, frame_w / 2.0),
        error_y=status.error_y / max(1.0, frame_h / 2.0),
        confidence=fresh_target.confidence,
        sequence=sequence,
    )


class TrackingWebSocketServer:
    """Runs a small asyncio WebSocket server without blocking Tkinter."""

    def __init__(
        self,
        config: TrackingServerConfig | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config or TrackingServerConfig()
        self.on_status = on_status
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._clients: set[Any] = set()
        self._frame_lock = threading.Lock()
        self._latest_frame_bytes: bytes | None = None
        self._received_frame_count = 0
        self._sequence = 0
        self._last_publish_at = 0.0
        self._running = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def local_urls(self) -> list[str]:
        interface_addresses = self._active_interface_addresses()
        addresses: set[str] = {address for _, address in interface_addresses}
        hostname = socket.gethostname()
        local_name = hostname if hostname.endswith(".local") else f"{hostname}.local"
        try:
            addresses.update(socket.gethostbyname_ex(hostname)[2])
        except OSError:
            pass
        usable = [address for address in addresses if ":" not in address and not address.startswith("127.")]
        link_local = sorted(address for address in usable if ipaddress.ip_address(address).is_link_local)
        private = sorted(
            address
            for address in usable
            if ipaddress.ip_address(address).is_private and address not in link_local
        )
        other = sorted(address for address in usable if address not in link_local and address not in private)
        # Prefer the normal LAN address. A 169.254 link-local address may be
        # present whenever an iPhone is attached by USB, but it is not always
        # routable from the app and previously became the misleading default.
        urls = [
            f"ws://{address}:{self.config.port}{self.config.path}"
            for address in (*private, *link_local, *other)
        ]
        urls.append(f"ws://{local_name}:{self.config.port}{self.config.path}")
        return urls

    @property
    def preferred_url(self) -> str:
        return self.local_urls[0]

    @staticmethod
    def _active_interface_addresses() -> list[tuple[str, str]]:
        """Return active macOS IPv4 interfaces, including USB link-local addresses."""

        try:
            result = subprocess.run(
                ["ifconfig"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return []

        interfaces: dict[str, dict[str, Any]] = {}
        current: str | None = None
        for line in result.stdout.splitlines():
            match = re.match(r"^([a-zA-Z0-9]+):", line)
            if match:
                current = match.group(1)
                interfaces[current] = {"addresses": [], "active": False}
                continue
            if current is None:
                continue
            address_match = re.match(r"\s+inet (\d+\.\d+\.\d+\.\d+)\b", line)
            if address_match:
                interfaces[current]["addresses"].append(address_match.group(1))
            if line.strip() == "status: active":
                interfaces[current]["active"] = True

        return [
            (name, address)
            for name, state in interfaces.items()
            if state["active"]
            for address in state["addresses"]
            if not address.startswith("127.")
        ]

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._thread_main, name="tracking-websocket", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        loop = self._loop
        stop_event = self._stop_event
        if loop is not None and stop_event is not None and loop.is_running():
            loop.call_soon_threadsafe(stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._running.clear()

    def publish_frame(self, frame_data, frame_shape) -> None:
        interval = 1.0 / max(1.0, self.config.publish_hz)
        now = monotonic()
        if now - self._last_publish_at < interval:
            return
        self._last_publish_at = now
        self._sequence += 1
        self.publish(frame_tracking_message(frame_data, frame_shape, self._sequence))

    def publish_test_pulse(self, error_x: float = 0.12) -> None:
        self._sequence += 1
        self.publish(
            tracking_message(
                target_locked=True,
                target_id=999,
                error_x=error_x,
                confidence=1.0,
                sequence=self._sequence,
            )
        )

    def publish_stop(self) -> None:
        self._sequence += 1
        self.publish(tracking_message(target_locked=False, sequence=self._sequence))

    def read_latest_frame(self):
        """Decode and consume only the newest iPhone JPEG frame."""

        with self._frame_lock:
            data = self._latest_frame_bytes
            self._latest_frame_bytes = None
        if data is None:
            return None

        import cv2
        import numpy as np

        encoded = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def publish(self, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as exc:  # pragma: no cover - surfaced in the desktop UI
            self._notify(f"iPhone server failed: {exc}")
        finally:
            self._running.clear()
            self._loop = None
            self._stop_event = None

    async def _serve(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:
            raise RuntimeError("Install the 'websockets' dependency first") from exc

        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        async with serve(self._handle_client, self.config.host, self.config.port):
            self._running.set()
            self._notify("Waiting for iPhone")
            await self._stop_event.wait()

        self._clients.clear()

    async def _handle_client(self, websocket) -> None:
        from websockets.exceptions import ConnectionClosed

        request_path = getattr(getattr(websocket, "request", None), "path", "")
        if request_path.split("?", 1)[0] != self.config.path:
            await websocket.close(code=1008, reason="Unsupported path")
            return

        self._clients.add(websocket)
        self._notify(f"iPhone connected ({len(self._clients)})")
        await websocket.send(json.dumps(tracking_message(target_locked=False, sequence=self._sequence)))
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    self._accept_camera_frame(message)
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            self._notify("iPhone disconnected" if not self._clients else f"iPhone connected ({len(self._clients)})")

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return
        message = json.dumps(payload, separators=(",", ":"))
        clients = list(self._clients)
        results = await asyncio.gather(*(client.send(message) for client in clients), return_exceptions=True)
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                self._clients.discard(client)

    def _accept_camera_frame(self, data: bytes) -> None:
        if len(data) < 4 or len(data) > 2_000_000 or not data.startswith(b"\xff\xd8"):
            return
        with self._frame_lock:
            self._latest_frame_bytes = data
            self._received_frame_count += 1
            first_frame = self._received_frame_count == 1
        if first_frame:
            self._notify("iPhone video receiving")

    def _notify(self, message: str) -> None:
        if self.on_status is not None:
            self.on_status(message)
