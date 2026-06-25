"""Background detector worker used by the Tk desktop application."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import time
from typing import Any, Callable, TypeVar


T = TypeVar("T")


@dataclass
class DetectionWorkerResult:
    frame: Any | None
    detections: list[Any]
    inference_time_ms: float
    error: Exception | None = None


class DetectionWorker:
    """Runs one requested detector step at a time away from Tk's main thread."""

    def __init__(self, detector) -> None:
        self.detector = detector
        self._request_event = Event()
        self._stop_event = Event()
        self._busy = Event()
        self._operation_lock = Lock()
        self._results: Queue[DetectionWorkerResult] = Queue(maxsize=1)
        self._thread = Thread(target=self._run, name="autocam-detection", daemon=True)
        self._thread.start()

    @property
    def is_busy(self) -> bool:
        return self._busy.is_set() or self._request_event.is_set()

    def request_frame(self) -> bool:
        if self._stop_event.is_set() or self.is_busy:
            return False
        self._request_event.set()
        return True

    def poll(self) -> DetectionWorkerResult | None:
        try:
            return self._results.get_nowait()
        except Empty:
            return None

    def discard_results(self) -> None:
        while True:
            try:
                self._results.get_nowait()
            except Empty:
                return

    def run_locked(self, callback: Callable[[], T]) -> T:
        """Serialize occasional seek/skip/reset operations with inference."""

        with self._operation_lock:
            return callback()

    def close(self) -> None:
        self._stop_event.set()
        self._request_event.set()
        self._thread.join(timeout=10.0)
        self.discard_results()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._request_event.wait(timeout=0.1):
                continue
            self._request_event.clear()
            if self._stop_event.is_set():
                return

            self._busy.set()
            started_at = time()
            try:
                with self._operation_lock:
                    frame, detections = self.detector.read_and_track()
                result = DetectionWorkerResult(
                    frame=frame,
                    detections=detections,
                    inference_time_ms=(time() - started_at) * 1000.0,
                )
            except Exception as exc:
                result = DetectionWorkerResult(
                    frame=None,
                    detections=[],
                    inference_time_ms=(time() - started_at) * 1000.0,
                    error=exc,
                )
            finally:
                self._busy.clear()
            self._put_latest(result)

    def _put_latest(self, result: DetectionWorkerResult) -> None:
        try:
            self._results.put_nowait(result)
            return
        except Full:
            pass
        try:
            self._results.get_nowait()
        except Empty:
            pass
        self._results.put_nowait(result)
