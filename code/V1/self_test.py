"""Self-test for the AutoCamTracker V1 runtime.

Run from VSCode with the "AutoCamTracker V1 Self Test" launch config, or:

    .venv/bin/python code/V1/self_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "code" / "model" / "yolo11n.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEST_VIDEO = OUTPUT_DIR / "self_test_input.mp4"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    results.append(run_check("dependencies", check_dependencies))
    results.append(run_check("model_load", check_model_load))
    results.append(run_check("video_input_pipeline", check_video_input_pipeline))
    results.append(run_check("webcam_probe", check_webcam_probe, warning_ok=True))

    print("\nAutoCamTracker V1 self-test summary")
    print("=" * 40)
    failed = False
    for name, status, detail in results:
        print(f"{status:>6}  {name}")
        if detail:
            print(f"        {detail}")
        if status == "FAIL":
            failed = True

    return 1 if failed else 0


def run_check(name, func, warning_ok: bool = False) -> tuple[str, str, str]:
    try:
        detail = func()
        return (name, "PASS", detail or "")
    except CameraPermissionBlocked as exc:
        return (name, "WARN" if warning_ok else "FAIL", str(exc))
    except Exception as exc:  # pragma: no cover - command-line diagnostic
        traceback.print_exc()
        return (name, "FAIL", str(exc))


def check_dependencies() -> str:
    import cv2
    import filterpy
    import mss
    import PIL
    import ultralytics

    return (
        f"python={sys.executable}; "
        f"ultralytics={ultralytics.__version__}; "
        f"cv2={cv2.__version__}; "
        f"filterpy={filterpy.__version__}; "
        f"mss={mss.__version__}; "
        f"Pillow={PIL.__version__}"
    )


def check_model_load() -> str:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "V1"))
    from video_detector import InputConfig, VideoDetector

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    detector = VideoDetector(
        InputConfig(
            model_path="yolo11n.pt",
            tracker_name="deepocsort",
            source_type="video_file",
        )
    )
    detector.load_model()
    return f"loaded {detector.config.model_path}"


def check_video_input_pipeline() -> str:
    import cv2
    import numpy as np

    sys.path.insert(0, str(PROJECT_ROOT / "code" / "V1"))
    from video_detector import InputConfig, VideoDetector

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(TEST_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        60.0,
        (320, 240),
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create self-test video file")

    for index in range(8):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = 30 + index * 8
        cv2.rectangle(frame, (x, 80), (x + 70, 150), (0, 255, 0), -1)
        cv2.putText(frame, "AutoCamTracker", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()

    detector = VideoDetector(
        InputConfig(
            source_type="video_file",
            video_path=str(TEST_VIDEO),
            model_path="yolo11n.pt",
            tracker_name="deepocsort",
        )
    )
    detector.load_model()
    detector.open_source()
    source_fps = detector.get_source_fps()
    if source_fps is None or abs(source_fps - 60.0) > 1.0:
        detector.close()
        raise RuntimeError(f"Expected a 60fps source, got {source_fps}")
    frame, detections = detector.read_and_track()
    skipped = detector.skip_video_frames(2)
    detector.close()
    if frame is None:
        raise RuntimeError("Video input opened but returned no frame")
    return f"source_fps={source_fps:.1f}; read frame {frame.shape}; detections={len(detections)}; skipped={skipped}"


def check_webcam_probe() -> str:
    import cv2

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    capture = cv2.VideoCapture(0, backend)
    opened = capture.isOpened()
    ok, frame = capture.read() if opened else (False, None)
    capture.release()

    if opened and ok and frame is not None:
        return f"camera index 0 ok; frame={frame.shape}"

    if sys.platform == "darwin":
        raise CameraPermissionBlocked(
            "Camera is not available to this Python process. "
            "Enable Camera permission for Visual Studio Code or Terminal in "
            "System Settings > Privacy & Security > Camera, then restart VSCode."
        )
    raise RuntimeError("Camera index 0 did not open")


class CameraPermissionBlocked(RuntimeError):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
