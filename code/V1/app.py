"""Tkinter UI + Recording + Debug Log module for AutoCamTracker V1.

Responsibilities:
- Create the Tkinter desktop UI.
- Wire together input, YOLO tracking, data store, target tracking, and reframe.
- Show before and after views.
- Expose controls for source, tracker, framing mode, and recording.

This file is intentionally a V1 integration scaffold. The core logic lives in
the other four modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from time import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageGrab, ImageTk
except ImportError:  # pragma: no cover
    Image = None
    ImageGrab = None
    ImageTk = None

try:
    from video_detector import InputConfig, VideoDetector
    from detection_store import DetectionStore
    from target_tracker import TargetTracker, TrackingConfig
    from reframer import FramingConfig, Reframer
except ImportError:  # pragma: no cover
    from .video_detector import InputConfig, VideoDetector
    from .detection_store import DetectionStore
    from .target_tracker import TargetTracker, TrackingConfig
    from .reframer import FramingConfig, Reframer


@dataclass
class AppConfig:
    window_title: str = "AutoCamTracker V1"
    update_interval_ms: int = 15
    output_width: int = 640
    output_height: int = 360
    log_dir: Path = Path("outputs")
    model_dir: Path = Path(__file__).resolve().parents[1] / "model"
    default_model: str = "yolo11n.pt"


class AutoCamTrackerApp:
    """Tkinter integration shell for the five V1 modules."""

    def __init__(self, root: tk.Tk, config: AppConfig | None = None) -> None:
        self.root = root
        self.config = config or AppConfig()
        self.root.title(self.config.window_title)

        self.input_config = InputConfig()
        self.detector: VideoDetector | None = None
        self.store = DetectionStore()
        self.target_tracker = TargetTracker(TrackingConfig())
        self.reframer = Reframer(
            FramingConfig(
                output_width=self.config.output_width,
                output_height=self.config.output_height,
            )
        )

        self.running = False
        self.recording = False
        self.last_frame_time = time()
        self.loop_started_at = time()
        self.fps = 0.0
        self.skipped_frames = 0
        self.model_options: dict[str, str] = {}

        self.before_image_ref = None
        self.after_image_ref = None
        self._build_ui()
        self.refresh_model_options()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Button(controls, text="▶", width=4, command=self.start).grid(row=0, column=0, padx=4)
        ttk.Button(controls, text="⏸", width=4, command=self.pause).grid(row=0, column=1, padx=4)
        ttk.Button(controls, text="⏹", width=4, command=self.stop).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="↺", width=4, command=self.reset_tracking).grid(row=0, column=3, padx=4)
        ttk.Button(controls, text="Browse Video", command=self.choose_video_file).grid(row=0, column=4, padx=4)
        ttk.Button(controls, text="Select Screen Region", command=self.select_screen_region).grid(row=0, column=5, padx=4)
        ttk.Button(controls, text="Clear Selection", command=self.clear_selection).grid(row=0, column=6, padx=4)
        ttk.Button(controls, text="Auto One", command=self.auto_select_one).grid(row=0, column=7, padx=4)
        ttk.Button(controls, text="Record", command=self.toggle_recording).grid(row=0, column=8, padx=4)

        self.source_var = tk.StringVar(value="webcam")
        self.tracker_var = tk.StringVar(value="botsort")
        self.framing_var = tk.StringVar(value="medium")
        self.model_var = tk.StringVar(value=self.config.default_model)
        self.playback_speed_var = tk.StringVar(value="1x")
        self.camera_index_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar(value="No video selected")
        self.screen_region_var = tk.StringVar(value="No screen region selected")

        ttk.Label(controls, text="Input").grid(row=1, column=0, padx=4, pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.source_var,
            values=["webcam", "video_file", "screen_region"],
            width=14,
            state="readonly",
        ).grid(row=1, column=1, padx=4, pady=(8, 0))

        ttk.Label(controls, text="Tracker").grid(row=1, column=2, padx=4, pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.tracker_var,
            values=["botsort", "deepocsort"],
            width=14,
            state="readonly",
        ).grid(row=1, column=3, padx=4, pady=(8, 0))

        ttk.Label(controls, text="Framing").grid(row=1, column=4, padx=4, pady=(8, 0))
        framing_box = ttk.Combobox(
            controls,
            textvariable=self.framing_var,
            values=["wide", "medium", "close"],
            width=10,
            state="readonly",
        )
        framing_box.grid(row=1, column=5, padx=4, pady=(8, 0))
        framing_box.bind("<<ComboboxSelected>>", lambda _: self.apply_ui_config())

        ttk.Label(controls, text="Camera").grid(row=1, column=6, padx=4, pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.camera_index_var,
            values=["0", "1", "2", "3", "4"],
            width=6,
            state="readonly",
        ).grid(row=1, column=7, padx=4, pady=(8, 0))
        ttk.Button(controls, text="Test Camera", command=self.test_camera).grid(row=1, column=8, padx=4, pady=(8, 0))

        ttk.Label(controls, text="Model").grid(row=2, column=0, padx=4, pady=(8, 0))
        self.model_box = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=[],
            width=42,
            state="readonly",
        )
        self.model_box.grid(row=2, column=1, columnspan=4, padx=4, pady=(8, 0), sticky="ew")
        ttk.Button(controls, text="Refresh Models", command=self.refresh_model_options).grid(row=2, column=5, padx=4, pady=(8, 0))
        ttk.Button(controls, text="Browse Model", command=self.choose_model_file).grid(row=2, column=6, padx=4, pady=(8, 0))

        ttk.Label(controls, text="Speed").grid(row=2, column=7, padx=4, pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.playback_speed_var,
            values=["0.25x", "0.5x", "1x", "1.25x", "1.5x"],
            width=8,
            state="readonly",
        ).grid(row=2, column=8, padx=4, pady=(8, 0))

        ttk.Label(controls, textvariable=self.video_path_var).grid(row=3, column=0, columnspan=6, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(controls, textvariable=self.screen_region_var).grid(row=3, column=6, columnspan=5, sticky="w", padx=4, pady=(8, 0))

        views = ttk.Frame(main)
        views.grid(row=1, column=0, columnspan=2, sticky="nsew")
        views.columnconfigure(0, weight=1)
        views.columnconfigure(1, weight=1)

        ttk.Label(views, text="Before: raw + detection").grid(row=0, column=0)
        ttk.Label(views, text="After: reframe output").grid(row=0, column=1)

        self.before_label = ttk.Label(views)
        self.before_label.grid(row=1, column=0, padx=6, pady=6)
        self.after_label = ttk.Label(views)
        self.after_label.grid(row=1, column=1, padx=6, pady=6)

        self.status_var = tk.StringVar(value="Status: idle")
        ttk.Label(main, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="w")

    def apply_ui_config(self) -> None:
        self.input_config.source_type = self.source_var.get()
        self.input_config.tracker_name = self.tracker_var.get()
        self.input_config.model_path = self.model_options.get(
            self.model_var.get(),
            self.model_var.get() or self.config.default_model,
        )
        try:
            self.input_config.camera_index = int(self.camera_index_var.get())
        except ValueError:
            self.input_config.camera_index = 0
        self.reframer.set_framing_mode(self.framing_var.get())

    def start(self) -> None:
        try:
            if self.detector is not None and not self.running:
                self.running = True
                self.last_frame_time = time()
                self._loop()
                return

            self.apply_ui_config()
            if self.detector is not None:
                self.detector.close()
            self.detector = VideoDetector(self.input_config)
            self.detector.load_model()
            self.detector.open_source()
            self.running = True
            self.last_frame_time = time()
            self.skipped_frames = 0
            self._loop()
        except Exception as exc:
            self.running = False
            if self.detector is not None:
                self.detector.close()
                self.detector = None
            messagebox.showerror("Start failed", str(exc))

    def pause(self) -> None:
        self.running = False

    def stop(self) -> None:
        self.running = False
        if self.detector is not None:
            self.detector.close()
        self.detector = None

    def reset_tracking(self) -> None:
        self.target_tracker.clear_selection()
        self.reframer.reset()

    def clear_selection(self) -> None:
        self.target_tracker.clear_selection()

    def choose_video_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.source_var.set("video_file")
            self.input_config.video_path = path
            self.video_path_var.set(f"Video: {path}")

    def choose_model_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose YOLO model",
            initialdir=str(self.config.model_dir) if self.config.model_dir.exists() else str(Path.cwd()),
            filetypes=[
                ("YOLO model files", "*.pt *.pth *.onnx *.engine *.mlpackage *.torchscript"),
                ("All files", "*.*"),
            ],
        )
        if path:
            label = self._model_label(Path(path))
            self.model_options[label] = path
            self.model_box.configure(values=list(self.model_options.keys()))
            self.model_var.set(label)

    def refresh_model_options(self) -> None:
        model_files = self._discover_model_files()
        options = {self.config.default_model: self.config.default_model}
        for path in model_files:
            options[self._model_label(path)] = str(path)
        self.model_options = options
        if hasattr(self, "model_box"):
            self.model_box.configure(values=list(self.model_options.keys()))
        if self.model_var.get() not in self.model_options:
            self.model_var.set(next(iter(self.model_options)))

    def select_screen_region(self) -> None:
        self.pause()
        screenshot = self._capture_screen_selection_background()
        selector = tk.Toplevel(self.root)
        selector.title("Select screen region")
        selector.attributes("-fullscreen", True)
        selector.attributes("-topmost", True)

        canvas = tk.Canvas(selector, cursor="crosshair", bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        if screenshot is not None:
            selector._screen_selection_image = ImageTk.PhotoImage(screenshot)
            canvas.create_image(0, 0, anchor="nw", image=selector._screen_selection_image)
        canvas.create_text(
            30,
            30,
            anchor="nw",
            text="Drag to select screen region. Press Esc to cancel.",
            fill="white",
            font=("Arial", 24),
        )

        state: dict[str, int | None] = {"start_x": None, "start_y": None, "rect": None}

        def on_press(event) -> None:
            state["start_x"] = event.x_root
            state["start_y"] = event.y_root
            if state["rect"] is not None:
                canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(
                event.x,
                event.y,
                event.x,
                event.y,
                outline="yellow",
                width=4,
            )

        def on_drag(event) -> None:
            if state["rect"] is None or state["start_x"] is None or state["start_y"] is None:
                return
            local_start_x = state["start_x"] - selector.winfo_rootx()
            local_start_y = state["start_y"] - selector.winfo_rooty()
            canvas.coords(state["rect"], local_start_x, local_start_y, event.x, event.y)

        def on_release(event) -> None:
            if state["start_x"] is None or state["start_y"] is None:
                selector.destroy()
                return
            x1 = int(min(state["start_x"], event.x_root))
            y1 = int(min(state["start_y"], event.y_root))
            x2 = int(max(state["start_x"], event.x_root))
            y2 = int(max(state["start_y"], event.y_root))
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            self.input_config.screen_region = (x1, y1, width, height)
            self.source_var.set("screen_region")
            self.screen_region_var.set(f"Screen region: x={x1}, y={y1}, w={width}, h={height}")
            selector.destroy()

        selector.bind("<Escape>", lambda _: selector.destroy())
        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

    def _capture_screen_selection_background(self):
        if Image is None or ImageGrab is None:
            return None

        try:
            screenshot = ImageGrab.grab()
        except Exception:
            return None

        screen_width = max(1, self.root.winfo_screenwidth())
        screen_height = max(1, self.root.winfo_screenheight())
        screenshot = screenshot.resize((screen_width, screen_height))
        overlay = Image.new("RGB", screenshot.size, (0, 0, 0))
        return Image.blend(screenshot.convert("RGB"), overlay, 0.22)

    def test_camera(self) -> None:
        import cv2

        try:
            camera_index = int(self.camera_index_var.get())
        except ValueError:
            camera_index = 0
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        capture = cv2.VideoCapture(camera_index, backend)
        ok, _ = capture.read() if capture.isOpened() else (False, None)
        capture.release()
        if ok:
            messagebox.showinfo("Camera test", f"Camera index {camera_index} opened successfully.")
        else:
            messagebox.showerror(
                "Camera test failed",
                "Unable to open camera. Check macOS camera permission, camera index, "
                "and whether another app is using the camera.",
            )

    def auto_select_one(self) -> None:
        candidates = self.store.rank_candidates(strategy="largest")
        self.target_tracker.auto_select_one(candidates)

    def toggle_recording(self) -> None:
        self.recording = not self.recording
        messagebox.showinfo(
            "Recording",
            "Recording scaffold toggled. VideoWriter implementation belongs here.",
        )

    def _loop(self) -> None:
        if not self.running or self.detector is None:
            return

        self.loop_started_at = time()
        frame, detections = self.detector.read_and_track()
        if frame is None:
            self.stop()
            return

        candidates = self.store.update(detections, frame.shape)
        selected_targets = self.target_tracker.update_from_store(self.store)
        after_frame, framing_status = self.reframer.render(frame, selected_targets)

        now = time()
        elapsed = max(1e-6, now - self.last_frame_time)
        self.fps = 1.0 / elapsed
        self.last_frame_time = now

        before_frame = self._draw_detections(frame, detections)
        self._update_images(before_frame, after_frame)
        state = self.target_tracker.get_state()
        self.status_var.set(
            "Status: "
            f"{state['status']} | Display FPS: {self.fps:.1f} | "
            f"Source: {self._source_fps_label()} | Speed: {self.playback_speed_var.get()} | "
            f"Skipped: {self.skipped_frames} | "
            f"Candidates: {len(candidates)} | Selected: {state['selected_track_ids']} | "
            f"Crop: {framing_status.crop_window}"
        )

        if state.get("lost_alert"):
            messagebox.showwarning("Tracking failed", str(state["lost_alert"]))

        self._drop_late_video_frames()
        self.root.after(self._next_loop_delay_ms(), self._loop)

    def _next_loop_delay_ms(self) -> int:
        if self.detector is None or self.input_config.source_type != "video_file":
            return self.config.update_interval_ms

        source_fps = self.detector.get_source_fps()
        if source_fps is None:
            return self.config.update_interval_ms

        target_interval_ms = 1000.0 / max(1.0, source_fps) / self._playback_speed()
        elapsed_ms = (time() - self.loop_started_at) * 1000.0
        return max(1, int(round(target_interval_ms - elapsed_ms)))

    def _drop_late_video_frames(self) -> None:
        if self.detector is None or self.input_config.source_type != "video_file":
            return

        source_fps = self.detector.get_source_fps()
        if source_fps is None:
            return

        target_interval_ms = 1000.0 / max(1.0, source_fps) / self._playback_speed()
        elapsed_ms = (time() - self.loop_started_at) * 1000.0
        frames_due = int(elapsed_ms // max(1.0, target_interval_ms))
        frames_to_skip = min(30, max(0, frames_due - 1))
        if frames_to_skip <= 0:
            return

        self.skipped_frames += self.detector.skip_video_frames(frames_to_skip)

    def _playback_speed(self) -> float:
        value = self.playback_speed_var.get().strip().lower().replace("x", "")
        try:
            speed = float(value)
        except ValueError:
            return 1.0
        return max(0.05, speed)

    def _source_fps_label(self) -> str:
        if self.detector is None:
            return "--"
        source_fps = self.detector.get_source_fps()
        if source_fps is None:
            return "--"
        return f"{source_fps:.1f}"

    def _discover_model_files(self) -> list[Path]:
        suffixes = {".pt", ".pth", ".onnx", ".engine", ".mlpackage", ".torchscript"}
        if not self.config.model_dir.exists():
            return []
        return sorted(
            path
            for path in self.config.model_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )

    def _model_label(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config.model_dir))
        except ValueError:
            return str(path)

    def _draw_detections(self, frame, detections):
        import cv2

        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(value) for value in detection.bbox]
            label = f"id:{detection.track_id} {detection.class_name} {detection.confidence:.2f}"
            color = (0, 220, 255) if detection.track_id in self.target_tracker.selected_track_ids else (80, 220, 80)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _update_images(self, before_frame, after_frame) -> None:
        if Image is None or ImageTk is None:
            return
        import cv2

        before_rgb = cv2.cvtColor(before_frame, cv2.COLOR_BGR2RGB)
        after_rgb = cv2.cvtColor(after_frame, cv2.COLOR_BGR2RGB)

        before_image = Image.fromarray(before_rgb).resize(
            (self.config.output_width, self.config.output_height)
        )
        after_image = Image.fromarray(after_rgb).resize(
            (self.config.output_width, self.config.output_height)
        )

        self.before_image_ref = ImageTk.PhotoImage(before_image)
        self.after_image_ref = ImageTk.PhotoImage(after_image)
        self.before_label.configure(image=self.before_image_ref)
        self.after_label.configure(image=self.after_image_ref)


def main() -> None:
    root = tk.Tk()
    app = AutoCamTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
