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

from dataclasses import dataclass, replace
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
        self.active_input_signature: tuple[object, ...] | None = None
        self.last_frame_shape: tuple[int, int, int] | tuple[int, int] | None = None
        self.display_width = self.config.output_width
        self.display_height = self.config.output_height
        self.timeline_dragging = False

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

        source_controls = ttk.LabelFrame(controls, text="Source", padding=8, width=395, height=190)
        source_controls.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        tracking_controls = ttk.LabelFrame(controls, text="Tracking", padding=8, width=395, height=190)
        tracking_controls.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        playback_controls = ttk.LabelFrame(controls, text="Playback", padding=8, width=395, height=190)
        playback_controls.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        view_controls = ttk.LabelFrame(controls, text="View", padding=8, width=395, height=190)
        view_controls.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        for panel in (source_controls, tracking_controls, playback_controls, view_controls):
            panel.grid_propagate(False)
        controls.columnconfigure(0, weight=0, minsize=407)
        controls.columnconfigure(1, weight=0, minsize=407)

        self.source_var = tk.StringVar(value="webcam")
        self.tracker_var = tk.StringVar(value="botsort")
        self.framing_var = tk.StringVar(value="medium")
        self.model_var = tk.StringVar(value=self.config.default_model)
        self.playback_speed_var = tk.StringVar(value="1x")
        self.camera_index_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar(value="No video selected")
        self.screen_region_var = tk.StringVar(value="No screen region selected")
        self.view_width_var = tk.StringVar(value=str(self.config.output_width))
        self.view_height_var = tk.StringVar(value=str(self.config.output_height))
        self.auto_stretch_var = tk.BooleanVar(value=True)
        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline_label_var = tk.StringVar(value="00:00 / 00:00")

        ttk.Label(source_controls, text="Input").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Combobox(
            source_controls,
            textvariable=self.source_var,
            values=["webcam", "video_file", "screen_region"],
            width=21,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(source_controls, text="Browse Video", width=18, command=self.choose_video_file).grid(row=1, column=0, sticky="ew", padx=4, pady=(8, 0))
        ttk.Button(source_controls, text="Screen Region", width=18, command=self.select_screen_region).grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))

        ttk.Label(source_controls, textvariable=self.video_path_var, wraplength=320).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(source_controls, textvariable=self.screen_region_var, wraplength=320).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=(3, 0))
        source_controls.columnconfigure(1, weight=1)

        ttk.Label(tracking_controls, text="Model").grid(row=0, column=0, sticky="w", padx=4)
        self.model_box = ttk.Combobox(
            tracking_controls,
            textvariable=self.model_var,
            values=[],
            width=21,
            state="readonly",
        )
        self.model_box.grid(row=0, column=1, padx=4, sticky="ew")
        ttk.Button(tracking_controls, text="Refresh", width=12, command=self.refresh_model_options).grid(row=0, column=2, padx=4)

        ttk.Label(tracking_controls, text="Tracker").grid(row=1, column=0, sticky="w", padx=4, pady=(8, 0))
        ttk.Combobox(
            tracking_controls,
            textvariable=self.tracker_var,
            values=["botsort", "deepocsort"],
            width=15,
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))

        ttk.Label(tracking_controls, text="Framing").grid(row=2, column=0, sticky="w", padx=4, pady=(8, 0))
        framing_box = ttk.Combobox(
            tracking_controls,
            textvariable=self.framing_var,
            values=["wide", "medium", "close"],
            width=15,
            state="readonly",
        )
        framing_box.grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 0))
        framing_box.bind("<<ComboboxSelected>>", lambda _: self.apply_ui_config())
        ttk.Button(tracking_controls, text="Clear Selection", width=18, command=self.clear_selection).grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=(10, 0))
        ttk.Button(tracking_controls, text="Auto Track", width=18, command=self.auto_select_one).grid(row=4, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 0))
        ttk.Button(tracking_controls, text="Reset", width=14, command=self.reset_tracking).grid(row=4, column=2, sticky="ew", padx=4, pady=(8, 0))
        tracking_controls.columnconfigure(1, weight=1)
        tracking_controls.columnconfigure(2, weight=1)

        ttk.Button(playback_controls, text="Start", width=16, command=self.start).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(playback_controls, text="Pause", width=16, command=self.pause).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(playback_controls, text="Stop", width=16, command=self.stop).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(playback_controls, text="Record", width=16, command=self.toggle_recording).grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(playback_controls, text="Speed").grid(row=2, column=0, sticky="w", padx=4, pady=(8, 0))
        ttk.Combobox(
            playback_controls,
            textvariable=self.playback_speed_var,
            values=["0.25x", "0.5x", "1x", "1.25x", "1.5x", "3x", "4x", "5x", "6x"],
            width=15,
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 0))
        playback_controls.columnconfigure(0, weight=1)
        playback_controls.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            view_controls,
            text="Stretch",
            variable=self.auto_stretch_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        ttk.Label(view_controls, text="Width").grid(row=1, column=0, sticky="w", padx=4, pady=(8, 0))
        ttk.Entry(view_controls, textvariable=self.view_width_var, width=10).grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))
        ttk.Label(view_controls, text="Height").grid(row=2, column=0, sticky="w", padx=4, pady=(8, 0))
        ttk.Entry(view_controls, textvariable=self.view_height_var, width=10).grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 0))
        ttk.Button(view_controls, text="Apply Size", width=16, command=self.apply_view_size).grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 0))
        view_controls.columnconfigure(1, weight=1)

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)
        views = ttk.Frame(main)
        views.grid(row=1, column=0, columnspan=2, sticky="nsew")
        views.columnconfigure(0, weight=1)
        views.columnconfigure(1, weight=1)
        views.rowconfigure(1, weight=1)
        views.bind("<Configure>", self.on_views_resize)

        ttk.Label(views, text="Before: raw + detection").grid(row=0, column=0)
        ttk.Label(views, text="After: reframe output").grid(row=0, column=1)

        self.before_label = ttk.Label(views)
        self.before_label.grid(row=1, column=0, padx=6, pady=6, sticky="nsew")
        self.before_label.configure(cursor="hand2", anchor="nw")
        self.before_label.bind("<Button-1>", self.on_before_click)
        self.after_label = ttk.Label(views)
        self.after_label.grid(row=1, column=1, padx=6, pady=6, sticky="nsew")
        self.after_label.configure(anchor="nw")

        timeline = ttk.Frame(views)
        timeline.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        timeline.columnconfigure(0, weight=1)
        self.timeline_scale = ttk.Scale(
            timeline,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self.timeline_var,
            command=self.on_timeline_drag,
        )
        self.timeline_scale.grid(row=0, column=0, sticky="ew")
        self.timeline_scale.bind("<ButtonPress-1>", self.on_timeline_press)
        self.timeline_scale.bind("<ButtonRelease-1>", self.on_timeline_release)
        ttk.Label(timeline, textvariable=self.timeline_label_var, width=16).grid(row=0, column=1, padx=(8, 0))

        self.status_var = tk.StringVar(value="Status: idle")
        ttk.Label(main, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="w")

    def apply_ui_config(self) -> None:
        self.input_config = self._ui_input_config()
        self.reframer.set_framing_mode(self.framing_var.get())

    def _ui_input_config(self) -> InputConfig:
        try:
            camera_index = int(self.camera_index_var.get())
        except ValueError:
            camera_index = 0

        return InputConfig(
            source_type=self.source_var.get(),
            camera_index=camera_index,
            video_path=self.input_config.video_path,
            screen_region=self.input_config.screen_region,
            model_path=self.model_options.get(
                self.model_var.get(),
                self.model_var.get() or self.config.default_model,
            ),
            tracker_name=self.tracker_var.get(),
            confidence_threshold=self.input_config.confidence_threshold,
            iou_threshold=self.input_config.iou_threshold,
            vehicle_classes_only=self.input_config.vehicle_classes_only,
        )

    def start(self) -> None:
        try:
            self.apply_ui_config()
            desired_signature = self._input_signature(self.input_config)
            can_resume_current_source = (
                self.detector is not None
                and not self.running
                and self.active_input_signature == desired_signature
            )
            if can_resume_current_source:
                self.running = True
                self.last_frame_time = time()
                self._loop()
                return

            if self.detector is not None:
                self._close_detector()
            self._reset_runtime_state()
            self.detector = VideoDetector(replace(self.input_config))
            self.detector.load_model()
            self.detector.open_source()
            self.active_input_signature = desired_signature
            self.running = True
            self.last_frame_time = time()
            self.skipped_frames = 0
            self._loop()
        except Exception as exc:
            self.running = False
            if self.detector is not None:
                self._close_detector()
                self.detector = None
            messagebox.showerror("Start failed", str(exc))

    def pause(self) -> None:
        self.running = False

    def stop(self) -> None:
        self.running = False
        if self.detector is not None:
            self._close_detector()
        self.detector = None
        self.active_input_signature = None
        self._reset_runtime_state()

    def reset_tracking(self) -> None:
        self._reset_runtime_state()

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
            self.video_path_var.set(f"Video: {self._short_label(Path(path).name)}")

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

    def auto_select_one(self) -> None:
        candidates = self.store.rank_candidates(self.last_frame_shape, strategy="stable")
        self.target_tracker.auto_select_one(candidates)

    def apply_view_size(self) -> None:
        width = self._parse_dimension(self.view_width_var.get(), self.display_width)
        height = self._parse_dimension(self.view_height_var.get(), self.display_height)
        self.auto_stretch_var.set(False)
        self._set_display_size(width, height, update_fields=True)

    def on_views_resize(self, event) -> None:
        if not self.auto_stretch_var.get():
            return
        width = max(160, (event.width - 24) // 2)
        height = max(90, event.height - 72)
        self._set_display_size(width, height, update_fields=True)

    def on_timeline_press(self, _event) -> None:
        self.timeline_dragging = True

    def on_timeline_drag(self, value: str) -> None:
        if self.timeline_dragging:
            self._update_timeline_label(int(float(value)))

    def on_timeline_release(self, _event) -> None:
        self.timeline_dragging = False
        if self.detector is None or self.input_config.source_type != "video_file":
            return

        target_frame = int(self.timeline_var.get())
        if not self.detector.seek_video_frame(target_frame):
            return

        self.store.reset()
        self.target_tracker.clear_selection()
        self.reframer.reset()
        self.skipped_frames = 0
        self._render_current_video_frame()

    def on_before_click(self, event) -> None:
        if self.last_frame_shape is None:
            return

        frame_height, frame_width = self.last_frame_shape[:2]
        image_width = max(1, self.display_width)
        image_height = max(1, self.display_height)
        frame_x = event.x * frame_width / image_width
        frame_y = event.y * frame_height / image_height

        candidate = self.store.get_candidate_at_point(frame_x, frame_y, self.last_frame_shape)
        if candidate is None:
            self.status_var.set("Status: no tracked vehicle at clicked point")
            return

        self.target_tracker.select_track(candidate.track_id)
        self.target_tracker.update_from_store(self.store)
        self.status_var.set(f"Status: selected track id {candidate.track_id}")

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

        candidates, framing_status = self._process_frame(frame, detections)
        now = time()
        elapsed = max(1e-6, now - self.last_frame_time)
        self.fps = 1.0 / elapsed
        self.last_frame_time = now

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
        self._sync_timeline_from_detector()
        self.root.after(self._next_loop_delay_ms(), self._loop)

    def _process_frame(self, frame, detections):
        self.last_frame_shape = frame.shape
        candidates = self.store.update(detections, frame.shape)
        selected_targets = self.target_tracker.update_from_store(self.store)
        after_frame, framing_status = self.reframer.render(frame, selected_targets)
        before_frame = self._draw_detections(frame, detections)
        self._update_images(before_frame, after_frame)
        return candidates, framing_status

    def _render_current_video_frame(self) -> None:
        if self.detector is None:
            return
        frame, detections = self.detector.read_and_track()
        if frame is None:
            return
        self._process_frame(frame, detections)
        self._sync_timeline_from_detector()

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

    def _sync_timeline_from_detector(self) -> None:
        if self.detector is None or self.input_config.source_type != "video_file":
            self.timeline_scale.configure(to=0)
            self.timeline_var.set(0)
            self.timeline_label_var.set("00:00 / 00:00")
            return

        frame_count = self.detector.get_source_frame_count() or 0
        current_frame = self.detector.get_current_frame_index()
        self.timeline_scale.configure(to=max(0, frame_count - 1))
        if not self.timeline_dragging:
            self.timeline_var.set(min(max(0, current_frame), max(0, frame_count - 1)))
            self._update_timeline_label(current_frame)

    def _update_timeline_label(self, frame_index: int) -> None:
        frame_count = self.detector.get_source_frame_count() if self.detector is not None else None
        fps = self.detector.get_source_fps() if self.detector is not None else None
        if not frame_count or not fps:
            self.timeline_label_var.set(f"{frame_index}")
            return
        current_seconds = max(0.0, frame_index / fps)
        total_seconds = max(0.0, frame_count / fps)
        self.timeline_label_var.set(
            f"{self._format_time(current_seconds)} / {self._format_time(total_seconds)}"
        )

    def _reset_runtime_state(self) -> None:
        self.store.reset()
        self.target_tracker.clear_selection()
        self.reframer.reset()
        self.last_frame_shape = None
        self.skipped_frames = 0

    def _set_display_size(self, width: int, height: int, update_fields: bool = False) -> None:
        width = max(160, min(3840, int(width)))
        height = max(90, min(2160, int(height)))
        if width == self.display_width and height == self.display_height:
            return

        self.display_width = width
        self.display_height = height
        self.config.output_width = width
        self.config.output_height = height
        self.reframer.config.output_width = width
        self.reframer.config.output_height = height
        if update_fields:
            self.view_width_var.set(str(width))
            self.view_height_var.set(str(height))

    @staticmethod
    def _parse_dimension(value: str, fallback: int) -> int:
        try:
            return int(value)
        except ValueError:
            return fallback

    @staticmethod
    def _format_time(seconds: float) -> str:
        total_seconds = int(round(seconds))
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _close_detector(self) -> None:
        if self.detector is None:
            return
        clear_temp_cache = self.detector.config.source_type == "video_file"
        self.detector.close(clear_temp_cache=clear_temp_cache)

    @staticmethod
    def _input_signature(config: InputConfig) -> tuple[object, ...]:
        return (
            config.source_type,
            config.camera_index,
            config.video_path,
            config.screen_region,
            config.model_path,
            config.tracker_name,
            config.confidence_threshold,
            config.iou_threshold,
            config.vehicle_classes_only,
        )

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

    @staticmethod
    def _short_label(value: str, max_length: int = 38) -> str:
        if len(value) <= max_length:
            return value
        keep = max(8, max_length - 3)
        return f"{value[:keep]}..."

    def _draw_detections(self, frame, detections):
        import cv2

        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(value) for value in detection.bbox]
            label = f"id:{detection.track_id} {detection.class_name} {detection.confidence:.2f}"
            color = (0, 220, 255) if detection.track_id in self.target_tracker.selected_track_ids else (80, 220, 80)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = cv2.getFontScaleFromHeight(font_face, 20, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(24, y1 - 10)),
                font_face,
                font_scale,
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

        size = (self.display_width, self.display_height)
        before_image = Image.fromarray(before_rgb).resize(size)
        after_image = Image.fromarray(after_rgb).resize(size)

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
