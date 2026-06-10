"""Tkinter UI + Recording + Debug Log module for AutoCamTracker V1.

Responsibilities:
- Create the Tkinter desktop UI.
- Wire together input, YOLO tracking, data store, target tracking, and reframe.
- Show before and after views.
- Expose controls for source, tracker, selection mode, framing mode, and recording.

This file is intentionally a V1 integration scaffold. The core logic lives in
the other four modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    Image = None
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
        self.fps = 0.0

        self.before_image_ref = None
        self.after_image_ref = None
        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Button(controls, text="Start", command=self.start).grid(row=0, column=0, padx=4)
        ttk.Button(controls, text="Pause", command=self.pause).grid(row=0, column=1, padx=4)
        ttk.Button(controls, text="Stop", command=self.stop).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="Reset", command=self.reset_tracking).grid(row=0, column=3, padx=4)
        ttk.Button(controls, text="Open Video", command=self.choose_video_file).grid(row=0, column=4, padx=4)
        ttk.Button(controls, text="Clear Selection", command=self.clear_selection).grid(row=0, column=5, padx=4)
        ttk.Button(controls, text="Auto One", command=self.auto_select_one).grid(row=0, column=6, padx=4)
        ttk.Button(controls, text="Auto Multiple", command=self.auto_select_multiple).grid(row=0, column=7, padx=4)
        ttk.Button(controls, text="Record", command=self.toggle_recording).grid(row=0, column=8, padx=4)

        self.source_var = tk.StringVar(value="webcam")
        self.tracker_var = tk.StringVar(value="botsort")
        self.selection_var = tk.StringVar(value="single")
        self.framing_var = tk.StringVar(value="medium")

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

        ttk.Label(controls, text="Select").grid(row=1, column=4, padx=4, pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.selection_var,
            values=["single", "multi"],
            width=10,
            state="readonly",
        ).grid(row=1, column=5, padx=4, pady=(8, 0))

        ttk.Label(controls, text="Framing").grid(row=1, column=6, padx=4, pady=(8, 0))
        framing_box = ttk.Combobox(
            controls,
            textvariable=self.framing_var,
            values=["wide", "medium", "close"],
            width=10,
            state="readonly",
        )
        framing_box.grid(row=1, column=7, padx=4, pady=(8, 0))
        framing_box.bind("<<ComboboxSelected>>", lambda _: self.apply_ui_config())

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
        self.target_tracker.set_selection_mode(self.selection_var.get())
        self.reframer.set_framing_mode(self.framing_var.get())

    def start(self) -> None:
        try:
            self.apply_ui_config()
            if self.detector is None:
                self.detector = VideoDetector(self.input_config)
                self.detector.load_model()
                self.detector.open_source()
            self.running = True
            self._loop()
        except Exception as exc:
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

    def auto_select_one(self) -> None:
        candidates = self.store.rank_candidates(strategy="largest")
        self.target_tracker.auto_select_one(candidates)

    def auto_select_multiple(self) -> None:
        candidates = self.store.rank_candidates(strategy="largest")
        self.target_tracker.auto_select_multiple(candidates)

    def toggle_recording(self) -> None:
        self.recording = not self.recording
        messagebox.showinfo(
            "Recording",
            "Recording scaffold toggled. VideoWriter implementation belongs here.",
        )

    def _loop(self) -> None:
        if not self.running or self.detector is None:
            return

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

        self._update_images(frame, after_frame)
        state = self.target_tracker.get_state()
        self.status_var.set(
            "Status: "
            f"{state['status']} | FPS: {self.fps:.1f} | "
            f"Candidates: {len(candidates)} | Selected: {state['selected_track_ids']} | "
            f"Crop: {framing_status.crop_window}"
        )

        if state.get("lost_alert"):
            messagebox.showwarning("Tracking failed", str(state["lost_alert"]))

        self.root.after(self.config.update_interval_ms, self._loop)

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
