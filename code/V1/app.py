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
from io import BytesIO
from pathlib import Path
import sys
from time import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    from PIL import Image, ImageGrab, ImageTk
except ImportError:  # pragma: no cover
    Image = None
    ImageGrab = None
    ImageTk = None

try:
    from auto_feature_sampler import AutoFeatureMode, AutoFeatureSampler
    from video_detector import InputConfig, VideoDetector
    from detection_store import DetectionStore
    from feature_gallery import FeatureGallery
    from frame_data import FrameData
    from identity_manager import GlobalIdentityManager
    from pipeline_processor import PipelineProcessor
    from reframer import FramingConfig, Reframer
    from scene_cut import SceneCutDetector
    from vehicle_identity_store import VehicleIdentityStore
except ImportError:  # pragma: no cover
    from .auto_feature_sampler import AutoFeatureMode, AutoFeatureSampler
    from .video_detector import InputConfig, VideoDetector
    from .detection_store import DetectionStore
    from .feature_gallery import FeatureGallery
    from .frame_data import FrameData
    from .identity_manager import GlobalIdentityManager
    from .pipeline_processor import PipelineProcessor
    from .reframer import FramingConfig, Reframer
    from .scene_cut import SceneCutDetector
    from .vehicle_identity_store import VehicleIdentityStore


@dataclass
class AppConfig:
    window_title: str = "AutoCamTracker V1.3"
    update_interval_ms: int = 15
    output_width: int = 640
    output_height: int = 360
    log_dir: Path = Path("outputs")
    identity_db_path: Path = Path("outputs") / "vehicle_identity.sqlite3"
    model_dir: Path = Path(__file__).resolve().parents[1] / "model"
    default_model: str = "yolo26s.pt"
    default_reid_model: str = "yolo26s-reid.onnx"


class AutoCamTrackerApp:
    """Tkinter integration shell for the five V1 modules."""

    def __init__(self, root: tk.Tk, config: AppConfig | None = None) -> None:
        self.root = root
        self.config = config or AppConfig()
        self.root.title(self.config.window_title)
        self.root.minsize(1120, 720)

        self.input_config = InputConfig()
        self.detector: VideoDetector | None = None
        self.store = DetectionStore()
        self.identity_store = VehicleIdentityStore(self.config.identity_db_path)
        self.feature_gallery = FeatureGallery(
            self.config.identity_db_path,
            reid_model_path=str(self.config.model_dir / self.config.default_reid_model),
        )
        self.identity_manager = GlobalIdentityManager(
            identity_store=self.identity_store,
            feature_gallery=self.feature_gallery,
        )
        self.auto_feature_sampler = AutoFeatureSampler(self.feature_gallery)
        self.scene_cut_detector = SceneCutDetector()
        self.reframer = Reframer(
            FramingConfig(
                output_width=self.config.output_width,
                output_height=self.config.output_height,
            )
        )
        self.pipeline = PipelineProcessor(
            store=self.store,
            identity_manager=self.identity_manager,
            scene_cut_detector=self.scene_cut_detector,
            reframer=self.reframer,
        )

        self.running = False
        self.recording = False
        self.last_frame_time = time()
        self.loop_started_at = time()
        self.fps = 0.0
        self.skipped_frames = 0
        self.last_inference_time_ms = 0.0
        self.model_options: dict[str, str] = {}
        self.reid_model_options: dict[str, str] = {}
        self.active_input_signature: tuple[object, ...] | None = None
        self.last_frame_shape: tuple[int, int, int] | tuple[int, int] | None = None
        self.last_raw_frame = None
        self.current_frame_data: FrameData | None = None
        self.display_width = self.config.output_width
        self.display_height = self.config.output_height
        self.rendered_image_width = self.display_width
        self.rendered_image_height = self.display_height
        self.timeline_dragging = False
        self.refreshing_identity_panel = False
        self.link_bbox_vehicle_id: int | None = None
        self.selected_identity_tree_ids: set[int] = set()
        self.identity_preview_window: tk.Toplevel | None = None
        self.identity_preview_label: ttk.Label | None = None
        self.identity_preview_photo = None
        self.identity_preview_vehicle_id: int | None = None
        self.auto_feature_status_message = ""

        self.before_image_ref = None
        self.after_image_ref = None
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_model_options()
        self.refresh_reid_model_options()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        source_controls = ttk.LabelFrame(controls, text="Source", padding=8)
        source_controls.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        tracking_controls = ttk.LabelFrame(controls, text="Tracking", padding=8)
        tracking_controls.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        playback_controls = ttk.LabelFrame(controls, text="Playback", padding=8)
        playback_controls.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        identity_controls = ttk.LabelFrame(controls, text="Identity DB", padding=6)
        identity_controls.grid(row=0, column=2, rowspan=2, sticky="nw", padx=4, pady=4)
        for column in range(2):
            controls.columnconfigure(column, weight=0, minsize=207)
        controls.columnconfigure(2, weight=0, minsize=414)

        self.source_var = tk.StringVar(value="webcam")
        self.tracker_var = tk.StringVar(value="botsort")
        self.framing_var = tk.StringVar(value="medium")
        self.model_var = tk.StringVar(value=self.config.default_model)
        self.reid_model_var = tk.StringVar(value=self.config.default_reid_model)
        self.auto_reid_threshold_var = tk.StringVar(value=f"{self.identity_manager.auto_reid_min_score:.2f}")
        self.auto_feature_mode_var = tk.StringVar(value=self.auto_feature_sampler.config.mode)
        self.playback_speed_var = tk.StringVar(value="1x")
        self.camera_index_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar(value="No video selected")
        self.video_url_var = tk.StringVar(value="")
        self.video_url_status_var = tk.StringVar(value="No video URL selected")
        self.screen_region_var = tk.StringVar(value="No screen region selected")
        self.identity_summary_var = tk.StringVar(value="Vehicles: 0 | Master: 0 | Pending: 0 | Candidate: 0")
        self.identity_mode_var = tk.StringVar(value="Identity Mode: click bbox to select/create GID")
        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline_label_var = tk.StringVar(value="00:00 / 00:00")

        ttk.Label(source_controls, text="Input").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Combobox(
            source_controls,
            textvariable=self.source_var,
            values=["webcam", "video_file", "video_url", "screen_region"],
            width=17,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(source_controls, text="Browse Video", command=self.choose_video_file).grid(row=1, column=0, sticky="ew", padx=4, pady=(8, 0))
        ttk.Button(source_controls, text="Screen Region", command=self.select_screen_region).grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))

        ttk.Label(source_controls, text="URL").grid(row=2, column=0, sticky="w", padx=4, pady=(8, 0))
        url_entry = ttk.Entry(source_controls, textvariable=self.video_url_var)
        url_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 0))
        url_entry.bind("<Return>", self.apply_video_url)
        url_entry.bind("<FocusOut>", self.apply_video_url)

        ttk.Label(source_controls, textvariable=self.video_path_var, wraplength=220).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(source_controls, textvariable=self.video_url_status_var, wraplength=220).grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=(3, 0))
        ttk.Label(source_controls, textvariable=self.screen_region_var, wraplength=220).grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=(3, 0))
        source_controls.columnconfigure(0, weight=1)
        source_controls.columnconfigure(1, weight=1)

        ttk.Label(tracking_controls, text="Model").grid(row=0, column=0, sticky="w", padx=4)
        self.model_box = ttk.Combobox(
            tracking_controls,
            textvariable=self.model_var,
            values=[],
            width=17,
            state="readonly",
        )
        self.model_box.grid(row=0, column=1, padx=4, sticky="ew")
        ttk.Button(tracking_controls, text="Refresh", command=self.refresh_model_options).grid(row=0, column=2, sticky="ew", padx=4)

        ttk.Label(tracking_controls, text="Tracker").grid(row=1, column=0, sticky="w", padx=4, pady=(8, 0))
        ttk.Combobox(
            tracking_controls,
            textvariable=self.tracker_var,
            values=["botsort", "deepocsort"],
            width=13,
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))

        ttk.Label(tracking_controls, text="Framing").grid(row=2, column=0, sticky="w", padx=4, pady=(8, 0))
        framing_box = ttk.Combobox(
            tracking_controls,
            textvariable=self.framing_var,
            values=["wide", "medium", "close"],
            width=13,
            state="readonly",
        )
        framing_box.grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 0))
        framing_box.bind("<<ComboboxSelected>>", lambda _: self.apply_ui_config())
        ttk.Button(tracking_controls, text="Clear Selection", command=self.clear_selection).grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=(10, 0))
        ttk.Button(tracking_controls, text="Auto Track", command=self.auto_select_one).grid(row=4, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 0))
        ttk.Button(tracking_controls, text="Reset", command=self.reset_tracking).grid(row=4, column=2, sticky="ew", padx=4, pady=(8, 0))
        tracking_controls.columnconfigure(1, weight=1)
        tracking_controls.columnconfigure(2, weight=1)

        ttk.Button(playback_controls, text="Start", command=self.start).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(playback_controls, text="Pause", command=self.pause).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(playback_controls, text="Stop", command=self.stop).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(playback_controls, text="Record", command=self.toggle_recording).grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(playback_controls, text="Speed").grid(row=2, column=0, sticky="w", padx=4, pady=(8, 0))
        ttk.Combobox(
            playback_controls,
            textvariable=self.playback_speed_var,
            values=["0.25x", "0.5x", "1x", "1.25x", "1.5x", "3x", "4x", "5x", "6x"],
            width=13,
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 0))
        playback_controls.columnconfigure(0, weight=1)
        playback_controls.columnconfigure(1, weight=1)

        ttk.Label(identity_controls, textvariable=self.identity_summary_var).grid(row=0, column=0, columnspan=4, sticky="w", padx=3)
        ttk.Button(identity_controls, text="Refresh", width=9, command=self.refresh_identity_db_panel).grid(row=0, column=4, sticky="ew", padx=2)
        ttk.Button(identity_controls, text="Delete ID", width=9, command=self.delete_selected_identity).grid(row=0, column=5, sticky="ew", padx=2)
        ttk.Button(identity_controls, text="Auto Add Feature", command=self.start_auto_add_feature).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=(4, 0)
        )
        ttk.Button(
            identity_controls,
            text="Manual Add",
            command=self.add_feature_to_selected_identity,
        ).grid(row=1, column=2, columnspan=2, sticky="ew", padx=2, pady=(4, 0))
        ttk.Button(identity_controls, text="Find GID", command=self.track_selected_identity_from_db).grid(
            row=1, column=4, columnspan=2, sticky="ew", padx=2, pady=(4, 0)
        )
        ttk.Label(identity_controls, text="Auto ReID Th").grid(row=2, column=0, columnspan=2, sticky="w", padx=3, pady=(5, 0))
        threshold_entry = ttk.Entry(identity_controls, textvariable=self.auto_reid_threshold_var, width=7)
        threshold_entry.grid(row=2, column=2, sticky="ew", padx=2, pady=(5, 0))
        threshold_entry.bind("<Return>", self.apply_auto_reid_threshold)
        threshold_entry.bind("<FocusOut>", self.apply_auto_reid_threshold)
        ttk.Label(identity_controls, text="0.00-1.00").grid(row=2, column=3, columnspan=3, sticky="w", padx=3, pady=(5, 0))
        ttk.Label(identity_controls, text="Auto Feature Mode").grid(row=3, column=0, columnspan=3, sticky="w", padx=3, pady=(5, 0))
        feature_mode_box = ttk.Combobox(
            identity_controls,
            textvariable=self.auto_feature_mode_var,
            values=["Balanced", "Diverse", "Strict"],
            width=10,
            state="readonly",
        )
        feature_mode_box.grid(row=3, column=3, columnspan=3, sticky="ew", padx=3, pady=(5, 0))
        feature_mode_box.bind("<<ComboboxSelected>>", self.apply_auto_feature_mode)
        self.identity_tree = ttk.Treeview(
            identity_controls,
            columns=("gid", "type", "lid", "master", "pending", "candidate", "frame", "conf"),
            show="headings",
            height=4,
        )
        headings = {
            "gid": "GID",
            "type": "Type",
            "lid": "LID",
            "master": "Master",
            "pending": "Pending",
            "candidate": "Candidate",
            "frame": "Frame",
            "conf": "DetConf",
        }
        widths = {"gid": 38, "type": 50, "lid": 38, "master": 48, "pending": 52, "candidate": 58, "frame": 48, "conf": 44}
        for column, label in headings.items():
            self.identity_tree.heading(column, text=label)
            self.identity_tree.column(column, width=widths[column], minwidth=36, anchor="center", stretch=False)
        self.identity_tree.tag_configure("selected", background="#d7ecff")
        self.identity_tree.tag_configure("tree_selected", background="#b9dcff")
        self.identity_tree.tag_configure("no_master", background="#fff4cc")
        self.identity_tree.bind("<<TreeviewSelect>>", self.on_identity_tree_select)
        self.identity_tree.bind("<Double-1>", self.edit_identity_display_name)
        self.identity_tree.bind("<Delete>", self.delete_selected_identity)
        self.identity_tree.bind("<BackSpace>", self.delete_selected_identity)
        self.identity_tree.bind("<Motion>", self.on_identity_tree_motion)
        self.identity_tree.bind("<Leave>", self.hide_identity_preview)
        self.identity_tree.grid(row=4, column=0, columnspan=6, sticky="nw", padx=3, pady=(6, 0))
        ttk.Label(identity_controls, text="ReID Model").grid(row=5, column=0, sticky="w", padx=3, pady=(6, 0))
        self.reid_model_box = ttk.Combobox(
            identity_controls,
            textvariable=self.reid_model_var,
            values=[],
            width=18,
            state="readonly",
        )
        self.reid_model_box.grid(row=5, column=1, columnspan=3, sticky="ew", padx=3, pady=(6, 0))
        self.reid_model_box.bind("<<ComboboxSelected>>", lambda _: self.apply_reid_model_config())
        ttk.Button(identity_controls, text="Refresh Models", width=15, command=self.refresh_reid_model_options).grid(
            row=5,
            column=4,
            columnspan=2,
            sticky="ew",
            padx=3,
            pady=(6, 0),
        )
        ttk.Label(identity_controls, textvariable=self.identity_mode_var, wraplength=390).grid(
            row=6,
            column=0,
            columnspan=6,
            sticky="e",
            padx=3,
            pady=(5, 0),
        )
        for column in range(6):
            identity_controls.columnconfigure(column, weight=0)
        identity_controls.rowconfigure(4, weight=0)

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
        self.refresh_identity_db_panel()

    def apply_ui_config(self) -> None:
        self.input_config = self._ui_input_config()
        self.reframer.set_framing_mode(self.framing_var.get())

    def apply_reid_model_config(self) -> None:
        model_path = self.reid_model_options.get(
            self.reid_model_var.get(),
            str(self.config.model_dir / self.reid_model_var.get()),
        )
        self.feature_gallery.set_reid_model(model_path)
        self._set_identity_mode(f"ReID model: {self.reid_model_var.get()}")
        self.status_var.set(f"Status: ReID model set to {self.reid_model_var.get()}")

    def apply_auto_reid_threshold(self, _event=None) -> str:
        raw_value = self.auto_reid_threshold_var.get().strip()
        try:
            threshold = float(raw_value)
        except ValueError:
            threshold = self.identity_manager.auto_reid_min_score

        threshold = max(0.0, min(1.0, threshold))
        self.identity_manager.set_auto_reid_threshold(threshold)
        self.auto_reid_threshold_var.set(f"{threshold:.2f}")
        self._set_identity_mode(f"Find GID threshold: {threshold:.2f}")
        self.status_var.set(f"Status: Auto ReID threshold set to {threshold:.2f}")
        return "break"

    def apply_auto_feature_mode(self, _event=None) -> str:
        mode = self.auto_feature_mode_var.get()
        self.auto_feature_sampler.set_mode(mode)
        self.auto_feature_mode_var.set(self.auto_feature_sampler.config.mode)
        config = self.auto_feature_sampler.config
        self._set_identity_mode(
            f"Auto Feature Mode: {config.mode} "
            f"(quality {config.min_quality_score:.2f}, area {config.min_area_ratio:.3f})"
        )
        self.status_var.set(f"Status: Auto Feature Mode set to {config.mode}")
        return "break"

    def _set_identity_mode(self, message: str) -> None:
        self.identity_mode_var.set(f"Identity Mode: {message}")

    def _ui_input_config(self) -> InputConfig:
        try:
            camera_index = int(self.camera_index_var.get())
        except ValueError:
            camera_index = 0

        return InputConfig(
            source_type=self.source_var.get(),
            camera_index=camera_index,
            video_path=self.input_config.video_path,
            video_url=self._normalized_video_url(),
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
        self.refresh_identity_db_panel()

    def clear_selection(self) -> None:
        self.identity_manager.reset()
        self.auto_feature_sampler.stop()
        self.auto_feature_status_message = ""
        self.link_bbox_vehicle_id = None
        self._set_identity_mode("click bbox to select/create GID")
        self.refresh_identity_db_panel()

    def choose_video_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.source_var.set("video_file")
            self.input_config.video_path = path
            self.video_path_var.set(f"Video: {self._short_label(Path(path).name)}")

    def apply_video_url(self, _event=None) -> None:
        video_url = self._normalized_video_url()
        if video_url is None:
            self.input_config.video_url = None
            self.video_url_status_var.set("No video URL selected")
            return
        self.source_var.set("video_url")
        self.input_config.video_url = video_url
        self.video_url_status_var.set(f"URL: {self._short_label(video_url)}")

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

    def refresh_reid_model_options(self) -> None:
        asset_names = [
            "yolo26n-reid.onnx",
            "yolo26s-reid.onnx",
            "yolo26m-reid.onnx",
            "yolo26l-reid.onnx",
            "yolo26x-reid.onnx",
        ]
        options = {name: str(self.config.model_dir / name) for name in asset_names}
        if self.config.model_dir.exists():
            for path in sorted(self.config.model_dir.rglob("*-reid.onnx")):
                options[self._model_label(path)] = str(path)
        self.reid_model_options = options
        if hasattr(self, "reid_model_box"):
            self.reid_model_box.configure(values=list(self.reid_model_options.keys()))
        if self.reid_model_var.get() not in self.reid_model_options:
            self.reid_model_var.set(self.config.default_reid_model)
        self.apply_reid_model_config()

    def select_screen_region(self) -> None:
        self.pause()
        self._clear_screen_region_selection()
        screenshot = self._capture_screen_selection_background()
        screen_width = max(1, self.root.winfo_screenwidth())
        screen_height = max(1, self.root.winfo_screenheight())
        selector = tk.Toplevel(self.root)
        selector.withdraw()
        selector.title("Select screen region")
        selector.overrideredirect(True)
        selector.geometry(f"{screen_width}x{screen_height}+0+0")
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
        selector.deiconify()
        selector.lift()
        selector.focus_force()

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
        if not candidates or self.last_raw_frame is None:
            self.identity_manager.reset()
            self.refresh_identity_db_panel()
            return
        detection = self._detection_for_track(candidates[0].track_id)
        if detection is None:
            self.identity_manager.reset()
            self.refresh_identity_db_panel()
            return
        identity = self.identity_manager.select_detection(detection, self.last_raw_frame, persist=False)
        self.refresh_identity_db_panel()
        self.status_var.set(
            "Status: auto tracking local track "
            f"{identity.last_track_id if identity.last_track_id is not None else '--'} without writing Identity DB"
        )

    def on_views_resize(self, event) -> None:
        width_limit = max(160, (event.width - 24) // 2)
        height_limit = max(90, event.height - 72)
        width, height = self._fit_size_to_source_aspect(width_limit, height_limit)
        if self._set_display_size(width, height) and self.current_frame_data is not None:
            self._update_images(
                self.current_frame_data.before_frame,
                self.current_frame_data.after_frame,
            )

    def on_timeline_press(self, _event) -> None:
        self.timeline_dragging = True

    def on_timeline_drag(self, value: str) -> None:
        if self.timeline_dragging:
            self._update_timeline_label(int(float(value)))

    def on_timeline_release(self, _event) -> None:
        self.timeline_dragging = False
        if not self._is_video_source_active():
            return

        target_frame = int(self.timeline_var.get())
        if not self.detector.seek_video_frame(target_frame):
            return

        self.store.reset()
        self.identity_manager.reset()
        self.scene_cut_detector.reset()
        self.reframer.reset()
        self.skipped_frames = 0
        self._render_current_video_frame()

    def on_before_click(self, event) -> None:
        if self.last_frame_shape is None:
            return

        frame_height, frame_width = self.last_frame_shape[:2]
        image_width = max(1, self.rendered_image_width)
        image_height = max(1, self.rendered_image_height)
        if event.x < 0 or event.y < 0 or event.x > image_width or event.y > image_height:
            return
        frame_x = event.x * frame_width / image_width
        frame_y = event.y * frame_height / image_height

        candidate = self.store.get_candidate_at_point(frame_x, frame_y, self.last_frame_shape)
        if candidate is None:
            self.status_var.set("Status: no tracked vehicle at clicked point")
            self._set_identity_mode("no tracked vehicle at clicked point")
            return

        detection = self._detection_for_track(candidate.track_id)
        if detection is None or self.last_raw_frame is None:
            self.status_var.set("Status: selected candidate is no longer visible")
            return

        if self.link_bbox_vehicle_id is not None:
            vehicle_id = self.link_bbox_vehicle_id
            self.link_bbox_vehicle_id = None
            identity = self.identity_manager.link_detection(vehicle_id, detection, self.last_raw_frame)
            self.refresh_identity_db_panel()
            if identity is None:
                self.status_var.set(f"Status: vehicle id {vehicle_id} was not found for bbox link")
                self._set_identity_mode(f"GID {vehicle_id} not found")
                return
            label = self.identity_store.display_label(vehicle_id)
            self._activate_auto_feature_capture(vehicle_id, detection, self.last_raw_frame)
            self.refresh_identity_db_panel()
            self._set_identity_mode(f"linked bbox to GID {label}")
            self.status_var.set(
                f"Status: linked bbox local track {candidate.track_id} to GID {label}"
            )
            return

        identity = self.identity_manager.select_detection(detection, self.last_raw_frame)
        self.refresh_identity_db_panel()
        if identity.global_vehicle_id is not None:
            self._activate_auto_feature_capture(identity.global_vehicle_id, detection, self.last_raw_frame)
            self.refresh_identity_db_panel()
            self._set_identity_mode(f"tracking GID {identity.global_vehicle_id}")
        self.status_var.set(
            f"Status: selected global id {identity.global_vehicle_id} "
            f"(local track {candidate.track_id})"
        )

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
        inference_started_at = time()
        frame, detections = self.detector.read_and_track()
        self.last_inference_time_ms = (time() - inference_started_at) * 1000.0
        if frame is None:
            self.stop()
            return

        frame_data = self._process_frame(frame, detections, self.last_inference_time_ms)
        now = time()
        elapsed = max(1e-6, now - self.last_frame_time)
        self.fps = 1.0 / elapsed
        self.last_frame_time = now
        frame_data.display_fps = self.fps
        frame_data.source_fps = self.detector.get_source_fps()
        frame_data.skipped_frames = self.skipped_frames

        auto_feature_note = (
            f" | AutoFeat: {self.auto_feature_status_message}"
            if self.auto_feature_status_message
            else ""
        )
        self.status_var.set(
            "Status: "
            f"{frame_data.tracking_status} | Display FPS: {self.fps:.1f} | "
            f"Source: {self._source_fps_label()} | Speed: {self.playback_speed_var.get()} | "
            f"Skipped: {self.skipped_frames} | "
            f"Candidates: {len(frame_data.candidates)} | "
            f"GID: {frame_data.selected_global_vehicle_id or '--'} | "
            f"LID: {frame_data.selected_local_track_id if frame_data.selected_local_track_id is not None else '--'} | "
            f"Lost: {frame_data.lost_frames} | ReID: {frame_data.reacquire_score:.2f} | "
            f"Cut: {'yes' if frame_data.camera_cut_detected else 'no'} | "
            f"Crop: {frame_data.framing_status.crop_window}"
            f"{auto_feature_note}"
        )

        self._drop_late_video_frames()
        self._sync_timeline_from_detector()
        self.root.after(self._next_loop_delay_ms(), self._loop)

    def _process_frame(self, frame, detections, inference_time_ms: float = 0.0) -> FrameData:
        self.last_frame_shape = frame.shape
        self.last_raw_frame = frame
        self._sync_reframer_to_source_size(frame.shape)
        frame_data = self.pipeline.process(
            frame=frame,
            detections=detections,
            draw_detections=self._draw_detections,
            reset_tracker_state=self.detector.reset_tracker_state if self.detector is not None else None,
            inference_time_ms=inference_time_ms,
            source_fps=self.detector.get_source_fps() if self.detector is not None else None,
            skipped_frames=self.skipped_frames,
        )
        if frame_data.camera_cut_detected:
            self._stop_auto_feature_capture_for_scene_change()
        self._update_images(frame_data.before_frame, frame_data.after_frame)
        self.current_frame_data = frame_data
        self._run_auto_feature_sampling(frame)
        self.refresh_identity_db_panel()
        return frame_data

    def _render_current_video_frame(self) -> None:
        if self.detector is None:
            return
        frame, detections = self.detector.read_and_track()
        if frame is None:
            return
        self._process_frame(frame, detections, self.last_inference_time_ms)
        self._sync_timeline_from_detector()

    def _next_loop_delay_ms(self) -> int:
        if not self._is_video_source_active():
            return self.config.update_interval_ms

        source_fps = self.detector.get_source_fps()
        if source_fps is None:
            return self.config.update_interval_ms

        target_interval_ms = 1000.0 / max(1.0, source_fps) / self._playback_speed()
        elapsed_ms = (time() - self.loop_started_at) * 1000.0
        return max(1, int(round(target_interval_ms - elapsed_ms)))

    def _drop_late_video_frames(self) -> None:
        if not self._is_video_source_active():
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
        if not self._is_video_source_active():
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
        self.pipeline.reset()
        self.last_frame_shape = None
        self.last_raw_frame = None
        self.current_frame_data = None
        self.skipped_frames = 0
        self.link_bbox_vehicle_id = None
        self.auto_feature_sampler.stop()
        self.auto_feature_status_message = ""
        self._set_identity_mode("click bbox to select/create GID")

    def _clear_screen_region_selection(self) -> None:
        self.input_config.screen_region = None
        self.screen_region_var.set("No screen region selected")
        if self.detector is not None and self.detector.config.source_type == "screen_region":
            self._close_detector()
            self.detector = None
            self.active_input_signature = None
        self._reset_runtime_state()

    def _is_video_source_active(self) -> bool:
        return (
            self.detector is not None
            and self.input_config.source_type in {"video_file", "video_url"}
        )

    def _set_display_size(self, width: int, height: int) -> bool:
        width = max(160, min(3840, int(width)))
        height = max(90, min(2160, int(height)))
        if width == self.display_width and height == self.display_height:
            return False

        self.display_width = width
        self.display_height = height
        return True

    def _sync_reframer_to_source_size(
        self,
        frame_shape: tuple[int, int, int] | tuple[int, int],
    ) -> None:
        frame_h, frame_w = frame_shape[:2]
        self.config.output_width = frame_w
        self.config.output_height = frame_h
        self.reframer.config.output_width = frame_w
        self.reframer.config.output_height = frame_h
        width, height = self._fit_size_to_source_aspect(self.display_width, self.display_height)
        self._set_display_size(width, height)

    def refresh_identity_db_panel(self) -> None:
        if not hasattr(self, "identity_tree"):
            return
        self.refreshing_identity_panel = True
        selected_tree_ids = set(self.selected_identity_tree_ids)
        summary = self.identity_store.summary(feature_counts=self.feature_gallery.summary_by_vehicle())
        self.identity_summary_var.set(
            "Vehicles: "
            f"{summary.vehicle_count} | Master: {summary.master_feature_count} | "
            f"Pending: {summary.pending_feature_count} | Candidate: {summary.candidate_feature_count}"
        )
        for item in self.identity_tree.get_children():
            self.identity_tree.delete(item)

        selected_gid = self.identity_manager.selected_global_vehicle_id
        for vehicle in summary.vehicles:
            tags: tuple[str, ...] = ()
            if vehicle.vehicle_id in selected_tree_ids:
                tags = ("tree_selected",)
            elif vehicle.vehicle_id == selected_gid:
                tags = ("selected",)
            elif vehicle.master_feature_count <= 0:
                tags = ("no_master",)
            self.identity_tree.insert(
                "",
                "end",
                iid=str(vehicle.vehicle_id),
                values=(
                    vehicle.display_name,
                    vehicle.class_name,
                    vehicle.last_track_id if vehicle.last_track_id is not None else "--",
                    vehicle.master_feature_count,
                    vehicle.pending_feature_count,
                    vehicle.candidate_feature_count,
                    vehicle.last_frame_index,
                    f"{vehicle.confidence:.2f}",
                ),
                tags=tags,
            )
        existing_items = set(self.identity_tree.get_children())
        restore_selection = [str(vehicle_id) for vehicle_id in selected_tree_ids if str(vehicle_id) in existing_items]
        if restore_selection:
            self.identity_tree.selection_set(restore_selection)
        self.refreshing_identity_panel = False

    def on_identity_tree_select(self, _event=None) -> str:
        if self.refreshing_identity_panel:
            return "break"
        self.selected_identity_tree_ids = set(self._selected_identity_vehicle_ids())
        if self.selected_identity_tree_ids:
            vehicle_id = sorted(self.selected_identity_tree_ids)[0]
            self.link_bbox_vehicle_id = vehicle_id
            label = self.identity_store.display_label(vehicle_id)
            self._set_identity_mode(f"Link BBox to GID {label}; click a visible bbox")
        else:
            self.link_bbox_vehicle_id = None
            self._set_identity_mode("click bbox to select/create GID")
        selected_gid = self.identity_manager.selected_global_vehicle_id
        for item in self.identity_tree.get_children():
            try:
                vehicle_id = int(item)
                master_count = int(self.identity_tree.set(item, "master") or 0)
            except ValueError:
                continue

            tags: tuple[str, ...] = ()
            if vehicle_id in self.selected_identity_tree_ids:
                tags = ("tree_selected",)
            elif vehicle_id == selected_gid:
                tags = ("selected",)
            elif master_count <= 0:
                tags = ("no_master",)
            self.identity_tree.item(item, tags=tags)
        return "break"

    def on_identity_tree_motion(self, event) -> None:
        item = self.identity_tree.identify_row(event.y)
        if not item:
            self.hide_identity_preview()
            return
        try:
            vehicle_id = int(item)
        except ValueError:
            self.hide_identity_preview()
            return
        self.show_identity_preview(vehicle_id, event.x_root + 18, event.y_root + 12)

    def show_identity_preview(self, vehicle_id: int, screen_x: int, screen_y: int) -> None:
        if Image is None or ImageTk is None:
            return
        if self.identity_preview_vehicle_id == vehicle_id and self.identity_preview_window is not None:
            self.identity_preview_window.geometry(f"+{screen_x}+{screen_y}")
            return

        crop_jpeg = self.feature_gallery.first_feature_crop_jpeg(vehicle_id)
        if crop_jpeg is None:
            self.hide_identity_preview()
            return

        try:
            image = Image.open(BytesIO(crop_jpeg)).convert("RGB")
        except Exception:
            self.hide_identity_preview()
            return

        image.thumbnail((220, 150))
        photo = ImageTk.PhotoImage(image)
        if self.identity_preview_window is None:
            self.identity_preview_window = tk.Toplevel(self.root)
            self.identity_preview_window.overrideredirect(True)
            self.identity_preview_window.attributes("-topmost", True)
            self.identity_preview_label = ttk.Label(self.identity_preview_window, padding=4)
            self.identity_preview_label.pack()

        assert self.identity_preview_label is not None
        self.identity_preview_photo = photo
        self.identity_preview_vehicle_id = vehicle_id
        self.identity_preview_label.configure(image=photo)
        self.identity_preview_window.geometry(f"+{screen_x}+{screen_y}")
        self.identity_preview_window.deiconify()

    def hide_identity_preview(self, _event=None) -> None:
        self.identity_preview_vehicle_id = None
        if self.identity_preview_window is not None:
            self.identity_preview_window.withdraw()

    def track_selected_identity_from_db(self, _event=None) -> str:
        if self.refreshing_identity_panel:
            return "break"

        vehicle_ids = self._selected_identity_vehicle_ids()
        if not vehicle_ids:
            self._set_identity_mode("select a GID row before Find GID")
            return "break"
        vehicle_id = vehicle_ids[0]

        if self.last_raw_frame is None:
            self._set_identity_mode("Find GID waiting for current frame")
            self.status_var.set("Status: no current frame available for DB identity tracking")
            return "break"

        identity, score = self.identity_manager.select_stored_vehicle(
            vehicle_id,
            self.store.current_detections,
            self.last_raw_frame,
            min_score=self.identity_manager.auto_reid_min_score,
        )
        self.refresh_identity_db_panel()
        if identity is None:
            self._set_identity_mode(f"GID {vehicle_id} was not found")
            self.status_var.set(f"Status: vehicle id {vehicle_id} was not found in Identity DB")
            return "break"

        label = self.identity_store.display_label(vehicle_id)
        if identity.last_track_id is None:
            self._set_identity_mode(f"Find GID searching for GID {label}")
            self.status_var.set(
                f"Status: no Master feature match for GID {label}; searching "
                f"(score {score:.2f})"
            )
        else:
            self._set_identity_mode(f"Find GID tracking GID {label}")
            self.status_var.set(
                f"Status: tracking GID {label} on local track {identity.last_track_id} "
                f"(score {score:.2f})"
            )
            if self.current_frame_data is not None:
                self._update_images(
                    self._draw_detections(self.last_raw_frame, self.store.current_detections),
                    self.current_frame_data.after_frame,
                )
        return "break"

    def start_link_bbox(self) -> str:
        vehicle_ids = self._selected_identity_vehicle_ids()
        if not vehicle_ids:
            self.status_var.set("Status: select a GID before Link BBox")
            return "break"
        self.link_bbox_vehicle_id = vehicle_ids[0]
        label = self.identity_store.display_label(self.link_bbox_vehicle_id)
        self._set_identity_mode(f"Link BBox to GID {label}; click a visible bbox")
        self.status_var.set(f"Status: click a visible bbox to link it to GID {label}")
        return "break"

    def start_auto_add_feature(self) -> str:
        vehicle_ids = self._selected_identity_vehicle_ids()
        vehicle_id = vehicle_ids[0] if vehicle_ids else self.identity_manager.selected_global_vehicle_id
        if vehicle_id is None:
            self._set_identity_mode("select a GID before Auto Add Feature")
            self.status_var.set("Status: select a GID before Auto Add Feature")
            return "break"
        if self.last_raw_frame is None:
            self._set_identity_mode("Auto Add Feature waiting for current frame")
            self.status_var.set("Status: no current frame available for Auto Add Feature")
            return "break"

        detection = self._detection_for_vehicle_id(vehicle_id)
        if detection is None:
            self._set_identity_mode("Auto Add Feature needs visible linked bbox")
            self.status_var.set("Status: Link BBox to a visible vehicle before Auto Add Feature")
            return "break"

        result = self._activate_auto_feature_capture(vehicle_id, detection, self.last_raw_frame)
        self.refresh_identity_db_panel()
        label = self.identity_store.display_label(vehicle_id)
        if result.accepted:
            self._set_identity_mode(f"Auto Add Feature active for GID {label}")
            self.status_var.set(
                f"Status: auto feature capture active for GID {label}; "
                f"added master feature {result.feature_id} (quality {result.quality_score:.2f})"
            )
        else:
            self._set_identity_mode(f"Auto Add Feature waiting for clean GID {label} crop")
            self.status_var.set(
                f"Status: auto feature capture active for GID {label}; first sample rejected: {result.reason}"
            )
        return "break"

    def _activate_auto_feature_capture(self, vehicle_id: int, detection, frame):
        result = self.auto_feature_sampler.start(vehicle_id, detection, frame, self.store)
        label = self.identity_store.display_label(vehicle_id)
        if result.accepted:
            self.auto_feature_status_message = f"GID {label} added {result.feature_id}"
        else:
            self.auto_feature_status_message = f"GID {label} active; waiting for clean crop"
        return result

    def _run_auto_feature_sampling(self, frame) -> None:
        vehicle_id = self.auto_feature_sampler.active_vehicle_id
        if vehicle_id is None:
            return
        detection = self._detection_for_vehicle_id(vehicle_id)
        result = self.auto_feature_sampler.update(detection, frame, self.store)
        if not result.accepted:
            return
        label = self.identity_store.display_label(vehicle_id)
        self.auto_feature_status_message = f"GID {label} added {result.feature_id} q{result.quality_score:.2f}"
        self._set_identity_mode(f"Auto Add Feature added feature {result.feature_id} to GID {label}")

    def _stop_auto_feature_capture_for_scene_change(self) -> None:
        vehicle_id = self.auto_feature_sampler.active_vehicle_id
        if vehicle_id is None:
            return
        label = self.identity_store.display_label(vehicle_id)
        self.auto_feature_sampler.stop()
        self.auto_feature_status_message = ""
        self._set_identity_mode(f"camera changed; Auto Add Feature stopped for GID {label}")

    def add_feature_to_selected_identity(self) -> str:
        vehicle_ids = self._selected_identity_vehicle_ids()
        vehicle_id = vehicle_ids[0] if vehicle_ids else self.identity_manager.selected_global_vehicle_id
        if vehicle_id is None:
            self.status_var.set("Status: select a GID before Add Feature")
            return "break"
        if self.last_raw_frame is None:
            self.status_var.set("Status: no current frame available for Add Feature")
            return "break"

        detection = self._detection_for_vehicle_id(vehicle_id)
        if detection is None:
            self.status_var.set("Status: Link BBox to a visible vehicle before Add Feature")
            return "break"

        result = self.feature_gallery.add_master_feature(vehicle_id, detection, self.last_raw_frame)
        self.refresh_identity_db_panel()
        label = self.identity_store.display_label(vehicle_id)
        if result.accepted:
            self.status_var.set(
                f"Status: added master feature {result.feature_id} to GID {label} "
                f"(quality {result.quality.score:.2f})"
            )
            return "break"
        duplicate = (
            f", duplicate {result.duplicate_score:.3f}"
            if result.duplicate_score is not None
            else ""
        )
        self.status_var.set(
            f"Status: rejected Add Feature for GID {label}: {result.reason}{duplicate}"
        )
        return "break"

    def edit_identity_display_name(self, event) -> str:
        if self.identity_tree.identify_column(event.x) != "#1":
            return "break"

        item = self.identity_tree.identify_row(event.y)
        if not item:
            return "break"
        try:
            vehicle_id = int(item)
        except ValueError:
            return "break"

        current_name = self.identity_tree.set(item, "gid")
        new_name = simpledialog.askstring(
            "Edit GID",
            "Vehicle ID label:",
            initialvalue=current_name,
            parent=self.root,
        )
        if new_name is None:
            return "break"

        if self.identity_store.update_display_name(vehicle_id, new_name):
            self.refresh_identity_db_panel()
            label = self.identity_store.display_label(vehicle_id)
            self._set_identity_mode(f"renamed GID {vehicle_id} to {label}")
            self.status_var.set(f"Status: renamed vehicle id {vehicle_id} to {label}")
        return "break"

    def delete_selected_identity(self, _event=None) -> str:
        if not hasattr(self, "identity_tree"):
            return "break"

        vehicle_ids = self._selected_identity_vehicle_ids()
        if not vehicle_ids:
            self._set_identity_mode("select an Identity DB row before deleting")
            self.status_var.set("Status: select an Identity DB row before deleting")
            return "break"

        deleted_ids: list[int] = []
        for vehicle_id in vehicle_ids:
            if self.identity_store.delete_vehicle(vehicle_id):
                self.feature_gallery.delete_vehicle_features(vehicle_id)
                deleted_ids.append(vehicle_id)

        if self.identity_manager.selected_global_vehicle_id in deleted_ids:
            self.identity_manager.reset()
        if self.link_bbox_vehicle_id in deleted_ids:
            self.link_bbox_vehicle_id = None

        self.refresh_identity_db_panel()
        if deleted_ids:
            ids = ", ".join(str(vehicle_id) for vehicle_id in deleted_ids)
            self._set_identity_mode(f"deleted GID {ids}")
            self.status_var.set(f"Status: deleted vehicle id {ids}")
        else:
            self._set_identity_mode("selected GID was already deleted")
            self.status_var.set("Status: selected vehicle id was already deleted")
        return "break"

    def _selected_identity_vehicle_ids(self) -> list[int]:
        if not hasattr(self, "identity_tree"):
            return []
        vehicle_ids: list[int] = []
        for item in self.identity_tree.selection():
            try:
                vehicle_ids.append(int(item))
            except ValueError:
                continue
        return vehicle_ids

    def _fit_size_to_source_aspect(self, width_limit: int, height_limit: int) -> tuple[int, int]:
        if self.last_frame_shape is not None:
            frame_h, frame_w = self.last_frame_shape[:2]
        else:
            frame_w, frame_h = self.config.output_width, self.config.output_height
        aspect = max(1, frame_w) / max(1, frame_h)
        width = max(160, int(width_limit))
        height = max(90, int(round(width / aspect)))
        if height > height_limit:
            height = max(90, int(height_limit))
            width = max(160, int(round(height * aspect)))
        return width, height

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
        clear_temp_cache = self.detector.config.source_type in {"video_file", "video_url"}
        self.detector.close(clear_temp_cache=clear_temp_cache)

    def on_close(self) -> None:
        self.running = False
        if self.identity_preview_window is not None:
            self.identity_preview_window.destroy()
            self.identity_preview_window = None
        if self.detector is not None:
            self._close_detector()
            self.detector = None
        self.feature_gallery.close()
        self.identity_store.close()
        self.root.destroy()

    @staticmethod
    def _input_signature(config: InputConfig) -> tuple[object, ...]:
        return (
            config.source_type,
            config.camera_index,
            config.video_path,
            config.video_url,
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

    def _normalized_video_url(self) -> str | None:
        value = self.video_url_var.get().strip()
        return value or None

    def _detection_for_track(self, track_id: int):
        for detection in self.store.current_detections:
            if detection.track_id == track_id:
                return detection
        return None

    def _detection_for_vehicle_id(self, vehicle_id: int):
        identity = self.identity_manager.selected_identity
        if (
            identity is not None
            and identity.global_vehicle_id == vehicle_id
            and identity.last_track_id is not None
        ):
            detection = self._detection_for_track(identity.last_track_id)
            if detection is not None:
                return detection

        stored = self.identity_store.get_vehicle(vehicle_id)
        if stored is not None and stored.last_track_id is not None:
            return self._detection_for_track(stored.last_track_id)
        return None

    def _draw_detections(self, frame, detections):
        import cv2

        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(value) for value in detection.bbox]
            global_id = self.identity_manager.global_id_for_detection(detection)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            selected_gid = self.identity_manager.selected_global_vehicle_id
            is_selected = global_id is not None and global_id == selected_gid
            box_color = (0, 0, 255) if is_selected else (80, 220, 80)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 4 if is_selected else 3)

            gid_height = 50
            lid_height = 30
            gid_scale = cv2.getFontScaleFromHeight(font_face, gid_height, 3)
            lid_scale = cv2.getFontScaleFromHeight(font_face, lid_height, 2)
            text_x = max(0, x1)
            gid_y = max(gid_height + 4, y1 - lid_height - 14)
            lid_y = max(gid_height + lid_height + 10, y1 - 6)

            if global_id is not None:
                cv2.putText(
                    annotated,
                    str(global_id),
                    (text_x, gid_y),
                    font_face,
                    gid_scale,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
            cv2.putText(
                annotated,
                f"LID {detection.track_id}",
                (text_x, lid_y),
                font_face,
                lid_scale,
                box_color,
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
        self.rendered_image_width, self.rendered_image_height = size

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
