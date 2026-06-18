from __future__ import annotations

import queue
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from simulator.backend import (
    CancelledError,
    IntegratedAnalyzer,
    build_output_root,
    format_seconds,
    list_video_files,
    next_table_id,
    read_first_frame,
    save_image,
)
from simulator.models import AnalysisSessionResult, TableBox


def representative_temporal_frame_indices(frames: list[Any]) -> list[int]:
    if not frames:
        return []
    if len(frames) <= 1:
        return [0]

    indices = {0, len(frames) - 1}
    previous_frame_state = str(getattr(frames[0], "frame_state", ""))
    previous_final_state = str(getattr(frames[0], "final_state", ""))
    for index, frame in enumerate(frames[1:], start=1):
        current_frame_state = str(getattr(frame, "frame_state", ""))
        current_final_state = str(getattr(frame, "final_state", ""))
        if current_frame_state != previous_frame_state or current_final_state != previous_final_state:
            indices.add(index)
        previous_frame_state = current_frame_state
        previous_final_state = current_final_state
    return sorted(indices)


class SimulatorApp(tk.Tk):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.video_root = project_root / "video"
        self.output_root = build_output_root(project_root)
        self.analyzer = IntegratedAnalyzer()

        self.title("Restaurant Table Cleanliness Simulator")
        self.geometry("1640x980")
        self.minsize(1320, 840)
        self.configure(bg="#f2f4f7")

        self.queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.cancel_event = threading.Event()

        self.video_files = list_video_files(self.video_root)
        self.selected_video: Path | None = None
        self.selected_video_item: str | None = None
        self.first_frame_array = None
        self.first_frame_path: Path | None = None
        self.first_frame_size = (1, 1)
        self.display_photo: ImageTk.PhotoImage | None = None
        self.stage2_photo: ImageTk.PhotoImage | None = None
        self.stage1_photo: ImageTk.PhotoImage | None = None
        self.boxes: list[TableBox] = []
        self.undo_stack: list[list[TableBox]] = []
        self.redo_stack: list[list[TableBox]] = []
        self.selected_box_id: str | None = None
        self.edit_mode = "view"
        self.drag_start: tuple[int, int] | None = None
        self.preview_rect: int | None = None
        self.analysis_result: AnalysisSessionResult | None = None
        self.stage2_scale = 1.0
        self.stage2_offset = (0, 0)
        self.processing_complete = False
        self.result_photos: list[ImageTk.PhotoImage] = []
        self.font_family = self._choose_font_family()
        self._configure_application_fonts()

        self._build_style()
        self._build_layout()
        self._load_video_tree()
        self.after(120, self._poll_queue)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#f2f4f7")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#f2f4f7", foreground="#111827", font=(self.font_family, 18, "bold"))
        style.configure("Sub.TLabel", background="#ffffff", foreground="#374151", font=(self.font_family, 11))
        style.configure("MethodTitle.TLabel", background="#ffffff", foreground="#111827", font=(self.font_family, 15, "bold"))
        style.configure("Metric.TLabel", background="#ffffff", foreground="#111827", font=(self.font_family, 26, "bold"))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#6b7280", font=(self.font_family, 10))
        style.configure("Caption.TLabel", background="#ffffff", foreground="#4b5563", font=(self.font_family, 9))
        style.configure("Evidence.TLabel", background="#ffffff", foreground="#374151", font=(self.font_family, 10))
        style.configure("Headline.TLabel", background="#ffffff", foreground="#111827", font=(self.font_family, 10, "bold"))
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("Action.TButton", font=(self.font_family, 12, "bold"), padding=(12, 8))
        style.configure("Green.Action.TButton", foreground="#ffffff", background="#16a34a")
        style.map("Green.Action.TButton", background=[("active", "#15803d")])
        style.configure("Red.Action.TButton", foreground="#ffffff", background="#dc2626")
        style.map("Red.Action.TButton", background=[("active", "#b91c1c")])
        style.configure("Tool.TButton", font=(self.font_family, 11), padding=(10, 6))
        style.configure("Stage.TLabel", background="#f2f4f7", foreground="#6b7280", font=(self.font_family, 11, "bold"))
        style.configure("Treeview", rowheight=28, font=(self.font_family, 11))
        style.configure("Treeview.Heading", font=(self.font_family, 11, "bold"))

    def _choose_font_family(self) -> str:
        preferred = [
            "DejaVu Sans",
            "Liberation Sans",
            "Arial",
            "Helvetica",
        ]
        available = set(tkfont.families(self))
        matched = next((font for font in preferred if font in available), "")
        if matched:
            return matched
        return "DejaVu Sans"

    def _configure_application_fonts(self) -> None:
        try:
            self.tk.call("tk", "scaling", 1.1)
        except tk.TclError:
            pass

        named_fonts = {
            "TkDefaultFont": (10, "normal"),
            "TkTextFont": (10, "normal"),
            "TkMenuFont": (10, "normal"),
            "TkHeadingFont": (11, "bold"),
            "TkTooltipFont": (9, "normal"),
        }
        for name, (size, weight) in named_fonts.items():
            try:
                tkfont.nametofont(name).configure(family=self.font_family, size=size, weight=weight)
            except tk.TclError:
                continue

        self.option_add("*Font", (self.font_family, 10))
        self.option_add("*Text.Font", (self.font_family, 10))
        self.option_add("*Listbox.Font", (self.font_family, 10))

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="App.TFrame", padding=16)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root, style="App.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Integrated Restaurant Cleanliness Simulator", style="Title.TLabel").pack(side="left")

        self.stage_label = ttk.Label(header, text="1. Video Select", style="Stage.TLabel")
        self.stage_label.pack(side="right")

        self.stage_host = ttk.Frame(root, style="App.TFrame")
        self.stage_host.pack(fill="both", expand=True, pady=(12, 0))

        self.stage_frames: list[ttk.Frame] = []
        for _ in range(4):
            frame = ttk.Frame(self.stage_host, style="App.TFrame")
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.stage_frames.append(frame)

        self._build_stage1(self.stage_frames[0])
        self._build_stage2(self.stage_frames[1])
        self._build_stage3(self.stage_frames[2])
        self._build_stage4(self.stage_frames[3])

        self._show_stage(0)

    def _build_stage1(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        panel.pack(fill="both", expand=True)

        toolbar = ttk.Frame(panel, style="Panel.TFrame")
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Video Explorer", style="Sub.TLabel").pack(side="left")
        self.stage1_next_button = ttk.Button(
            toolbar,
            text="Next",
            style="Green.Action.TButton",
            command=lambda: self._show_stage(1),
            state="disabled",
        )
        self.stage1_next_button.pack(side="right")

        body = ttk.Panedwindow(panel, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(12, 0))

        left = ttk.Frame(body, style="Panel.TFrame")
        right = ttk.Frame(body, style="Panel.TFrame")
        body.add(left, weight=1)
        body.add(right, weight=4)

        self.video_tree = ttk.Treeview(left, columns=("path",), show="tree")
        self.video_tree.pack(fill="both", expand=True)
        self.video_tree.bind("<<TreeviewSelect>>", self._on_video_selected)

        preview_header = ttk.Frame(right, style="Panel.TFrame")
        preview_header.pack(fill="x")
        self.preview_title = ttk.Label(preview_header, text="Select a video from ./video", style="Sub.TLabel")
        self.preview_title.pack(side="left")

        self.stage1_canvas = tk.Canvas(right, bg="#111827", highlightthickness=0)
        self.stage1_canvas.pack(fill="both", expand=True, pady=(8, 0))
        self.stage1_canvas.bind("<Configure>", lambda _event: self._render_stage1_preview())

    def _build_stage2(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        panel.pack(fill="both", expand=True)

        toolbar = ttk.Frame(panel, style="Panel.TFrame")
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Back", style="Tool.TButton", command=lambda: self._show_stage(0)).pack(side="left")
        ttk.Button(toolbar, text="Pointer", style="Tool.TButton", command=lambda: self._set_edit_mode("view")).pack(side="left", padx=(12, 4))
        ttk.Button(toolbar, text="+ Add", style="Tool.TButton", command=lambda: self._set_edit_mode("add")).pack(side="left", padx=4)
        ttk.Button(toolbar, text="- Remove", style="Tool.TButton", command=lambda: self._set_edit_mode("remove")).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Auto Detect", style="Tool.TButton", command=self._auto_detect_tables).pack(side="left", padx=(12, 4))
        ttk.Button(toolbar, text="Undo", style="Tool.TButton", command=self._undo_boxes).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Redo", style="Tool.TButton", command=self._redo_boxes).pack(side="left", padx=4)

        ttk.Label(toolbar, text="Frame Interval (s)", style="Sub.TLabel").pack(side="right", padx=(0, 6))
        self.interval_var = tk.StringVar(value="10")
        ttk.Entry(toolbar, width=8, textvariable=self.interval_var).pack(side="right", padx=(0, 12))

        self.stage2_next_button = ttk.Button(
            toolbar,
            text="Next",
            style="Green.Action.TButton",
            command=self._start_processing_stage,
            state="disabled",
        )
        self.stage2_next_button.pack(side="right")

        self.stage2_status = ttk.Label(panel, text="Use auto-detect or draw red boxes for each table.", style="Sub.TLabel")
        self.stage2_status.pack(fill="x", pady=(10, 6))

        self.stage2_canvas = tk.Canvas(panel, bg="#0f172a", highlightthickness=0, cursor="crosshair")
        self.stage2_canvas.pack(fill="both", expand=True)
        self.stage2_canvas.bind("<Configure>", lambda _event: self._render_stage2_canvas())
        self.stage2_canvas.bind("<ButtonPress-1>", self._on_stage2_mouse_down)
        self.stage2_canvas.bind("<B1-Motion>", self._on_stage2_mouse_drag)
        self.stage2_canvas.bind("<ButtonRelease-1>", self._on_stage2_mouse_up)

    def _build_stage3(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        panel.pack(fill="both", expand=True)

        toolbar = ttk.Frame(panel, style="Panel.TFrame")
        toolbar.pack(fill="x")
        self.stage3_title = ttk.Label(toolbar, text="Processing pipelines", style="Sub.TLabel")
        self.stage3_title.pack(side="left")
        self.stage3_action_button = ttk.Button(toolbar, text="Stop", style="Red.Action.TButton", command=self._cancel_processing)
        self.stage3_action_button.pack(side="right")

        progress_grid = ttk.Frame(panel, style="Panel.TFrame")
        progress_grid.pack(fill="x", pady=(16, 12))
        progress_grid.columnconfigure(1, weight=1)

        ttk.Label(progress_grid, text="YOLO + LLM", style="Sub.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 6))
        self.object_progress = ttk.Progressbar(progress_grid, maximum=100)
        self.object_progress.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        self.object_progress_text = ttk.Label(progress_grid, text="0 / 0", style="Sub.TLabel")
        self.object_progress_text.grid(row=0, column=2, sticky="e", padx=(12, 0), pady=(0, 6))

        ttk.Label(progress_grid, text="Action / VLM", style="Sub.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 12))
        self.temporal_progress = ttk.Progressbar(progress_grid, maximum=100)
        self.temporal_progress.grid(row=1, column=1, sticky="ew")
        self.temporal_progress_text = ttk.Label(progress_grid, text="0 / 0", style="Sub.TLabel")
        self.temporal_progress_text.grid(row=1, column=2, sticky="e", padx=(12, 0))

        self.log_text = tk.Text(panel, wrap="word", bg="#111827", fg="#d1d5db", font=("Consolas", 11), height=28)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _build_stage4(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        panel.pack(fill="both", expand=True)

        toolbar = ttk.Frame(panel, style="Panel.TFrame")
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Start Over", style="Tool.TButton", command=self._restart_workflow).pack(side="left")
        self.result_summary_label = ttk.Label(toolbar, text="No results yet.", style="Sub.TLabel")
        self.result_summary_label.pack(side="left", fill="x", expand=True, padx=(12, 0))

        body = ttk.Panedwindow(panel, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(12, 0))
        yolo_panel = ttk.Frame(body, style="Panel.TFrame", padding=(0, 0, 8, 0))
        action_panel = ttk.Frame(body, style="Panel.TFrame", padding=(8, 0, 0, 0))
        body.add(yolo_panel, weight=1)
        body.add(action_panel, weight=1)

        self.yolo_results_inner = self._build_result_column(yolo_panel, "YOLO + LLM", "object evidence")
        self.action_results_inner = self._build_result_column(action_panel, "Action + VLM", "timeline evidence")

    def _build_result_column(self, parent: ttk.Frame, title: str, subtitle: str) -> ttk.Frame:
        header = ttk.Frame(parent, style="Panel.TFrame")
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text=title, style="MethodTitle.TLabel").pack(side="left")
        ttk.Label(header, text=subtitle, style="Muted.TLabel").pack(side="right")

        canvas = tk.Canvas(parent, bg="#ffffff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Panel.TFrame")
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        inner.bind("<Configure>", on_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return inner

    def _show_stage(self, index: int) -> None:
        labels = ["1. Video Select", "2. Table Edit", "3. Processing", "4. Result"]
        self.stage_frames[index].tkraise()
        self.stage_label.configure(text=labels[index])
        if index == 1:
            self._prepare_stage2()
        elif index == 3:
            self._populate_results()

    def _load_video_tree(self) -> None:
        self.video_tree.delete(*self.video_tree.get_children())
        root_item = self.video_tree.insert("", "end", text=str(self.video_root.name), open=True, values=(str(self.video_root),))
        nodes: dict[Path, str] = {self.video_root: root_item}
        for path in self.video_files:
            relative_parts = path.relative_to(self.video_root).parts
            current_path = self.video_root
            parent = root_item
            for part in relative_parts[:-1]:
                current_path = current_path / part
                if current_path not in nodes:
                    nodes[current_path] = self.video_tree.insert(parent, "end", text=part, open=True, values=(str(current_path),))
                parent = nodes[current_path]
            self.video_tree.insert(parent, "end", text=relative_parts[-1], values=(str(path),))

    def _on_video_selected(self, _event: Any) -> None:
        selection = self.video_tree.selection()
        if not selection:
            return
        item = selection[0]
        path_text = self.video_tree.item(item, "values")
        if not path_text:
            return
        path = Path(path_text[0])
        if path.is_dir() or path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
            return
        self.selected_video = path
        self.selected_video_item = item
        try:
            self.first_frame_array = read_first_frame(path)
            self.first_frame_size = (self.first_frame_array.shape[1], self.first_frame_array.shape[0])
            preview_dir = self.output_root / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            self.first_frame_path = save_image(preview_dir / f"{path.stem}_first_frame.png", self.first_frame_array)
            self.preview_title.configure(text=f"Selected video: {path.name}")
            self.stage1_next_button.configure(state="normal")
            self.stage2_status.configure(text=f"Loaded {path.name}. Detect or edit table boxes.")
            self._render_stage1_preview()
        except Exception as exc:
            messagebox.showerror("Video Load Error", str(exc))

    def _render_stage1_preview(self) -> None:
        if self.first_frame_array is None:
            return
        self.stage1_canvas.delete("all")
        image = Image.fromarray(cv2_to_rgb(self.first_frame_array))
        canvas_w = max(1, self.stage1_canvas.winfo_width())
        canvas_h = max(1, self.stage1_canvas.winfo_height())
        fitted = fit_image(image, canvas_w, canvas_h)
        self.stage1_photo = ImageTk.PhotoImage(fitted)
        self.stage1_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.stage1_photo)

    def _prepare_stage2(self) -> None:
        if self.first_frame_array is None:
            return
        self._render_stage2_canvas()
        self._update_stage2_buttons()

    def _render_stage2_canvas(self) -> None:
        if self.first_frame_array is None:
            return
        self.stage2_canvas.delete("all")
        image = Image.fromarray(cv2_to_rgb(self.first_frame_array))
        canvas_w = max(1, self.stage2_canvas.winfo_width())
        canvas_h = max(1, self.stage2_canvas.winfo_height())
        fitted, scale, offset = fit_image_with_geometry(image, canvas_w, canvas_h)
        self.stage2_scale = scale
        self.stage2_offset = offset
        self.stage2_photo = ImageTk.PhotoImage(fitted)
        self.stage2_canvas.create_image(offset[0], offset[1], image=self.stage2_photo, anchor="nw")
        for box in self.boxes:
            self._draw_box(box)

    def _draw_box(self, box: TableBox) -> None:
        x1, y1 = self._image_to_canvas(box.x1, box.y1)
        x2, y2 = self._image_to_canvas(box.x2, box.y2)
        width = 3 if box.table_id == self.selected_box_id else 2
        self.stage2_canvas.create_rectangle(x1, y1, x2, y2, outline="#ef4444", width=width)
        self.stage2_canvas.create_text(x1 + 8, y1 + 8, text=box.table_id, fill="#ef4444", anchor="nw", font=("Helvetica", 11, "bold"))

    def _image_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        offset_x, offset_y = self.stage2_offset
        return int(round(offset_x + x * self.stage2_scale)), int(round(offset_y + y * self.stage2_scale))

    def _canvas_to_image(self, x: int, y: int) -> tuple[int, int]:
        offset_x, offset_y = self.stage2_offset
        image_x = int(round((x - offset_x) / max(0.001, self.stage2_scale)))
        image_y = int(round((y - offset_y) / max(0.001, self.stage2_scale)))
        width, height = self.first_frame_size
        return max(0, min(image_x, width - 1)), max(0, min(image_y, height - 1))

    def _set_edit_mode(self, mode: str) -> None:
        self.edit_mode = mode
        self.stage2_status.configure(text=f"Mode: {mode}. {'Drag to add boxes.' if mode == 'add' else 'Click a red box to remove it.' if mode == 'remove' else 'Preview and inspect current boxes.'}")

    def _push_history(self) -> None:
        self.undo_stack.append(deepcopy(self.boxes))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _undo_boxes(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(deepcopy(self.boxes))
        self.boxes = self.undo_stack.pop()
        self.selected_box_id = None
        self._render_stage2_canvas()
        self._update_stage2_buttons()

    def _redo_boxes(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(deepcopy(self.boxes))
        self.boxes = self.redo_stack.pop()
        self.selected_box_id = None
        self._render_stage2_canvas()
        self._update_stage2_buttons()

    def _auto_detect_tables(self) -> None:
        if self.first_frame_path is None:
            return
        try:
            detected = self.analyzer.auto_detect_tables(self.first_frame_path)
        except Exception as exc:
            messagebox.showerror("Auto Detect Error", str(exc))
            return
        self._push_history()
        self.boxes = detected
        self.selected_box_id = None
        self._render_stage2_canvas()
        self._update_stage2_buttons()
        self.stage2_status.configure(text=f"Auto-detected {len(detected)} tables. Adjust boxes if needed.")

    def _on_stage2_mouse_down(self, event: tk.Event) -> None:
        if self.first_frame_array is None:
            return
        if self.edit_mode == "add":
            self.drag_start = self._canvas_to_image(event.x, event.y)
            if self.preview_rect is not None:
                self.stage2_canvas.delete(self.preview_rect)
                self.preview_rect = None
        elif self.edit_mode == "remove":
            image_point = self._canvas_to_image(event.x, event.y)
            target = self._find_box_at(image_point)
            if target is not None:
                self._push_history()
                self.boxes = [box for box in self.boxes if box.table_id != target.table_id]
                self.selected_box_id = None
                self._render_stage2_canvas()
                self._update_stage2_buttons()
        else:
            image_point = self._canvas_to_image(event.x, event.y)
            target = self._find_box_at(image_point)
            self.selected_box_id = target.table_id if target else None
            self._render_stage2_canvas()

    def _on_stage2_mouse_drag(self, event: tk.Event) -> None:
        if self.edit_mode != "add" or self.drag_start is None:
            return
        start_x, start_y = self._image_to_canvas(*self.drag_start)
        if self.preview_rect is not None:
            self.stage2_canvas.delete(self.preview_rect)
        self.preview_rect = self.stage2_canvas.create_rectangle(start_x, start_y, event.x, event.y, outline="#f59e0b", width=2, dash=(4, 4))

    def _on_stage2_mouse_up(self, event: tk.Event) -> None:
        if self.edit_mode != "add" or self.drag_start is None:
            return
        end_point = self._canvas_to_image(event.x, event.y)
        x1 = min(self.drag_start[0], end_point[0])
        y1 = min(self.drag_start[1], end_point[1])
        x2 = max(self.drag_start[0], end_point[0])
        y2 = max(self.drag_start[1], end_point[1])
        self.drag_start = None
        if self.preview_rect is not None:
            self.stage2_canvas.delete(self.preview_rect)
            self.preview_rect = None
        if x2 - x1 < 12 or y2 - y1 < 12:
            return
        self._push_history()
        new_box = TableBox(next_table_id(self.boxes), x1, y1, x2, y2)
        self.boxes.append(new_box)
        self.boxes.sort(key=lambda item: item.table_id)
        self.selected_box_id = new_box.table_id
        self._render_stage2_canvas()
        self._update_stage2_buttons()

    def _find_box_at(self, point: tuple[int, int]) -> TableBox | None:
        x, y = point
        for box in reversed(self.boxes):
            if box.contains(x, y):
                return box
        return None

    def _update_stage2_buttons(self) -> None:
        self.stage2_next_button.configure(state="normal" if self.selected_video and self.boxes else "disabled")

    def _validate_interval(self) -> float | None:
        try:
            value = float(self.interval_var.get().strip())
        except ValueError:
            value = -1
        if value <= 0:
            messagebox.showerror("Invalid Interval", "Frame interval must be a positive number.")
            return None
        return value

    def _start_processing_stage(self) -> None:
        interval = self._validate_interval()
        if interval is None or self.selected_video is None:
            return
        self.processing_complete = False
        self.analysis_result = None
        self._reset_stage3()
        self._show_stage(2)
        self.cancel_event.clear()
        self.worker_thread = threading.Thread(target=self._run_processing, args=(interval,), daemon=True)
        self.worker_thread.start()

    def _reset_stage3(self) -> None:
        self.object_progress["value"] = 0
        self.temporal_progress["value"] = 0
        self.object_progress_text.configure(text="0 / 0")
        self.temporal_progress_text.configure(text="0 / 0")
        self.stage3_action_button.configure(text="Stop", style="Red.Action.TButton", command=self._cancel_processing)
        self._set_log_text("")

    def _run_processing(self, interval: float) -> None:
        assert self.selected_video is not None
        try:
            result = self.analyzer.run(
                video_path=self.selected_video,
                tables=self.boxes,
                frame_interval_seconds=interval,
                output_root=self.output_root,
                progress_callback=self.queue.put,
                cancel_event=self.cancel_event,
            )
            self.queue.put({"type": "done", "result": result})
        except CancelledError as exc:
            self.queue.put({"type": "cancelled", "message": str(exc)})
        except Exception as exc:
            self.queue.put({"type": "error", "message": str(exc)})

    def _cancel_processing(self) -> None:
        self.cancel_event.set()
        self.queue.put({"type": "log", "message": "Cancellation requested. Waiting for active tasks to stop..."})

    def _poll_queue(self) -> None:
        try:
            while True:
                event = self.queue.get_nowait()
                self._handle_queue_event(event)
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _handle_queue_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "log":
            self._append_log(str(event.get("message", "")))
        elif event_type == "progress":
            self._update_progress(event)
        elif event_type == "done":
            self.processing_complete = True
            self.analysis_result = event["result"]
            self._append_log("Processing complete. Review the integrated results.")
            self.stage3_action_button.configure(text="Next", style="Green.Action.TButton", command=lambda: self._show_stage(3))
        elif event_type == "cancelled":
            self._append_log(f"Processing cancelled: {event.get('message', '')}")
            self._show_stage(1)
        elif event_type == "error":
            self._append_log(f"Error: {event.get('message', '')}")
            messagebox.showerror("Processing Error", str(event.get("message", "Unknown error")))
            self._show_stage(1)

    def _update_progress(self, event: dict[str, Any]) -> None:
        pipeline = str(event.get("pipeline"))
        fraction = float(event.get("fraction", 0.0))
        completed = int(event.get("completed", 0))
        total = int(event.get("total", 0))
        eta = event.get("eta_seconds")
        if pipeline == "object":
            self.object_progress["value"] = fraction * 100.0
            suffix = f" | ETA {format_seconds(float(eta))}" if isinstance(eta, (int, float)) else ""
            self.object_progress_text.configure(text=f"{completed} / {total}{suffix}")
        elif pipeline == "temporal":
            self.temporal_progress["value"] = fraction * 100.0
            suffix = f" | ETA {format_seconds(float(eta))}" if isinstance(eta, (int, float)) else ""
            self.temporal_progress_text.configure(text=f"{completed} / {total}{suffix}")

    def _set_log_text(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", text)
        self.log_text.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _populate_results(self) -> None:
        self.result_photos.clear()
        self._clear_container(self.yolo_results_inner)
        self._clear_container(self.action_results_inner)
        if self.analysis_result is None:
            self.result_summary_label.configure(text="No results available.")
            return
        metadata = self.analysis_result.metadata if isinstance(self.analysis_result.metadata, dict) else {}
        sampling_note = ""
        if metadata.get("sample_mode") in {"dynamic_per_table", "dynamic_shared_decode"}:
            observed_count = int(metadata.get("observed_frame_count", 0))
            selected_count = int(metadata.get("sampled_frame_count", 0))
            decoded_count = int(metadata.get("decoded_frame_count", observed_count))
            sampling_note = (
                f" | dynamic sampler decoded {decoded_count} timestamps, observed {observed_count} table-frames, "
                f"and reviewed {selected_count} candidates"
            )
        self.result_summary_label.configure(
            text=(
                self._compact_text(
                    f"Overall score {self.analysis_result.overall_score:.1f} | {self.analysis_result.overall_label} | "
                    f"{self.analysis_result.overall_summary}{sampling_note}",
                    260,
                )
            ),
            wraplength=max(720, self.winfo_width() - 260),
        )
        self._render_yolo_results()
        self._render_action_results()

    def _render_yolo_results(self) -> None:
        assert self.analysis_result is not None
        scores = [item.object_summary.aggregate_score for item in self.analysis_result.tables]
        confidence = [item.object_summary.aggregate_confidence for item in self.analysis_result.tables]
        self._render_method_summary(
            self.yolo_results_inner,
            score=sum(scores) / max(1, len(scores)),
            label="object score",
            confidence=sum(confidence) / max(1, len(confidence)),
            note="lowest sampled frame per table drives the method score",
        )
        for item in sorted(self.analysis_result.tables, key=lambda value: value.object_summary.aggregate_score):
            ordered_frames = sorted(item.object_summary.frame_results, key=lambda value: (value.score, value.confidence))[:3]
            rows = [
                ("Score", f"{item.object_summary.aggregate_score:.1f}"),
                ("Confidence", f"{item.object_summary.aggregate_confidence:.2f}"),
                ("Detections", str(sum(frame.detection_count for frame in item.object_summary.frame_results))),
            ]
            self._render_result_card(
                self.yolo_results_inner,
                title=f"{item.table.table_id} | {self._ascii_text(item.overall_label)}",
                rows=rows,
                headline=item.object_summary.headline,
                evidence=item.object_summary.evidence[:2],
                image_items=[
                    (
                        frame.evidence_path,
                        [
                            format_seconds(frame.timestamp_seconds),
                            self._short_sampler_type(frame.frame_type),
                            self._short_selection_reason(frame.selection_reasons),
                            f"score {frame.score}",
                            f"det {frame.detection_count} / {frame.person_relevance_reason}",
                        ],
                    )
                    for frame in ordered_frames
                ],
            )

    def _render_action_results(self) -> None:
        assert self.analysis_result is not None
        scores = [item.temporal_summary.aggregate_score for item in self.analysis_result.tables]
        confidence = [item.temporal_summary.aggregate_confidence for item in self.analysis_result.tables]
        self._render_method_summary(
            self.action_results_inner,
            score=sum(scores) / max(1, len(scores)),
            label="timeline score",
            confidence=sum(confidence) / max(1, len(confidence)),
            note="state tracker summarizes VLM frame states over time",
        )
        for item in sorted(self.analysis_result.tables, key=lambda value: value.table.table_id):
            frames = self._representative_temporal_frames(item.temporal_summary.frame_results)
            rows = [
                ("State", item.temporal_summary.final_state),
                ("Score", f"{item.temporal_summary.aggregate_score:.1f}"),
                ("Confidence", f"{item.temporal_summary.aggregate_confidence:.2f}"),
            ]
            self._render_result_card(
                self.action_results_inner,
                title=f"{item.table.table_id} | {self._ascii_text(item.temporal_summary.final_state)}",
                rows=rows,
                headline=item.temporal_summary.headline,
                evidence=item.temporal_summary.evidence[:3],
                image_items=[
                    (
                        frame.evidence_path,
                        [
                            format_seconds(frame.timestamp_seconds),
                            self._short_sampler_type(frame.frame_type),
                            self._short_selection_reason(frame.selection_reasons),
                            self._short_state(frame.frame_state),
                            f"final {self._short_state(frame.final_state)} / {frame.person_relevance_reason}",
                        ],
                    )
                    for frame in frames
                ],
            )

    def _render_method_summary(self, parent: ttk.Frame, *, score: float, label: str, confidence: float, note: str) -> None:
        summary = ttk.Frame(parent, style="Card.TFrame", padding=12)
        summary.pack(fill="x", pady=(0, 10))
        top = ttk.Frame(summary, style="Panel.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=f"{score:.1f}", style="Metric.TLabel").pack(side="left")
        ttk.Label(top, text=label, style="Sub.TLabel").pack(side="left", padx=(10, 0), pady=(9, 0))
        ttk.Label(top, text=f"confidence {confidence:.2f}", style="Muted.TLabel").pack(side="right", pady=(12, 0))
        ttk.Label(summary, text=note, style="Muted.TLabel", wraplength=620, justify="left").pack(anchor="w", pady=(6, 0))

    def _render_result_card(
        self,
        parent: ttk.Frame,
        *,
        title: str,
        rows: list[tuple[str, str]],
        headline: str,
        evidence: list[str],
        image_items: list[tuple[Path, list[str]]],
    ) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.pack(fill="x", pady=(0, 10))
        ttk.Label(card, text=self._ascii_text(title), style="Sub.TLabel", font=(self.font_family, 12, "bold")).pack(anchor="w")

        row_frame = ttk.Frame(card, style="Panel.TFrame")
        row_frame.pack(fill="x", pady=(8, 4))
        for label, value in rows:
            metric = ttk.Frame(row_frame, style="Panel.TFrame")
            metric.pack(side="left", padx=(0, 18))
            ttk.Label(metric, text=self._ascii_text(label), style="Muted.TLabel").pack(anchor="w")
            ttk.Label(metric, text=self._ascii_text(value), style="Sub.TLabel", font=(self.font_family, 11, "bold")).pack(anchor="w")

        image_strip = ttk.Frame(card, style="Panel.TFrame")
        image_strip.pack(fill="x", pady=(8, 8))
        for image_path, caption in image_items:
            self._render_thumbnail(image_strip, image_path, caption)

        ttk.Label(card, text=self._compact_text(headline, 220), style="Headline.TLabel", wraplength=640, justify="left").pack(anchor="w", pady=(2, 6))
        for item in evidence:
            ttk.Label(card, text=f"- {self._compact_text(item, 180)}", style="Evidence.TLabel", wraplength=640, justify="left").pack(anchor="w", pady=(2, 0))

    def _render_thumbnail(self, parent: ttk.Frame, image_path: Path, caption_lines: list[str]) -> None:
        thumb_frame = ttk.Frame(parent, style="Panel.TFrame")
        thumb_frame.pack(side="left", padx=(0, 14), anchor="n")
        image_box = tk.Frame(thumb_frame, width=286, height=214, bg="#f8fafc", highlightbackground="#d1d5db", highlightthickness=1)
        image_box.pack_propagate(False)
        image_box.pack()
        if image_path.exists():
            with Image.open(image_path) as image:
                image.thumbnail((278, 206), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image.copy())
            self.result_photos.append(photo)
            tk.Label(image_box, image=photo, bg="#f8fafc").pack(expand=True)
        else:
            ttk.Label(image_box, text="missing", style="Muted.TLabel").pack(expand=True)
        caption = ttk.Frame(thumb_frame, style="Panel.TFrame")
        caption.pack(fill="x", pady=(6, 0))
        for line in caption_lines:
            ttk.Label(caption, text=self._ascii_text(line), style="Caption.TLabel", width=32, anchor="center").pack(fill="x")

    def _representative_temporal_frames(self, frames: list[Any]) -> list[Any]:
        indices = representative_temporal_frame_indices(frames)
        return [frames[index] for index in indices]

    def _clear_container(self, container: ttk.Frame) -> None:
        for child in container.winfo_children():
            child.destroy()

    def _short_state(self, state: str) -> str:
        return {
            "AFTER_MEAL_CANDIDATE": "AFTER_MEAL_CAND",
            "POSSIBLY_EMPTY": "POSSIBLY_EMPTY",
            "UNCERTAIN": "UNCERTAIN",
            "AFTER_MEAL": "AFTER_MEAL",
            "CLEANING": "CLEANING",
            "DINING": "DINING",
        }.get(state, state)

    def _short_sampler_type(self, frame_type: str) -> str:
        return {
            "occupied_representative": "occupied",
            "meal_end_candidate": "meal_end",
            "cleaning_before_candidate": "pre_clean",
            "cleaning_candidate": "cleaning",
            "post_check": "post_check",
            "periodic_sample": "periodic",
        }.get(frame_type, frame_type)

    def _short_selection_reason(self, reasons: list[str]) -> str:
        if not reasons:
            return "review"
        label = ",".join(reasons[:2])
        return self._compact_text(label, 22)

    def _compact_text(self, value: str, limit: int) -> str:
        text = self._ascii_text(value)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _ascii_text(self, value: object) -> str:
        text = " ".join(str(value).split())
        ascii_text = text.encode("ascii", "ignore").decode("ascii")
        ascii_text = " ".join(ascii_text.split())
        if not any(character.isalnum() for character in ascii_text):
            return "English summary unavailable."
        return ascii_text

    def _restart_workflow(self) -> None:
        self.cancel_event.set()
        self.boxes = []
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.selected_box_id = None
        self.analysis_result = None
        self.processing_complete = False
        if self.first_frame_path and self.first_frame_path.exists():
            self.first_frame_path.unlink(missing_ok=True)
        self._show_stage(0)
        self._render_stage2_canvas()
        self._update_stage2_buttons()


def cv2_to_rgb(image):
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted, _, _ = fit_image_with_geometry(image, width, height)
    return fitted


def fit_image_with_geometry(image: Image.Image, width: int, height: int) -> tuple[Image.Image, float, tuple[int, int]]:
    source_w, source_h = image.size
    scale = min(width / max(1, source_w), height / max(1, source_h))
    scale = max(0.05, scale)
    new_w = max(1, int(source_w * scale))
    new_h = max(1, int(source_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    offset = ((width - new_w) // 2, (height - new_h) // 2)
    return resized, scale, offset


def launch(project_root: Path) -> None:
    app = SimulatorApp(project_root)
    app.mainloop()
