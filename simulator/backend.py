from __future__ import annotations

import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable

import cv2
import numpy as np

from simulator.models import (
    AnalysisSessionResult,
    CombinedTableResult,
    ObjectFrameResult,
    ObjectTableSummary,
    SampledFrame,
    TableBox,
    TemporalFrameResult,
    TemporalTableSummary,
)
from vlm_classifier.state_tracker import RestaurantStateTracker

DEFAULT_CLEANLINESS_PROMPT_PROFILE = "restaurant"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
FRAME_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visible_people": {"type": "boolean"},
        "active_dining": {"type": "boolean"},
        "cleaning_action": {"type": "boolean"},
        "used_dishes_or_leftovers": {"type": "boolean"},
        "personal_items_or_food_remaining": {"type": "boolean"},
        "frame_state": {
            "type": "string",
            "enum": ["DINING", "CLEANING", "POSSIBLY_EMPTY", "AFTER_MEAL_CANDIDATE", "UNCERTAIN"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": [
        "visible_people",
        "active_dining",
        "cleaning_action",
        "used_dishes_or_leftovers",
        "personal_items_or_food_remaining",
        "frame_state",
        "confidence",
        "reason",
    ],
}


class CancelledError(RuntimeError):
    pass


def format_seconds(value: float) -> str:
    total = max(0, int(round(value)))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def list_video_files(video_root: Path) -> list[Path]:
    if not video_root.exists():
        return []
    files = [path for path in video_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(files)


def read_first_frame(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        success, frame = capture.read()
    finally:
        capture.release()
    if not success or frame is None:
        raise ValueError(f"cannot read first frame from {video_path}")
    return frame


def save_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower() or ".png"
    if extension == ".jpeg":
        extension = ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"cannot encode image for {path}")
    encoded.tofile(str(path))
    return path


def read_image_file(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def crop_box(image: np.ndarray, box: TableBox) -> np.ndarray:
    height, width = image.shape[:2]
    clamped = box.clamp(width, height)
    return image[clamped.y1 : clamped.y2, clamped.x1 : clamped.x2].copy()


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.36
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    image_h, image_w = image.shape[:2]
    x = max(0, min(x, max(0, image_w - text_w - 8)))
    y = max(text_h + 8, min(y, max(text_h + 8, image_h - baseline - 4)))
    cv2.rectangle(image, (x, y - text_h - 7), (x + text_w + 8, y + baseline + 4), color, -1)
    cv2.putText(image, text, (x + 4, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def dirty_detection_label(label: str) -> bool:
    lowered = label.lower()
    dirty_tokens = (
        "trash",
        "wrapper",
        "napkin",
        "tissue",
        "paper",
        "residue",
        "crumb",
        "stain",
        "spill",
        "leftover",
        "food",
    )
    return any(token in lowered for token in dirty_tokens)


def make_evidence_canvas(crop_image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    canvas_w, canvas_h = 560, 400
    image_area_h = 310
    canvas = np.full((canvas_h, canvas_w, 3), 248, dtype=np.uint8)
    crop_h, crop_w = crop_image.shape[:2]
    scale = min((canvas_w - 24) / max(1, crop_w), (image_area_h - 24) / max(1, crop_h))
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    offset = ((canvas_w - new_w) // 2, (image_area_h - new_h) // 2)
    resized = cv2.resize(crop_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    x0, y0 = offset
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas, scale, offset


def draw_footer_line(image: np.ndarray, text: str, row: int, color: tuple[int, int, int]) -> None:
    y = 342 + row * 24
    cv2.putText(image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)


def render_yolo_evidence_image(
    crop_image: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    score: int,
    confidence: float,
    output_path: Path,
) -> Path:
    canvas, scale, offset = make_evidence_canvas(crop_image)
    image_h, image_w = crop_image.shape[:2]
    offset_x, offset_y = offset
    dirty_count = 0
    for detection in detections:
        label = str(detection.get("label", "object"))
        bbox = detection.get("bbox", {})
        if not isinstance(bbox, dict):
            continue
        x1 = int(round(float(bbox.get("x1", 0))))
        y1 = int(round(float(bbox.get("y1", 0))))
        x2 = int(round(float(bbox.get("x2", 0))))
        y2 = int(round(float(bbox.get("y2", 0))))
        x1 = max(0, min(x1, image_w - 1))
        y1 = max(0, min(y1, image_h - 1))
        x2 = max(x1 + 1, min(x2, image_w))
        y2 = max(y1 + 1, min(y2, image_h))
        sx1 = int(round(offset_x + x1 * scale))
        sy1 = int(round(offset_y + y1 * scale))
        sx2 = int(round(offset_x + x2 * scale))
        sy2 = int(round(offset_y + y2 * scale))
        is_dirty = dirty_detection_label(label)
        if is_dirty:
            dirty_count += 1
        color = (35, 35, 220) if is_dirty or score <= 3 else (42, 157, 244)
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), color, 2)
        conf = float(detection.get("confidence", 0.0))
        draw_label(canvas, f"{label[:16]} {conf:.2f}", (sx1, max(18, sy1)), color)

    border_color = (35, 35, 220) if score <= 3 or dirty_count else (40, 167, 69)
    crop_x1, crop_y1 = offset
    crop_x2 = int(round(crop_x1 + image_w * scale))
    crop_y2 = int(round(crop_y1 + image_h * scale))
    cv2.rectangle(canvas, (crop_x1, crop_y1), (crop_x2, crop_y2), border_color, 3)
    cv2.rectangle(canvas, (0, 310), (560, 400), (255, 255, 255), -1)
    draw_footer_line(canvas, f"YOLO+LLM  score {score}  conf {confidence:.2f}  det {len(detections)}", 0, border_color)
    footer_note = "Dirty labels highlighted in red." if dirty_count else "No dirty object boxes found in this crop."
    if score <= 3 and not detections:
        footer_note = "Low score came from VLM image judgment, not a YOLO box."
    draw_footer_line(canvas, footer_note, 1, (75, 85, 99))
    save_image(output_path, canvas)
    return output_path


def state_color(final_state: str) -> tuple[int, int, int]:
    mapping = {
        "CLEANING": (40, 167, 69),
        "DINING": (42, 157, 244),
        "AFTER_MEAL": (35, 35, 220),
        "UNCERTAIN": (0, 180, 230),
    }
    return mapping.get(final_state.upper(), (107, 114, 128))


def render_action_evidence_image(
    crop_image: np.ndarray,
    payload: dict[str, Any],
    *,
    final_state: str,
    confidence: float,
    output_path: Path,
) -> Path:
    canvas, scale, offset = make_evidence_canvas(crop_image)
    image_h, image_w = crop_image.shape[:2]
    color = state_color(final_state)
    crop_x1, crop_y1 = offset
    crop_x2 = int(round(crop_x1 + image_w * scale))
    crop_y2 = int(round(crop_y1 + image_h * scale))
    cv2.rectangle(canvas, (crop_x1, crop_y1), (crop_x2, crop_y2), color, 3)

    chips = [
        ("PEOPLE", bool(payload.get("visible_people"))),
        ("DINING", bool(payload.get("active_dining"))),
        ("CLEANING", bool(payload.get("cleaning_action"))),
        ("USED ITEMS", bool(payload.get("used_dishes_or_leftovers"))),
        ("BELONGINGS", bool(payload.get("personal_items_or_food_remaining"))),
    ]
    cv2.rectangle(canvas, (0, 310), (560, 400), (255, 255, 255), -1)
    draw_footer_line(canvas, f"Action/VLM  {final_state}  conf {confidence:.2f}", 0, color)
    active_chips = [label for label, active in chips if active]
    chip_text = "Signals: " + (", ".join(active_chips) if active_chips else "none active")
    frame_state = str(payload.get("frame_state", "UNKNOWN"))
    draw_footer_line(canvas, chip_text[:58], 1, (75, 85, 99))
    draw_footer_line(canvas, f"Frame state: {frame_state}", 2, (75, 85, 99))
    save_image(output_path, canvas)
    return output_path


def next_table_id(boxes: list[TableBox]) -> str:
    used = {box.table_id for box in boxes}
    index = 1
    while True:
        candidate = f"T{index:02d}"
        if candidate not in used:
            return candidate
        index += 1


class TableDetector:
    def __init__(self) -> None:
        from app.yolo_module import yolo_module

        prompts = ["restaurant table", "dining table", "table", "desk", "booth table"]
        self.detector = yolo_module(
            confidence_threshold=0.15,
            tile_size=768,
            tile_overlap=0.2,
            prompt_labels=prompts,
            debris_prompt_labels=prompts,
            debris_confidence_threshold=0.12,
            debris_tile_size=512,
            debris_tile_overlap=0.25,
        )

    def detect(self, frame_path: Path) -> list[TableBox]:
        image = read_image_file(frame_path)
        image_height, image_width = image.shape[:2]
        image_area = max(1, image_height * image_width)
        raw = self.detector.detect_objects(frame_path)
        boxes: list[TableBox] = []
        for detection in raw:
            label = detection.label.strip().lower()
            if not any(token in label for token in ("table", "desk")):
                continue
            x1 = int(round(detection.bbox["x1"]))
            y1 = int(round(detection.bbox["y1"]))
            x2 = int(round(detection.bbox["x2"]))
            y2 = int(round(detection.bbox["y2"]))
            candidate = TableBox(table_id="", x1=min(x1, x2), y1=min(y1, y2), x2=max(x1, x2), y2=max(y1, y2)).clamp(
                image_width,
                image_height,
            )
            if candidate.area < int(image_area * 0.01):
                continue
            boxes.append(candidate)

        deduplicated: list[TableBox] = []
        for candidate in sorted(boxes, key=lambda item: item.area, reverse=True):
            if any(iou(candidate, kept) >= 0.55 for kept in deduplicated):
                continue
            deduplicated.append(candidate)

        deduplicated.sort(key=lambda item: (item.y1, item.x1))
        return [TableBox(f"T{index:02d}", box.x1, box.y1, box.x2, box.y2) for index, box in enumerate(deduplicated, start=1)]


class TemporalFrameClassifier:
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL) -> None:
        from app.analysis import OpenAIModelClient
        from app.config import OPENAI_MODEL

        self.model = model or OPENAI_MODEL
        self._client = OpenAIModelClient(model=self.model)

    def classify(self, crop_path: Path) -> dict[str, Any]:
        prompt = (
            "You are analyzing one low-resolution CCTV crop of a restaurant table.\n"
            "Your goal is not overall cleanliness yet. First infer the operational table state from the frame.\n"
            "Be conservative: a single frame should not overclaim that the meal is fully finished.\n"
            "Assess the following booleans and one frame_state label.\n"
            "visible_people: true when a person is visible inside the crop.\n"
            "active_dining: true when people appear to be eating or actively using food, dishes, or drinks.\n"
            "cleaning_action: true when a person is visibly wiping the table, collecting dishes, or clearing trash.\n"
            "used_dishes_or_leftovers: true when used dishes, leftovers, napkins, wrappers, or table debris are visible.\n"
            "personal_items_or_food_remaining: true when bags, coats, cups, bottles, food, or similar items suggest someone may return.\n"
            "frame_state must be one of DINING, CLEANING, POSSIBLY_EMPTY, AFTER_MEAL_CANDIDATE, UNCERTAIN.\n"
            "Return only JSON."
        )
        payload = self._client.create_json_response(
            prompt=prompt,
            image_paths=[crop_path],
            schema_name="simulator_frame_state",
            schema=FRAME_CLASSIFICATION_SCHEMA,
            temperature=0.1,
        )
        payload["confidence"] = float(max(0.0, min(1.0, float(payload["confidence"]))))
        return payload


def english_temporal_reason(final_state: str, payload: dict[str, Any]) -> str:
    state = final_state.upper()
    visible_people = bool(payload.get("visible_people"))
    active_dining = bool(payload.get("active_dining"))
    cleaning_action = bool(payload.get("cleaning_action"))
    used_items = bool(payload.get("used_dishes_or_leftovers"))
    personal_items = bool(payload.get("personal_items_or_food_remaining"))

    if cleaning_action or state == "CLEANING":
        return "Cleaning activity was detected, so the table is treated as being cleaned."
    if active_dining or state == "DINING":
        return "Visible people and active dining cues indicate the table is currently in use."
    if state == "AFTER_MEAL":
        return "People were absent long enough while used dishes, leftovers, or debris remained visible."
    if not visible_people and (used_items or personal_items):
        return "No people are visible, but dishes, food, or belongings remain on or near the table."
    if visible_people:
        return "People are visible, but dining or cleaning activity is not clear enough."
    return "The frame does not provide enough evidence for a stable table state."


def iou(left: TableBox, right: TableBox) -> float:
    inter_x1 = max(left.x1, right.x1)
    inter_y1 = max(left.y1, right.y1)
    inter_x2 = min(left.x2, right.x2)
    inter_y2 = min(left.y2, right.y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    union = left.area + right.area - intersection
    return float(intersection / max(1, union))


def sample_video_frames(
    video_path: Path,
    *,
    interval_seconds: float,
    output_dir: Path,
    max_frames: int,
    cancel_event: Event,
) -> list[SampledFrame]:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = total_frames / fps if total_frames > 0 else 0.0
    effective_interval = max(1.0 / fps, float(interval_seconds))

    frame_targets: list[float] = []
    current = 0.0
    while duration_seconds <= 0.0 or current <= duration_seconds + 1e-6:
        frame_targets.append(current)
        current += effective_interval
        if len(frame_targets) >= max_frames:
            break
        if duration_seconds <= 0 and len(frame_targets) >= 1:
            break

    sampled: list[SampledFrame] = []
    for index, timestamp_seconds in enumerate(frame_targets):
        if cancel_event.is_set():
            raise CancelledError("sampling cancelled")
        frame_index = max(0, int(round(timestamp_seconds * fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            continue
        frame_path = output_dir / f"frame_{index:03d}_{int(round(timestamp_seconds * 1000)):07d}ms.jpg"
        save_image(frame_path, frame)
        sampled.append(SampledFrame(index=index, timestamp_seconds=timestamp_seconds, frame_path=frame_path))
    capture.release()
    if not sampled:
        raise ValueError(f"no frames could be sampled from {video_path}")
    return sampled


@dataclass
class PipelineProgress:
    completed: int = 0
    total: int = 1
    started_at: float = 0.0

    def update(self, completed: int, total: int) -> dict[str, Any]:
        self.completed = completed
        self.total = max(1, total)
        now = time.time()
        elapsed = max(0.001, now - self.started_at)
        rate = completed / elapsed if completed else 0.0
        remaining = max(0, total - completed)
        eta = remaining / rate if rate > 0 else None
        return {
            "completed": completed,
            "total": total,
            "fraction": min(1.0, completed / max(1, total)),
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
        }


class IntegratedAnalyzer:
    def __init__(self) -> None:
        self.table_detector = TableDetector()

    def auto_detect_tables(self, frame_path: Path) -> list[TableBox]:
        return self.table_detector.detect(frame_path)

    def run(
        self,
        *,
        video_path: Path,
        tables: list[TableBox],
        frame_interval_seconds: float,
        output_root: Path,
        progress_callback: Callable[[dict[str, Any]], None],
        cancel_event: Event,
    ) -> AnalysisSessionResult:
        if not tables:
            raise ValueError("at least one table box is required")

        session_name = f"{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = output_root / session_name
        frames_dir = session_dir / "frames"
        tables_dir = session_dir / "tables"
        session_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        progress_callback({"type": "log", "message": f"Sampling frames from {video_path.name} every {frame_interval_seconds:.1f}s"})
        sampled_frames = sample_video_frames(
            video_path,
            interval_seconds=frame_interval_seconds,
            output_dir=frames_dir,
            max_frames=18,
            cancel_event=cancel_event,
        )
        progress_callback(
            {
                "type": "log",
                "message": f"Prepared {len(sampled_frames)} sampled frames for {len(tables)} tables.",
            }
        )

        object_results: dict[str, ObjectTableSummary] = {}
        temporal_results: dict[str, TemporalTableSummary] = {}
        object_lock = Lock()
        temporal_lock = Lock()

        def emit_progress(pipeline: str, payload: dict[str, Any]) -> None:
            progress_callback({"type": "progress", "pipeline": pipeline, **payload})

        object_thread = ThreadPoolExecutor(max_workers=1)
        temporal_thread = ThreadPoolExecutor(max_workers=1)
        try:
            object_future = object_thread.submit(
                self._run_object_pipeline,
                tables,
                sampled_frames,
                tables_dir,
                cancel_event,
                object_results,
                object_lock,
                emit_progress,
                progress_callback,
            )
            temporal_future = temporal_thread.submit(
                self._run_temporal_pipeline,
                tables,
                sampled_frames,
                tables_dir,
                cancel_event,
                temporal_results,
                temporal_lock,
                emit_progress,
                progress_callback,
            )
            object_future.result()
            temporal_future.result()
        finally:
            object_thread.shutdown(wait=False, cancel_futures=True)
            temporal_thread.shutdown(wait=False, cancel_futures=True)

        if cancel_event.is_set():
            raise CancelledError("analysis cancelled")

        combined = self._combine_results(video_path, frame_interval_seconds, sampled_frames, tables, object_results, temporal_results, session_dir)
        metadata = {
            "sampled_frame_count": len(sampled_frames),
            "table_count": len(tables),
            "session_dir": str(session_dir),
        }
        return AnalysisSessionResult(
            video_path=video_path,
            frame_interval_seconds=frame_interval_seconds,
            sampled_frames=sampled_frames,
            tables=combined["tables"],
            output_dir=session_dir,
            overall_score=combined["overall_score"],
            overall_label=combined["overall_label"],
            overall_summary=combined["overall_summary"],
            metadata=metadata,
        )

    def _run_object_pipeline(
        self,
        tables: list[TableBox],
        sampled_frames: list[SampledFrame],
        tables_dir: Path,
        cancel_event: Event,
        sink: dict[str, ObjectTableSummary],
        sink_lock: Lock,
        emit_progress: Callable[[str, dict[str, Any]], None],
        progress_callback: Callable[[dict[str, Any]], None],
    ) -> None:
        total = len(tables) * len(sampled_frames)
        progress = PipelineProgress(total=total, started_at=time.time())
        completed = 0

        def worker(table: TableBox) -> ObjectTableSummary:
            from app.cleanliness import CleanlinessService
            from app.yolo_module import yolo_module

            service = CleanlinessService(
                yolo_helper=yolo_module(
                    confidence_threshold=0.18,
                    tile_size=640,
                    tile_overlap=0.2,
                    debris_confidence_threshold=0.2,
                    debris_tile_size=512,
                    debris_tile_overlap=0.2,
                )
            )
            frame_results: list[ObjectFrameResult] = []
            table_dir = tables_dir / table.table_id / "object"
            table_dir.mkdir(parents=True, exist_ok=True)
            nonlocal completed
            for sampled in sampled_frames:
                if cancel_event.is_set():
                    raise CancelledError("object pipeline cancelled")
                frame_image = read_image_file(sampled.frame_path)
                crop_path = table_dir / f"crop_{sampled.index:03d}.png"
                crop_image = crop_box(frame_image, table)
                save_image(crop_path, crop_image)
                result = service.inspect_image(
                    crop_path,
                    inspected_path=crop_path,
                    prompt_profile=DEFAULT_CLEANLINESS_PROMPT_PROFILE,
                    use_yolo=True,
                    roi=None,
                    output_stem=None,
                )
                raw_detections = (result.yolo_payload or {}).get("detections", [])
                detections = [item for item in raw_detections if isinstance(item, dict)]
                detection_count = len(detections)
                evidence_path = table_dir / f"evidence_{sampled.index:03d}.png"
                render_yolo_evidence_image(
                    crop_image,
                    detections,
                    score=result.score,
                    confidence=result.confidence,
                    output_path=evidence_path,
                )
                frame_results.append(
                    ObjectFrameResult(
                        table_id=table.table_id,
                        timestamp_seconds=sampled.timestamp_seconds,
                        frame_path=sampled.frame_path,
                        crop_path=crop_path,
                        evidence_path=evidence_path,
                        score=result.score,
                        confidence=result.confidence,
                        summary=result.summary,
                        findings=result.findings,
                        exact_objects=result.exact_objects,
                        estimated_objects=result.estimated_objects,
                        detection_count=detection_count,
                    )
                )
                with sink_lock:
                    completed += 1
                    payload = progress.update(completed, total)
                emit_progress("object", payload)
                progress_callback(
                    {
                        "type": "log",
                        "message": (
                            f"[YOLO+LLM] {table.table_id} @ {format_seconds(sampled.timestamp_seconds)} "
                            f"score={result.score} conf={result.confidence:.2f} det={detection_count}"
                        ),
                    }
                )
            return self._summarize_object_table(table, frame_results)

        with ThreadPoolExecutor(max_workers=min(2, max(1, len(tables)))) as executor:
            futures = {executor.submit(worker, table): table for table in tables}
            for future in as_completed(futures):
                table = futures[future]
                summary = future.result()
                sink[table.table_id] = summary

    def _run_temporal_pipeline(
        self,
        tables: list[TableBox],
        sampled_frames: list[SampledFrame],
        tables_dir: Path,
        cancel_event: Event,
        sink: dict[str, TemporalTableSummary],
        sink_lock: Lock,
        emit_progress: Callable[[str, dict[str, Any]], None],
        progress_callback: Callable[[dict[str, Any]], None],
    ) -> None:
        total = len(tables) * len(sampled_frames)
        progress = PipelineProgress(total=total, started_at=time.time())
        completed = 0

        def worker(table: TableBox) -> TemporalTableSummary:
            classifier = TemporalFrameClassifier(model=DEFAULT_OPENAI_MODEL)
            tracker = RestaurantStateTracker(empty_threshold_seconds=600)
            frame_results: list[TemporalFrameResult] = []
            table_dir = tables_dir / table.table_id / "temporal"
            table_dir.mkdir(parents=True, exist_ok=True)
            nonlocal completed
            for sampled in sampled_frames:
                if cancel_event.is_set():
                    raise CancelledError("temporal pipeline cancelled")
                frame_image = read_image_file(sampled.frame_path)
                crop_path = table_dir / f"crop_{sampled.index:03d}.png"
                crop_image = crop_box(frame_image, table)
                save_image(crop_path, crop_image)
                payload = classifier.classify(crop_path)
                tracker_result = tracker.update(int(round(sampled.timestamp_seconds)), payload)
                final_state = str(tracker_result["final_state"])
                temporal_reason = english_temporal_reason(final_state, payload)
                evidence_path = table_dir / f"evidence_{sampled.index:03d}.png"
                render_action_evidence_image(
                    crop_image,
                    payload,
                    final_state=final_state,
                    confidence=float(payload["confidence"]),
                    output_path=evidence_path,
                )
                frame_results.append(
                    TemporalFrameResult(
                        table_id=table.table_id,
                        timestamp_seconds=sampled.timestamp_seconds,
                        frame_path=sampled.frame_path,
                        crop_path=crop_path,
                        evidence_path=evidence_path,
                        visible_people=bool(payload["visible_people"]),
                        active_dining=bool(payload["active_dining"]),
                        cleaning_action=bool(payload["cleaning_action"]),
                        used_dishes_or_leftovers=bool(payload["used_dishes_or_leftovers"]),
                        personal_items_or_food_remaining=bool(payload["personal_items_or_food_remaining"]),
                        frame_state=str(payload["frame_state"]),
                        final_state=final_state,
                        confidence=float(payload["confidence"]),
                        reason=str(payload["reason"]),
                        temporal_reason=temporal_reason,
                    )
                )
                with sink_lock:
                    completed += 1
                    progress_payload = progress.update(completed, total)
                emit_progress("temporal", progress_payload)
                progress_callback(
                    {
                        "type": "log",
                        "message": (
                            f"[Action/VLM] {table.table_id} @ {format_seconds(sampled.timestamp_seconds)} "
                            f"state={final_state} conf={float(payload['confidence']):.2f}"
                        ),
                    }
                )
            return self._summarize_temporal_table(table, frame_results)

        with ThreadPoolExecutor(max_workers=min(4, max(1, len(tables)))) as executor:
            futures = {executor.submit(worker, table): table for table in tables}
            for future in as_completed(futures):
                table = futures[future]
                summary = future.result()
                sink[table.table_id] = summary

    def _summarize_object_table(self, table: TableBox, frame_results: list[ObjectFrameResult]) -> ObjectTableSummary:
        ordered = sorted(frame_results, key=lambda item: (item.score, item.confidence))
        worst = ordered[0]
        average_confidence = sum(item.confidence for item in frame_results) / max(1, len(frame_results))
        aggregate_score = float(min(item.score for item in frame_results))
        evidence = []
        for item in ordered[:3]:
            evidence.append(
                f"{format_seconds(item.timestamp_seconds)} score={item.score} det={item.detection_count}: {item.summary}"
            )
        return ObjectTableSummary(
            table=table,
            frame_results=frame_results,
            aggregate_score=aggregate_score,
            aggregate_confidence=average_confidence,
            headline=worst.summary,
            evidence=evidence,
        )

    def _summarize_temporal_table(self, table: TableBox, frame_results: list[TemporalFrameResult]) -> TemporalTableSummary:
        latest = frame_results[-1]
        average_confidence = sum(item.confidence for item in frame_results) / max(1, len(frame_results))
        aggregate_score = temporal_state_score(latest.final_state)
        evidence = []
        for item in frame_results[:4]:
            evidence.append(
                f"{format_seconds(item.timestamp_seconds)} frame={item.frame_state} final={item.final_state}: {item.temporal_reason}"
            )
        return TemporalTableSummary(
            table=table,
            frame_results=frame_results,
            aggregate_score=aggregate_score,
            aggregate_confidence=average_confidence,
            final_state=latest.final_state,
            headline=latest.temporal_reason,
            evidence=evidence,
        )

    def _combine_results(
        self,
        video_path: Path,
        frame_interval_seconds: float,
        sampled_frames: list[SampledFrame],
        tables: list[TableBox],
        object_results: dict[str, ObjectTableSummary],
        temporal_results: dict[str, TemporalTableSummary],
        session_dir: Path,
    ) -> dict[str, Any]:
        combined_tables: list[CombinedTableResult] = []
        for table in tables:
            object_summary = object_results[table.table_id]
            temporal_summary = temporal_results[table.table_id]
            overall_score = round(object_summary.aggregate_score * 0.7 + temporal_summary.aggregate_score * 0.3, 2)
            overall_label = cleanliness_label(overall_score)
            overall_summary = (
                f"YOLO+LLM score {object_summary.aggregate_score:.1f} and temporal state {temporal_summary.final_state} "
                f"were combined into {overall_score:.1f}."
            )
            combined_tables.append(
                CombinedTableResult(
                    table=table,
                    object_summary=object_summary,
                    temporal_summary=temporal_summary,
                    overall_score=overall_score,
                    overall_label=overall_label,
                    overall_summary=overall_summary,
                )
            )

        overall_score = round(sum(item.overall_score for item in combined_tables) / max(1, len(combined_tables)), 2)
        worst_table = min(combined_tables, key=lambda item: item.overall_score)
        overall_label = cleanliness_label(overall_score)
        overall_summary = (
            f"Average integrated score is {overall_score:.1f}. The riskiest table is {worst_table.table.table_id} "
            f"with {worst_table.overall_score:.1f}."
        )
        return {
            "tables": combined_tables,
            "overall_score": overall_score,
            "overall_label": overall_label,
            "overall_summary": overall_summary,
        }


def temporal_state_score(state: str) -> float:
    mapping = {
        "CLEANING": 4.0,
        "DINING": 3.0,
        "AFTER_MEAL": 2.0,
        "UNCERTAIN": 3.0,
    }
    return mapping.get(state.upper(), 3.0)


def cleanliness_label(score: float) -> str:
    if score >= 4.5:
        return "Very Clean"
    if score >= 3.5:
        return "Clean"
    if score >= 2.5:
        return "Watch"
    if score >= 1.5:
        return "Needs Attention"
    return "Critical"


def build_output_root(project_root: Path) -> Path:
    output_root = project_root / "simulator" / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root
