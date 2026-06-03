from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.analysis import ordered_points, read_image, write_image
from app.config import ANALYSIS_CROP_DIR
from app.schemas import ROI


DEFAULT_OPEN_VOCAB_PROMPTS = [
    "used napkin",
    "paper napkin",
    "food wrapper",
    "snack wrapper",
    "plastic wrapper",
    "small trash",
    "trash",
    "food residue",
    "crumbs",
    "stain",
    "spill",
    "tray",
    "metal tray",
    "bowl",
    "metal bowl",
    "cup",
    "bottle",
    "container",
    "plate",
    "utensil",
]

DEBRIS_FOCUSED_PROMPTS = [
    "used tissue",
    "tissue",
    "crumpled tissue",
    "paper tissue",
    "used napkin",
    "paper napkin",
    "crumpled napkin",
    "food wrapper",
    "plastic wrapper",
    "snack wrapper",
    "paper scrap",
    "crumpled paper",
    "small white trash",
    "small paper trash",
    "food residue",
    "crumbs",
]


@dataclass(frozen=True)
class YoloDetection:
    label: str
    confidence: float
    bbox: dict[str, float]
    position: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": {key: round(value, 1) for key, value in self.bbox.items()},
            "position": self.position,
            "source": self.source,
        }


@dataclass(frozen=True)
class YoloAugmentation:
    image_path: Path
    detector_name: str
    weights: str
    confidence_threshold: float
    iou_threshold: float
    tile_size: int
    tile_overlap: float
    prompt_labels: list[str]
    debris_prompt_labels: list[str]
    detections: list[YoloDetection]
    crop_annotated_path: Path | None
    source_annotated_path: Path | None
    prompt_instruction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_path.name,
            "detector_name": self.detector_name,
            "weights": self.weights,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "prompt_labels": self.prompt_labels,
            "debris_prompt_labels": self.debris_prompt_labels,
            "crop_annotated_path": str(self.crop_annotated_path) if self.crop_annotated_path is not None else "",
            "source_annotated_path": str(self.source_annotated_path) if self.source_annotated_path is not None else "",
            "detections": [item.to_dict() for item in self.detections],
        }


class yolo_module:
    def __init__(
        self,
        model_name: str = "yoloe-26n-seg.pt",
        fallback_model_name: str = "yolov8s-worldv2.pt",
        confidence_threshold: float = 0.12,
        iou_threshold: float = 0.45,
        tile_size: int = 512,
        tile_overlap: float = 0.25,
        prompt_labels: list[str] | None = None,
        debris_prompt_labels: list[str] | None = None,
        debris_confidence_threshold: float = 0.08,
        debris_tile_size: int = 256,
        debris_tile_overlap: float = 0.35,
    ) -> None:
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.prompt_labels = list(prompt_labels or DEFAULT_OPEN_VOCAB_PROMPTS)
        self.debris_prompt_labels = list(debris_prompt_labels or DEBRIS_FOCUSED_PROMPTS)
        self.debris_confidence_threshold = debris_confidence_threshold
        self.debris_tile_size = debris_tile_size
        self.debris_tile_overlap = debris_tile_overlap
        self._model: Any | None = None
        self._detector_name = "uninitialized"

    def _set_model_classes(self, model: Any, prompt_labels: list[str]) -> None:
        if hasattr(model, "set_classes"):
            model.set_classes(prompt_labels)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLOE  # type: ignore

            model = YOLOE(self.model_name)
            self._set_model_classes(model, self.prompt_labels)
            self._model = model
            self._detector_name = "YOLOE"
            return self._model
        except Exception:
            pass

        try:
            from ultralytics import YOLOWorld  # type: ignore

            model = YOLOWorld(self.fallback_model_name)
            self._set_model_classes(model, self.prompt_labels)
            self._model = model
            self._detector_name = "YOLO-World"
            return self._model
        except ImportError as exc:
            raise RuntimeError(
                "YOLO inference requires the `ultralytics` package. Install dependencies and try again."
            ) from exc

    def _position_label(self, *, center_x: float, center_y: float, width: int, height: int) -> str:
        horizontal = "left" if center_x < width / 3 else "right" if center_x > (width * 2 / 3) else "center"
        vertical = "top" if center_y < height / 3 else "bottom" if center_y > (height * 2 / 3) else "middle"
        return f"{vertical}-{horizontal}"

    def _generate_slices(self, width: int, height: int, tile_size: int, tile_overlap: float) -> list[tuple[int, int, int, int]]:
        if width <= tile_size and height <= tile_size:
            return [(0, 0, width, height)]

        stride = max(64, int(round(tile_size * (1.0 - tile_overlap))))
        x_starts = list(range(0, max(width - tile_size, 0) + 1, stride))
        y_starts = list(range(0, max(height - tile_size, 0) + 1, stride))
        if not x_starts or x_starts[-1] != max(width - tile_size, 0):
            x_starts.append(max(width - tile_size, 0))
        if not y_starts or y_starts[-1] != max(height - tile_size, 0):
            y_starts.append(max(height - tile_size, 0))

        slices: list[tuple[int, int, int, int]] = []
        for y1 in y_starts:
            for x1 in x_starts:
                x2 = min(width, x1 + tile_size)
                y2 = min(height, y1 + tile_size)
                slices.append((x1, y1, x2, y2))
        return slices

    def _extract_detections(
        self,
        result: Any,
        *,
        offset_x: int,
        offset_y: int,
        image_width: int,
        image_height: int,
        source: str,
    ) -> list[YoloDetection]:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: list[YoloDetection] = []
        for box in boxes:
            cls_value = getattr(box, "cls", None)
            conf_value = getattr(box, "conf", None)
            xyxy_value = getattr(box, "xyxy", None)
            if cls_value is None or conf_value is None or xyxy_value is None:
                continue

            class_index = int(cls_value[0].item())
            label = str(names.get(class_index, class_index))
            confidence = float(conf_value[0].item())
            x1, y1, x2, y2 = [float(item) for item in xyxy_value[0].tolist()]
            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            detections.append(
                YoloDetection(
                    label=label,
                    confidence=confidence,
                    bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    position=self._position_label(center_x=center_x, center_y=center_y, width=image_width, height=image_height),
                    source=source,
                )
            )
        return detections

    def _iou(self, left: YoloDetection, right: YoloDetection) -> float:
        lx1 = left.bbox["x1"]
        ly1 = left.bbox["y1"]
        lx2 = left.bbox["x2"]
        ly2 = left.bbox["y2"]
        rx1 = right.bbox["x1"]
        ry1 = right.bbox["y1"]
        rx2 = right.bbox["x2"]
        ry2 = right.bbox["y2"]

        inter_x1 = max(lx1, rx1)
        inter_y1 = max(ly1, ry1)
        inter_x2 = min(lx2, rx2)
        inter_y2 = min(ly2, ry2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        if intersection <= 0:
            return 0.0

        left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        union = left_area + right_area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _deduplicate(self, detections: list[YoloDetection]) -> list[YoloDetection]:
        ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
        kept: list[YoloDetection] = []
        for candidate in ordered:
            candidate_group = self._label_group(candidate.label)
            if any(candidate_group == self._label_group(existing.label) and self._iou(candidate, existing) >= self.iou_threshold for existing in kept):
                continue
            kept.append(candidate)
        return kept

    def _label_group(self, label: str) -> str:
        normalized = label.strip().lower()
        if any(token in normalized for token in ("tissue", "napkin")):
            return "napkin_tissue_debris"
        if any(token in normalized for token in ("wrapper", "scrap", "paper trash", "small white trash", "crumpled paper")):
            return "wrapper_paper_debris"
        if any(token in normalized for token in ("crumb", "residue", "spill", "stain")):
            return "residue_debris"
        return normalized

    def _color_for_label(self, label: str) -> tuple[int, int, int]:
        seed = sum((index + 1) * ord(character) for index, character in enumerate(label))
        blue = 60 + (seed * 37) % 156
        green = 60 + (seed * 67) % 156
        red = 60 + (seed * 97) % 156
        return int(blue), int(green), int(red)

    def _draw_label(self, image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        box_x1 = max(0, x)
        box_y1 = max(0, y - text_height - 8)
        box_x2 = min(image.shape[1] - 1, box_x1 + text_width + 8)
        box_y2 = min(image.shape[0] - 1, box_y1 + text_height + 6)
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, thickness=-1)
        cv2.putText(image, text, (box_x1 + 4, box_y2 - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _build_crop_box_polygon(self, detection: YoloDetection) -> np.ndarray:
        x1 = detection.bbox["x1"]
        y1 = detection.bbox["y1"]
        x2 = detection.bbox["x2"]
        y2 = detection.bbox["y2"]
        return np.array(
            [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
            dtype=np.float32,
        )

    def _crop_to_source_polygon(self, crop_polygon: np.ndarray, roi: ROI, crop_width: int, crop_height: int) -> np.ndarray:
        destination = np.array(
            [
                [0, 0],
                [crop_width - 1, 0],
                [crop_width - 1, crop_height - 1],
                [0, crop_height - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(destination, ordered_points(roi))
        projected = cv2.perspectiveTransform(crop_polygon, transform)
        return np.round(projected).astype(np.int32)

    def _render_crop_annotations(self, crop_image: np.ndarray, detections: list[YoloDetection], output_path: Path) -> Path:
        annotated = crop_image.copy()
        for detection in detections:
            color = self._color_for_label(detection.label)
            x1 = int(round(detection.bbox["x1"]))
            y1 = int(round(detection.bbox["y1"]))
            x2 = int(round(detection.bbox["x2"]))
            y2 = int(round(detection.bbox["y2"]))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=1)
            self._draw_label(annotated, detection.label, x1, y1, color)
        write_image(output_path, annotated)
        return output_path

    def _render_source_annotations(
        self,
        source_image: np.ndarray,
        roi: ROI,
        crop_width: int,
        crop_height: int,
        detections: list[YoloDetection],
        output_path: Path,
    ) -> Path:
        annotated = source_image.copy()
        for detection in detections:
            color = self._color_for_label(detection.label)
            crop_polygon = self._build_crop_box_polygon(detection)
            source_polygon = self._crop_to_source_polygon(crop_polygon, roi, crop_width, crop_height)
            cv2.polylines(annotated, [source_polygon.reshape(-1, 2)], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
            anchor_x = int(source_polygon[0][0][0])
            anchor_y = int(source_polygon[0][0][1])
            self._draw_label(annotated, detection.label, anchor_x, anchor_y, color)
        write_image(output_path, annotated)
        return output_path

    def _run_detection_pass(
        self,
        model: Any,
        image: np.ndarray,
        *,
        prompt_labels: list[str],
        confidence_threshold: float,
        tile_size: int,
        tile_overlap: float,
        source_prefix: str,
    ) -> list[YoloDetection]:
        height, width = image.shape[:2]
        self._set_model_classes(model, prompt_labels)

        detections: list[YoloDetection] = []
        full_results = model.predict(source=image, conf=confidence_threshold, verbose=False)
        if full_results:
            detections.extend(
                self._extract_detections(
                    full_results[0],
                    offset_x=0,
                    offset_y=0,
                    image_width=width,
                    image_height=height,
                    source=f"{source_prefix}_full_image",
                )
            )

        for index, (x1, y1, x2, y2) in enumerate(self._generate_slices(width, height, tile_size, tile_overlap)):
            tile = image[y1:y2, x1:x2]
            tile_results = model.predict(source=tile, conf=confidence_threshold, verbose=False)
            if not tile_results:
                continue
            detections.extend(
                self._extract_detections(
                    tile_results[0],
                    offset_x=x1,
                    offset_y=y1,
                    image_width=width,
                    image_height=height,
                    source=f"{source_prefix}_tile_{index}",
                )
            )
        return detections

    def detect_objects(self, image_path: Path) -> list[YoloDetection]:
        model = self._load_model()
        image = read_image(image_path)

        detections = self._run_detection_pass(
            model,
            image,
            prompt_labels=self.prompt_labels,
            confidence_threshold=self.confidence_threshold,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            source_prefix="general",
        )
        detections.extend(
            self._run_detection_pass(
                model,
                image,
                prompt_labels=self.debris_prompt_labels,
                confidence_threshold=self.debris_confidence_threshold,
                tile_size=self.debris_tile_size,
                tile_overlap=self.debris_tile_overlap,
                source_prefix="debris",
            )
        )

        return self._deduplicate(detections)

    def build_prompt_instruction(self, detection_payload: dict[str, Any]) -> str:
        detection_json = json.dumps(detection_payload, ensure_ascii=False, indent=2)
        return (
            "You are also given auxiliary open-vocabulary object-detection output for the same ROI crop.\n"
            "The detector was prompted with restaurant cleanliness objects such as napkins, wrappers, residue, trays, bowls, and small trash.\n"
            "A second debris-focused detection pass used tighter tiling and tissue/wrapper-specific prompts to catch tiny crumpled paper or napkin-like trash.\n"
            "Use the detection labels and bounding-box positions as supporting evidence when reasoning about visible objects and cleanliness.\n"
            "Treat the detector as auxiliary evidence only: if the detector conflicts with the image, trust the image and explain uncertainty.\n"
            "Pay special attention to small dirty items such as napkins, wrappers, crumbs, residue, spills, and other table debris.\n"
            "Do not invent objects that are absent from both the image and detector output.\n"
            "Detection JSON:\n"
            f"{detection_json}"
        )

    def prepare_for_cleanliness(
        self,
        crop_image_path: Path,
        *,
        source_image_path: Path | None = None,
        roi: ROI | None = None,
        output_stem: str | None = None,
    ) -> YoloAugmentation:
        detections = self.detect_objects(crop_image_path)
        crop_annotated_path: Path | None = None
        source_annotated_path: Path | None = None
        crop_image = read_image(crop_image_path)

        if output_stem:
            crop_annotated_path = ANALYSIS_CROP_DIR / f"{output_stem}_yolo_crop_boxes.png"
            self._render_crop_annotations(crop_image, detections, crop_annotated_path)

        if output_stem and source_image_path is not None and roi is not None:
            source_image = read_image(source_image_path)
            crop_height, crop_width = crop_image.shape[:2]
            source_annotated_path = ANALYSIS_CROP_DIR / f"{output_stem}_yolo_source_boxes.png"
            self._render_source_annotations(
                source_image,
                roi,
                crop_width,
                crop_height,
                detections,
                source_annotated_path,
            )

        payload = {
            "image_name": crop_image_path.name,
            "detector_name": self._detector_name,
            "weights": self.model_name if self._detector_name == "YOLOE" else self.fallback_model_name,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "prompt_labels": self.prompt_labels,
            "debris_prompt_labels": self.debris_prompt_labels,
            "debris_confidence_threshold": self.debris_confidence_threshold,
            "debris_tile_size": self.debris_tile_size,
            "debris_tile_overlap": self.debris_tile_overlap,
            "crop_annotated_path": crop_annotated_path.name if crop_annotated_path is not None else "",
            "source_annotated_path": source_annotated_path.name if source_annotated_path is not None else "",
            "detections": [item.to_dict() for item in detections],
        }
        return YoloAugmentation(
            image_path=crop_image_path,
            detector_name=self._detector_name,
            weights=payload["weights"],
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            prompt_labels=self.prompt_labels,
            debris_prompt_labels=self.debris_prompt_labels,
            detections=detections,
            crop_annotated_path=crop_annotated_path,
            source_annotated_path=source_annotated_path,
            prompt_instruction=self.build_prompt_instruction(payload),
        )
