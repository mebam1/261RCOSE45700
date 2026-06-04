from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from app.action_cleanliness import CLUTTER_OBJECT_CLASSES, YoloTableState


VISUAL_TRASH_LIKE_CLASSES = CLUTTER_OBJECT_CLASSES | {"wrapper", "napkin", "food_waste"}


OBJECT_CLASS_ALIASES = {
    "trash": "trash",
    "litter": "trash",
    "paper": "wrapper",
    "wrapper": "wrapper",
    "food wrapper": "wrapper",
    "snack wrapper": "wrapper",
    "plastic wrapper": "wrapper",
    "tissue": "napkin",
    "napkin": "napkin",
    "used tissue": "napkin",
    "paper tissue": "napkin",
    "crumpled tissue": "napkin",
    "used napkin": "napkin",
    "paper napkin": "napkin",
    "crumpled napkin": "napkin",
    "plastic_cup": "cup",
    "cup": "cup",
    "plate": "dish",
    "dish": "dish",
    "bowl": "dish",
    "metal bowl": "dish",
    "tray": "tray",
    "metal tray": "tray",
    "leftover": "food_waste",
    "food_residue": "food_waste",
    "food residue": "food_waste",
    "crumbs": "food_waste",
    "person": "person",
}

VISIBLE_CONTAMINATION_LABELS = {"stain", "spill"}


@dataclass(frozen=True)
class VisualMetricResult:
    visual_score: int
    visual_clean_score: float
    visual_mess_score: float
    penalties_breakdown: list[dict[str, Any]]
    bonuses_breakdown: list[dict[str, Any]]
    reason_codes: list[str]
    visual_metric_grade: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_score": self.visual_score,
            "visual_clean_score": round(self.visual_clean_score, 3),
            "visual_mess_score": round(self.visual_mess_score, 3),
            "penalties_breakdown": self.penalties_breakdown,
            "bonuses_breakdown": self.bonuses_breakdown,
            "reason_codes": self.reason_codes,
            "visual_metric_grade": self.visual_metric_grade,
        }


def build_visual_metric_result(visual_state: YoloTableState) -> VisualMetricResult:
    penalties = 0
    bonuses = 0
    penalties_breakdown: list[dict[str, Any]] = []
    bonuses_breakdown: list[dict[str, Any]] = []
    reason_codes: list[str] = []

    trash_like_count = sum(
        item.count for item in visual_state.detected_objects if item.object_class.lower() in VISUAL_TRASH_LIKE_CLASSES
    )
    if trash_like_count > 0:
        penalty_points = min(trash_like_count * 5, 30)
        penalties += penalty_points
        penalties_breakdown.append(
            {
                "code": "TRASH_LIKE_OBJECTS",
                "points": -penalty_points,
                "count": trash_like_count,
            }
        )
        reason_codes.append("TRASH_LIKE_OBJECTS_DETECTED")

    if visual_state.visible_contamination:
        penalties += 10
        penalties_breakdown.append({"code": "VISIBLE_CONTAMINATION", "points": -10})
        reason_codes.append("VISIBLE_CONTAMINATION")

    contamination_penalty = contamination_area_penalty(visual_state.contamination_area_ratio)
    if contamination_penalty > 0:
        penalties += contamination_penalty
        penalties_breakdown.append(
            {
                "code": "CONTAMINATION_AREA_RATIO",
                "points": -contamination_penalty,
                "ratio": round(visual_state.contamination_area_ratio, 3),
            }
        )
        reason_codes.append("CONTAMINATION_AREA_RATIO_HIGH")

    if visual_state.messy_arrangement:
        penalties += 10
        penalties_breakdown.append({"code": "MESSY_ARRANGEMENT", "points": -10})
        reason_codes.append("MESSY_ARRANGEMENT")

    if visual_state.hazardous_contamination:
        penalties += 20
        penalties_breakdown.append({"code": "HAZARDOUS_CONTAMINATION", "points": -20})
        reason_codes.append("HAZARDOUS_CONTAMINATION")

    duration_penalty = tracked_dirty_duration_penalty(visual_state.tracked_dirty_duration_seconds)
    if duration_penalty > 0:
        penalties += duration_penalty
        penalties_breakdown.append(
            {
                "code": "TRACKED_DIRTY_DURATION",
                "points": -duration_penalty,
                "seconds": round(visual_state.tracked_dirty_duration_seconds, 1),
            }
        )
        reason_codes.append("DIRTY_DURATION_PERSISTED")

    if visual_state.removed_objects_count > 0 or visual_state.contamination_removed:
        bonuses += 10
        bonuses_breakdown.append(
            {
                "code": "REMOVED_OBJECTS_OR_CONTAMINATION",
                "points": 10,
                "removed_objects_count": visual_state.removed_objects_count,
                "contamination_removed": visual_state.contamination_removed,
            }
        )
        reason_codes.append("REMOVAL_EVIDENCE")

    if visual_state.clear_cleaning_action_evidence:
        bonuses += 10
        bonuses_breakdown.append({"code": "CLEAR_CLEANING_ACTION_EVIDENCE", "points": 10})
        reason_codes.append("CLEAR_CLEANING_ACTION_EVIDENCE")

    if visual_state.vision_confidence_missing:
        reason_codes.append("MISSING_VISION_CONFIDENCE")

    visual_score = clamp_score(100 - penalties + bonuses)
    visual_clean_score = visual_score / 100.0
    visual_mess_score = 1.0 - visual_clean_score

    return VisualMetricResult(
        visual_score=visual_score,
        visual_clean_score=visual_clean_score,
        visual_mess_score=visual_mess_score,
        penalties_breakdown=penalties_breakdown,
        bonuses_breakdown=bonuses_breakdown,
        reason_codes=reason_codes,
        visual_metric_grade=visual_metric_grade_for_score(visual_score),
    )


def normalize_visual_metric_input(
    payload: YoloTableState | dict[str, Any],
    *,
    table_id: str | None = None,
) -> YoloTableState:
    if isinstance(payload, YoloTableState):
        return payload

    normalized_payload = dict(payload)
    normalized_payload.setdefault("table_id", table_id or str(payload.get("table_id") or "unknown"))
    normalized_payload.setdefault("yolo_mess_score", float(payload.get("yolo_mess_score", 0.0)))
    if "vision_confidence" in payload and payload.get("vision_confidence") is not None:
        normalized_payload.setdefault("vision_confidence", float(payload["vision_confidence"]))
        normalized_payload.setdefault("vision_confidence_missing", False)
    else:
        normalized_payload.setdefault("vision_confidence", 0.5)
        normalized_payload.setdefault("vision_confidence_missing", True)
    normalized_payload["detected_objects"] = normalize_detected_objects(payload)

    scene_flags = payload.get("scene_flags")
    if isinstance(scene_flags, dict):
        normalized_payload.setdefault("visible_contamination", bool(scene_flags.get("visible_contamination", False)))
        normalized_payload.setdefault("messy_arrangement", bool(scene_flags.get("messy_arrangement", False)))
        normalized_payload.setdefault("hazardous_contamination", bool(scene_flags.get("hazardous_contamination", False)))

    flags = payload.get("flags")
    if isinstance(flags, dict):
        normalized_payload.setdefault("visible_contamination", bool(flags.get("visible_contamination", False)))
        normalized_payload.setdefault("messy_arrangement", bool(flags.get("messy_arrangement", False)))
        normalized_payload.setdefault("hazardous_contamination", bool(flags.get("hazardous_contamination", False)))

    normalized_payload["person_exposed"] = bool(
        normalized_payload.get("person_exposed", False) or payload_contains_person(payload)
    )
    return YoloTableState.from_dict(normalized_payload)


def build_visual_metric_result_from_payload(
    payload: YoloTableState | dict[str, Any],
    *,
    table_id: str | None = None,
) -> VisualMetricResult:
    return build_visual_metric_result(normalize_visual_metric_input(payload, table_id=table_id))


def build_visual_payload_from_yolo_detections(
    detections: Sequence[Any],
    *,
    table_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "yolo_objects": [],
    }
    if table_id is not None:
        payload["table_id"] = table_id

    max_score: float | None = None
    visible_contamination = False
    for detection in detections:
        raw_label, score, bbox = parse_yolo_detection(detection)
        max_score = score if max_score is None else max(max_score, score)
        normalized_label = canonical_object_class(raw_label)
        if normalized_label is None:
            if str(raw_label or "").strip().lower() in VISIBLE_CONTAMINATION_LABELS:
                visible_contamination = True
            continue
        payload["yolo_objects"].append(
            {
                "label": normalized_label,
                "score": score,
                "bbox": bbox,
            }
        )

    if max_score is not None:
        payload["vision_confidence"] = max_score
    if visible_contamination:
        payload["visible_contamination"] = True
    return payload


def normalize_detected_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    detected_objects = payload.get("detected_objects")
    if isinstance(detected_objects, list):
        return [normalize_detected_object_summary(item) for item in detected_objects if isinstance(item, dict)]

    raw_objects = payload.get("objects")
    if isinstance(raw_objects, list):
        return summarize_raw_objects(raw_objects, label_key="class", confidence_key="confidence")

    yolo_objects = payload.get("yolo_objects")
    if isinstance(yolo_objects, list):
        return summarize_raw_objects(yolo_objects, label_key="label", confidence_key="score")

    return []


def parse_yolo_detection(detection: Any) -> tuple[str, float, Any]:
    if isinstance(detection, dict):
        label = detection.get("label")
        score = detection.get("score", detection.get("confidence", 0.0))
        bbox = detection.get("bbox", {})
        return str(label or "").strip(), clamp_confidence(score), bbox

    label = getattr(detection, "label", "")
    score = getattr(detection, "score", getattr(detection, "confidence", 0.0))
    bbox = getattr(detection, "bbox", {})
    return str(label or "").strip(), clamp_confidence(score), bbox


def normalize_detected_object_summary(item: dict[str, Any]) -> dict[str, Any]:
    canonical_class = canonical_object_class(item.get("class"))
    if canonical_class is None:
        canonical_class = str(item.get("class") or "unknown").strip().lower() or "unknown"
    return {
        "class": canonical_class,
        "count": max(0, int(item.get("count", 0))),
        "max_confidence": clamp_confidence(item.get("max_confidence", 0.0)),
    }


def summarize_raw_objects(
    raw_objects: list[dict[str, Any]],
    *,
    label_key: str,
    confidence_key: str,
) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for item in raw_objects:
        canonical_class = canonical_object_class(item.get(label_key))
        if canonical_class is None:
            continue
        bucket = summary.setdefault(
            canonical_class,
            {
                "class": canonical_class,
                "count": 0,
                "max_confidence": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["max_confidence"] = max(bucket["max_confidence"], clamp_confidence(item.get(confidence_key, 0.0)))
    return list(summary.values())


def canonical_object_class(raw_label: Any) -> str | None:
    label = str(raw_label or "").strip().lower()
    if not label:
        return None
    return OBJECT_CLASS_ALIASES.get(label, label if label in VISUAL_TRASH_LIKE_CLASSES else None)


def payload_contains_person(payload: dict[str, Any]) -> bool:
    for key in ("objects", "yolo_objects", "detected_objects"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            label = item.get("class")
            if key == "yolo_objects":
                label = item.get("label")
            if canonical_object_class(label) == "person":
                return True
    return False


def clamp_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def contamination_area_penalty(contamination_area_ratio: float) -> int:
    if contamination_area_ratio >= 0.20:
        return 20
    if contamination_area_ratio >= 0.10:
        return 10
    if contamination_area_ratio >= 0.05:
        return 5
    return 0


def tracked_dirty_duration_penalty(tracked_dirty_duration_seconds: float) -> int:
    if tracked_dirty_duration_seconds >= 1200:
        return 15
    if tracked_dirty_duration_seconds >= 600:
        return 10
    if tracked_dirty_duration_seconds >= 300:
        return 5
    return 0


def visual_metric_grade_for_score(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 70:
        return "B"
    if score >= 50:
        return "C"
    if score >= 30:
        return "D"
    return "E"


def clamp_score(score: int) -> int:
    return max(0, min(100, score))
