from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from app.schemas import ROI


ACTION_DECISION_CLEANED = "cleaned_likely"
ACTION_DECISION_NEEDS_CHECK = "needs_check"
ACTION_DECISION_UNKNOWN = "unknown"
ACTION_DECISION_IN_USE = "in_use"

ACTION_CLEANED_DWELL_SECONDS = 35.0
ACTION_NEEDS_CHECK_DWELL_SECONDS = 15.0
ACTION_SCORE_2_DWELL_SECONDS = 5.0
ACTION_SCORE_5_DWELL_SECONDS = 50.0

MEAL_STATUS_IDLE = "IDLE"
MEAL_STATUS_CUSTOMER_IN_USE = "CUSTOMER_IN_USE"
MEAL_STATUS_POSSIBLY_LEFT = "POSSIBLY_LEFT"
MEAL_STATUS_MEAL_ENDED = "MEAL_ENDED"

CLEANING_STATUS_NEED_CLEANING = "NEED_CLEANING"
CLEANING_STATUS_CLEANING_CANDIDATE = "CLEANING_CANDIDATE"
CLEANING_STATUS_CLEANED_LIKELY = "CLEANED_LIKELY"
CLEANING_STATUS_CLEANING_UNVERIFIED = "CLEANING_UNVERIFIED"
CLEANING_STATUS_MISSED_SUSPECTED = "MISSED_SUSPECTED"
CLEANING_STATUS_UNCERTAIN = "UNCERTAIN"

ACTION_SCORE_BY_CLEANING_STATUS = {
    CLEANING_STATUS_CLEANED_LIKELY: 0.85,
    CLEANING_STATUS_CLEANING_UNVERIFIED: 0.60,
    CLEANING_STATUS_NEED_CLEANING: 0.45,
    CLEANING_STATUS_MISSED_SUSPECTED: 0.15,
    CLEANING_STATUS_UNCERTAIN: 0.50,
}

CLEANING_REASON_CUSTOMER_LEFT_CONFIRMED = "CUSTOMER_LEFT_CONFIRMED"
CLEANING_REASON_STAFF_ZONE_DWELL_CONFIRMED = "STAFF_ZONE_DWELL_CONFIRMED"
CLEANING_REASON_TABLE_STATE_CHANGED = "TABLE_STATE_CHANGED"
CLEANING_REASON_MESS_SCORE_REDUCED = "MESS_SCORE_REDUCED"
CLEANING_REASON_OBJECT_COUNT_REDUCED = "OBJECT_COUNT_REDUCED"
CLEANING_REASON_CLUTTER_OBJECTS_REDUCED = "CLUTTER_OBJECTS_REDUCED"
CLEANING_REASON_POST_FRAME_CLEANUP_CONFIRMED = "POST_FRAME_CLEANUP_CONFIRMED"
CLEANING_REASON_SLA_EXPIRED = "SLA_EXPIRED"

CLUTTER_OBJECT_CLASSES = {"trash", "cup", "tray", "dish"}

SAMPLE_ORDER_TIME = "2026-05-21T12:00:00"
SAMPLE_PAYMENT_COMPLETED_TIME = "2026-05-21T12:00:30"
SAMPLE_TRAJECTORY_JSON = """[
  {"t": "2026-05-21T12:00:00", "x": 120, "y": 260, "staff_id": "staff-a"},
  {"t": "2026-05-21T12:00:08", "x": 165, "y": 250, "staff_id": "staff-a"},
  {"t": "2026-05-21T12:00:18", "x": 235, "y": 245, "staff_id": "staff-a"},
  {"t": "2026-05-21T12:00:30", "x": 310, "y": 255, "staff_id": "staff-a"},
  {"t": "2026-05-21T12:00:42", "x": 315, "y": 330, "staff_id": "staff-a"},
  {"t": "2026-05-21T12:00:55", "x": 225, "y": 340, "staff_id": "staff-a"},
  {"t": "2026-05-21T12:01:08", "x": 150, "y": 330, "staff_id": "staff-a"},
  {"t": "2026-05-21T12:01:22", "x": 110, "y": 285, "staff_id": "staff-a"}
]"""


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parse_timeline_value(raw_value: str | int | float, field_name: str) -> float:
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")

    try:
        return float(value)
    except ValueError:
        pass

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a datetime-local, ISO datetime, or numeric timeline value") from exc


def parse_datetime_value(raw_value: str | datetime, field_name: str) -> datetime:
    if isinstance(raw_value, datetime):
        return raw_value

    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO datetime string") from exc


def format_datetime_value(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat(timespec="seconds")


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float
    x: float
    y: float
    staff_id: str = "staff"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrajectoryPoint":
        if "t" not in payload or "x" not in payload or "y" not in payload:
            raise ValueError("trajectory point must include t, x, and y")
        return cls(
            t=parse_timeline_value(payload["t"], "trajectory point t"),
            x=float(payload["x"]),
            y=float(payload["y"]),
            staff_id=str(payload.get("staff_id") or payload.get("marker_id") or "staff"),
        )


@dataclass(frozen=True)
class ActionCleanlinessResult:
    decision: str
    score: int | None
    confidence: float
    summary: str
    evidence: list[str]
    features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "score": self.score,
            "score_label": "사용중" if self.score is None else f"{self.score} / 5",
            "confidence": round(self.confidence, 3),
            "summary": self.summary,
            "evidence": self.evidence,
            "features": self.features,
        }


@dataclass(frozen=True)
class TableOccupancySample:
    timestamp: datetime
    customer_present: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TableOccupancySample":
        return cls(
            timestamp=parse_datetime_value(payload["timestamp"], "timestamp"),
            customer_present=bool(payload["customer_present"]),
        )


@dataclass(frozen=True)
class StaffZoneVisit:
    visit_id: str
    store_id: str
    staff_id: str
    zone_id: str
    entered_at: datetime
    left_at: datetime
    dwell_seconds: float
    mean_confidence: float
    sample_count: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StaffZoneVisit":
        entered_at = parse_datetime_value(payload["entered_at"], "entered_at")
        left_at = parse_datetime_value(payload["left_at"], "left_at")
        computed_dwell = max(0.0, (left_at - entered_at).total_seconds())
        return cls(
            visit_id=str(payload["visit_id"]),
            store_id=str(payload["store_id"]),
            staff_id=str(payload["staff_id"]),
            zone_id=str(payload["zone_id"]),
            entered_at=entered_at,
            left_at=left_at,
            dwell_seconds=float(payload.get("dwell_seconds", computed_dwell)),
            mean_confidence=clamp(float(payload.get("mean_confidence", 0.0)), 0.0, 1.0),
            sample_count=max(0, int(payload.get("sample_count", 0))),
        )


@dataclass(frozen=True)
class ZoneDefinition:
    zone_id: str
    beacon_ids: list[str]
    table_ids: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ZoneDefinition":
        return cls(
            zone_id=str(payload["zone_id"]),
            beacon_ids=[str(item) for item in payload.get("beacon_ids", [])],
            table_ids=[str(item) for item in payload.get("table_ids", [])],
        )


@dataclass(frozen=True)
class ZoneTableMapping:
    store_id: str
    zones: list[ZoneDefinition]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ZoneTableMapping":
        return cls(
            store_id=str(payload["store_id"]),
            zones=[ZoneDefinition.from_dict(item) for item in payload.get("zones", [])],
        )

    def zone_for_table(self, table_id: str) -> str:
        for zone in self.zones:
            if table_id in zone.table_ids:
                return zone.zone_id
        raise KeyError(f"table_id {table_id} is not mapped to a zone")


@dataclass(frozen=True)
class MealSession:
    meal_session_id: str
    table_id: str
    status: str
    meal_started_at: datetime
    last_customer_seen_at: datetime
    possibly_left_started_at: datetime | None = None
    meal_ended_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "meal_session_id": self.meal_session_id,
            "table_id": self.table_id,
            "status": self.status,
            "meal_started_at": format_datetime_value(self.meal_started_at),
            "last_customer_seen_at": format_datetime_value(self.last_customer_seen_at),
            "possibly_left_started_at": format_datetime_value(self.possibly_left_started_at),
            "meal_ended_at": format_datetime_value(self.meal_ended_at),
        }


@dataclass(frozen=True)
class CleaningTask:
    cleaning_task_id: str
    meal_session_id: str
    table_id: str
    zone_id: str
    created_at: datetime
    status: str
    sla_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cleaning_task_id": self.cleaning_task_id,
            "meal_session_id": self.meal_session_id,
            "table_id": self.table_id,
            "zone_id": self.zone_id,
            "created_at": format_datetime_value(self.created_at),
            "status": self.status,
            "sla_seconds": self.sla_seconds,
        }


@dataclass(frozen=True)
class CleaningCandidate:
    table_id: str
    zone_id: str
    staff_zone_visit: StaffZoneVisit

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "zone_id": self.zone_id,
            "staff_zone_visit": {
                "visit_id": self.staff_zone_visit.visit_id,
                "staff_id": self.staff_zone_visit.staff_id,
                "entered_at": format_datetime_value(self.staff_zone_visit.entered_at),
                "left_at": format_datetime_value(self.staff_zone_visit.left_at),
                "dwell_seconds": round(self.staff_zone_visit.dwell_seconds, 2),
                "mean_confidence": round(self.staff_zone_visit.mean_confidence, 3),
            },
        }


@dataclass(frozen=True)
class DetectedObjectSummary:
    object_class: str
    count: int
    max_confidence: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DetectedObjectSummary":
        return cls(
            object_class=str(payload["class"]),
            count=max(0, int(payload.get("count", 0))),
            max_confidence=clamp(float(payload.get("max_confidence", 0.0)), 0.0, 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.object_class,
            "count": self.count,
            "max_confidence": round(self.max_confidence, 3),
        }


@dataclass(frozen=True)
class YoloTableState:
    table_id: str
    yolo_mess_score: float
    visual_clean_score: float
    vision_confidence: float
    detected_objects: list[DetectedObjectSummary]
    vision_confidence_missing: bool = False
    captured_at: datetime | None = None
    camera_occluded: bool = False
    cleanup_evidence_confirmed: bool = False
    visible_contamination: bool = False
    contamination_area_ratio: float = 0.0
    messy_arrangement: bool = False
    hazardous_contamination: bool = False
    tracked_dirty_duration_seconds: float = 0.0
    person_exposed: bool = False
    removed_objects_count: int = 0
    contamination_removed: bool = False
    clear_cleaning_action_evidence: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "YoloTableState":
        mess_score = clamp(float(payload["yolo_mess_score"]), 0.0, 1.0)
        visual_clean_score = payload.get("visual_clean_score")
        if visual_clean_score is None:
            visual_clean_score = 1.0 - mess_score
        captured_at = payload.get("captured_at")
        return cls(
            table_id=str(payload["table_id"]),
            yolo_mess_score=mess_score,
            visual_clean_score=clamp(float(visual_clean_score), 0.0, 1.0),
            vision_confidence=clamp(float(payload.get("vision_confidence", 0.0)), 0.0, 1.0),
            vision_confidence_missing=bool(payload.get("vision_confidence_missing", False)),
            detected_objects=[DetectedObjectSummary.from_dict(item) for item in payload.get("detected_objects", [])],
            captured_at=parse_datetime_value(captured_at, "captured_at") if captured_at else None,
            camera_occluded=bool(payload.get("camera_occluded", False)),
            cleanup_evidence_confirmed=bool(payload.get("cleanup_evidence_confirmed", False)),
            visible_contamination=bool(payload.get("visible_contamination", False)),
            contamination_area_ratio=clamp(float(payload.get("contamination_area_ratio", 0.0)), 0.0, 1.0),
            messy_arrangement=bool(payload.get("messy_arrangement", False)),
            hazardous_contamination=bool(payload.get("hazardous_contamination", False)),
            tracked_dirty_duration_seconds=max(0.0, float(payload.get("tracked_dirty_duration_seconds", 0.0))),
            person_exposed=bool(payload.get("person_exposed", False)),
            removed_objects_count=max(0, int(payload.get("removed_objects_count", 0))),
            contamination_removed=bool(payload.get("contamination_removed", False)),
            clear_cleaning_action_evidence=bool(payload.get("clear_cleaning_action_evidence", False)),
        )

    @property
    def total_detected_object_count(self) -> int:
        return sum(item.count for item in self.detected_objects)

    @property
    def clutter_object_count(self) -> int:
        return sum(item.count for item in self.detected_objects if item.object_class.lower() in CLUTTER_OBJECT_CLASSES)


@dataclass(frozen=True)
class TableStateChange:
    changed: bool
    reason_codes: list[str]
    mess_score_delta: float
    object_count_before: int
    object_count_after: int
    clutter_count_before: int
    clutter_count_after: int


@dataclass(frozen=True)
class ActionWorkflowResult:
    table_id: str
    zone_id: str
    meal_status: str
    cleaning_status: str
    action_score: float
    action_confidence: float
    meal_started_at: datetime | None
    meal_ended_at: datetime | None
    cleaning_started_at: datetime | None
    cleaning_ended_at: datetime | None
    reason_codes: list[str]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "zone_id": self.zone_id,
            "meal_status": self.meal_status,
            "cleaning_status": self.cleaning_status,
            "action_score": round(self.action_score, 2),
            "action_confidence": round(self.action_confidence, 2),
            "meal_started_at": format_datetime_value(self.meal_started_at),
            "meal_ended_at": format_datetime_value(self.meal_ended_at),
            "cleaning_started_at": format_datetime_value(self.cleaning_started_at),
            "cleaning_ended_at": format_datetime_value(self.cleaning_ended_at),
            "reason_codes": self.reason_codes,
            "explanation": self.explanation,
        }


def parse_trajectory_json(raw_json: str) -> list[TrajectoryPoint]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid trajectory JSON: {exc}") from exc

    if isinstance(payload, dict):
        payload = payload.get("points")
    if not isinstance(payload, list):
        raise ValueError("trajectory JSON must be a list or an object with a points list")

    points: list[TrajectoryPoint] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("trajectory points must be objects")
        points.append(TrajectoryPoint.from_dict(item))
    return points


def parse_action_time_value(raw_value: str | int | float | None, field_name: str) -> float | None:
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if not value:
        return None

    return parse_timeline_value(value, field_name)


def distance_to_rect(point: TrajectoryPoint, bounds: dict[str, int]) -> float:
    x_min = float(bounds["x"])
    y_min = float(bounds["y"])
    x_max = x_min + float(bounds["width"])
    y_max = y_min + float(bounds["height"])
    dx = max(x_min - point.x, 0.0, point.x - x_max)
    dy = max(y_min - point.y, 0.0, point.y - y_max)
    return math.hypot(dx, dy)


def point_sides_near_rect(point: TrajectoryPoint, bounds: dict[str, int], margin: float) -> set[str]:
    x_min = float(bounds["x"])
    y_min = float(bounds["y"])
    x_max = x_min + float(bounds["width"])
    y_max = y_min + float(bounds["height"])
    inside_expanded_y = y_min - margin <= point.y <= y_max + margin
    inside_expanded_x = x_min - margin <= point.x <= x_max + margin

    sides: set[str] = set()
    if inside_expanded_y and abs(point.x - x_min) <= margin:
        sides.add("left")
    if inside_expanded_y and abs(point.x - x_max) <= margin:
        sides.add("right")
    if inside_expanded_x and abs(point.y - y_min) <= margin:
        sides.add("top")
    if inside_expanded_x and abs(point.y - y_max) <= margin:
        sides.add("bottom")
    return sides


class ActionCleanlinessService:
    def __init__(
        self,
        near_margin: float = 80.0,
        max_gap_seconds: float = 30.0,
        stop_speed_pixels_per_second: float = 18.0,
        customer_in_use_seconds: int = 60,
        meal_min_duration_seconds: int | None = None,
        meal_end_seconds: int = 90,
        cleaning_candidate_dwell_seconds: int = 10,
        cleaning_sla_seconds: int = 300,
    ) -> None:
        self.near_margin = near_margin
        self.max_gap_seconds = max_gap_seconds
        self.stop_speed_pixels_per_second = stop_speed_pixels_per_second
        self.customer_in_use_seconds = customer_in_use_seconds
        self.meal_min_duration_seconds = meal_min_duration_seconds
        self.meal_end_seconds = meal_end_seconds
        self.cleaning_candidate_dwell_seconds = cleaning_candidate_dwell_seconds
        self.cleaning_sla_seconds = cleaning_sla_seconds

    def evaluate(
        self,
        roi: ROI,
        trajectory_points: list[TrajectoryPoint],
        *,
        near_margin: float | None = None,
        manual_confirmed: bool = False,
        order_time: float | None = None,
        payment_completed_time: float | None = None,
    ) -> ActionCleanlinessResult:
        if (
            order_time is not None
            and payment_completed_time is not None
            and payment_completed_time < order_time
        ):
            raise ValueError("payment_completed_time must be greater than or equal to order_time")

        if (
            trajectory_points
            and order_time is not None
            and payment_completed_time is not None
            and max(point.t for point in trajectory_points) < payment_completed_time
        ):
            return self.build_in_use_result(
                order_time=order_time,
                payment_completed_time=payment_completed_time,
                latest_trajectory_time=max(point.t for point in trajectory_points),
                near_margin=near_margin,
            )

        if not trajectory_points:
            return ActionCleanlinessResult(
                decision=ACTION_DECISION_UNKNOWN,
                score=1,
                confidence=0.0,
                summary="결제 이후 직원 궤적이 제공되지 않아 청소 완료 추정을 할 수 없습니다.",
                evidence=["결제 이후 직원 위치 데이터가 없습니다."],
                features=self.empty_features(),
            )

        margin = float(near_margin if near_margin is not None else self.near_margin)
        bounds = roi.bounds
        points_by_staff = self.group_points(trajectory_points)
        analysis_start_time = payment_completed_time

        near_point_count = 0
        visited_sides: set[str] = set()
        dwell_seconds = 0.0
        stopped_seconds = 0.0
        approach_count = 0
        side_transition_count = 0
        staff_with_near_visit: set[str] = set()

        all_times = [point.t for point in trajectory_points]
        if all_times and analysis_start_time is not None:
            observation_seconds = max(0.0, max(all_times) - analysis_start_time)
        else:
            observation_seconds = max(all_times) - min(all_times) if all_times else 0.0

        for staff_id, points in points_by_staff.items():
            sorted_points = sorted(points, key=lambda item: item.t)
            previous_near = False
            previous_side: str | None = None

            for index, point in enumerate(sorted_points):
                point_in_analysis_window = analysis_start_time is None or point.t >= analysis_start_time
                is_near = distance_to_rect(point, bounds) <= margin
                sides = point_sides_near_rect(point, bounds, margin)

                if point_in_analysis_window and is_near:
                    near_point_count += 1
                    staff_with_near_visit.add(staff_id)
                    visited_sides.update(sides)
                    current_side = sorted(sides)[0] if sides else None
                    if previous_side and current_side and previous_side != current_side:
                        side_transition_count += 1
                    if current_side:
                        previous_side = current_side
                if point_in_analysis_window and is_near and not previous_near:
                    approach_count += 1
                previous_near = is_near

                if index + 1 >= len(sorted_points):
                    continue
                next_point = sorted_points[index + 1]
                interval = next_point.t - point.t
                if interval <= 0 or interval > self.max_gap_seconds:
                    continue
                interval_start = point.t
                interval_end = next_point.t
                if analysis_start_time is not None:
                    if interval_end <= analysis_start_time:
                        continue
                    interval_start = max(interval_start, analysis_start_time)
                next_near = distance_to_rect(next_point, bounds) <= margin
                if not (is_near and next_near):
                    continue
                counted_interval = interval_end - interval_start
                if counted_interval <= 0:
                    continue
                dwell_seconds += counted_interval
                distance = math.hypot(next_point.x - point.x, next_point.y - point.y)
                speed = distance / interval
                if speed <= self.stop_speed_pixels_per_second:
                    stopped_seconds += counted_interval

        coverage_ratio = len(visited_sides) / 4.0
        features: dict[str, Any] = {
            "analysis_status": "evaluated",
            "order_time": order_time,
            "payment_completed_time": payment_completed_time,
            "excluded_order_payment_seconds": round(payment_completed_time - order_time, 2)
            if order_time is not None and payment_completed_time is not None
            else 0.0,
            "analysis_start_time": analysis_start_time,
            "observation_seconds": round(observation_seconds, 2),
            "near_point_count": near_point_count,
            "dwell_seconds": round(dwell_seconds, 2),
            "stopped_seconds": round(stopped_seconds, 2),
            "coverage_ratio": round(coverage_ratio, 3),
            "visited_sides": sorted(visited_sides),
            "approach_count": approach_count,
            "side_transition_count": side_transition_count,
            "staff_count": len(points_by_staff),
            "staff_with_near_visit_count": len(staff_with_near_visit),
            "near_margin": round(margin, 2),
        }
        return self.classify(features, manual_confirmed=manual_confirmed)

    def evaluate_meal_session(
        self,
        table_id: str,
        observations: Sequence[TableOccupancySample],
        *,
        meal_session_id: str | None = None,
    ) -> MealSession | None:
        if not observations:
            return None

        sorted_observations = sorted(observations, key=lambda item: item.timestamp)
        meal_confirmation_seconds = max(
            self.customer_in_use_seconds,
            self.meal_min_duration_seconds or self.customer_in_use_seconds,
        )
        presence_started_at: datetime | None = None
        meal_started_at: datetime | None = None
        last_customer_seen_at: datetime | None = None
        possibly_left_started_at: datetime | None = None

        for observation in sorted_observations:
            if observation.customer_present:
                if presence_started_at is None:
                    presence_started_at = observation.timestamp
                last_customer_seen_at = observation.timestamp
                if meal_started_at is None:
                    present_duration = (observation.timestamp - presence_started_at).total_seconds()
                    if present_duration >= meal_confirmation_seconds:
                        meal_started_at = presence_started_at
                if meal_started_at is not None:
                    possibly_left_started_at = None
                continue

            presence_started_at = None
            if meal_started_at is None or last_customer_seen_at is None:
                continue

            if possibly_left_started_at is None:
                possibly_left_started_at = observation.timestamp
                continue

            absent_duration = (observation.timestamp - possibly_left_started_at).total_seconds()
            if absent_duration >= self.meal_end_seconds:
                meal_ended_at = possibly_left_started_at + timedelta(seconds=self.meal_end_seconds)
                return MealSession(
                    meal_session_id=meal_session_id or f"meal_{table_id}_{meal_started_at.strftime('%Y%m%d%H%M%S')}",
                    table_id=table_id,
                    status=MEAL_STATUS_MEAL_ENDED,
                    meal_started_at=meal_started_at,
                    last_customer_seen_at=last_customer_seen_at,
                    possibly_left_started_at=possibly_left_started_at,
                    meal_ended_at=meal_ended_at,
                )

        if meal_started_at is None or last_customer_seen_at is None:
            return None

        status = MEAL_STATUS_POSSIBLY_LEFT if possibly_left_started_at is not None else MEAL_STATUS_CUSTOMER_IN_USE
        return MealSession(
            meal_session_id=meal_session_id or f"meal_{table_id}_{meal_started_at.strftime('%Y%m%d%H%M%S')}",
            table_id=table_id,
            status=status,
            meal_started_at=meal_started_at,
            last_customer_seen_at=last_customer_seen_at,
            possibly_left_started_at=possibly_left_started_at,
            meal_ended_at=None,
        )

    def create_cleaning_task(
        self,
        meal_session: MealSession,
        zone_table_mapping: ZoneTableMapping,
        *,
        cleaning_task_id: str | None = None,
    ) -> CleaningTask:
        if meal_session.status != MEAL_STATUS_MEAL_ENDED or meal_session.meal_ended_at is None:
            raise ValueError("cleaning task can only be created after MEAL_ENDED")

        zone_id = zone_table_mapping.zone_for_table(meal_session.table_id)
        return CleaningTask(
            cleaning_task_id=cleaning_task_id or f"clean_task_{meal_session.meal_session_id}",
            meal_session_id=meal_session.meal_session_id,
            table_id=meal_session.table_id,
            zone_id=zone_id,
            created_at=meal_session.meal_ended_at,
            status=CLEANING_STATUS_NEED_CLEANING,
            sla_seconds=self.cleaning_sla_seconds,
        )

    def find_cleaning_candidate(
        self,
        cleaning_task: CleaningTask,
        staff_zone_visits: Sequence[StaffZoneVisit],
    ) -> CleaningCandidate | None:
        candidates = sorted(staff_zone_visits, key=lambda item: item.entered_at)
        for visit in candidates:
            if visit.zone_id != cleaning_task.zone_id:
                continue
            if visit.entered_at < cleaning_task.created_at:
                continue
            if visit.dwell_seconds < self.cleaning_candidate_dwell_seconds:
                continue
            return CleaningCandidate(
                table_id=cleaning_task.table_id,
                zone_id=cleaning_task.zone_id,
                staff_zone_visit=visit,
            )
        return None

    def compare_table_states(
        self,
        before_state: YoloTableState,
        after_state: YoloTableState,
    ) -> TableStateChange:
        if before_state.table_id != after_state.table_id:
            raise ValueError("before_state and after_state must refer to the same table")

        reason_codes: list[str] = []
        mess_score_delta = before_state.yolo_mess_score - after_state.yolo_mess_score
        if mess_score_delta >= 0.25:
            reason_codes.append(CLEANING_REASON_MESS_SCORE_REDUCED)
        if after_state.total_detected_object_count < before_state.total_detected_object_count:
            reason_codes.append(CLEANING_REASON_OBJECT_COUNT_REDUCED)
        if after_state.clutter_object_count < before_state.clutter_object_count:
            reason_codes.append(CLEANING_REASON_CLUTTER_OBJECTS_REDUCED)
        if after_state.cleanup_evidence_confirmed:
            reason_codes.append(CLEANING_REASON_POST_FRAME_CLEANUP_CONFIRMED)

        return TableStateChange(
            changed=bool(reason_codes),
            reason_codes=reason_codes,
            mess_score_delta=round(mess_score_delta, 3),
            object_count_before=before_state.total_detected_object_count,
            object_count_after=after_state.total_detected_object_count,
            clutter_count_before=before_state.clutter_object_count,
            clutter_count_after=after_state.clutter_object_count,
        )

    def evaluate_cleaning_task(
        self,
        cleaning_task: CleaningTask,
        *,
        before_state: YoloTableState,
        after_state: YoloTableState,
        staff_zone_visits: Sequence[StaffZoneVisit],
        meal_session: MealSession | None = None,
        as_of: datetime | None = None,
    ) -> ActionWorkflowResult:
        if cleaning_task.status != CLEANING_STATUS_NEED_CLEANING:
            raise ValueError("cleaning_task status must be NEED_CLEANING")
        if before_state.table_id != cleaning_task.table_id or after_state.table_id != cleaning_task.table_id:
            raise ValueError("YOLO states must match the cleaning task table")

        reference_time = as_of or after_state.captured_at or cleaning_task.created_at
        candidate = self.find_cleaning_candidate(cleaning_task, staff_zone_visits)
        state_change = self.compare_table_states(before_state, after_state)

        reason_codes = [CLEANING_REASON_CUSTOMER_LEFT_CONFIRMED]
        cleaning_started_at: datetime | None = None
        cleaning_ended_at: datetime | None = None

        if candidate is not None:
            reason_codes.append(CLEANING_REASON_STAFF_ZONE_DWELL_CONFIRMED)
            cleaning_started_at = candidate.staff_zone_visit.entered_at
            cleaning_ended_at = candidate.staff_zone_visit.left_at
            if state_change.changed:
                reason_codes.append(CLEANING_REASON_TABLE_STATE_CHANGED)
                reason_codes.extend(state_change.reason_codes)
                cleaning_status = CLEANING_STATUS_CLEANED_LIKELY
            else:
                cleaning_status = CLEANING_STATUS_CLEANING_UNVERIFIED
        else:
            elapsed_since_task = max(0.0, (reference_time - cleaning_task.created_at).total_seconds())
            if elapsed_since_task >= cleaning_task.sla_seconds and not state_change.changed:
                reason_codes.append(CLEANING_REASON_SLA_EXPIRED)
                cleaning_status = CLEANING_STATUS_MISSED_SUSPECTED
            else:
                cleaning_status = CLEANING_STATUS_NEED_CLEANING

        action_score = ACTION_SCORE_BY_CLEANING_STATUS[cleaning_status]
        action_confidence = self.compute_action_confidence(
            cleaning_status=cleaning_status,
            candidate=candidate,
            after_state=after_state,
            state_change=state_change,
        )
        explanation = self.build_cleaning_explanation(
            cleaning_status=cleaning_status,
            candidate=candidate,
            state_change=state_change,
        )

        return ActionWorkflowResult(
            table_id=cleaning_task.table_id,
            zone_id=cleaning_task.zone_id,
            meal_status=meal_session.status if meal_session is not None else MEAL_STATUS_MEAL_ENDED,
            cleaning_status=cleaning_status,
            action_score=action_score,
            action_confidence=action_confidence,
            meal_started_at=meal_session.meal_started_at if meal_session is not None else None,
            meal_ended_at=meal_session.meal_ended_at if meal_session is not None else cleaning_task.created_at,
            cleaning_started_at=cleaning_started_at,
            cleaning_ended_at=cleaning_ended_at,
            reason_codes=reason_codes,
            explanation=explanation,
        )

    @staticmethod
    def group_points(points: list[TrajectoryPoint]) -> dict[str, list[TrajectoryPoint]]:
        grouped: dict[str, list[TrajectoryPoint]] = {}
        for point in points:
            grouped.setdefault(point.staff_id, []).append(point)
        return grouped

    @staticmethod
    def empty_features() -> dict[str, Any]:
        return {
            "analysis_status": "evaluated",
            "order_time": None,
            "payment_completed_time": None,
            "excluded_order_payment_seconds": 0.0,
            "analysis_start_time": None,
            "observation_seconds": 0.0,
            "near_point_count": 0,
            "dwell_seconds": 0.0,
            "stopped_seconds": 0.0,
            "coverage_ratio": 0.0,
            "visited_sides": [],
            "approach_count": 0,
            "side_transition_count": 0,
            "staff_count": 0,
            "staff_with_near_visit_count": 0,
            "near_margin": 0.0,
        }

    @staticmethod
    def build_in_use_result(
        *,
        order_time: float,
        payment_completed_time: float,
        latest_trajectory_time: float,
        near_margin: float | None,
    ) -> ActionCleanlinessResult:
        features: dict[str, Any] = {
            **ActionCleanlinessService.empty_features(),
            "analysis_status": "in_use",
            "order_time": order_time,
            "payment_completed_time": payment_completed_time,
            "excluded_order_payment_seconds": round(payment_completed_time - order_time, 2),
            "latest_trajectory_time": round(latest_trajectory_time, 2),
            "near_margin": round(float(near_margin or 0.0), 2),
        }
        return ActionCleanlinessResult(
            decision=ACTION_DECISION_IN_USE,
            score=None,
            confidence=1.0,
            summary="주문 시점부터 결제 완료 시점까지는 고객 사용 구간이므로 청결도 점수를 계산하지 않습니다.",
            evidence=[
                "해당 테이블은 아직 고객 사용 구간입니다.",
                "주문 시점부터 결제 완료 시점까지의 궤적은 청소 판단에서 제외됩니다.",
                "점수 표시는 사용중으로 유지합니다.",
            ],
            features=features,
        )

    @staticmethod
    def score_from_dwell_seconds(dwell_seconds: float) -> int:
        if dwell_seconds < ACTION_SCORE_2_DWELL_SECONDS:
            return 1
        if dwell_seconds < ACTION_NEEDS_CHECK_DWELL_SECONDS:
            return 2
        if dwell_seconds < ACTION_CLEANED_DWELL_SECONDS:
            return 3
        if dwell_seconds < ACTION_SCORE_5_DWELL_SECONDS:
            return 4
        return 5

    @staticmethod
    def classify(
        features: dict[str, Any],
        *,
        manual_confirmed: bool,
    ) -> ActionCleanlinessResult:
        dwell_seconds = float(features["dwell_seconds"])
        score = ActionCleanlinessService.score_from_dwell_seconds(dwell_seconds)
        evidence = [
            f"테이블 주변 체류 시간 {dwell_seconds:.1f}초",
            f"Action 청결도 점수 {score} / 5",
            f"청소 완료 추정 기준 {ACTION_CLEANED_DWELL_SECONDS:.0f}초 이상",
            f"확인 필요 기준 {ACTION_NEEDS_CHECK_DWELL_SECONDS:.0f}초 미만",
        ]

        if manual_confirmed:
            return ActionCleanlinessResult(
                decision=ACTION_DECISION_CLEANED,
                score=score,
                confidence=0.95,
                summary="직원 확인 입력이 있어 청소 완료 추정 가능성이 높습니다.",
                evidence=evidence + ["직원 수동 확인이 포함되었습니다."],
                features=features,
            )

        if dwell_seconds >= ACTION_CLEANED_DWELL_SECONDS:
            confidence = min(0.95, 0.7 + min((dwell_seconds - ACTION_CLEANED_DWELL_SECONDS) / 60.0, 0.25))
            return ActionCleanlinessResult(
                decision=ACTION_DECISION_CLEANED,
                score=score,
                confidence=confidence,
                summary="결제 이후 직원이 테이블 주변에 충분히 머문 것으로 보입니다.",
                evidence=evidence,
                features=features,
            )

        if dwell_seconds < ACTION_NEEDS_CHECK_DWELL_SECONDS:
            confidence = min(0.9, 0.65 + min((ACTION_NEEDS_CHECK_DWELL_SECONDS - dwell_seconds) / 60.0, 0.25))
            return ActionCleanlinessResult(
                decision=ACTION_DECISION_NEEDS_CHECK,
                score=score,
                confidence=confidence,
                summary="결제 이후 직원 체류 시간이 짧아 추가 확인이 필요합니다.",
                evidence=evidence,
                features=features,
            )

        confidence = min(0.62, max(0.3, dwell_seconds / ACTION_CLEANED_DWELL_SECONDS * 0.6))
        return ActionCleanlinessResult(
            decision=ACTION_DECISION_UNKNOWN,
            score=score,
            confidence=confidence,
            summary="테이블 주변 체류 시간만으로는 청소 완료 추정을 확정하기 어렵습니다.",
            evidence=evidence,
            features=features,
        )

    @staticmethod
    def compute_action_confidence(
        *,
        cleaning_status: str,
        candidate: CleaningCandidate | None,
        after_state: YoloTableState,
        state_change: TableStateChange,
    ) -> float:
        base = {
            CLEANING_STATUS_CLEANED_LIKELY: 0.78,
            CLEANING_STATUS_CLEANING_UNVERIFIED: 0.66,
            CLEANING_STATUS_NEED_CLEANING: 0.58,
            CLEANING_STATUS_MISSED_SUSPECTED: 0.74,
            CLEANING_STATUS_UNCERTAIN: 0.50,
        }[cleaning_status]

        if candidate is not None:
            base += max(0.0, candidate.staff_zone_visit.mean_confidence - 0.5) * 0.2
        if state_change.changed:
            base += 0.08
        base += max(0.0, after_state.vision_confidence - 0.5) * 0.2
        if after_state.camera_occluded:
            base -= 0.12
        return round(clamp(base, 0.0, 0.95), 2)

    @staticmethod
    def build_cleaning_explanation(
        *,
        cleaning_status: str,
        candidate: CleaningCandidate | None,
        state_change: TableStateChange,
    ) -> str:
        if cleaning_status == CLEANING_STATUS_CLEANED_LIKELY and candidate is not None:
            return (
                "식사 종료 이후 해당 구역에 직원 체류가 확인되었고, "
                "전후 프레임에서 테이블 상태 개선이 관찰되어 청소 완료 추정으로 판단했습니다."
            )
        if cleaning_status == CLEANING_STATUS_CLEANING_UNVERIFIED and candidate is not None:
            return (
                "직원 체류는 확인되었지만, 전후 프레임에서 테이블 상태 변화가 충분하지 않아 "
                "청소 완료 추정 대신 CLEANING_UNVERIFIED로 남겼습니다."
            )
        if cleaning_status == CLEANING_STATUS_MISSED_SUSPECTED:
            return "식사 종료 이후 SLA 안에 직원 체류 또는 청소 증거가 없어 청소 누락 의심으로 판단했습니다."
        if cleaning_status == CLEANING_STATUS_NEED_CLEANING:
            return "식사 종료는 확인되었지만 아직 청소 완료 추정을 뒷받침할 직원 체류와 상태 변화가 부족합니다."
        if state_change.changed:
            return "전후 프레임 변화는 감지되었지만 규칙을 모두 만족하지 않아 추가 확인이 필요합니다."
        return "입력 증거가 부족하여 청소 상태를 확정하지 못했습니다."
