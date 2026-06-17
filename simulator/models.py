from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.schemas import ROI


@dataclass(frozen=True)
class TableBox:
    table_id: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def to_roi(self) -> ROI:
        return ROI.from_rectangle(self.table_id, self.x1, self.y1, self.width, self.height)

    def clamp(self, width: int, height: int) -> "TableBox":
        x1 = max(0, min(self.x1, width - 1))
        y1 = max(0, min(self.y1, height - 1))
        x2 = max(x1 + 1, min(self.x2, width))
        y2 = max(y1 + 1, min(self.y2, height))
        return TableBox(self.table_id, x1, y1, x2, y2)


@dataclass(frozen=True)
class SampledFrame:
    index: int
    timestamp_seconds: float
    frame_path: Path


@dataclass(frozen=True)
class ObjectFrameResult:
    table_id: str
    timestamp_seconds: float
    frame_path: Path
    crop_path: Path
    evidence_path: Path
    score: int
    confidence: float
    summary: str
    findings: list[str]
    exact_objects: list[str]
    estimated_objects: list[str]
    detection_count: int


@dataclass(frozen=True)
class TemporalFrameResult:
    table_id: str
    timestamp_seconds: float
    frame_path: Path
    crop_path: Path
    evidence_path: Path
    visible_people: bool
    active_dining: bool
    cleaning_action: bool
    used_dishes_or_leftovers: bool
    personal_items_or_food_remaining: bool
    frame_state: str
    final_state: str
    confidence: float
    reason: str
    temporal_reason: str


@dataclass(frozen=True)
class ObjectTableSummary:
    table: TableBox
    frame_results: list[ObjectFrameResult]
    aggregate_score: float
    aggregate_confidence: float
    headline: str
    evidence: list[str]


@dataclass(frozen=True)
class TemporalTableSummary:
    table: TableBox
    frame_results: list[TemporalFrameResult]
    aggregate_score: float
    aggregate_confidence: float
    final_state: str
    headline: str
    evidence: list[str]


@dataclass(frozen=True)
class CombinedTableResult:
    table: TableBox
    object_summary: ObjectTableSummary
    temporal_summary: TemporalTableSummary
    overall_score: float
    overall_label: str
    overall_summary: str


@dataclass(frozen=True)
class AnalysisSessionResult:
    video_path: Path
    frame_interval_seconds: float
    sampled_frames: list[SampledFrame]
    tables: list[CombinedTableResult]
    output_dir: Path
    overall_score: float
    overall_label: str
    overall_summary: str
    metadata: dict[str, object] = field(default_factory=dict)
