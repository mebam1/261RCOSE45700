from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.action_cleanliness import (
    ACTION_DECISION_CLEANED,
    ACTION_DECISION_NEEDS_CHECK,
    ACTION_DECISION_UNKNOWN,
    ActionCleanlinessResult,
    ActionWorkflowResult,
    CLEANING_STATUS_CLEANING_UNVERIFIED,
    CLEANING_STATUS_MISSED_SUSPECTED,
    CLEANING_STATUS_NEED_CLEANING,
    YoloTableState,
)
from app.cleanliness import CleanlinessResult
from app.cleanliness_metric import build_visual_metric_result


OBJECT_CONFIDENCE_THRESHOLD = 0.6
FINAL_LOW_CONFIDENCE_THRESHOLD = 0.5

HYBRID_STAGE_OBJECT = "object_based"
HYBRID_STAGE_ACTION = "action_fallback"

FINAL_DECISION_CLEAN_OK = "CLEAN_OK"
FINAL_DECISION_CHECK_REQUIRED = "CHECK_REQUIRED"
FINAL_DECISION_CLEANING_REQUIRED = "CLEANING_REQUIRED"
FINAL_DECISION_CLEANING_MISSED = "CLEANING_MISSED_SUSPECTED"


@dataclass(frozen=True)
class HybridCleanlinessResult:
    final_decision: str
    final_confidence: float
    final_stage: str
    summary: str
    fallback_triggered: bool
    object_confidence_threshold: float
    object_result: CleanlinessResult
    action_result: ActionCleanlinessResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_decision": self.final_decision,
            "final_confidence": round(self.final_confidence, 3),
            "final_stage": self.final_stage,
            "summary": self.summary,
            "fallback_triggered": self.fallback_triggered,
            "object_confidence_threshold": self.object_confidence_threshold,
            "object_result": self.object_result.to_dict(),
            "action_result": self.action_result.to_dict() if self.action_result is not None else None,
        }


@dataclass(frozen=True)
class FinalCleanlinessResult:
    table_id: str
    final_cleanliness_score: int
    final_grade: str
    decision: str
    visual_score: int
    visual_clean_score: float
    visual_mess_score: float
    action_score: float
    penalties_breakdown: list[dict[str, Any]]
    bonuses_breakdown: list[dict[str, Any]]
    reason_codes: list[str]
    visual_metric_grade: str
    applied_caps: list[str]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "final_cleanliness_score": self.final_cleanliness_score,
            "final_grade": self.final_grade,
            "decision": self.decision,
            "visual_score": self.visual_score,
            "visual_clean_score": round(self.visual_clean_score, 2),
            "visual_mess_score": round(self.visual_mess_score, 2),
            "action_score": round(self.action_score, 2),
            "penalties_breakdown": self.penalties_breakdown,
            "bonuses_breakdown": self.bonuses_breakdown,
            "reason_codes": self.reason_codes,
            "visual_metric_grade": self.visual_metric_grade,
            "applied_caps": self.applied_caps,
            "explanation": self.explanation,
        }


def object_score_to_decision(score: int) -> str:
    if score >= 4:
        return ACTION_DECISION_CLEANED
    if score <= 2:
        return ACTION_DECISION_NEEDS_CHECK
    return ACTION_DECISION_UNKNOWN


def build_hybrid_cleanliness_result(
    object_result: CleanlinessResult,
    action_result: ActionCleanlinessResult | None,
    *,
    object_confidence_threshold: float = OBJECT_CONFIDENCE_THRESHOLD,
) -> HybridCleanlinessResult:
    fallback_triggered = object_result.confidence < object_confidence_threshold

    if fallback_triggered:
        if action_result is None:
            action_result = ActionCleanlinessResult(
                decision=ACTION_DECISION_UNKNOWN,
                score=1,
                confidence=0.0,
                summary="객체 기반 청결도 confidence가 낮지만 Action 보조 입력이 없어 최종 판정을 보류했습니다.",
                evidence=["객체 기반 confidence가 기준보다 낮습니다.", "Action trajectory가 제공되지 않았습니다."],
                features={},
            )
        return HybridCleanlinessResult(
            final_decision=action_result.decision,
            final_confidence=action_result.confidence,
            final_stage=HYBRID_STAGE_ACTION,
            summary=(
                f"객체 기반 confidence {object_result.confidence:.2f}가 "
                f"기준치 {object_confidence_threshold:.2f}보다 낮아 Action 평가로 전환했습니다. "
                f"{action_result.summary}"
            ),
            fallback_triggered=True,
            object_confidence_threshold=object_confidence_threshold,
            object_result=object_result,
            action_result=action_result,
        )

    decision = object_score_to_decision(object_result.score)
    return HybridCleanlinessResult(
        final_decision=decision,
        final_confidence=object_result.confidence,
        final_stage=HYBRID_STAGE_OBJECT,
        summary=(
            f"객체 기반 confidence {object_result.confidence:.2f}가 "
            f"기준치 {object_confidence_threshold:.2f} 이상이므로 객체 기반 청결도 평가를 최종 결과로 사용했습니다. "
            f"{object_result.summary}"
        ),
        fallback_triggered=False,
        object_confidence_threshold=object_confidence_threshold,
        object_result=object_result,
        action_result=None,
    )


def build_final_cleanliness_result(
    action_result: ActionWorkflowResult,
    visual_state: YoloTableState,
    *,
    low_confidence_threshold: float = FINAL_LOW_CONFIDENCE_THRESHOLD,
) -> FinalCleanlinessResult:
    if action_result.table_id != visual_state.table_id:
        raise ValueError("action_result and visual_state must refer to the same table")

    visual_metric = build_visual_metric_result(visual_state)
    raw_score = int(round(100 * (0.5 * visual_metric.visual_clean_score + 0.5 * action_result.action_score)))
    capped_score = raw_score
    applied_caps: list[str] = []

    if visual_state.yolo_mess_score >= 0.75:
        capped_score = min(capped_score, 35)
        applied_caps.append("HIGH_YOLO_MESS_CAP_35")
    if action_result.cleaning_status == CLEANING_STATUS_MISSED_SUSPECTED:
        capped_score = min(capped_score, 30)
        applied_caps.append("MISSED_SUSPECTED_CAP_30")
    if action_result.cleaning_status == CLEANING_STATUS_NEED_CLEANING:
        capped_score = min(capped_score, 55)
        applied_caps.append("NEED_CLEANING_CAP_55")
    if action_result.cleaning_status == CLEANING_STATUS_CLEANING_UNVERIFIED:
        capped_score = min(capped_score, 70)
        applied_caps.append("CLEANING_UNVERIFIED_CAP_70")
    if visual_state.camera_occluded or visual_state.vision_confidence < low_confidence_threshold:
        capped_score = min(capped_score, 60)
        applied_caps.append("LOW_VISUAL_CONFIDENCE_CAP_60")

    final_grade, decision = grade_for_score(capped_score)
    explanation = action_result.explanation
    if applied_caps:
        explanation = f"{explanation} 최종 점수에는 {' ,'.join(applied_caps)} 규칙을 적용했습니다."

    return FinalCleanlinessResult(
        table_id=action_result.table_id,
        final_cleanliness_score=capped_score,
        final_grade=final_grade,
        decision=decision,
        visual_score=visual_metric.visual_score,
        visual_clean_score=visual_metric.visual_clean_score,
        visual_mess_score=visual_metric.visual_mess_score,
        action_score=action_result.action_score,
        penalties_breakdown=visual_metric.penalties_breakdown,
        bonuses_breakdown=visual_metric.bonuses_breakdown,
        reason_codes=visual_metric.reason_codes,
        visual_metric_grade=visual_metric.visual_metric_grade,
        applied_caps=applied_caps,
        explanation=explanation,
    )


def grade_for_score(score: int) -> tuple[str, str]:
    if score >= 80:
        return "정상", FINAL_DECISION_CLEAN_OK
    if score >= 60:
        return "확인 필요", FINAL_DECISION_CHECK_REQUIRED
    if score >= 40:
        return "청소 필요", FINAL_DECISION_CLEANING_REQUIRED
    return "청소 누락 의심", FINAL_DECISION_CLEANING_MISSED
