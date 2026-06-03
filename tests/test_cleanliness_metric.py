from __future__ import annotations

import unittest

from app.action_cleanliness import ActionWorkflowResult, YoloTableState
from app.cleanliness_metric import (
    build_visual_metric_result,
    build_visual_metric_result_from_payload,
    canonical_object_class,
    normalize_visual_metric_input,
    visual_metric_grade_for_score,
)
from app.hybrid_cleanliness import build_final_cleanliness_result


def build_visual_state(**overrides: object) -> YoloTableState:
    payload: dict[str, object] = {
        "table_id": "T06",
        "yolo_mess_score": 0.10,
        "vision_confidence": 0.90,
        "detected_objects": [],
    }
    payload.update(overrides)
    return YoloTableState.from_dict(payload)


def build_action_result(action_score: float = 0.80) -> ActionWorkflowResult:
    return ActionWorkflowResult(
        table_id="T06",
        zone_id="zone_B",
        meal_status="MEAL_ENDED",
        cleaning_status="CLEANED_LIKELY",
        action_score=action_score,
        action_confidence=0.82,
        meal_started_at=None,
        meal_ended_at=None,
        cleaning_started_at=None,
        cleaning_ended_at=None,
        reason_codes=["TABLE_STATE_CHANGED"],
        explanation="Visual metric final score test.",
    )


class CleanlinessMetricTest(unittest.TestCase):
    def test_existing_visual_metric_function_is_used_for_raw_payload(self) -> None:
        raw_payload = {
            "table_id": "T06",
            "objects": [
                {"class": "trash", "confidence": 0.91, "bbox": [0, 0, 10, 10]},
                {"class": "cup", "confidence": 0.81, "bbox": [10, 10, 20, 20]},
            ],
            "visible_contamination": True,
        }

        normalized_state = normalize_visual_metric_input(raw_payload)
        direct_result = build_visual_metric_result(normalized_state)
        adapter_result = build_visual_metric_result_from_payload(raw_payload)

        self.assertEqual(adapter_result.visual_score, direct_result.visual_score)
        self.assertEqual(adapter_result.reason_codes, direct_result.reason_codes)

    def test_penalizes_three_trash_like_objects_by_fifteen(self) -> None:
        result = build_visual_metric_result(
            build_visual_state(
                detected_objects=[{"class": "trash", "count": 3, "max_confidence": 0.9}],
            )
        )

        self.assertEqual(result.visual_score, 85)
        self.assertEqual(result.penalties_breakdown[0]["points"], -15)

    def test_caps_trash_like_object_penalty_at_thirty(self) -> None:
        result = build_visual_metric_result(
            build_visual_state(
                detected_objects=[{"class": "trash", "count": 10, "max_confidence": 0.9}],
            )
        )

        self.assertEqual(result.visual_score, 70)
        self.assertEqual(result.penalties_breakdown[0]["points"], -30)

    def test_visible_contamination_penalizes_ten_points(self) -> None:
        result = build_visual_metric_result(build_visual_state(visible_contamination=True))

        self.assertEqual(result.visual_score, 90)

    def test_contamination_area_ratio_six_percent_penalizes_five_points(self) -> None:
        result = build_visual_metric_result(build_visual_state(contamination_area_ratio=0.06))

        self.assertEqual(result.visual_score, 95)

    def test_contamination_area_ratio_twelve_percent_penalizes_ten_points(self) -> None:
        result = build_visual_metric_result(build_visual_state(contamination_area_ratio=0.12))

        self.assertEqual(result.visual_score, 90)

    def test_contamination_area_ratio_twenty_five_percent_penalizes_twenty_points(self) -> None:
        result = build_visual_metric_result(build_visual_state(contamination_area_ratio=0.25))

        self.assertEqual(result.visual_score, 80)

    def test_messy_arrangement_penalizes_ten_points(self) -> None:
        result = build_visual_metric_result(build_visual_state(messy_arrangement=True))

        self.assertEqual(result.visual_score, 90)

    def test_hazardous_contamination_penalizes_twenty_points(self) -> None:
        result = build_visual_metric_result(build_visual_state(hazardous_contamination=True))

        self.assertEqual(result.visual_score, 80)

    def test_dirty_duration_over_five_minutes_penalizes_five_points(self) -> None:
        result = build_visual_metric_result(build_visual_state(tracked_dirty_duration_seconds=301))

        self.assertEqual(result.visual_score, 95)

    def test_dirty_duration_over_ten_minutes_penalizes_ten_points(self) -> None:
        result = build_visual_metric_result(build_visual_state(tracked_dirty_duration_seconds=601))

        self.assertEqual(result.visual_score, 90)

    def test_dirty_duration_over_twenty_minutes_penalizes_fifteen_points(self) -> None:
        result = build_visual_metric_result(build_visual_state(tracked_dirty_duration_seconds=1201))

        self.assertEqual(result.visual_score, 85)

    def test_removed_objects_bonus_adds_ten_points(self) -> None:
        result = build_visual_metric_result(
            build_visual_state(
                hazardous_contamination=True,
                removed_objects_count=2,
            )
        )

        self.assertEqual(result.visual_score, 90)
        self.assertEqual(result.bonuses_breakdown[0]["points"], 10)

    def test_clear_cleaning_action_evidence_bonus_adds_ten_points(self) -> None:
        result = build_visual_metric_result(
            build_visual_state(
                visible_contamination=True,
                clear_cleaning_action_evidence=True,
            )
        )

        self.assertEqual(result.visual_score, 100)
        self.assertEqual(result.bonuses_breakdown[0]["points"], 10)

    def test_visual_score_is_clamped_between_zero_and_one_hundred(self) -> None:
        low_result = build_visual_metric_result(
            build_visual_state(
                detected_objects=[{"class": "trash", "count": 10, "max_confidence": 0.9}],
                visible_contamination=True,
                contamination_area_ratio=0.25,
                messy_arrangement=True,
                hazardous_contamination=True,
                tracked_dirty_duration_seconds=1201,
            )
        )
        high_result = build_visual_metric_result(
            build_visual_state(
                removed_objects_count=3,
                contamination_removed=True,
                clear_cleaning_action_evidence=True,
            )
        )

        self.assertEqual(low_result.visual_score, 0)
        self.assertEqual(high_result.visual_score, 100)

    def test_person_exposed_does_not_change_score(self) -> None:
        base_result = build_visual_metric_result(build_visual_state(visible_contamination=True))
        person_result = build_visual_metric_result(build_visual_state(visible_contamination=True, person_exposed=True))

        self.assertEqual(base_result.visual_score, person_result.visual_score)

    def test_raw_yolo_like_result_is_normalized_to_visual_metric_input(self) -> None:
        normalized_state = normalize_visual_metric_input(
            {
                "table_id": "T06",
                "yolo_objects": [
                    {"label": "plastic_cup", "score": 0.77},
                    {"label": "plastic_cup", "score": 0.61},
                    {"label": "tray", "score": 0.88},
                ],
                "scene_flags": {
                    "visible_contamination": True,
                    "messy_arrangement": True,
                },
                "contamination_area_ratio": 0.12,
            }
        )

        self.assertEqual(normalized_state.detected_objects[0].object_class, "cup")
        self.assertEqual(normalized_state.detected_objects[0].count, 2)
        self.assertTrue(normalized_state.visible_contamination)
        self.assertTrue(normalized_state.messy_arrangement)
        self.assertEqual(normalized_state.contamination_area_ratio, 0.12)

    def test_missing_vision_confidence_does_not_default_to_one(self) -> None:
        normalized_state = normalize_visual_metric_input(
            {
                "table_id": "T06",
                "objects": [{"class": "trash", "confidence": 0.91}],
            }
        )

        self.assertEqual(normalized_state.vision_confidence, 0.5)
        self.assertNotEqual(normalized_state.vision_confidence, 1.0)
        self.assertTrue(normalized_state.vision_confidence_missing)

    def test_missing_vision_confidence_adds_reason_code(self) -> None:
        result = build_visual_metric_result_from_payload(
            {
                "table_id": "T06",
                "objects": [{"class": "trash", "confidence": 0.91}],
            }
        )

        self.assertIn("MISSING_VISION_CONFIDENCE", result.reason_codes)

    def test_explicit_vision_confidence_is_preserved(self) -> None:
        normalized_state = normalize_visual_metric_input(
            {
                "table_id": "T06",
                "vision_confidence": 0.73,
                "objects": [{"class": "trash", "confidence": 0.91}],
            }
        )

        self.assertEqual(normalized_state.vision_confidence, 0.73)
        self.assertFalse(normalized_state.vision_confidence_missing)

    def test_alias_mapping_normalizes_supported_labels(self) -> None:
        self.assertEqual(canonical_object_class("litter"), "trash")
        self.assertEqual(canonical_object_class("plastic_cup"), "cup")
        self.assertEqual(canonical_object_class("plate"), "dish")
        self.assertEqual(canonical_object_class("leftover"), "food_waste")

        result = build_visual_metric_result_from_payload(
            {
                "table_id": "T06",
                "objects": [
                    {"class": "paper", "confidence": 0.8},
                    {"class": "tissue", "confidence": 0.7},
                ],
            }
        )
        self.assertEqual(result.visual_score, 90)

    def test_person_object_sets_person_exposed_without_direct_penalty(self) -> None:
        normalized_state = normalize_visual_metric_input(
            {
                "table_id": "T06",
                "objects": [
                    {"class": "person", "confidence": 0.99},
                ],
                "visible_contamination": True,
            }
        )
        result = build_visual_metric_result(normalized_state)

        self.assertTrue(normalized_state.person_exposed)
        self.assertEqual(result.visual_score, 90)

    def test_unknown_class_is_not_penalized(self) -> None:
        normalized_state = normalize_visual_metric_input(
            {
                "table_id": "T06",
                "objects": [
                    {"class": "forklift", "confidence": 0.75},
                ],
            }
        )
        result = build_visual_metric_result(normalized_state)

        self.assertEqual(len(normalized_state.detected_objects), 0)
        self.assertEqual(result.visual_score, 100)

    def test_adapter_output_is_compatible_with_visual_metric(self) -> None:
        result = build_visual_metric_result_from_payload(
            {
                "table_id": "T06",
                "detected_objects": [
                    {"class": "trash", "count": 2, "max_confidence": 0.88},
                ],
                "flags": {
                    "hazardous_contamination": True,
                },
            }
        )

        self.assertEqual(result.visual_score, 70)
        self.assertIn("HAZARDOUS_CONTAMINATION", result.reason_codes)

    def test_visual_metric_grade_mapping_uses_a_through_e(self) -> None:
        self.assertEqual(visual_metric_grade_for_score(95), "A")
        self.assertEqual(visual_metric_grade_for_score(75), "B")
        self.assertEqual(visual_metric_grade_for_score(55), "C")
        self.assertEqual(visual_metric_grade_for_score(35), "D")
        self.assertEqual(visual_metric_grade_for_score(10), "E")

    def test_adapter_visual_metric_and_final_score_flow_works(self) -> None:
        visual_state = normalize_visual_metric_input(
            {
                "table_id": "T06",
                "objects": [
                    {"class": "trash", "confidence": 0.91},
                    {"class": "plastic_cup", "confidence": 0.72},
                    {"class": "person", "confidence": 0.95},
                ],
                "scene_flags": {
                    "visible_contamination": True,
                },
                "contamination_area_ratio": 0.06,
            }
        )
        final_result = build_final_cleanliness_result(
            build_action_result(action_score=0.80),
            visual_state,
        )

        self.assertTrue(visual_state.person_exposed)
        self.assertEqual(final_result.visual_score, 75)
        self.assertEqual(final_result.visual_clean_score, 0.75)
        self.assertAlmostEqual(final_result.visual_mess_score, 0.25)
        self.assertEqual(final_result.final_cleanliness_score, 78)
        self.assertEqual(final_result.visual_metric_grade, "B")


if __name__ == "__main__":
    unittest.main()
