from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from app.action_cleanliness import (
    ActionCleanlinessService,
    CLEANING_STATUS_CLEANED_LIKELY,
    CLEANING_STATUS_CLEANING_UNVERIFIED,
    CLEANING_STATUS_MISSED_SUSPECTED,
    MEAL_STATUS_MEAL_ENDED,
    TableOccupancySample,
    StaffZoneVisit,
    YoloTableState,
    ZoneTableMapping,
)
from app.hybrid_cleanliness import build_final_cleanliness_result


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def build_mapping() -> ZoneTableMapping:
    return ZoneTableMapping.from_dict(
        {
            "store_id": "store_001",
            "zones": [
                {
                    "zone_id": "zone_A",
                    "beacon_ids": ["beacon_A_01"],
                    "table_ids": ["T01", "T02", "T03", "T04"],
                },
                {
                    "zone_id": "zone_B",
                    "beacon_ids": ["beacon_B_01"],
                    "table_ids": ["T05", "T06", "T07", "T08"],
                },
            ],
        }
    )


def build_before_state() -> YoloTableState:
    return YoloTableState.from_dict(
        {
            "table_id": "T06",
            "captured_at": "2026-06-03T14:44:00",
            "yolo_mess_score": 0.72,
            "vision_confidence": 0.84,
            "detected_objects": [
                {"class": "trash", "count": 2, "max_confidence": 0.82},
                {"class": "cup", "count": 1, "max_confidence": 0.70},
                {"class": "chair", "count": 4, "max_confidence": 0.91},
            ],
        }
    )


def build_visit(start: str, seconds: int = 26) -> StaffZoneVisit:
    entered_at = dt(start)
    left_at = entered_at + timedelta(seconds=seconds)
    return StaffZoneVisit.from_dict(
        {
            "visit_id": "visit_000341",
            "store_id": "store_001",
            "staff_id": "staff_03",
            "zone_id": "zone_B",
            "entered_at": entered_at.isoformat(timespec="seconds"),
            "left_at": left_at.isoformat(timespec="seconds"),
            "dwell_seconds": seconds,
            "mean_confidence": 0.76,
            "sample_count": 8,
        }
    )


class CleaningWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ActionCleanlinessService()
        self.mapping = build_mapping()

    def build_meal_observations(self) -> list[TableOccupancySample]:
        return [
            TableOccupancySample(timestamp=dt("2026-06-03T14:10:20"), customer_present=True),
            TableOccupancySample(timestamp=dt("2026-06-03T14:11:20"), customer_present=True),
            TableOccupancySample(timestamp=dt("2026-06-03T14:41:05"), customer_present=False),
            TableOccupancySample(timestamp=dt("2026-06-03T14:42:35"), customer_present=False),
        ]

    def test_does_not_create_meal_session_when_customer_presence_is_under_60_seconds(self) -> None:
        observations = [
            TableOccupancySample(timestamp=dt("2026-06-03T14:10:20"), customer_present=True),
            TableOccupancySample(timestamp=dt("2026-06-03T14:11:19"), customer_present=True),
        ]

        meal_session = self.service.evaluate_meal_session("T06", observations)

        self.assertIsNone(meal_session)

    def test_marks_meal_ended_after_90_seconds_of_absence(self) -> None:
        meal_session = self.service.evaluate_meal_session("T06", self.build_meal_observations())

        self.assertIsNotNone(meal_session)
        assert meal_session is not None
        self.assertEqual(meal_session.status, MEAL_STATUS_MEAL_ENDED)
        self.assertEqual(meal_session.meal_started_at, dt("2026-06-03T14:10:20"))
        self.assertEqual(meal_session.meal_ended_at, dt("2026-06-03T14:42:35"))

    def test_creates_cleaning_task_immediately_after_meal_end(self) -> None:
        meal_session = self.service.evaluate_meal_session("T06", self.build_meal_observations())
        assert meal_session is not None

        task = self.service.create_cleaning_task(meal_session, self.mapping, cleaning_task_id="clean_task_000912")

        self.assertEqual(task.table_id, "T06")
        self.assertEqual(task.zone_id, "zone_B")
        self.assertEqual(task.created_at, dt("2026-06-03T14:42:35"))
        self.assertEqual(task.status, "NEED_CLEANING")
        self.assertEqual(task.sla_seconds, 300)

    def test_creates_cleaning_candidate_for_same_zone_visit_of_10_seconds_or_more(self) -> None:
        meal_session = self.service.evaluate_meal_session("T06", self.build_meal_observations())
        assert meal_session is not None
        task = self.service.create_cleaning_task(meal_session, self.mapping)

        candidate = self.service.find_cleaning_candidate(
            task,
            [build_visit("2026-06-03T14:44:10", seconds=10)],
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.zone_id, "zone_B")
        self.assertEqual(candidate.staff_zone_visit.dwell_seconds, 10)

    def test_keeps_cleaning_unverified_when_staff_visit_exists_but_yolo_change_does_not(self) -> None:
        meal_session = self.service.evaluate_meal_session("T06", self.build_meal_observations())
        assert meal_session is not None
        task = self.service.create_cleaning_task(meal_session, self.mapping)
        before_state = build_before_state()
        after_state = YoloTableState.from_dict(
            {
                "table_id": "T06",
                "captured_at": "2026-06-03T14:45:00",
                "yolo_mess_score": 0.70,
                "vision_confidence": 0.84,
                "detected_objects": [
                    {"class": "trash", "count": 2, "max_confidence": 0.81},
                    {"class": "cup", "count": 1, "max_confidence": 0.72},
                    {"class": "chair", "count": 4, "max_confidence": 0.91},
                ],
            }
        )

        result = self.service.evaluate_cleaning_task(
            task,
            before_state=before_state,
            after_state=after_state,
            staff_zone_visits=[build_visit("2026-06-03T14:44:10", seconds=26)],
            meal_session=meal_session,
        )

        self.assertEqual(result.cleaning_status, CLEANING_STATUS_CLEANING_UNVERIFIED)

    def test_marks_cleaned_likely_when_staff_visit_and_yolo_mess_drop_both_exist(self) -> None:
        meal_session = self.service.evaluate_meal_session("T06", self.build_meal_observations())
        assert meal_session is not None
        task = self.service.create_cleaning_task(meal_session, self.mapping)
        before_state = build_before_state()
        after_state = YoloTableState.from_dict(
            {
                "table_id": "T06",
                "captured_at": "2026-06-03T14:45:00",
                "yolo_mess_score": 0.22,
                "vision_confidence": 0.84,
                "detected_objects": [
                    {"class": "trash", "count": 1, "max_confidence": 0.71},
                ],
            }
        )

        result = self.service.evaluate_cleaning_task(
            task,
            before_state=before_state,
            after_state=after_state,
            staff_zone_visits=[build_visit("2026-06-03T14:44:10", seconds=26)],
            meal_session=meal_session,
        )

        self.assertEqual(result.cleaning_status, CLEANING_STATUS_CLEANED_LIKELY)
        self.assertGreaterEqual(result.action_score, 0.85)

    def test_marks_missed_suspected_when_sla_expires_without_staff_visit_or_cleaning_evidence(self) -> None:
        meal_session = self.service.evaluate_meal_session("T06", self.build_meal_observations())
        assert meal_session is not None
        task = self.service.create_cleaning_task(meal_session, self.mapping)
        before_state = build_before_state()
        after_state = YoloTableState.from_dict(
            {
                "table_id": "T06",
                "captured_at": "2026-06-03T14:48:00",
                "yolo_mess_score": 0.71,
                "vision_confidence": 0.82,
                "detected_objects": [
                    {"class": "trash", "count": 2, "max_confidence": 0.79},
                    {"class": "cup", "count": 1, "max_confidence": 0.70},
                    {"class": "chair", "count": 4, "max_confidence": 0.91},
                ],
            }
        )

        result = self.service.evaluate_cleaning_task(
            task,
            before_state=before_state,
            after_state=after_state,
            staff_zone_visits=[],
            meal_session=meal_session,
            as_of=dt("2026-06-03T14:47:36"),
        )

        self.assertEqual(result.cleaning_status, CLEANING_STATUS_MISSED_SUSPECTED)

    def test_caps_final_score_at_35_when_yolo_mess_score_is_high(self) -> None:
        meal_session = self.service.evaluate_meal_session("T06", self.build_meal_observations())
        assert meal_session is not None
        task = self.service.create_cleaning_task(meal_session, self.mapping)
        action_result = self.service.evaluate_cleaning_task(
            task,
            before_state=build_before_state(),
            after_state=YoloTableState.from_dict(
                {
                    "table_id": "T06",
                    "captured_at": "2026-06-03T14:45:00",
                    "yolo_mess_score": 0.80,
                    "vision_confidence": 0.88,
                    "detected_objects": [
                        {"class": "trash", "count": 1, "max_confidence": 0.71},
                    ],
                }
            ),
            staff_zone_visits=[build_visit("2026-06-03T14:44:10", seconds=26)],
            meal_session=meal_session,
        )
        final_result = build_final_cleanliness_result(
            action_result,
            YoloTableState.from_dict(
                {
                    "table_id": "T06",
                    "captured_at": "2026-06-03T14:45:00",
                    "yolo_mess_score": 0.80,
                    "vision_confidence": 0.88,
                    "detected_objects": [
                        {"class": "trash", "count": 1, "max_confidence": 0.71},
                    ],
                }
            ),
        )

        self.assertEqual(final_result.final_cleanliness_score, 35)

    def test_caps_final_score_at_30_when_status_is_missed_suspected(self) -> None:
        meal_session = self.service.evaluate_meal_session("T06", self.build_meal_observations())
        assert meal_session is not None
        task = self.service.create_cleaning_task(meal_session, self.mapping)
        action_result = self.service.evaluate_cleaning_task(
            task,
            before_state=YoloTableState.from_dict(
                {
                    "table_id": "T06",
                    "captured_at": "2026-06-03T14:44:00",
                    "yolo_mess_score": 0.12,
                    "vision_confidence": 0.90,
                    "detected_objects": [
                        {"class": "chair", "count": 4, "max_confidence": 0.91},
                    ],
                }
            ),
            after_state=YoloTableState.from_dict(
                {
                    "table_id": "T06",
                    "captured_at": "2026-06-03T14:48:00",
                    "yolo_mess_score": 0.10,
                    "vision_confidence": 0.90,
                    "detected_objects": [
                        {"class": "chair", "count": 4, "max_confidence": 0.91},
                    ],
                }
            ),
            staff_zone_visits=[],
            meal_session=meal_session,
            as_of=dt("2026-06-03T14:47:36"),
        )
        final_result = build_final_cleanliness_result(
            action_result,
            YoloTableState.from_dict(
                {
                    "table_id": "T06",
                    "captured_at": "2026-06-03T14:48:00",
                    "yolo_mess_score": 0.10,
                    "vision_confidence": 0.90,
                    "detected_objects": [
                        {"class": "chair", "count": 4, "max_confidence": 0.91},
                    ],
                }
            ),
        )

        self.assertEqual(final_result.final_cleanliness_score, 30)


if __name__ == "__main__":
    unittest.main()
