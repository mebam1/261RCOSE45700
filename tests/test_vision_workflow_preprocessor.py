from __future__ import annotations

import unittest
from datetime import datetime

import numpy as np

from app.action_cleanliness import ActionCleanlinessService, TableOccupancySample
from app.person_masking import PersonDetection
from app.schemas import ActionWorkflowRequest, ROI
from app.vision_workflow_preprocessor import (
    build_meal_occupancy_sequence,
    build_workflow_frame_from_image,
    build_workflow_frames_from_images,
    build_workflow_frames_from_video,
    captured_at_for_video_frame,
    sample_video_workflow_frames,
)


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class FakePersonMaskService:
    def __init__(self, detection_batches: list[list[PersonDetection]]) -> None:
        self.detection_batches = detection_batches
        self.call_index = 0
        self.received_shapes: list[tuple[int, ...]] = []

    def apply_black_mask(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[PersonDetection]]:
        self.received_shapes.append(image.shape)
        detections = self.detection_batches[min(self.call_index, len(self.detection_batches) - 1)]
        self.call_index += 1
        return image.copy(), np.zeros(image.shape[:2], dtype=np.uint8), list(detections)


class VisionWorkflowPreprocessorTest(unittest.TestCase):
    def test_sample_video_workflow_frames_uses_fake_extractor_output(self) -> None:
        def fake_extractor(video_path, *, interval_seconds, max_frames):
            self.assertEqual(str(video_path), "demo.avi")
            self.assertEqual(interval_seconds, 30.0)
            self.assertEqual(max_frames, 10)
            return [
                {"image": np.zeros((8, 8, 3), dtype=np.uint8), "offset_seconds": 0.0},
                {"image": np.zeros((8, 8, 3), dtype=np.uint8), "offset_seconds": 30.0},
                {"image": np.zeros((8, 8, 3), dtype=np.uint8), "offset_seconds": 60.0},
            ]

        samples = sample_video_workflow_frames(
            video_path="demo.avi",
            frame_extractor=fake_extractor,
        )

        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["offset_seconds"], 0.0)
        self.assertEqual(samples[2]["offset_seconds"], 60.0)

    def test_captured_at_for_video_frame_increases_by_interval(self) -> None:
        captured_at = captured_at_for_video_frame(
            captured_at_start="2026-06-03T14:00:00",
            interval_seconds=30,
            sample_index=2,
        )

        self.assertEqual(captured_at, "2026-06-03T14:01:00")

    def test_builds_frame_with_no_person_when_mask_service_returns_no_detections(self) -> None:
        frame = build_workflow_frame_from_image(
            captured_at="2026-06-03T14:10:20",
            image=np.zeros((32, 32, 3), dtype=np.uint8),
            person_mask_service=FakePersonMaskService([[]]),
        )

        self.assertFalse(frame["person_present"])
        self.assertEqual(frame["person_count"], 0)
        self.assertEqual(frame["occupancy_source"], "person_masking")
        self.assertEqual(frame["occupancy_reason_codes"], ["ROI_NOT_PROVIDED"])

    def test_builds_frame_with_person_when_mask_service_returns_detections(self) -> None:
        detections = [
            PersonDetection(x=1, y=2, width=10, height=20, score=0.91, source="fake"),
            PersonDetection(x=20, y=4, width=12, height=18, score=0.88, source="fake"),
        ]
        frame = build_workflow_frame_from_image(
            captured_at="2026-06-03T14:10:20",
            image=np.zeros((48, 48, 3), dtype=np.uint8),
            person_mask_service=FakePersonMaskService([detections]),
        )

        self.assertTrue(frame["person_present"])
        self.assertEqual(frame["person_count"], 2)

    def test_uses_interaction_roi_crop_before_person_counting(self) -> None:
        roi = ROI.from_rectangle("TABLE_1", 10, 20, 30, 40)
        fake_service = FakePersonMaskService(
            [[PersonDetection(x=0, y=0, width=5, height=5, score=0.9, source="fake")]]
        )

        frame = build_workflow_frame_from_image(
            captured_at="2026-06-03T14:10:20",
            image=np.zeros((120, 120, 3), dtype=np.uint8),
            interaction_roi=roi,
            person_mask_service=fake_service,
        )

        self.assertEqual(fake_service.received_shapes[0][:2], (40, 30))
        self.assertEqual(frame["occupancy_reason_codes"], [])
        self.assertEqual(frame["person_count"], 1)

    def test_preserves_payload_when_building_workflow_frame(self) -> None:
        frame = build_workflow_frame_from_image(
            captured_at="2026-06-03T14:10:20",
            image=np.zeros((32, 32, 3), dtype=np.uint8),
            table_id="T06",
            payload={"objects": [{"class": "cup", "confidence": 0.9}]},
            person_mask_service=FakePersonMaskService([[]]),
        )

        self.assertEqual(frame["payload"]["table_id"], "T06")
        self.assertEqual(frame["payload"]["objects"][0]["class"], "cup")

    def test_builds_frame_list_compatible_with_action_workflow_request(self) -> None:
        frames = build_workflow_frames_from_images(
            images=[
                np.zeros((32, 32, 3), dtype=np.uint8),
                np.zeros((32, 32, 3), dtype=np.uint8),
            ],
            start_time="2026-06-03T14:10:20",
            interval_seconds=120,
            table_id="T06",
            payloads=[{"objects": []}, {"objects": [{"class": "trash", "confidence": 0.8}]}],
            person_mask_service=FakePersonMaskService([[], []]),
        )

        request = ActionWorkflowRequest(
            store_id="store_001",
            table_id="T06",
            zone_id="zone_B",
            frames=frames,
            staff_zone_visits=[],
        )

        self.assertEqual(len(request.frames), 2)
        self.assertEqual(request.frames[0].occupancy_source, "person_masking")
        self.assertEqual(request.frames[1].payload["objects"][0]["class"], "trash")

    def test_builds_workflow_frames_from_video_with_fake_extractor(self) -> None:
        def fake_extractor(video_path, *, interval_seconds, max_frames):
            self.assertEqual(str(video_path), "demo.avi")
            self.assertEqual(interval_seconds, 30)
            self.assertEqual(max_frames, 10)
            return [
                {"image": np.zeros((32, 32, 3), dtype=np.uint8), "offset_seconds": 0.0},
                {"image": np.zeros((32, 32, 3), dtype=np.uint8), "offset_seconds": 30.0},
                {"image": np.zeros((32, 32, 3), dtype=np.uint8), "offset_seconds": 60.0},
            ]

        detections = [
            [],
            [PersonDetection(x=1, y=1, width=4, height=4, score=0.9, source="fake")],
            [PersonDetection(x=2, y=2, width=5, height=5, score=0.88, source="fake")],
        ]
        frames = build_workflow_frames_from_video(
            video_path="demo.avi",
            captured_at_start="2026-06-03T14:00:00",
            frame_extractor=fake_extractor,
            person_mask_service=FakePersonMaskService(detections),
        )

        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0]["captured_at"], "2026-06-03T14:00:00")
        self.assertEqual(frames[1]["captured_at"], "2026-06-03T14:00:30")
        self.assertEqual(frames[2]["captured_at"], "2026-06-03T14:01:00")
        self.assertFalse(frames[0]["person_present"])
        self.assertEqual(frames[1]["person_count"], 1)
        self.assertEqual(frames[2]["person_count"], 1)

    def test_video_workflow_frames_use_interaction_roi_for_person_counting(self) -> None:
        roi = ROI.from_rectangle("TABLE_1", 5, 10, 20, 30)

        def fake_extractor(video_path, *, interval_seconds, max_frames):
            _ = (video_path, interval_seconds, max_frames)
            return [{"image": np.zeros((80, 80, 3), dtype=np.uint8), "offset_seconds": 0.0}]

        fake_service = FakePersonMaskService(
            [[PersonDetection(x=0, y=0, width=5, height=5, score=0.9, source="fake")]]
        )
        frames = build_workflow_frames_from_video(
            video_path="demo.avi",
            captured_at_start="2026-06-03T14:00:00",
            interaction_roi=roi,
            frame_extractor=fake_extractor,
            person_mask_service=fake_service,
        )

        self.assertEqual(len(frames), 1)
        self.assertEqual(fake_service.received_shapes[0][:2], (30, 20))
        self.assertEqual(frames[0]["occupancy_reason_codes"], [])

    def test_video_workflow_frames_apply_payload_builder_output(self) -> None:
        def fake_extractor(video_path, *, interval_seconds, max_frames):
            _ = (video_path, interval_seconds, max_frames)
            return [
                {"image": np.zeros((32, 32, 3), dtype=np.uint8), "offset_seconds": 0.0},
                {"image": np.zeros((32, 32, 3), dtype=np.uint8), "offset_seconds": 30.0},
            ]

        def payload_builder(frame_image, *, index, captured_at, frame_index, offset_seconds):
            _ = frame_image
            return {
                "objects": [{"class": "cup", "confidence": 0.9 + index * 0.01}],
                "captured_at_echo": captured_at,
                "offset_seconds": offset_seconds,
                "frame_index": frame_index,
            }

        frames = build_workflow_frames_from_video(
            video_path="demo.avi",
            captured_at_start="2026-06-03T14:00:00",
            frame_extractor=fake_extractor,
            payload_builder=payload_builder,
            person_mask_service=FakePersonMaskService([[], []]),
        )

        self.assertEqual(frames[0]["payload"]["objects"][0]["class"], "cup")
        self.assertEqual(frames[1]["payload"]["captured_at_echo"], "2026-06-03T14:00:30")
        self.assertEqual(frames[1]["payload"]["offset_seconds"], 30.0)

    def test_video_workflow_frames_use_minimal_payload_without_builder(self) -> None:
        def fake_extractor(video_path, *, interval_seconds, max_frames):
            _ = (video_path, interval_seconds, max_frames)
            return [{"image": np.zeros((32, 32, 3), dtype=np.uint8), "offset_seconds": 0.0}]

        frames = build_workflow_frames_from_video(
            video_path="demo.avi",
            captured_at_start="2026-06-03T14:00:00",
            table_id="T06",
            frame_extractor=fake_extractor,
            person_mask_service=FakePersonMaskService([[]]),
        )

        self.assertEqual(frames[0]["payload"]["table_id"], "T06")

    def test_video_workflow_frames_are_schema_compatible(self) -> None:
        def fake_extractor(video_path, *, interval_seconds, max_frames):
            _ = (video_path, interval_seconds, max_frames)
            return [
                {"image": np.zeros((32, 32, 3), dtype=np.uint8), "offset_seconds": 0.0},
                {"image": np.zeros((32, 32, 3), dtype=np.uint8), "offset_seconds": 60.0},
            ]

        frames = build_workflow_frames_from_video(
            video_path="demo.avi",
            captured_at_start="2026-06-03T14:00:00",
            interval_seconds=60,
            table_id="T06",
            frame_extractor=fake_extractor,
            person_mask_service=FakePersonMaskService([[], []]),
        )

        request = ActionWorkflowRequest(
            store_id="store_001",
            table_id="T06",
            zone_id="zone_B",
            frames=frames,
            staff_zone_visits=[],
        )

        self.assertEqual(len(request.frames), 2)
        self.assertEqual(request.frames[0].captured_at.isoformat(timespec="seconds"), "2026-06-03T14:00:00")
        self.assertEqual(request.frames[1].captured_at.isoformat(timespec="seconds"), "2026-06-03T14:01:00")

    def test_video_generated_frames_can_form_long_dwell_sequence(self) -> None:
        def fake_extractor(video_path, *, interval_seconds, max_frames):
            _ = (video_path, interval_seconds, max_frames)
            return [
                {"image": np.zeros((16, 16, 3), dtype=np.uint8), "offset_seconds": float(index * 60)}
                for index in range(11)
            ]

        frames = build_workflow_frames_from_video(
            video_path="demo.avi",
            captured_at_start="2026-06-03T14:00:00",
            interval_seconds=60,
            frame_extractor=fake_extractor,
            person_mask_service=FakePersonMaskService(
                [[PersonDetection(x=0, y=0, width=3, height=3, score=0.9, source="fake")] for _ in range(11)]
            ),
        )

        sequence = build_meal_occupancy_sequence(
            frames,
            table_id="T06",
            occupancy_candidate_seconds=60,
            meal_min_duration_seconds=600,
            meal_absence_seconds=90,
        )

        self.assertTrue(sequence["meal_confirmed"])
        self.assertEqual(sequence["meal_status"], "CUSTOMER_IN_USE")

    def test_long_dwell_mode_does_not_confirm_meal_before_ten_minutes(self) -> None:
        frames = build_workflow_frames_from_images(
            images=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(10)],
            start_time="2026-06-03T14:00:00",
            interval_seconds=60,
            person_mask_service=FakePersonMaskService(
                [[PersonDetection(x=0, y=0, width=3, height=3, score=0.9, source="fake")] for _ in range(10)]
            ),
        )

        sequence = build_meal_occupancy_sequence(
            frames,
            table_id="T06",
            occupancy_candidate_seconds=60,
            meal_min_duration_seconds=600,
            meal_absence_seconds=90,
        )

        self.assertTrue(sequence["occupancy_candidate_detected"])
        self.assertFalse(sequence["meal_confirmed"])
        self.assertEqual(sequence["meal_status"], "IDLE")

    def test_long_dwell_mode_confirms_meal_after_ten_minutes(self) -> None:
        frames = build_workflow_frames_from_images(
            images=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(11)],
            start_time="2026-06-03T14:00:00",
            interval_seconds=60,
            person_mask_service=FakePersonMaskService(
                [[PersonDetection(x=0, y=0, width=3, height=3, score=0.9, source="fake")] for _ in range(11)]
            ),
        )

        sequence = build_meal_occupancy_sequence(
            frames,
            table_id="T06",
            occupancy_candidate_seconds=60,
            meal_min_duration_seconds=600,
            meal_absence_seconds=90,
        )

        self.assertTrue(sequence["meal_confirmed"])
        self.assertEqual(sequence["meal_status"], "CUSTOMER_IN_USE")

    def test_long_dwell_mode_can_reach_meal_ended_after_absence(self) -> None:
        presence_detection = [PersonDetection(x=0, y=0, width=3, height=3, score=0.9, source="fake")]
        frames = build_workflow_frames_from_images(
            images=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(13)],
            captured_at_values=[
                "2026-06-03T14:00:00",
                "2026-06-03T14:01:00",
                "2026-06-03T14:02:00",
                "2026-06-03T14:03:00",
                "2026-06-03T14:04:00",
                "2026-06-03T14:05:00",
                "2026-06-03T14:06:00",
                "2026-06-03T14:07:00",
                "2026-06-03T14:08:00",
                "2026-06-03T14:09:00",
                "2026-06-03T14:10:00",
                "2026-06-03T14:10:30",
                "2026-06-03T14:12:00",
            ],
            person_mask_service=FakePersonMaskService(
                [presence_detection for _ in range(11)] + [[], []]
            ),
        )

        sequence = build_meal_occupancy_sequence(
            frames,
            table_id="T06",
            occupancy_candidate_seconds=60,
            meal_min_duration_seconds=600,
            meal_absence_seconds=90,
        )

        self.assertTrue(sequence["meal_confirmed"])
        self.assertEqual(sequence["meal_status"], "MEAL_ENDED")
        self.assertEqual(sequence["meal_ended_at"], "2026-06-03T14:12:00")

    def test_default_action_service_keeps_existing_sixty_second_meal_confirmation(self) -> None:
        service = ActionCleanlinessService()
        meal_session = service.evaluate_meal_session(
            "T06",
            [
                TableOccupancySample(timestamp=dt("2026-06-03T14:00:00"), customer_present=True),
                TableOccupancySample(timestamp=dt("2026-06-03T14:01:00"), customer_present=True),
            ],
        )

        self.assertIsNotNone(meal_session)


if __name__ == "__main__":
    unittest.main()
