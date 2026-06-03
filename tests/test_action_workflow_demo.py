from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import app.main as main_module
from app.database import clear_cleanliness_results, fetch_cleanliness_results, init_db
from scripts.generate_test_data import generate_test_data


class FakePersonMaskService:
    def __init__(self, detection_counts: list[int]) -> None:
        self._detection_counts = list(detection_counts)
        self.received_shapes: list[tuple[int, ...]] = []

    def apply_black_mask(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[object]]:
        self.received_shapes.append(image.shape)
        index = min(len(self.received_shapes) - 1, len(self._detection_counts) - 1)
        detections = [object() for _ in range(self._detection_counts[index])]
        return image, image.copy(), detections


def demo_staff_zone_visits_json() -> str:
    return json.dumps(
        [
            {
                "visit_id": "visit_000341",
                "staff_id": "staff_03",
                "zone_id": "zone_B",
                "entered_at": "2026-06-03T14:44:10",
                "left_at": "2026-06-03T14:44:36",
                "dwell_seconds": 26,
                "mean_confidence": 0.76,
                "sample_count": 8,
            }
        ]
    )


def image_upload_files() -> list[tuple[str, tuple[str, bytes, str]]]:
    image_bytes = Path("data/test_data/StoreAlpha_FrontCam_present_frame.png").read_bytes()
    return [
        ("image_files", ("frame_0.png", image_bytes, "image/png")),
        ("image_files", ("frame_1.png", image_bytes, "image/png")),
        ("image_files", ("frame_2.png", image_bytes, "image/png")),
    ]


class ActionWorkflowDemoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()
        generate_test_data()

    def setUp(self) -> None:
        clear_cleanliness_results()

    def test_action_workflow_demo_page_returns_200(self) -> None:
        client = TestClient(main_module.app)

        response = client.get("/action-workflow-demo")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/action-cleanliness/workflow-from-images", response.text)

    def test_workflow_from_images_returns_result_for_three_uploaded_images(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])

        try:
            response = client.post(
                "/api/action-cleanliness/workflow-from-images",
                data={
                    "store_id": "store_001",
                    "table_id": "T06",
                    "zone_id": "zone_B",
                    "captured_at_start": "2026-06-03T14:10:20",
                    "interval_seconds": "120",
                    "preset": "cleaned_likely",
                    "staff_zone_visits_json": demo_staff_zone_visits_json(),
                },
                files=image_upload_files(),
            )
        finally:
            main_module.person_mask_service = original_person_mask_service

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload["frames"]), 3)
        self.assertEqual(payload["cleaning_status"], "CLEANED_LIKELY")

    def test_workflow_from_images_increases_captured_at_by_interval_seconds(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])

        try:
            response = client.post(
                "/api/action-cleanliness/workflow-from-images",
                data={
                    "store_id": "store_001",
                    "table_id": "T06",
                    "zone_id": "zone_B",
                    "captured_at_start": "2026-06-03T14:10:20",
                    "interval_seconds": "120",
                    "preset": "cleaned_likely",
                    "staff_zone_visits_json": demo_staff_zone_visits_json(),
                },
                files=image_upload_files(),
            )
        finally:
            main_module.person_mask_service = original_person_mask_service

        frames = response.json()["frames"]
        self.assertEqual(frames[0]["captured_at"], "2026-06-03T14:10:20")
        self.assertEqual(frames[1]["captured_at"], "2026-06-03T14:12:20")
        self.assertEqual(frames[2]["captured_at"], "2026-06-03T14:14:20")

    def test_workflow_from_images_uses_staff_zone_visits_json(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])

        try:
            response = client.post(
                "/api/action-cleanliness/workflow-from-images",
                data={
                    "store_id": "store_001",
                    "table_id": "T06",
                    "zone_id": "zone_B",
                    "captured_at_start": "2026-06-03T14:10:20",
                    "interval_seconds": "120",
                    "preset": "cleaned_likely",
                    "staff_zone_visits_json": demo_staff_zone_visits_json(),
                },
                files=image_upload_files(),
            )
        finally:
            main_module.person_mask_service = original_person_mask_service

        payload = response.json()
        self.assertEqual(payload["cleaning_status"], "CLEANED_LIKELY")
        self.assertIn("STAFF_ZONE_DWELL_CONFIRMED", payload["reason_codes"])

    def test_workflow_from_images_applies_visual_preset_to_frame_payloads(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])

        try:
            response = client.post(
                "/api/action-cleanliness/workflow-from-images",
                data={
                    "store_id": "store_001",
                    "table_id": "T06",
                    "zone_id": "zone_B",
                    "captured_at_start": "2026-06-03T14:10:20",
                    "interval_seconds": "120",
                    "preset": "high_mess_cap",
                },
                files=image_upload_files(),
            )
        finally:
            main_module.person_mask_service = original_person_mask_service

        payload = response.json()
        self.assertEqual(payload["frames"][-1]["payload"]["yolo_mess_score"], 0.8)
        self.assertIn("HIGH_YOLO_MESS_CAP_35", payload["applied_caps"])

    def test_workflow_from_images_does_not_save_when_save_result_is_false(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])

        try:
            response = client.post(
                "/api/action-cleanliness/workflow-from-images",
                data={
                    "store_id": "store_001",
                    "table_id": "T06",
                    "zone_id": "zone_B",
                    "captured_at_start": "2026-06-03T14:10:20",
                    "interval_seconds": "120",
                    "preset": "cleaned_likely",
                    "save_result": "false",
                    "staff_zone_visits_json": demo_staff_zone_visits_json(),
                },
                files=image_upload_files(),
            )
        finally:
            main_module.person_mask_service = original_person_mask_service

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["saved"])
        self.assertIsNone(payload["result_id"])
        self.assertEqual(fetch_cleanliness_results({"mode": "action_workflow"}), [])

    def test_workflow_from_images_saves_when_save_result_is_true(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])

        try:
            response = client.post(
                "/api/action-cleanliness/workflow-from-images",
                data={
                    "store_id": "store_001",
                    "table_id": "T06",
                    "zone_id": "zone_B",
                    "captured_at_start": "2026-06-03T14:10:20",
                    "interval_seconds": "120",
                    "preset": "cleaned_likely",
                    "save_result": "true",
                    "staff_zone_visits_json": demo_staff_zone_visits_json(),
                },
                files=image_upload_files(),
            )
        finally:
            main_module.person_mask_service = original_person_mask_service

        payload = response.json()
        records = fetch_cleanliness_results({"mode": "action_workflow"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["saved"])
        self.assertIsInstance(payload["result_id"], int)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], payload["result_id"])

    def test_workflow_from_images_rejects_invalid_json(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])

        try:
            response = client.post(
                "/api/action-cleanliness/workflow-from-images",
                data={
                    "store_id": "store_001",
                    "table_id": "T06",
                    "zone_id": "zone_B",
                    "captured_at_start": "2026-06-03T14:10:20",
                    "interval_seconds": "120",
                    "staff_zone_visits_json": "{bad-json",
                },
                files=image_upload_files(),
            )
        finally:
            main_module.person_mask_service = original_person_mask_service

        self.assertEqual(response.status_code, 400)

    def test_workflow_from_video_returns_result(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        original_sampler = main_module.sample_video_workflow_frames
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])
        main_module.sample_video_workflow_frames = lambda *args, **kwargs: [
            {"image": np.zeros((24, 24, 3), dtype=np.uint8), "offset_seconds": 0.0},
            {"image": np.zeros((24, 24, 3), dtype=np.uint8), "offset_seconds": 120.0},
            {"image": np.zeros((24, 24, 3), dtype=np.uint8), "offset_seconds": 240.0},
        ]

        try:
            response = client.post(
                "/api/action-cleanliness/workflow-from-video",
                data={
                    "store_id": "store_001",
                    "table_id": "T06",
                    "zone_id": "zone_B",
                    "captured_at_start": "2026-06-03T14:10:20",
                    "interval_seconds": "120",
                    "max_frames": "3",
                    "preset": "cleaned_likely",
                    "staff_zone_visits_json": demo_staff_zone_visits_json(),
                },
                files={"video_file": ("demo.avi", b"fake-video", "video/x-msvideo")},
            )
        finally:
            main_module.person_mask_service = original_person_mask_service
            main_module.sample_video_workflow_frames = original_sampler

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload["frames"]), 3)
        self.assertEqual(payload["cleaning_status"], "CLEANED_LIKELY")


if __name__ == "__main__":
    unittest.main()
