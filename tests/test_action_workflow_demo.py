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


class FakeYoloHelper:
    def __init__(self, detections_by_index: list[list[dict[str, object]]]) -> None:
        self._detections_by_index = list(detections_by_index)
        self.detected_paths: list[str] = []

    def detect_objects(self, image_path: Path) -> list[dict[str, object]]:
        self.detected_paths.append(image_path.name)
        index = min(len(self.detected_paths) - 1, len(self._detections_by_index) - 1)
        return list(self._detections_by_index[index])


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

    def test_workflow_from_images_builds_visual_payloads_from_yolo(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        original_yolo_helper = main_module.workflow_yolo_helper
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])
        fake_yolo_helper = FakeYoloHelper(
            [
                [{"label": "cup", "confidence": 0.91, "bbox": {"x1": 1, "y1": 1, "x2": 10, "y2": 10}}],
                [{"label": "tray", "confidence": 0.82, "bbox": {"x1": 2, "y1": 2, "x2": 11, "y2": 11}}],
                [{"label": "paper napkin", "confidence": 0.77, "bbox": {"x1": 3, "y1": 3, "x2": 12, "y2": 12}}],
            ]
        )
        main_module.workflow_yolo_helper = fake_yolo_helper

        try:
            response = client.post(
                "/api/action-cleanliness/workflow-from-images",
                data={
                    "store_id": "store_001",
                    "table_id": "T06",
                    "zone_id": "zone_B",
                    "captured_at_start": "2026-06-03T14:10:20",
                    "interval_seconds": "120",
                    "visual_payload_source": "yolo",
                    "staff_zone_visits_json": demo_staff_zone_visits_json(),
                },
                files=image_upload_files(),
            )
        finally:
            main_module.person_mask_service = original_person_mask_service
            main_module.workflow_yolo_helper = original_yolo_helper

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["frames"][2]["payload"]["yolo_objects"][0]["label"], "napkin")
        self.assertEqual(len(fake_yolo_helper.detected_paths), 3)

    def test_workflow_from_images_keeps_json_path_working(self) -> None:
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
                    "visual_payload_source": "json",
                    "visual_payloads_json": json.dumps(
                        [
                            {"objects": [{"class": "cup", "confidence": 0.91}]},
                            {"objects": [{"class": "cup", "confidence": 0.91}]},
                            {"yolo_mess_score": 0.80, "detected_objects": [{"class": "trash", "count": 1, "max_confidence": 0.71}]},
                        ]
                    ),
                },
                files=image_upload_files(),
            )
        finally:
            main_module.person_mask_service = original_person_mask_service

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["frames"][2]["payload"]["yolo_mess_score"], 0.8)
        self.assertIn("HIGH_YOLO_MESS_CAP_35", payload["applied_caps"])

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

    def test_workflow_video_candidates_returns_preview_metadata(self) -> None:
        client = TestClient(main_module.app)
        original_sampler = main_module.sample_dynamic_video_workflow_frames
        main_module.sample_dynamic_video_workflow_frames = lambda *args, **kwargs: [
            {
                "image": np.zeros((24, 24, 3), dtype=np.uint8),
                "crop_image": np.zeros((16, 16, 3), dtype=np.uint8),
                "offset_seconds": 0.0,
                "frame_type": "periodic_sample",
                "sampling_state": "idle",
                "priority": 0.15,
                "reason_codes": ["periodic_sample"],
                "features": {"change_score": 0.0, "person_present": False, "person_count": 0},
            },
            {
                "image": np.full((24, 24, 3), 255, dtype=np.uint8),
                "crop_image": np.full((16, 16, 3), 255, dtype=np.uint8),
                "offset_seconds": 12.0,
                "frame_type": "meal_end_candidate",
                "sampling_state": "meal_end_candidate",
                "priority": 0.83,
                "reason_codes": ["person_left"],
                "features": {"change_score": 0.22, "person_present": False, "person_count": 0},
            },
        ]

        try:
            response = client.post(
                "/api/action-cleanliness/workflow-video-candidates",
                data={
                    "table_id": "T06",
                    "captured_at_start": "2026-06-03T14:10:20",
                    "interval_seconds": "10",
                    "max_frames": "2",
                },
                files={"video_file": ("demo.avi", b"fake-video", "video/x-msvideo")},
            )
        finally:
            main_module.sample_dynamic_video_workflow_frames = original_sampler

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["debug_trace_count"], 2)
        self.assertEqual(payload["selected_candidates"][0]["frame_type"], "meal_end_candidate")
        self.assertTrue(payload["selected_candidates"][0]["preview_url"].startswith("data:image/"))
        self.assertEqual(payload["debug_trace"][0]["frame_type"], "periodic_sample")

    def test_workflow_from_video_dynamic_sampling_includes_candidate_summary(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        original_sampler = main_module.sample_dynamic_video_workflow_frames
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])
        main_module.sample_dynamic_video_workflow_frames = lambda *args, **kwargs: [
            {
                "image": np.zeros((24, 24, 3), dtype=np.uint8),
                "crop_image": np.zeros((16, 16, 3), dtype=np.uint8),
                "offset_seconds": 0.0,
                "frame_type": "occupied_representative",
                "sampling_state": "occupied",
                "priority": 0.42,
                "reason_codes": ["person_present"],
                "features": {"change_score": 0.0, "person_present": True, "person_count": 2},
            },
            {
                "image": np.zeros((24, 24, 3), dtype=np.uint8),
                "crop_image": np.zeros((16, 16, 3), dtype=np.uint8),
                "offset_seconds": 120.0,
                "frame_type": "meal_end_candidate",
                "sampling_state": "meal_end_candidate",
                "priority": 0.81,
                "reason_codes": ["person_left"],
                "features": {"change_score": 0.12, "person_present": False, "person_count": 0},
            },
            {
                "image": np.zeros((24, 24, 3), dtype=np.uint8),
                "crop_image": np.zeros((16, 16, 3), dtype=np.uint8),
                "offset_seconds": 240.0,
                "frame_type": "post_check",
                "sampling_state": "post_check",
                "priority": 0.76,
                "reason_codes": ["post_check_stable"],
                "features": {"change_score": 0.02, "person_present": False, "person_count": 0},
            },
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
                    "dynamic_sampling": "true",
                    "preset": "cleaned_likely",
                    "staff_zone_visits_json": demo_staff_zone_visits_json(),
                },
                files={"video_file": ("demo.avi", b"fake-video", "video/x-msvideo")},
            )
        finally:
            main_module.person_mask_service = original_person_mask_service
            main_module.sample_dynamic_video_workflow_frames = original_sampler

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["dynamic_sampling"])
        self.assertEqual(payload["candidate_count"], 3)
        self.assertEqual(payload["candidate_summary"][1]["frame_type"], "meal_end_candidate")
        self.assertEqual(payload["debug_trace_count"], 3)
        self.assertEqual(payload["episodes"][0]["post_check_at"], 240.0)
        self.assertEqual(payload["frames"][1]["payload"]["sampler_frame_type"], "meal_end_candidate")
        self.assertEqual(payload["frames"][2]["payload"]["sampler_reason_codes"], ["post_check_stable"])

    def test_workflow_from_video_builds_visual_payloads_from_yolo(self) -> None:
        client = TestClient(main_module.app)
        original_person_mask_service = main_module.person_mask_service
        original_sampler = main_module.sample_video_workflow_frames
        original_yolo_helper = main_module.workflow_yolo_helper
        fake_yolo_helper = FakeYoloHelper(
            [
                [{"label": "cup", "confidence": 0.91, "bbox": {"x1": 1, "y1": 1, "x2": 8, "y2": 8}}],
                [{"label": "tray", "confidence": 0.84, "bbox": {"x1": 2, "y1": 2, "x2": 9, "y2": 9}}],
                [{"label": "spill", "confidence": 0.87, "bbox": {"x1": 3, "y1": 3, "x2": 10, "y2": 10}}],
            ]
        )
        main_module.person_mask_service = FakePersonMaskService([2, 0, 0])
        main_module.sample_video_workflow_frames = lambda *args, **kwargs: [
            {"image": np.zeros((24, 24, 3), dtype=np.uint8), "offset_seconds": 0.0},
            {"image": np.zeros((24, 24, 3), dtype=np.uint8), "offset_seconds": 120.0},
            {"image": np.zeros((24, 24, 3), dtype=np.uint8), "offset_seconds": 240.0},
        ]
        main_module.workflow_yolo_helper = fake_yolo_helper

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
                    "visual_payload_source": "yolo",
                    "staff_zone_visits_json": demo_staff_zone_visits_json(),
                },
                files={"video_file": ("demo.avi", b"fake-video", "video/x-msvideo")},
            )
        finally:
            main_module.person_mask_service = original_person_mask_service
            main_module.sample_video_workflow_frames = original_sampler
            main_module.workflow_yolo_helper = original_yolo_helper

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["frames"][2]["payload"]["visible_contamination"])
        self.assertEqual(len(fake_yolo_helper.detected_paths), 3)


if __name__ == "__main__":
    unittest.main()
