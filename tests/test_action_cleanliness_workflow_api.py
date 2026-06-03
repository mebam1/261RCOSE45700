from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

import app.main as main_module
from app.database import clear_cleanliness_results, fetch_cleanliness_results, fetch_cleanliness_store_summary, init_db


def build_request_payload() -> dict[str, object]:
    return {
        "store_id": "store_001",
        "table_id": "T06",
        "zone_id": "zone_B",
        "frames": [
            {
                "captured_at": "2026-06-03T14:10:20",
                "person_present": True,
                "person_count": 2,
                "payload": {
                    "objects": [
                        {"class": "cup", "confidence": 0.91},
                        {"class": "tray", "confidence": 0.84},
                    ]
                },
            },
            {
                "captured_at": "2026-06-03T14:42:35",
                "person_present": False,
                "person_count": 0,
                "payload": {
                    "objects": [
                        {"class": "cup", "confidence": 0.91},
                        {"class": "tray", "confidence": 0.84},
                    ]
                },
            },
            {
                "captured_at": "2026-06-03T14:44:40",
                "person_present": False,
                "person_count": 0,
                "payload": {
                    "objects": [
                        {"class": "trash", "confidence": 0.71},
                    ],
                    "scene_flags": {
                        "messy_arrangement": False,
                    },
                    "vision_confidence": 0.84,
                },
            },
        ],
        "staff_zone_visits": [
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
        ],
    }


class ActionCleanlinessWorkflowApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    def setUp(self) -> None:
        clear_cleanliness_results()

    def test_workflow_endpoint_returns_200_for_valid_request(self) -> None:
        client = TestClient(main_module.app)

        response = client.post("/api/action-cleanliness/workflow", json=build_request_payload())

        self.assertEqual(response.status_code, 200)

    def test_workflow_endpoint_does_not_save_when_save_result_is_false(self) -> None:
        client = TestClient(main_module.app)
        request_payload = build_request_payload()
        request_payload["save_result"] = False

        response = client.post("/api/action-cleanliness/workflow", json=request_payload)
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["saved"])
        self.assertIsNone(payload["result_id"])
        self.assertEqual(fetch_cleanliness_results({"mode": "action_workflow"}), [])

    def test_workflow_endpoint_saves_to_existing_cleanliness_results_when_requested(self) -> None:
        client = TestClient(main_module.app)
        request_payload = build_request_payload()
        request_payload["save_result"] = True

        response = client.post("/api/action-cleanliness/workflow", json=request_payload)
        payload = response.json()
        records = fetch_cleanliness_results({"mode": "action_workflow"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["saved"])
        self.assertIsInstance(payload["result_id"], int)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], payload["result_id"])
        self.assertEqual(records[0]["store_name"], "store_001")
        self.assertEqual(records[0]["cctv_id"], "store_001::zone_B")
        self.assertEqual(records[0]["roi_name"], "T06")
        self.assertEqual(records[0]["mode"], "action_workflow")
        self.assertEqual(records[0]["decision"], "cleaned_likely")
        self.assertIsNotNone(records[0]["score"])
        self.assertEqual(records[0]["score"], 5)
        self.assertEqual(records[0]["final_stage"], "workflow_api")
        action_features = json.loads(records[0]["action_features"])
        self.assertEqual(action_features["final_cleanliness_score"], payload["final_cleanliness_score"])
        self.assertEqual(action_features["result"]["final_cleanliness_score"], payload["final_cleanliness_score"])

    def test_saved_action_workflow_result_is_returned_by_fetch_cleanliness_results(self) -> None:
        client = TestClient(main_module.app)
        request_payload = build_request_payload()
        request_payload["save_result"] = True

        client.post("/api/action-cleanliness/workflow", json=request_payload)
        records = fetch_cleanliness_results({"mode": "action_workflow", "store_name": "store_001"})

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["store_name"], "store_001")
        self.assertEqual(records[0]["mode"], "action_workflow")

    def test_action_workflow_result_keeps_store_summary_compatible(self) -> None:
        client = TestClient(main_module.app)
        request_payload = build_request_payload()
        request_payload["save_result"] = True

        client.post("/api/action-cleanliness/workflow", json=request_payload)
        summary = fetch_cleanliness_store_summary({"store_name": "store_001"})

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["store_name"], "store_001")
        self.assertEqual(summary[0]["cleaned_count"], 1)
        self.assertEqual(summary[0]["needs_check_count"], 0)
        self.assertEqual(summary[0]["unknown_count"], 0)
        self.assertEqual(summary[0]["average_score"], 5.0)

    def test_workflow_endpoint_returns_cleaned_likely_when_staff_and_visual_improve(self) -> None:
        client = TestClient(main_module.app)

        response = client.post("/api/action-cleanliness/workflow", json=build_request_payload())
        payload = response.json()

        self.assertEqual(payload["cleaning_status"], "CLEANED_LIKELY")
        self.assertEqual(payload["meal_status"], "MEAL_ENDED")
        self.assertEqual(payload["decision"], "CLEAN_OK")

    def test_workflow_endpoint_returns_cleaning_unverified_when_only_staff_visit_exists(self) -> None:
        client = TestClient(main_module.app)
        request_payload = build_request_payload()
        request_payload["frames"][2]["payload"] = {
            "objects": [
                {"class": "cup", "confidence": 0.91},
                {"class": "tray", "confidence": 0.84},
            ],
            "vision_confidence": 0.84,
        }

        response = client.post("/api/action-cleanliness/workflow", json=request_payload)
        payload = response.json()

        self.assertEqual(payload["cleaning_status"], "CLEANING_UNVERIFIED")

    def test_workflow_endpoint_returns_missed_suspected_when_sla_expires_without_staff_visit(self) -> None:
        client = TestClient(main_module.app)
        request_payload = build_request_payload()
        request_payload["frames"][2]["captured_at"] = "2026-06-03T14:49:10"
        request_payload["frames"][2]["payload"] = {
            "objects": [
                {"class": "cup", "confidence": 0.91},
                {"class": "tray", "confidence": 0.84},
            ],
            "vision_confidence": 0.84,
        }
        request_payload["staff_zone_visits"] = []

        response = client.post("/api/action-cleanliness/workflow", json=request_payload)
        payload = response.json()

        self.assertEqual(payload["cleaning_status"], "MISSED_SUSPECTED")

    def test_workflow_endpoint_applies_high_mess_cap(self) -> None:
        client = TestClient(main_module.app)
        request_payload = build_request_payload()
        request_payload["frames"][2]["payload"] = {
            "yolo_mess_score": 0.80,
            "detected_objects": [
                {"class": "trash", "count": 1, "max_confidence": 0.71},
            ],
            "vision_confidence": 0.88,
        }

        response = client.post("/api/action-cleanliness/workflow", json=request_payload)
        payload = response.json()

        self.assertEqual(payload["final_cleanliness_score"], 35)
        self.assertIn("HIGH_YOLO_MESS_CAP_35", payload["applied_caps"])

    def test_workflow_endpoint_rejects_invalid_request(self) -> None:
        client = TestClient(main_module.app)

        response = client.post(
            "/api/action-cleanliness/workflow",
            json={
                "store_id": "store_001",
                "table_id": "T06",
                "zone_id": "zone_B",
                "frames": [],
                "staff_zone_visits": [],
            },
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
