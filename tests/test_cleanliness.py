from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main_module
from app.analysis import read_image
from app.cleanliness import (
    CleanlinessAssessment,
    CleanlinessResult,
    CleanlinessService,
    PROMPT_PROFILE_GENERAL,
    PROMPT_PROFILE_RESTAURANT,
    build_cleanliness_analysis_prompt,
)


class FakeCleanlinessClient:
    def __init__(self, assessment: CleanlinessAssessment) -> None:
        self.assessment = assessment
        self.received_path: Path | None = None
        self.received_prompt_profile: str | None = None
        self.received_prompt_suffix: str | None = None

    def analyze_cleanliness(
        self,
        image_path: Path,
        prompt_profile: str = PROMPT_PROFILE_RESTAURANT,
        prompt_suffix: str | None = None,
        image_paths: list[Path] | None = None,
    ) -> CleanlinessAssessment:
        self.received_path = image_path
        self.received_prompt_profile = prompt_profile
        self.received_prompt_suffix = prompt_suffix
        _ = image_paths
        return self.assessment


class FakeYoloModule:
    def prepare_for_cleanliness(
        self,
        image_path: Path,
        *,
        source_image_path: Path | None = None,
        roi: object | None = None,
        output_stem: str | None = None,
    ) -> object:
        _ = source_image_path
        _ = roi
        _ = output_stem

        class FakeYoloAugmentation:
            def __init__(self, target_path: Path) -> None:
                self.image_path = target_path
                self.crop_annotated_path = None
                self.source_annotated_path = None
                self.prompt_instruction = "YOLO detection JSON:\n" + json.dumps(
                    {
                        "image_name": target_path.name,
                        "detector_name": "YOLOE",
                        "weights": "fake-yoloe.pt",
                        "confidence_threshold": 0.25,
                        "iou_threshold": 0.45,
                        "tile_size": 512,
                        "tile_overlap": 0.25,
                        "prompt_labels": ["used napkin", "food wrapper"],
                        "detections": [
                            {
                                "label": "cup",
                                "confidence": 0.91,
                                "bbox": {"x1": 12.0, "y1": 18.0, "x2": 48.0, "y2": 72.0},
                                "position": "top-left",
                                "source": "tile_0",
                            }
                        ],
                    }
                )

            def to_dict(self) -> dict[str, object]:
                return {
                    "image_name": self.image_path.name,
                    "detector_name": "YOLOE",
                    "weights": "fake-yoloe.pt",
                    "confidence_threshold": 0.25,
                    "iou_threshold": 0.45,
                    "tile_size": 512,
                    "tile_overlap": 0.25,
                    "prompt_labels": ["used napkin", "food wrapper"],
                    "detections": [
                        {
                            "label": "cup",
                            "confidence": 0.91,
                            "bbox": {"x1": 12.0, "y1": 18.0, "x2": 48.0, "y2": 72.0},
                            "position": "top-left",
                            "source": "tile_0",
                        }
                    ],
                }

        return FakeYoloAugmentation(image_path)


class CleanlinessTest(unittest.TestCase):
    def test_inspect_image_sends_roi_crop_to_openai_client(self) -> None:
        assessment = CleanlinessAssessment(
            score=4,
            confidence=0.82,
            summary="The visible area is clean with only minor clutter.",
            findings=["Floor is clear", "Counters appear orderly"],
            exact_objects=["floor", "counter"],
            estimated_objects=["small scattered items"],
        )
        client = FakeCleanlinessClient(assessment)
        service = CleanlinessService(client)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "store.png"
            crop_path = Path(temp_dir) / "roi_crop.png"
            source_path.write_bytes(b"not decoded by service")
            crop_path.write_bytes(b"not decoded by service")

            result = service.inspect_image(source_path, inspected_path=crop_path, prompt_profile=PROMPT_PROFILE_RESTAURANT)

        self.assertEqual(client.received_path, crop_path)
        self.assertEqual(client.received_prompt_profile, PROMPT_PROFILE_RESTAURANT)
        self.assertEqual(result.source_path, source_path)
        self.assertEqual(result.inspected_path, crop_path)
        self.assertEqual(result.prompt_profile, PROMPT_PROFILE_RESTAURANT)
        self.assertEqual(result.score, 4)
        self.assertEqual(result.confidence, 0.82)
        self.assertEqual(result.findings, ["Floor is clear", "Counters appear orderly"])
        self.assertEqual(result.exact_objects, ["floor", "counter"])
        self.assertEqual(result.estimated_objects, ["small scattered items"])
        self.assertFalse(result.use_yolo)
        self.assertIsNone(result.yolo_payload)

    def test_rejects_unsupported_file_type(self) -> None:
        service = CleanlinessService(
            FakeCleanlinessClient(
                CleanlinessAssessment(
                    score=5,
                    confidence=1.0,
                    summary="",
                    findings=[],
                    exact_objects=[],
                    estimated_objects=[],
                )
            )
        )

        with self.assertRaises(ValueError):
            service.inspect_image(Path("store.txt"))

    def test_inspect_image_adds_yolo_prompt_and_payload_when_enabled(self) -> None:
        assessment = CleanlinessAssessment(
            score=2,
            confidence=0.68,
            summary="Visible debris remains on the surface.",
            findings=["A cup-like object appears near the top-left area."],
            exact_objects=["table"],
            estimated_objects=["cup"],
        )
        client = FakeCleanlinessClient(assessment)
        service = CleanlinessService(client=client, yolo_helper=FakeYoloModule())

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "store.png"
            crop_path = Path(temp_dir) / "roi_crop.png"
            source_path.write_bytes(b"not decoded by service")
            crop_path.write_bytes(b"not decoded by service")

            result = service.inspect_image(
                source_path,
                inspected_path=crop_path,
                prompt_profile=PROMPT_PROFILE_RESTAURANT,
                use_yolo=True,
            )

        self.assertIsNotNone(client.received_prompt_suffix)
        assert client.received_prompt_suffix is not None
        self.assertIn("YOLO detection JSON", client.received_prompt_suffix)
        self.assertTrue(result.use_yolo)
        self.assertIsNotNone(result.yolo_payload)
        assert result.yolo_payload is not None
        self.assertEqual(result.yolo_payload["detector_name"], "YOLOE")
        self.assertEqual(result.yolo_payload["weights"], "fake-yoloe.pt")
        self.assertEqual(result.yolo_payload["detections"][0]["label"], "cup")

    def test_cleanliness_route_uses_selected_roi_crop(self) -> None:
        class FakeCleanlinessService:
            def __init__(self) -> None:
                self.source_path: Path | None = None
                self.inspected_path: Path | None = None
                self.prompt_profile: str | None = None
                self.use_yolo: bool = False

            def inspect_image(
                self,
                source_path: Path,
                inspected_path: Path | None = None,
                prompt_profile: str = PROMPT_PROFILE_RESTAURANT,
                use_yolo: bool = False,
                roi: object | None = None,
                output_stem: str | None = None,
            ) -> CleanlinessResult:
                self.source_path = source_path
                self.inspected_path = inspected_path
                self.prompt_profile = prompt_profile
                self.use_yolo = use_yolo
                _ = roi
                _ = output_stem
                return CleanlinessResult(
                    source_path=source_path,
                    inspected_path=inspected_path or source_path,
                    llm_input_paths=[inspected_path or source_path],
                    score=3,
                    confidence=0.74,
                    summary="ROI has minor visible clutter.",
                    findings=["Some items are scattered on the surface"],
                    exact_objects=["table surface", "chair"],
                    estimated_objects=["small paper-like items"],
                    prompt_profile=prompt_profile,
                )

        fake_service = FakeCleanlinessService()
        original_service = main_module.cleanliness_service
        main_module.cleanliness_service = fake_service

        try:
            client = TestClient(main_module.app)
            image_path = Path("data/test_data/StoreAlpha_FrontCam_present_frame.png")
            with image_path.open("rb") as handle:
                response = client.post(
                    "/cleanliness",
                    data={
                        "config_id": "StoreAlpha_FrontCam",
                        "roi_name": "POP",
                        "prompt_profile": "restaurant",
                    },
                    files={"image_file": ("store.png", handle, "image/png")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn("식당 특화 프롬프트", response.text)
            self.assertIn("OpenAI 입력 이미지", response.text)
            self.assertIn("정확히 식별 가능한 물체", response.text)
            self.assertIn("합리적으로 추정 가능한 물체", response.text)
            self.assertIn("table surface", response.text)
            self.assertIn("small paper-like items", response.text)
            self.assertIsNotNone(fake_service.source_path)
            self.assertIsNotNone(fake_service.inspected_path)
            self.assertEqual(fake_service.prompt_profile, PROMPT_PROFILE_RESTAURANT)
            self.assertFalse(fake_service.use_yolo)
            assert fake_service.source_path is not None
            assert fake_service.inspected_path is not None
            self.assertTrue(fake_service.source_path.exists())
            self.assertTrue(fake_service.inspected_path.exists())
            self.assertNotEqual(fake_service.source_path, fake_service.inspected_path)
            self.assertEqual(fake_service.inspected_path.parent.name, "analysis_crops")
            roi_crop = read_image(fake_service.inspected_path)
            self.assertGreater(roi_crop.size, 0)
        finally:
            if fake_service.source_path is not None:
                fake_service.source_path.unlink(missing_ok=True)
            if fake_service.inspected_path is not None:
                fake_service.inspected_path.unlink(missing_ok=True)
            main_module.cleanliness_service = original_service

    def test_cleanliness_route_passes_yolo_flag_and_renders_json(self) -> None:
        class FakeCleanlinessService:
            def __init__(self) -> None:
                self.use_yolo: bool = False
                self.source_path: Path | None = None
                self.inspected_path: Path | None = None

            def inspect_image(
                self,
                source_path: Path,
                inspected_path: Path | None = None,
                prompt_profile: str = PROMPT_PROFILE_RESTAURANT,
                use_yolo: bool = False,
                roi: object | None = None,
                output_stem: str | None = None,
            ) -> CleanlinessResult:
                self.use_yolo = use_yolo
                self.source_path = source_path
                self.inspected_path = inspected_path
                _ = roi
                _ = output_stem
                return CleanlinessResult(
                    source_path=source_path,
                    inspected_path=inspected_path or source_path,
                    llm_input_paths=[
                        source_path,
                        inspected_path or source_path,
                    ],
                    score=2,
                    confidence=0.81,
                    summary="YOLO-assisted result.",
                    findings=["Detected cup near the top-left area."],
                    exact_objects=["table"],
                    estimated_objects=["cup"],
                    prompt_profile=prompt_profile,
                    use_yolo=use_yolo,
                    yolo_payload={
                        "image_name": (inspected_path or source_path).name,
                        "detector_name": "YOLOE",
                        "weights": "fake-yoloe.pt",
                        "confidence_threshold": 0.25,
                        "iou_threshold": 0.45,
                        "tile_size": 512,
                        "tile_overlap": 0.25,
                        "prompt_labels": ["used napkin", "food wrapper"],
                        "detections": [
                            {
                                "label": "cup",
                                "confidence": 0.91,
                                "bbox": {"x1": 12.0, "y1": 18.0, "x2": 48.0, "y2": 72.0},
                                "position": "top-left",
                                "source": "tile_0",
                            }
                        ],
                    },
                )

        fake_service = FakeCleanlinessService()
        original_service = main_module.cleanliness_service
        main_module.cleanliness_service = fake_service

        try:
            client = TestClient(main_module.app)
            image_path = Path("data/test_data/StoreAlpha_FrontCam_present_frame.png")
            with image_path.open("rb") as handle:
                response = client.post(
                    "/cleanliness",
                    data={
                        "config_id": "StoreAlpha_FrontCam",
                        "roi_name": "POP",
                        "prompt_profile": "restaurant",
                        "use_yolo": "true",
                    },
                    files={"image_file": ("store.png", handle, "image/png")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(fake_service.use_yolo)
            self.assertIn("YOLO 보조 추론", response.text)
            self.assertIn("YOLO Detection JSON", response.text)
            self.assertIn("fake-yoloe.pt", response.text)
            self.assertIn("YOLOE", response.text)
            self.assertIn("top-left", response.text)
        finally:
            if fake_service.source_path is not None:
                fake_service.source_path.unlink(missing_ok=True)
            if fake_service.inspected_path is not None:
                fake_service.inspected_path.unlink(missing_ok=True)
            main_module.cleanliness_service = original_service

    def test_restaurant_prompt_contains_restaurant_context(self) -> None:
        prompt = build_cleanliness_analysis_prompt(PROMPT_PROFILE_RESTAURANT)

        self.assertIn("restaurant dining area", prompt)
        self.assertIn("tables", prompt)
        self.assertIn("dishes", prompt)
        self.assertIn("food residue", prompt)
        self.assertIn("exact_objects", prompt)
        self.assertIn("estimated_objects", prompt)

    def test_general_prompt_penalizes_wrappers_and_debris_on_surfaces(self) -> None:
        prompt = build_cleanliness_analysis_prompt(PROMPT_PROFILE_GENERAL)

        self.assertIn("do not use 3 when identifiable trash", prompt)
        self.assertIn("wrappers, snack packets, food packaging, or debris", prompt)
        self.assertIn("desk, table, counter, tray, or floor should normally score 2 or lower", prompt)


if __name__ == "__main__":
    unittest.main()
