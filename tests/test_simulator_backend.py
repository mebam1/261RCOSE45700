from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from simulator import backend
from simulator.models import TableBox


class DummyPersonMaskService:
    def __init__(self) -> None:
        self.call_count = 0

    def apply_black_mask(self, image):
        self.call_count += 1
        return image, np.zeros(image.shape[:2], dtype=np.uint8), []


class SimulatorBackendTest(unittest.TestCase):
    def test_simulator_observation_budget_scales_with_video_duration(self) -> None:
        budget = backend.simulator_observation_budget(
            8,
            duration_seconds=20.0 * 60.0,
            base_interval_seconds=10.0,
        )

        self.assertGreaterEqual(budget, 241)

    def test_build_dynamic_sampled_frames_preserves_sampler_metadata_and_split_crops(self) -> None:
        table = TableBox("T01", 20, 30, 80, 90)
        image = np.zeros((180, 240, 3), dtype=np.uint8)
        original_builder = backend.build_shared_dynamic_sampled_frames

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                frame_path = backend.save_image(Path(tmpdir) / "frame.png", image)
                backend.build_shared_dynamic_sampled_frames = lambda **kwargs: (
                    {
                        table.table_id: [
                            backend.SampledFrame(
                                index=3,
                                timestamp_seconds=12.0,
                                frame_path=frame_path,
                                frame_type="cleaning_candidate",
                                sampling_state="cleaning_candidate",
                                priority=0.91,
                                reason_codes=["person_reentered", "high_table_change"],
                                selection_reasons=["post_exit_change"],
                                person_present=True,
                                person_count=1,
                                raw_person_count=2,
                                best_relevant_person_score=0.87,
                                person_relevance_reason="person_near_table",
                                episode_id="episode_01",
                                object_crop_box=table,
                                action_crop_box=TableBox("T01", 5, 10, 145, 130),
                            )
                        ]
                    },
                    {
                        table.table_id: {
                            "duration_seconds": 1200.0,
                            "observed_frame_count": 1,
                            "selected_frame_count": 1,
                            "debug_trace": [{"frame_type": "cleaning_candidate", "selected_for_review": True}],
                        }
                    },
                    1,
                )
                sampled_frames, metadata = backend.build_dynamic_sampled_frames(
                    video_path=Path("demo.mp4"),
                    table=table,
                    output_dir=Path(tmpdir),
                    base_interval_seconds=10.0,
                    cancel_event=backend.Event(),
                    person_mask_service=DummyPersonMaskService(),
                )
                self.assertEqual(len(sampled_frames), 1)
                sampled = sampled_frames[0]
                self.assertTrue(sampled.frame_path.exists())
        finally:
            backend.build_shared_dynamic_sampled_frames = original_builder

        sampled = sampled_frames[0]
        self.assertEqual(sampled.object_crop_box, table)
        self.assertEqual(sampled.action_crop_box, TableBox("T01", 5, 10, 145, 130))
        self.assertEqual(sampled.selection_reasons, ["post_exit_change"])
        self.assertEqual(sampled.person_relevance_reason, "person_near_table")
        self.assertEqual(metadata["duration_seconds"], 1200.0)
        self.assertEqual(metadata["observed_frame_count"], 1)
        self.assertEqual(metadata["selected_frame_count"], 1)
        self.assertEqual(metadata["debug_trace"][0]["frame_type"], "cleaning_candidate")
        self.assertTrue(metadata["debug_trace"][0]["selected_for_review"])

    def test_interaction_crop_box_falls_back_to_table_without_bounds(self) -> None:
        table = TableBox("T02", 10, 15, 40, 55)
        resolved = backend.interaction_crop_box(table, {"features": {}}, (100, 120, 3))
        self.assertEqual(resolved, table)

    def test_shared_dynamic_sampler_decodes_common_timestamps_once_for_multiple_tables(self) -> None:
        tables = [
            TableBox("T01", 10, 10, 50, 50),
            TableBox("T02", 60, 10, 100, 50),
        ]
        call_offsets: list[float] = []
        person_mask_service = DummyPersonMaskService()
        frames_by_offset = {
            0.0: np.zeros((120, 160, 3), dtype=np.uint8),
            10.0: np.zeros((120, 160, 3), dtype=np.uint8),
            20.0: np.zeros((120, 160, 3), dtype=np.uint8),
        }

        def frame_reader(offset_seconds: float):
            rounded = round(offset_seconds, 1)
            call_offsets.append(rounded)
            image = frames_by_offset.get(rounded)
            if image is None:
                return None
            return {"image": image.copy(), "offset_seconds": rounded, "fps": 1.0, "frame_index": int(round(rounded))}

        with tempfile.TemporaryDirectory() as tmpdir:
            sampled_frames_by_table, sampling_metadata_by_table, decoded_frame_count = (
                backend.build_shared_dynamic_sampled_frames_from_reader(
                    tables=tables,
                    output_dir=Path(tmpdir),
                    base_interval_seconds=10.0,
                    cancel_event=backend.Event(),
                    person_mask_service=person_mask_service,
                    duration_seconds=20.0,
                    frame_reader=frame_reader,
                )
            )

        self.assertEqual(decoded_frame_count, 3)
        self.assertEqual(person_mask_service.call_count, 3)
        self.assertEqual(call_offsets, [0.0, 10.0, 20.0])
        self.assertEqual(sorted(sampled_frames_by_table), ["T01", "T02"])
        self.assertEqual(sampling_metadata_by_table["T01"]["observed_frame_count"], 3)
        self.assertEqual(sampling_metadata_by_table["T02"]["observed_frame_count"], 3)


if __name__ == "__main__":
    unittest.main()
