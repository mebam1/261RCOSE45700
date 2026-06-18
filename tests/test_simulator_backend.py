from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from simulator import backend
from simulator.models import TableBox


class DummyPersonMaskService:
    pass


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
        sample = {
            "sample_index": 3,
            "offset_seconds": 12.0,
            "image": image,
            "frame_type": "cleaning_candidate",
            "sampling_state": "cleaning_candidate",
            "priority": 0.91,
            "reason_codes": ["person_reentered", "high_table_change"],
            "selection_reasons": ["post_exit_change"],
            "episode_id": "episode_01",
            "features": {
                "person_present": True,
                "person_count": 1,
                "raw_person_count": 2,
                "best_relevant_person_score": 0.87,
                "person_relevance_reason": "person_near_table",
                "interaction_halo_bounds": {"x": 5, "y": 10, "width": 140, "height": 120},
                "change_score": 0.22,
            },
        }

        original_sampler = backend.sample_dynamic_video_workflow_frames
        original_summary = backend.summarize_dynamic_video_samples
        original_probe_duration = backend.probe_video_duration_seconds
        backend.sample_dynamic_video_workflow_frames = lambda **kwargs: [sample]
        backend.summarize_dynamic_video_samples = lambda samples, target_count: {
            "selected_samples": [dict(sample)],
            "debug_trace": [dict(sample, selected_for_review=True)],
        }
        backend.probe_video_duration_seconds = lambda video_path: 1200.0

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
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
            backend.sample_dynamic_video_workflow_frames = original_sampler
            backend.summarize_dynamic_video_samples = original_summary
            backend.probe_video_duration_seconds = original_probe_duration

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


if __name__ == "__main__":
    unittest.main()
