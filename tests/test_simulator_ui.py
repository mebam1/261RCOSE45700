from __future__ import annotations

import unittest
from dataclasses import dataclass

from simulator.ui import representative_temporal_frame_indices


@dataclass(frozen=True)
class FakeTemporalFrame:
    frame_state: str
    final_state: str


class SimulatorUiTest(unittest.TestCase):
    def test_representative_temporal_frame_indices_include_each_state_transition(self) -> None:
        frames = [
            FakeTemporalFrame(frame_state="DINING", final_state="DINING"),
            FakeTemporalFrame(frame_state="DINING", final_state="DINING"),
            FakeTemporalFrame(frame_state="AFTER_MEAL_CANDIDATE", final_state="UNCERTAIN"),
            FakeTemporalFrame(frame_state="AFTER_MEAL_CANDIDATE", final_state="UNCERTAIN"),
            FakeTemporalFrame(frame_state="CLEANING", final_state="CLEANING"),
            FakeTemporalFrame(frame_state="CLEANING", final_state="CLEANING"),
            FakeTemporalFrame(frame_state="POSSIBLY_EMPTY", final_state="UNCERTAIN"),
        ]

        indices = representative_temporal_frame_indices(frames)

        self.assertEqual(indices, [0, 2, 4, 6])


if __name__ == "__main__":
    unittest.main()
