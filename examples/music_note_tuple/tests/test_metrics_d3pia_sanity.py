import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.note_tuple import NoteTupleMetadata
from eval.d3pia_pop909 import compute_grooving_similarity, compute_ook_ratio


class TestD3PIAMetricsSanity(unittest.TestCase):
    def setUp(self):
        self.metadata = NoteTupleMetadata(
            beats_per_bar=4,
            steps_per_beat=4,
            segment_bars=2,
            dur_max_steps=16,
            vel_bins=8,
            num_tracks=2,
            max_notes_per_segment=16,
            default_tempo=120.0,
        )

    def test_ook(self):
        # C major notes + one out-of-key C#
        sample = {
            "track": torch.tensor([1, 1, 1, 1] + [0] * 12),
            "bar": torch.tensor([1, 1, 1, 1] + [0] * 12),
            "pos": torch.tensor([1, 5, 9, 13] + [0] * 12),
            "pitch": torch.tensor([61, 63, 65, 64] + [0] * 12),
            "dur": torch.tensor([2, 2, 2, 2] + [0] * 12),
            "vel": torch.tensor([4, 4, 4, 4] + [0] * 12),
        }
        # pitches (MIDI): 60(C),62(D),64(E),63(D#) => exactly one out-of-key.

        ratio = compute_ook_ratio(sample, metadata=self.metadata, key_name="C:maj")
        self.assertAlmostEqual(ratio, 0.25, places=4)

    def test_gs(self):
        # Two bars with identical onset pattern => GS = 1.0
        sample = {
            "track": torch.tensor([1, 1, 1, 1] + [0] * 12),
            "bar": torch.tensor([1, 1, 2, 2] + [0] * 12),
            "pos": torch.tensor([1, 9, 1, 9] + [0] * 12),
            "pitch": torch.tensor([61, 65, 61, 65] + [0] * 12),
            "dur": torch.tensor([2, 2, 2, 2] + [0] * 12),
            "vel": torch.tensor([4, 4, 4, 4] + [0] * 12),
        }

        gs = compute_grooving_similarity(sample, metadata=self.metadata)
        self.assertAlmostEqual(gs, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
