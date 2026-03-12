import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.note_tuple import NoteTupleMetadata
from eval.common import SegmentRecord
from eval.wholesonggen_iclr24 import compute_ils


class TestILSSanity(unittest.TestCase):
    def test_repeated_phrase_has_high_ils(self):
        metadata = NoteTupleMetadata(
            beats_per_bar=4,
            steps_per_beat=4,
            segment_bars=16,
            dur_max_steps=16,
            vel_bins=8,
            num_tracks=2,
            max_notes_per_segment=128,
            default_tempo=120.0,
        )

        # Repeated rhythmic/pitch motif across all bars.
        n = 128
        sample = {
            "track": torch.zeros(n, dtype=torch.long),
            "bar": torch.zeros(n, dtype=torch.long),
            "pos": torch.zeros(n, dtype=torch.long),
            "pitch": torch.zeros(n, dtype=torch.long),
            "dur": torch.zeros(n, dtype=torch.long),
            "vel": torch.zeros(n, dtype=torch.long),
        }

        idx = 0
        for bar in range(1, 17):
            for pos, pitch in [(1, 61), (5, 65), (9, 68), (13, 65)]:
                sample["track"][idx] = 1
                sample["bar"][idx] = bar
                sample["pos"][idx] = pos
                sample["pitch"][idx] = pitch
                sample["dur"][idx] = 2
                sample["vel"][idx] = 4
                idx += 1

        segments = [
            SegmentRecord(song_id="song0", segment_id="song0_seg0", note_tuple=sample)
        ]
        summary, _ = compute_ils(
            song_segments=segments,
            metadata=metadata,
            phrase_bars=2,
            latent_dim=16,
            latent_encoder_ckpt_dir=None,
        )

        self.assertGreater(summary["p"]["mean"], 0.7)
        self.assertGreater(summary["r"]["mean"], 0.7)


if __name__ == "__main__":
    unittest.main()
