import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.note_tuple import NoteTupleMetadata
from eval.pianoroll import note_tuple_to_pianoroll, pianoroll_to_note_tuple


class TestPianorollRoundtrip(unittest.TestCase):
    def test_note_tuple_roundtrip(self):
        metadata = NoteTupleMetadata(
            beats_per_bar=4,
            steps_per_beat=4,
            segment_bars=4,
            dur_max_steps=16,
            vel_bins=8,
            num_tracks=4,
            max_notes_per_segment=16,
            default_tempo=120.0,
        )

        sample = {
            "track": torch.tensor([1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "bar": torch.tensor([1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "pos": torch.tensor([1, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "pitch": torch.tensor([61, 65, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "dur": torch.tensor([4, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "vel": torch.tensor([3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        }

        roll = note_tuple_to_pianoroll(sample, metadata)
        recovered = pianoroll_to_note_tuple(
            pianoroll=roll,
            metadata=metadata,
            max_notes=16,
            default_track=1,
        )

        original_active = int((sample["pitch"] > 0).sum().item())
        recovered_active = int((recovered["pitch"] > 0).sum().item())

        self.assertGreaterEqual(recovered_active, 1)
        self.assertLessEqual(abs(original_active - recovered_active), 2)


if __name__ == "__main__":
    unittest.main()
