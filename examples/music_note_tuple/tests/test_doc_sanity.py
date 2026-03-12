import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.note_tuple import NoteTupleMetadata
from eval.common import SegmentRecord
from eval.wholesonggen_iclr24 import build_doc_bank, compute_doc


class TestDoCSanity(unittest.TestCase):
    def test_identical_query_has_high_doc(self):
        metadata = NoteTupleMetadata(
            beats_per_bar=4,
            steps_per_beat=4,
            segment_bars=4,
            dur_max_steps=16,
            vel_bins=8,
            num_tracks=2,
            max_notes_per_segment=64,
            default_tempo=120.0,
        )

        sample = {
            "track": torch.tensor([1, 1, 1, 1] + [0] * 60),
            "bar": torch.tensor([1, 2, 3, 4] + [0] * 60),
            "pos": torch.tensor([1, 1, 1, 1] + [0] * 60),
            "pitch": torch.tensor([61, 63, 65, 68] + [0] * 60),
            "dur": torch.tensor([4, 4, 4, 4] + [0] * 60),
            "vel": torch.tensor([4, 4, 4, 4] + [0] * 60),
        }

        train = [SegmentRecord(song_id="s", segment_id="s_0", note_tuple=sample)]
        query = [SegmentRecord(song_id="q", segment_id="q_0", note_tuple=sample)]

        bank = build_doc_bank(
            train_segments=train,
            metadata=metadata,
            measure_bars=2,
            remove_all_rest=True,
        )
        doc = compute_doc(
            query_segments=query,
            metadata=metadata,
            measure_bars=2,
            bank=bank,
        )

        self.assertGreater(doc["simrb"]["mean"], 0.8)


if __name__ == "__main__":
    unittest.main()
