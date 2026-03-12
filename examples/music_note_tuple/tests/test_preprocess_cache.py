import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import data as data_module
from data.note_tuple import NoteTupleMetadata


def _dummy_sample(num_notes: int = 8):
    sample = {
        "track": torch.zeros(num_notes, dtype=torch.long),
        "bar": torch.zeros(num_notes, dtype=torch.long),
        "pos": torch.zeros(num_notes, dtype=torch.long),
        "pitch": torch.zeros(num_notes, dtype=torch.long),
        "dur": torch.zeros(num_notes, dtype=torch.long),
        "vel": torch.zeros(num_notes, dtype=torch.long),
    }
    sample["track"][0] = 1
    sample["bar"][0] = 1
    sample["pos"][0] = 1
    sample["pitch"][0] = 64
    sample["dur"][0] = 4
    sample["vel"][0] = 5
    sample["attention_mask"] = (sample["pitch"] != 0).long()
    return sample


class TestPreprocessCache(unittest.TestCase):
    def test_preprocess_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            midi_file = root / "toy.mid"
            midi_file.write_bytes(b"dummy")

            cache_dir = root / "cache"
            cfg = SimpleNamespace(
                data=SimpleNamespace(
                    preprocess_cache_dir=str(cache_dir),
                    rebuild_preprocess_cache=False,
                    preprocess_cache_version=1,
                    preprocess_cache_lock_timeout_seconds=5,
                )
            )

            metadata = NoteTupleMetadata(
                beats_per_bar=4,
                steps_per_beat=4,
                segment_bars=8,
                dur_max_steps=32,
                vel_bins=8,
                num_tracks=8,
                max_notes_per_segment=8,
                default_tempo=120.0,
            )

            sample = _dummy_sample(num_notes=metadata.max_notes_per_segment)
            with mock.patch.object(
                data_module,
                "midi_file_to_segment_tuples",
                return_value=[sample],
            ) as mocked_preprocess:
                first = data_module._load_or_build_preprocessed_samples(
                    midi_files=[midi_file],
                    metadata=metadata,
                    segment_stride_bars=8,
                    split_name="train",
                    config=cfg,
                )
                self.assertEqual(mocked_preprocess.call_count, 1)

            cache_files = list(cache_dir.glob("train_*.pt"))
            self.assertEqual(len(cache_files), 1)
            self.assertEqual(len(first), 1)

            with mock.patch.object(
                data_module,
                "midi_file_to_segment_tuples",
                side_effect=RuntimeError("cache should be reused"),
            ) as mocked_preprocess:
                second = data_module._load_or_build_preprocessed_samples(
                    midi_files=[midi_file],
                    metadata=metadata,
                    segment_stride_bars=8,
                    split_name="train",
                    config=cfg,
                )
                self.assertEqual(mocked_preprocess.call_count, 0)

            self.assertEqual(len(second), 1)
            self.assertTrue(torch.equal(first[0]["pitch"], second[0]["pitch"]))


if __name__ == "__main__":
    unittest.main()
