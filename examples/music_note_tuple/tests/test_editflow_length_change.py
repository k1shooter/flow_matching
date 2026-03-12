import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.note_tuple import ATTR_NAMES
from logic.edit_flow import generate_samples_edit_flow


class DummyEditModel(nn.Module):
    def __init__(self, vocab_sizes):
        super().__init__()
        self.vocab_sizes = vocab_sizes

    def forward(self, x_t, time, return_edit_logits=False):
        bsz, seq_len = x_t["pitch"].shape
        logits = {}
        for attr in ATTR_NAMES:
            v = self.vocab_sizes[attr]
            attr_logits = torch.zeros(
                (bsz, seq_len, v),
                dtype=torch.float32,
                device=time.device,
            )
            # Bias toward non-PAD token for generation.
            if v > 1:
                attr_logits[..., 1] = 5.0
            logits[attr] = attr_logits

        if not return_edit_logits:
            return logits

        edit_logits = {
            "insert": torch.full((bsz, seq_len), 6.0, device=time.device),
            "delete": torch.full((bsz, seq_len), -6.0, device=time.device),
            "substitute": torch.full((bsz, seq_len), -6.0, device=time.device),
        }
        return logits, edit_logits


class TestEditFlowLengthChange(unittest.TestCase):
    def test_sampler_changes_length(self):
        vocab_sizes = {
            "track": 5,
            "bar": 9,
            "pos": 17,
            "pitch": 129,
            "dur": 17,
            "vel": 9,
        }

        model = DummyEditModel(vocab_sizes=vocab_sizes)
        out = generate_samples_edit_flow(
            model=model,
            vocab_sizes=vocab_sizes,
            sample_batch_size=4,
            max_length=32,
            sampling_steps=4,
            device=torch.device("cpu"),
            sampler="tau_leaping",
            tau_leaping_step_size=0.25,
            max_events_per_step=32,
            init_mode="empty",
        )

        lengths = out["length"]
        self.assertTrue(torch.any(lengths > 0).item())
        self.assertTrue(torch.all(lengths <= 32).item())


if __name__ == "__main__":
    unittest.main()
