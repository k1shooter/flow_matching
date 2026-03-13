import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.note_tuple import ATTR_NAMES
from logic.edit_flow import derive_edit_targets
from logic.flow import (
    get_source_distribution,
    MixtureDiscreteNoteTuplePath,
    PolynomialAttributeScheduler,
)


class TestPadHeavySource(unittest.TestCase):
    def test_insert_targets_are_non_degenerate(self):
        torch.manual_seed(0)
        bsz, seq_len = 8, 64
        vocab_sizes = {
            "track": 9,
            "bar": 9,
            "pos": 17,
            "pitch": 129,
            "dur": 33,
            "vel": 9,
        }

        source = get_source_distribution(
            source_distribution="pad_heavy",
            vocab_sizes=vocab_sizes,
            include_pad=True,
            pitch_pad_prob=0.9,
        )
        schedulers = {
            attr: PolynomialAttributeScheduler(exponent=1.0, scale=1.0)
            for attr in ATTR_NAMES
        }
        path = MixtureDiscreteNoteTuplePath(schedulers=schedulers)

        x_1 = {
            attr: torch.ones((bsz, seq_len), dtype=torch.long) for attr in ATTR_NAMES
        }
        x_0 = source.sample_like(x_1)
        t = torch.zeros(bsz, dtype=torch.float32)
        x_t = path.sample(t=t, x_0=x_0, x_1=x_1).x_t

        targets = derive_edit_targets(x_t=x_t, x_1=x_1)
        insert_mean = float(targets.insert.mean().item())
        self.assertGreater(insert_mean, 0.2)


if __name__ == "__main__":
    unittest.main()
