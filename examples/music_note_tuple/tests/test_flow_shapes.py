import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.note_tuple import ATTR_NAMES
from logic.flow import (
    MixtureDiscreteNoteTuplePath,
    PolynomialAttributeScheduler,
    UniformSourceDistribution,
)


class TestFlowShapes(unittest.TestCase):
    def test_forward_corruption_shapes(self):
        torch.manual_seed(0)
        batch_size, sequence_length = 4, 32
        vocab_sizes = {
            "track": 9,
            "bar": 9,
            "pos": 17,
            "pitch": 129,
            "dur": 33,
            "vel": 9,
        }

        schedulers = {
            attr: PolynomialAttributeScheduler(exponent=1.0, scale=1.0)
            for attr in ATTR_NAMES
        }
        path = MixtureDiscreteNoteTuplePath(schedulers=schedulers)
        source = UniformSourceDistribution(vocab_sizes=vocab_sizes, include_pad=True)

        x_1 = {
            attr: torch.randint(
                low=0,
                high=vocab_sizes[attr],
                size=(batch_size, sequence_length),
            )
            for attr in ATTR_NAMES
        }
        x_0 = source.sample_like(x_1)
        t = torch.rand(batch_size)

        sample = path.sample(t=t, x_0=x_0, x_1=x_1)

        for attr in ATTR_NAMES:
            self.assertEqual(sample.x_t[attr].shape, x_1[attr].shape)
            from_x0_or_x1 = (sample.x_t[attr] == x_0[attr]) | (
                sample.x_t[attr] == x_1[attr]
            )
            self.assertTrue(from_x0_or_x1.all().item())


if __name__ == "__main__":
    unittest.main()
