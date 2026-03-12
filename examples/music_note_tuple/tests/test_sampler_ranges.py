import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
from logic.generate import generate_samples
from model import Transformer


class TestSamplerRanges(unittest.TestCase):
    def test_sampler_produces_valid_ranges(self):
        torch.manual_seed(1)

        vocab_sizes = {
            "track": 5,
            "bar": 9,
            "pos": 17,
            "pitch": 129,
            "dur": 17,
            "vel": 9,
        }

        config = SimpleNamespace(
            d_model=64,
            n_layers=2,
            n_heads=4,
            ff_mult=2,
            dropout=0.0,
            max_seq_len=64,
        )

        model = Transformer(vocab_sizes=vocab_sizes, config=config).eval()

        schedulers = {
            "track": PolynomialAttributeScheduler(exponent=1.0),
            "bar": PolynomialAttributeScheduler(exponent=1.0),
            "pos": PolynomialAttributeScheduler(exponent=1.0),
            "pitch": PolynomialAttributeScheduler(exponent=2.0),
            "dur": PolynomialAttributeScheduler(exponent=1.5),
            "vel": PolynomialAttributeScheduler(exponent=3.0),
        }
        path = MixtureDiscreteNoteTuplePath(schedulers=schedulers)
        source = UniformSourceDistribution(vocab_sizes=vocab_sizes, include_pad=True)

        samples = generate_samples(
            model=model,
            path=path,
            source_distribution=source,
            sample_batch_size=3,
            sequence_length=64,
            sampling_steps=8,
            device=torch.device("cpu"),
            time_epsilon=1e-3,
            temperature=1.0,
            rho_cap=0.95,
            final_full_resample=True,
        )

        for attr in ATTR_NAMES:
            self.assertGreaterEqual(int(samples[attr].min().item()), 0)
            self.assertLess(int(samples[attr].max().item()), vocab_sizes[attr])


if __name__ == "__main__":
    unittest.main()
