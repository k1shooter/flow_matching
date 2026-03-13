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
    MultiAttributeCrossEntropyLoss,
    PolynomialAttributeScheduler,
)


class TestFlowRhoAndLoss(unittest.TestCase):
    def test_update_prob_matches_exact_kappa_formula(self):
        schedulers = {
            attr: PolynomialAttributeScheduler(exponent=2.0, scale=1.0)
            for attr in ATTR_NAMES
        }
        path = MixtureDiscreteNoteTuplePath(schedulers=schedulers)

        t = torch.tensor([0.2, 0.7, 0.98], dtype=torch.float32)
        h = 0.1

        rho = path.update_prob(attr="pitch", t=t, h=h)
        k_t = schedulers["pitch"].kappa(t=t)
        k_t_next = schedulers["pitch"].kappa(t=torch.clamp(t + h, max=1.0))
        expected = (k_t_next - k_t) / torch.clamp(1.0 - k_t, min=1e-8)
        expected = torch.clamp(expected, min=0.0, max=1.0)

        self.assertTrue(torch.allclose(rho, expected, atol=1e-6))

    def test_non_pitch_loss_ignores_inactive_slots(self):
        vocab_sizes = {
            "track": 4,
            "bar": 4,
            "pos": 4,
            "pitch": 4,
            "dur": 4,
            "vel": 4,
        }
        loss_fn = MultiAttributeCrossEntropyLoss(vocab_sizes=vocab_sizes)

        target = {
            "pitch": torch.tensor([[0, 2]], dtype=torch.long),
            "track": torch.tensor([[0, 1]], dtype=torch.long),
            "bar": torch.tensor([[0, 1]], dtype=torch.long),
            "pos": torch.tensor([[0, 1]], dtype=torch.long),
            "dur": torch.tensor([[0, 1]], dtype=torch.long),
            "vel": torch.tensor([[0, 1]], dtype=torch.long),
        }

        logits_a = {
            attr: torch.zeros((1, 2, vocab_sizes[attr]), dtype=torch.float32)
            for attr in ATTR_NAMES
        }
        logits_b = {attr: tensor.clone() for attr, tensor in logits_a.items()}

        # Change only inactive-slot logits on non-pitch attributes.
        for attr in ATTR_NAMES:
            if attr == "pitch":
                continue
            logits_b[attr][0, 0, :] = torch.tensor([10.0, -10.0, 5.0, -5.0])

        loss_a = loss_fn(logits=logits_a, target=target)
        loss_b = loss_fn(logits=logits_b, target=target)
        self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
