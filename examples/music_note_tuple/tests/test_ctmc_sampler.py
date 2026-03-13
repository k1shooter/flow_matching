import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samplers.ctmc_sampler import (
    ctmc_step_attribute,
    rates_from_velocity_parameterization,
)


class TestCTMCSampler(unittest.TestCase):
    def test_probabilities_are_valid_and_tokens_in_range(self):
        torch.manual_seed(0)
        bsz, seq_len, vocab = 2, 4, 5
        current = torch.randint(0, vocab, (bsz, seq_len), dtype=torch.long)
        logits = torch.randn(bsz, seq_len, vocab)
        log_lambda = torch.randn(bsz, seq_len)

        rates, lambda_values, probs = rates_from_velocity_parameterization(
            token_logits=logits,
            log_lambda=log_lambda,
            current_tokens=current,
            disallow_zero_token=False,
            eps=1e-12,
        )

        self.assertTrue(torch.isfinite(rates).all().item())
        self.assertTrue(torch.isfinite(lambda_values).all().item())
        self.assertTrue(torch.all(lambda_values >= 0).item())
        self.assertTrue(
            torch.allclose(probs.sum(dim=-1), torch.ones_like(lambda_values), atol=1e-4)
        )

        h = 0.1
        p_stay = torch.exp(-h * lambda_values)
        total_prob = p_stay + (1.0 - p_stay) * probs.sum(dim=-1)
        self.assertTrue(
            torch.allclose(total_prob, torch.ones_like(total_prob), atol=1e-5)
        )

        next_tokens, _, _ = ctmc_step_attribute(
            current_tokens=current,
            token_logits=logits,
            log_lambda=log_lambda,
            h=h,
            disallow_zero_token=False,
            active_mask=None,
            eps=1e-12,
        )
        self.assertGreaterEqual(int(next_tokens.min().item()), 0)
        self.assertLess(int(next_tokens.max().item()), vocab)

    def test_rates_support_backward_without_inplace_errors(self):
        torch.manual_seed(1)
        bsz, seq_len, vocab = 2, 3, 7
        current = torch.randint(0, vocab, (bsz, seq_len), dtype=torch.long)
        logits = torch.randn(bsz, seq_len, vocab, requires_grad=True)
        log_lambda = torch.randn(bsz, seq_len, requires_grad=True)

        rates, _, _ = rates_from_velocity_parameterization(
            token_logits=logits,
            log_lambda=log_lambda,
            current_tokens=current,
            disallow_zero_token=False,
            eps=1e-12,
        )
        loss = rates.sum()
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(log_lambda.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue(torch.isfinite(log_lambda.grad).all().item())


if __name__ == "__main__":
    unittest.main()
