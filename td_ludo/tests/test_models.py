"""Smoke tests for moved model classes.

Tests the V6.2 transformer model that lives in td_ludo.models.v6_2.
V5 / AlphaLudoV5 tests are deferred until B2 unblocks (model.py is
in the sweep skip list — sweep is using src.model right now).

Run from inside td_ludo/:
    cd td_ludo
    python3 -m unittest tests.test_models -v
"""
import unittest

import torch


class TestV62Model(unittest.TestCase):
    def test_v6_2_constructs(self):
        from td_ludo.models.v6_2 import AlphaLudoV62
        m = AlphaLudoV62(context_length=4, num_res_blocks=10, in_channels=24)
        # NOTE: this is the pre-ReZero local AlphaLudoV62 — it has a
        # scalar `transformer_alpha` gate, NOT the `trans_out_proj`
        # Linear that exists only on GCP / in gcp_snapshots/. The
        # ReZero fix has not been merged back to local main yet.
        self.assertTrue(hasattr(m, "transformer_alpha"),
                        "local AlphaLudoV62 should have transformer_alpha")
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.assertGreater(n_params, 1_000_000, "expected >1M trainable params")
        self.assertLess(n_params, 10_000_000, "expected <10M trainable params")

    def test_v6_2_alpha_gate_initially_zero(self):
        """The transformer alpha gate must start at exactly 0.0 so
        tanh(alpha) = 0 → transformer contributes nothing on the
        first forward pass."""
        from td_ludo.models.v6_2 import AlphaLudoV62
        m = AlphaLudoV62(context_length=4, num_res_blocks=10, in_channels=24)
        self.assertEqual(m.transformer_alpha.abs().max().item(), 0.0)
        self.assertEqual(torch.tanh(m.transformer_alpha).abs().max().item(),
                         0.0)

    def test_v6_2_forward_random_input(self):
        from td_ludo.models.v6_2 import AlphaLudoV62
        m = AlphaLudoV62(context_length=4, num_res_blocks=10, in_channels=24)
        m.eval()
        # Build a tiny input batch in V6.2 sequence shape:
        # (B, K, in_channels, H, W) where K = context_length, H=W=15
        B, K, C, H, W = 2, 4, 24, 15, 15
        grids = torch.randn(B, K, C, H, W)
        prev_actions = torch.zeros(B, K, dtype=torch.long)
        seq_mask = torch.zeros(B, K, dtype=torch.bool)
        legal_mask = torch.ones(B, 4)
        with torch.no_grad():
            policy, value = m(grids, prev_actions, seq_mask, legal_mask)
        self.assertEqual(policy.shape, (B, 4))
        self.assertEqual(value.shape, (B, 1))
        # Policy must be a valid probability distribution
        self.assertTrue(torch.allclose(policy.sum(dim=1),
                                       torch.ones(B), atol=1e-5))
        # Output must be finite
        self.assertTrue(torch.isfinite(policy).all())
        self.assertTrue(torch.isfinite(value).all())

    def test_v6_2_shim_resolves_to_same_class(self):
        """The src/model_v6_2.py shim must re-export the same
        AlphaLudoV62 class as the canonical td_ludo.models.v6_2 path."""
        import warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        from src.model_v6_2 import AlphaLudoV62 as ShimClass
        from td_ludo.models.v6_2 import AlphaLudoV62 as CanonicalClass
        self.assertIs(ShimClass, CanonicalClass)


if __name__ == "__main__":
    unittest.main()
