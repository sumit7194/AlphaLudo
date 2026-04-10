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


class TestV5Model(unittest.TestCase):
    """Tests for AlphaLudoV5 (the V6/V6.1 backbone).

    Uses the legacy `from src.model import AlphaLudoV5` import because
    src/model.py is in the sweep skip list — not yet moved to
    td_ludo.models.v5. When B2 unblocks (post-sweep) this test will
    keep working through the shim without modification.
    """

    def test_v5_imports(self):
        from src.model import AlphaLudoV5
        m = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
        self.assertTrue(hasattr(m, "policy_fc1"))
        self.assertTrue(hasattr(m, "value_fc1"))

    def test_v5_param_count(self):
        from src.model import AlphaLudoV5
        m = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        # V6.1 spec: ~3M params (conv 24->128, 10 ResBlocks @ 128ch, heads)
        self.assertGreater(n, 2_500_000)
        self.assertLess(n, 4_000_000)

    def test_v5_forward_pass(self):
        from src.model import AlphaLudoV5
        m = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
        m.eval()
        # V5 takes a single-frame (B, in_channels, 15, 15) input,
        # not the K-step sequence that V6.2 takes.
        x = torch.randn(2, 24, 15, 15)
        legal = torch.ones(2, 4)
        with torch.no_grad():
            policy, value = m(x, legal)
        self.assertEqual(policy.shape, (2, 4))
        self.assertEqual(value.shape, (2, 1))
        self.assertTrue(torch.allclose(policy.sum(dim=1),
                                       torch.ones(2), atol=1e-5))
        self.assertTrue(torch.isfinite(policy).all())
        self.assertTrue(torch.isfinite(value).all())

    def test_v5_load_v6_1_checkpoint(self):
        """If the V6.1 checkpoint is present locally, we should be
        able to load it cleanly into AlphaLudoV5."""
        import os
        ckpt_path = "checkpoints/ac_v6_1_strategic/model_latest.pt"
        if not os.path.exists(ckpt_path):
            self.skipTest(f"checkpoint {ckpt_path} not present locally")
        from src.model import AlphaLudoV5
        m = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        # Strict load — any missing or unexpected key fails the test.
        result = m.load_state_dict(sd)
        self.assertEqual(len(result.missing_keys), 0,
                         f"missing keys: {result.missing_keys}")
        self.assertEqual(len(result.unexpected_keys), 0,
                         f"unexpected keys: {result.unexpected_keys}")


if __name__ == "__main__":
    unittest.main()
