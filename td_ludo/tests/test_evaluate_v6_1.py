"""Smoke test for evaluate_v6_1.evaluate_model.

This is a real end-to-end test: it loads V6.1 model_latest.pt from
the local checkpoints dir and runs a tiny 4-game eval against the
Expert bot. We don't assert a specific win-rate (eval is noisy in
Ludo) — we just verify the call returns a valid result dict.

Skipped automatically if the local V6.1 checkpoint isn't present.

Run from inside td_ludo/:
    cd td_ludo
    python3 -m unittest tests.test_evaluate_v6_1 -v
"""
import os
import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

CKPT_PATH = "checkpoints/ac_v6_1_strategic/model_latest.pt"


class TestEvaluateV61(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(CKPT_PATH),
                         f"V6.1 checkpoint not found at {CKPT_PATH}")
    def test_evaluate_4_games_smoke(self):
        import torch
        # Use the legacy import path — model.py is sweep-blocked,
        # not yet moved to td_ludo.models.v5.
        from src.model import AlphaLudoV5
        from evaluate_v6_1 import evaluate_model

        device = "cpu"
        model = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        model.to(device).eval()

        result = evaluate_model(
            model, device,
            num_games=4,
            verbose=False,
            bot_types=["Expert"],
        )
        # Minimal contract checks: result is a dict with the right keys
        # and the win_rate is in [0, 1].
        self.assertIsInstance(result, dict)
        self.assertIn("win_rate", result)
        self.assertGreaterEqual(result["win_rate"], 0.0)
        self.assertLessEqual(result["win_rate"], 1.0)
        # If the function reports a per-bot breakdown, we're happy.
        # Otherwise we just confirm the top-level shape.


if __name__ == "__main__":
    unittest.main()
