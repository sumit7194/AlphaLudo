"""Smoke test for inference-time MCTS via mcts_eval_sweep.run_matchup.

Runs a tiny 2-game MCTS(5)-vs-Expert matchup to verify the C++ MCTS
engine + Python loop integration still works. We don't assert on
the win rate (way too noisy at 2 games) — just on the result-dict
shape.

Skipped automatically if the V6.1 checkpoint isn't present locally.

NOTE: imports mcts_eval_sweep via the legacy path (it's in the
sweep skip list — sweep is currently running on GCP using this
exact file, on the GCP filesystem; local refactor is independent).

Run from inside td_ludo/:
    cd td_ludo
    python3 -m unittest tests.test_mcts_smoke -v
"""
import os
import unittest
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

CKPT_PATH = "checkpoints/ac_v6_1_strategic/model_latest.pt"


class TestMCTSSmoke(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(CKPT_PATH),
                         f"V6.1 checkpoint not found at {CKPT_PATH}")
    def test_run_matchup_2_games_smoke(self):
        # Legacy import path — mcts_eval_sweep.py is in the sweep
        # skip list and lives at td_ludo/ root. The wrapper in
        # td_ludo/scripts/mcts_eval_sweep.py was intentionally NOT
        # created in B8 for the same reason.
        from mcts_eval_sweep import run_matchup

        result, paused = run_matchup(
            label="C5_smoke",
            model_path=CKPT_PATH,
            num_sims=5,
            opponent_spec={"type": "bot", "bot": "Expert"},
            num_games=2,
            device="cpu",
            seed=12345,
            log_path="/tmp/td_ludo_c5_smoke.log",
        )
        # Contract checks
        self.assertIsInstance(result, dict)
        self.assertEqual(result["num_games"], 2)
        self.assertIn("wins", result)
        self.assertIn("losses", result)
        self.assertIn("win_rate", result)
        self.assertGreaterEqual(result["win_rate"], 0.0)
        self.assertLessEqual(result["win_rate"], 1.0)
        # Wins + losses + draws must equal num_games
        self.assertEqual(
            result["wins"] + result["losses"] + result.get("draws", 0),
            result["num_games"],
        )
        self.assertFalse(paused, "smoke test should complete naturally, not pause")


if __name__ == "__main__":
    unittest.main()
