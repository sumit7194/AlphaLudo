"""Unit tests for opponent_registry.

Tests the registry's lazy-loading + dispatch behaviour without requiring
real historical checkpoints. We monkeypatch `_ckpt_path` to point at a
tmp file containing a freshly-randomly-initialised model — same
architecture, just untrained — so the load path executes end-to-end.

Run from td_ludo/ root:
    /path/to/python -m td_ludo.game.players.test_opponent_registry
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch

import td_ludo_cpp as cpp

from td_ludo.game.players import opponent_registry as opp_reg


def _mid_game_state(seed: int = 0):
    """Roll 20 random moves into the game so legal_moves is non-trivial."""
    rng = np.random.RandomState(seed)
    g = cpp.create_initial_state()
    for _ in range(20):
        if cpp.get_winner(g) >= 0:
            break
        if g.current_dice_roll == 0:
            g.current_dice_roll = int(rng.randint(1, 7))
        moves = cpp.get_legal_moves(g)
        if not moves:
            g.current_player = (g.current_player + 1) % 4
            g.current_dice_roll = 0
            continue
        g = cpp.apply_move(g, int(rng.choice(moves)))
    if g.current_dice_roll == 0:
        g.current_dice_roll = int(rng.randint(1, 7))
    return g


def _save_random_init_ckpt(spec, path):
    """Build a model from spec, randomise it, save the state_dict to path."""
    model = spec.arch_class(**spec.arch_kwargs)
    torch.save({"model_state_dict": model.state_dict()}, path)


class TestOpponentRegistry(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.tmpdir = tempfile.mkdtemp(prefix="opp_reg_test_")
        # Stub all four tags' checkpoints with random-init weights.
        self.specs = opp_reg._build_specs()
        self.fake_ckpts = {}
        for tag, spec in self.specs.items():
            ckpt_path = os.path.join(self.tmpdir, f"{tag}.pt")
            _save_random_init_ckpt(spec, ckpt_path)
            self.fake_ckpts[tag] = ckpt_path
        # Override the checkpoint path resolver via env var.
        import json
        os.environ["HISTORICAL_OPPONENT_CKPTS"] = json.dumps(self.fake_ckpts)

    def tearDown(self):
        os.environ.pop("HISTORICAL_OPPONENT_CKPTS", None)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_available_tags(self):
        reg = opp_reg.OpponentRegistry(device=self.device)
        tags = reg.available_tags()
        self.assertIn("Hist_V6_3", tags)
        self.assertIn("Hist_V10", tags)
        self.assertIn("Hist_V11", tags)
        self.assertIn("Hist_V12", tags)

    def test_lazy_load_each_arch(self):
        """Every tag should load and produce a frozen, eval-mode model."""
        reg = opp_reg.OpponentRegistry(device=self.device)
        for tag in reg.available_tags():
            model = reg.get_model(tag)
            self.assertFalse(model.training, f"{tag} should be in eval mode")
            for p in model.parameters():
                self.assertFalse(p.requires_grad, f"{tag} params not frozen")

    def test_encoder_dispatch_correct_channels(self):
        """Each tag's encoder should produce its declared channel count."""
        reg = opp_reg.OpponentRegistry(device=self.device)
        g = _mid_game_state(seed=1)
        for tag in reg.available_tags():
            spec = self.specs[tag]
            enc = reg.encode(tag, g, consecutive_sixes=0)
            self.assertEqual(
                enc.shape, (spec.in_channels, 15, 15),
                f"{tag} produced wrong shape: got {enc.shape}",
            )

    def test_select_action_single_legal(self):
        """select_action_single returns a legal move for each tag."""
        reg = opp_reg.OpponentRegistry(device=self.device)
        for seed in range(5):
            g = _mid_game_state(seed=seed)
            legal = list(cpp.get_legal_moves(g))
            if not legal:
                continue
            for tag in reg.available_tags():
                a = reg.select_action_single(tag, g, consecutive_sixes=0)
                self.assertIn(
                    a, legal,
                    f"{tag} returned illegal action {a} (legal={legal})",
                )

    def test_select_actions_batched_correctness(self):
        """Batched call returns same actions as repeated single calls
        when the model is frozen (deterministic argmax)."""
        reg = opp_reg.OpponentRegistry(device=self.device)
        items = []
        for seed in range(8):
            g = _mid_game_state(seed=seed)
            tag = ["Hist_V6_3", "Hist_V10", "Hist_V11", "Hist_V12"][seed % 4]
            items.append((tag, g, 0))

        batched = reg.select_actions_batched(items)
        single = [
            reg.select_action_single(tag, g, csix)
            for (tag, g, csix) in items
        ]
        self.assertEqual(batched, single)

    def test_unknown_tag_raises(self):
        reg = opp_reg.OpponentRegistry(device=self.device)
        with self.assertRaises(KeyError):
            reg.get_model("Hist_VFakeVersion")


if __name__ == "__main__":
    unittest.main(verbosity=2)
