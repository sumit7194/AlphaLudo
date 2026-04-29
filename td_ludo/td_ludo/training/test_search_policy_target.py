"""Unit tests for search_policy_target.

Run from td_ludo/ root:
    /path/to/python -m td_ludo.training.test_search_policy_target
"""
from __future__ import annotations

import sys
import unittest

import numpy as np
import torch

import td_ludo_cpp as cpp

from td_ludo.training.search_policy_target import compute_pi_search_batch


class _ToyModel(torch.nn.Module):
    """Minimal stand-in for AlphaLudoV12 used in fast tests.

    Returns:
      - policy: argmax-able softmax over 4 actions (uniform for default ToyModel)
      - win_prob: deterministic function of input mean — lets us craft
        states with known relative leaf values for argmax verification.
    """
    def __init__(self, value_fn=None):
        super().__init__()
        self.value_fn = value_fn  # callable(state_tensor) -> tensor of [B] in [0,1]
        # Real parameter so .to(device) works.
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, legal_mask=None):
        B = x.shape[0]
        device = x.device
        # Uniform policy over 4 actions (legal-mask-aware below).
        logits = torch.zeros(B, 4, device=device)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask < 0.5, -1e9)
        policy = torch.softmax(logits, dim=1)
        if self.value_fn is not None:
            win_prob = self.value_fn(x).to(device)
        else:
            # Default: 0.5 for every state — neutral leaf evaluator.
            win_prob = torch.full((B,), 0.5, device=device)
        moves = torch.zeros(B, device=device)
        return policy, win_prob, moves

    def forward_policy_only(self, x, legal_mask=None):
        B = x.shape[0]
        device = x.device
        logits = torch.zeros(B, 4, device=device)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask < 0.5, -1e9)
        return logits


def _mid_game_state(seed: int = 0):
    """Roll a few moves into the game so legal_moves is meaningful."""
    rng = np.random.RandomState(seed)
    g = cpp.create_initial_state()
    for _ in range(20):
        if cpp.get_winner(g) >= 0:
            break
        if g.current_dice_roll == 0:
            g.current_dice_roll = int(rng.randint(1, 7))
        moves = cpp.get_legal_moves(g)
        if not moves:
            # advance to next player
            g.current_player = (g.current_player + 1) % 4
            g.current_dice_roll = 0
            continue
        a = int(rng.choice(moves))
        g = cpp.apply_move(g, a)
        if g.current_dice_roll == 0:
            g.current_dice_roll = int(rng.randint(1, 7))
    if g.current_dice_roll == 0:
        g.current_dice_roll = int(rng.randint(1, 7))
    return g


class TestSearchPolicyTarget(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")

    def test_no_input_mutation(self):
        """compute_pi_search_batch must not mutate the input GameState."""
        g = _mid_game_state(seed=1)
        positions_before = np.array(g.player_positions).copy()
        dice_before = g.current_dice_roll
        player_before = g.current_player

        model = _ToyModel()
        pi_search, _ = compute_pi_search_batch(
            [g], [g.current_player], model, self.device,
        )

        self.assertTrue(np.array_equal(
            positions_before, np.array(g.player_positions)
        ))
        self.assertEqual(g.current_dice_roll, dice_before)
        self.assertEqual(g.current_player, player_before)

    def test_pi_search_shape_and_legality(self):
        """pi_search is (N, 4); only legal first-action slots are nonzero."""
        games = [_mid_game_state(seed=s) for s in range(5)]
        roots = [g.current_player for g in games]
        legals = [list(cpp.get_legal_moves(g)) for g in games]

        model = _ToyModel()
        pi_search, diag = compute_pi_search_batch(
            games, roots, model, self.device,
        )
        self.assertEqual(tuple(pi_search.shape), (5, 4))
        for i, legal in enumerate(legals):
            for a in range(4):
                if a not in legal:
                    self.assertEqual(
                        float(pi_search[i, a]), 0.0,
                        f"Illegal action {a} got nonzero mass in row {i}",
                    )
        # Diagnostics shape
        self.assertEqual(len(diag['top_actions']), 5)
        self.assertEqual(len(diag['q_values']), 5)

    def test_pi_search_sums_to_one_for_multi_legal_states(self):
        """For states with ≥2 legal first actions, pi_search sums to 1.0."""
        games, roots = [], []
        for s in range(20):
            g = _mid_game_state(seed=s)
            if len(cpp.get_legal_moves(g)) >= 2:
                games.append(g)
                roots.append(g.current_player)
            if len(games) >= 5:
                break
        self.assertGreater(len(games), 0, "no multi-legal states found")

        model = _ToyModel()
        pi_search, _ = compute_pi_search_batch(games, roots, model, self.device)
        sums = pi_search.sum(dim=1).cpu().numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_argmax_consistency_with_crafted_value_fn(self):
        """If value_fn rewards a specific board pattern, the search should
        pick the first action whose downstream leaves max that pattern."""
        # Craft a value_fn that returns a high value when the sum of the
        # first 4 channels (own positions) is large — i.e., prefers states
        # where own tokens are spread on the board.
        def value_fn(x):
            # x: (B, 33, 15, 15)
            own_mass = x[:, :4].sum(dim=(1, 2, 3))
            # Squash to [0, 1] via sigmoid centered at the mean.
            return torch.sigmoid(0.05 * (own_mass - own_mass.mean()))

        model = _ToyModel(value_fn=value_fn)
        games = []
        for s in range(50):
            g = _mid_game_state(seed=s)
            if len(cpp.get_legal_moves(g)) >= 2:
                games.append(g)
            if len(games) >= 3:
                break
        self.assertGreater(len(games), 0, "no multi-legal seed states")
        roots = [g.current_player for g in games]

        pi_search, diag = compute_pi_search_batch(
            games, roots, model, self.device, label_smoothing=0.1,
        )
        # The argmax row should match the action with the largest Q.
        for gi, q_dict in enumerate(diag['q_values']):
            if not q_dict:
                continue
            best_action = max(q_dict, key=q_dict.get)
            row = pi_search[gi].cpu().numpy()
            self.assertEqual(int(np.argmax(row)), best_action)
            self.assertAlmostEqual(float(row[best_action]), 0.9, places=4)

    def test_label_smoothing_zero_gives_pure_onehot(self):
        g = _mid_game_state(seed=42)
        model = _ToyModel()
        pi_search, _ = compute_pi_search_batch(
            [g], [g.current_player], model, self.device, label_smoothing=0.0,
        )
        row = pi_search[0].cpu().numpy()
        # Exactly one entry is 1.0, rest are 0.
        nonzero = (row > 1e-9).sum()
        self.assertEqual(nonzero, 1)
        self.assertAlmostEqual(float(row.max()), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
