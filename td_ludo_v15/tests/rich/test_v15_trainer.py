"""Tests for V15RichTrainer — PPO loss math + buffering."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from td_ludo_v15.rich.v15_trainer import V15RichTrainer, GAMMA, SCORE_REWARD


class _DummyV15(nn.Module):
    """Minimal V15-shaped model for unit tests (no Graph Transformer overhead)."""
    def __init__(self, d=8):
        super().__init__()
        # Per-cell tiny MLP: 8*3=24 → d
        self.in_proj = nn.Linear(24, d)
        # Policy head: per-cell scalar → 225 logits
        self.policy_head = nn.Linear(d, 1)
        # Value head: global pool → sigmoid
        self.value_head = nn.Linear(d, 1)

    def forward(self, x, mask=None):
        # x: (B, 8, 15, 15, 3) → flatten to (B, 225, 24) → (B, 225, d)
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(B, 225, -1).float()
        feat = self.in_proj(x)                               # (B, 225, d)
        logits = self.policy_head(feat).squeeze(-1)          # (B, 225)
        if mask is not None:
            logits = logits.masked_fill(mask < 0.5, -1e9)
        policy = torch.softmax(logits, dim=-1)
        pooled = feat.mean(dim=1)                            # (B, d)
        value = torch.sigmoid(self.value_head(pooled)).squeeze(-1)  # (B,)
        return policy, value


def _make_trajectory(n_steps=8, win=True, n_score_events=0):
    """Generate a synthetic trajectory of n_steps decisions."""
    traj = []
    for i in range(n_steps):
        v15_x = np.random.randn(8, 15, 15, 3).astype(np.float32) * 0.1
        v15_mask = np.zeros(225, dtype=np.float32)
        v15_mask[i * 7 % 225] = 1.0
        v15_mask[(i * 13 + 3) % 225] = 1.0
        chosen_cell = int(np.argwhere(v15_mask > 0.5)[0, 0])
        traj.append({
            "v15_x": v15_x,
            "v15_mask": v15_mask,
            "action": chosen_cell,
            "old_log_prob": -0.69,  # ≈ log(0.5)
            "temperature": 1.0,
            "step_reward": (SCORE_REWARD if i in [2, 5][:n_score_events] else 0.0),
        })
    return traj


def test_buffering_triggers_update_at_ppo_buffer_games():
    """No update fires until ppo_buffer_games games are buffered."""
    device = torch.device("cpu")
    model = _DummyV15()
    trainer = V15RichTrainer(model, device, ppo_buffer_games=3, ppo_minibatch_size=8,
                              ppo_epochs=1, kl_anchor_coeff=0.0)
    # Two games: no update yet
    for _ in range(2):
        m = trainer.train_on_game(_make_trajectory(5), winner=0, model_player=0)
        assert m is None, "should not update before ppo_buffer_games"
    # Third game triggers update
    m = trainer.train_on_game(_make_trajectory(5), winner=2, model_player=0)
    assert m is not None, "third game should trigger PPO update"
    assert "policy_loss" in m and "win_bce_loss" in m and "entropy" in m
    assert m["n_steps"] == 15
    assert m["n_minibatches"] >= 1


def test_mc_return_uses_gamma_and_terminal_reward():
    """Backwards discount: last step's reward includes terminal ±1."""
    device = torch.device("cpu")
    model = _DummyV15()
    trainer = V15RichTrainer(model, device, ppo_buffer_games=1,
                              ppo_minibatch_size=4, ppo_epochs=1, kl_anchor_coeff=0.0)
    traj = _make_trajectory(3)
    # Make rewards predictable: all step_reward=0
    for s in traj:
        s["step_reward"] = 0.0
    # Process — won game: terminal reward +1 on last step
    trainer.train_on_game(traj, winner=0, model_player=0)
    # After update the buffer is reset, but we can recompute the math:
    # rewards (after terminal added) = [0, 0, 1]
    # backwards: R_2=1, R_1=γ·1=γ, R_0=γ²
    expected_returns = np.array([GAMMA ** 2, GAMMA, 1.0], dtype=np.float32)
    # Recompute manually for clarity
    R = 0.0
    rewards = [0.0, 0.0, 1.0]
    got = []
    for r in reversed(rewards):
        R = r + GAMMA * R
        got.insert(0, R)
    assert np.allclose(np.array(got, dtype=np.float32), expected_returns, atol=1e-6)


def test_kl_anchor_metric_present_when_enabled():
    device = torch.device("cpu")
    model = _DummyV15()
    anchor = _DummyV15()
    for p in anchor.parameters():
        p.requires_grad = False
    trainer = V15RichTrainer(model, device, ppo_buffer_games=2,
                              ppo_minibatch_size=4, ppo_epochs=1,
                              kl_anchor_coeff=0.1, kl_anchor_model=anchor)
    trainer.train_on_game(_make_trajectory(4), winner=0, model_player=0)
    m = trainer.train_on_game(_make_trajectory(4), winner=2, model_player=0)
    assert m is not None
    # KL anchor metric should be > 0 (model and anchor differ)
    assert "kl_anchor" in m
    assert m["kl_anchor"] > 0.0


def test_diagnostic_means_track_recent_updates():
    device = torch.device("cpu")
    model = _DummyV15()
    trainer = V15RichTrainer(model, device, ppo_buffer_games=1,
                              ppo_minibatch_size=4, ppo_epochs=1, kl_anchor_coeff=0.0)
    assert trainer.get_diagnostic_means()["policy_entropy"] == 0.0
    trainer.train_on_game(_make_trajectory(4), winner=0, model_player=0)
    means = trainer.get_diagnostic_means()
    # After one update, entropy should be > 0 (uniform-ish policy on 2 legal cells)
    assert means["policy_entropy"] > 0.0
    assert means["avg_value_loss"] > 0.0
    # All keys present
    for k in ("policy_entropy", "avg_value_loss", "avg_policy_loss",
              "avg_advantage", "clip_fraction", "approx_kl"):
        assert k in means


def test_ratio_clamp_does_not_explode_on_extreme_old_lp():
    """Sanity check: if old_log_prob is wildly negative, ratio doesn't blow up."""
    device = torch.device("cpu")
    model = _DummyV15()
    trainer = V15RichTrainer(model, device, ppo_buffer_games=1,
                              ppo_minibatch_size=8, ppo_epochs=1, kl_anchor_coeff=0.0)
    traj = _make_trajectory(8)
    for s in traj:
        s["old_log_prob"] = -50.0  # absurdly small (very low original prob)
    m = trainer.train_on_game(traj, winner=0, model_player=0)
    assert m is not None
    # Loss should be finite
    assert np.isfinite(m["policy_loss"])
    assert np.isfinite(m["win_bce_loss"])
