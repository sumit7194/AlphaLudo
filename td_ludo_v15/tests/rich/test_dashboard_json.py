"""Tests for stats JSON shape — must match what v13_dashboard.html consumes."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
LEGACY = ROOT.parent / "td_ludo"
sys.path.insert(0, str(LEGACY))

import train_v15_rich as t
from td_ludo_v15.rich.v15_trainer import V15RichTrainer


REQUIRED_STATS_FIELDS = {
    "total_games", "total_updates", "win_rate_100",
    "policy_entropy", "avg_value_loss", "avg_policy_loss",
    "avg_advantage", "clip_fraction", "approx_kl",
    "temperature", "games_per_minute", "best_eval_win_rate",
    "ghost_count", "is_stagnated", "play_alarm", "timestamp",
    "main_elo", "elo_rankings", "opponent_stats", "db_total",
    "recent_opponent_stats",
}

REQUIRED_METRICS_FIELDS = {
    "games", "updates", "win_rate", "policy_entropy",
    "avg_value_loss", "avg_policy_loss", "timestamp",
}


class _StubPlayer:
    pass


def _make_trainer():
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))  # at least one param for optimizer
        def forward(self, x, mask=None):
            B = x.shape[0]
            policy = torch.ones(B, 225) / 225.0
            value = torch.full((B,), 0.5)
            return policy, value
    m = _M()
    return V15RichTrainer(m, torch.device("cpu"),
                          ppo_buffer_games=1, ppo_minibatch_size=4,
                          ppo_epochs=1, kl_anchor_coeff=0.0)


def test_stats_json_has_all_required_fields():
    """`write_stats_json` writes a payload matching v13_dashboard's expectations."""
    trainer = _make_trainer()
    # Fake elo & db
    elo = MagicMock()
    elo.ratings = {"Model": 1542.3, "Hist_V13_2": 1600.0}
    elo.get_rankings = MagicMock(return_value=[("Hist_V13_2", 1600.0), ("Model", 1542.3)])
    db = MagicMock()
    db.get_opponent_stats = MagicMock(return_value={"Hist_V13_2": {"wins": 5, "games": 10}})
    db.get_recent_games = MagicMock(return_value=[
        {"p0": "Model", "p2": "Hist_V13_2", "winner": 0, "model_player_idx": 0},
        {"p0": "Hist_V13_2", "p2": "Model", "winner": 2, "model_player_idx": 2},
    ])
    db.get_total_games = MagicMock(return_value=2)
    player = _StubPlayer()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "stats.json")
        t.write_stats_json(path, trainer, player, elo, db,
                            win_rate_100=58.4, gpm=120.0, best_eval_wr=0.83,
                            eval_wr=0.81)
        with open(path) as f:
            data = json.load(f)
    missing = REQUIRED_STATS_FIELDS - set(data.keys())
    assert not missing, f"missing stats fields: {missing}"
    # Specific shape checks the dashboard reads:
    assert isinstance(data["elo_rankings"], list)
    if data["elo_rankings"]:
        assert "name" in data["elo_rankings"][0]
        assert "elo" in data["elo_rankings"][0]
    assert data["main_elo"] == 1542.3
    assert isinstance(data["recent_opponent_stats"], dict)
    # eval_win_rate set when passed
    assert "eval_win_rate" in data
    assert data["eval_win_rate"] == 81.0


def test_recent_opp_stats_aggregation():
    """_compute_recent_opp_stats correctly aggregates per-opponent WR.

    Rows match legacy GameDB.get_recent_games shape: 'players' list +
    'winner' int + 'model_player_idx' int.
    """
    db = MagicMock()
    db.get_recent_games = MagicMock(return_value=[
        {"players": ["Model", None, "Hist_V13_2", None],
         "winner": 0, "model_player_idx": 0},
        {"players": ["Model", None, "Hist_V13_2", None],
         "winner": 2, "model_player_idx": 0},
        {"players": ["Hist_V13_5_SL", None, "Model", None],
         "winner": 2, "model_player_idx": 2},
    ])
    out = t._compute_recent_opp_stats(db, n_recent=10)
    assert "Hist_V13_2" in out
    assert out["Hist_V13_2"]["games"] == 2
    assert out["Hist_V13_2"]["wins"] == 1
    assert out["Hist_V13_2"]["win_rate"] == 50.0
    assert "Hist_V13_5_SL" in out
    assert out["Hist_V13_5_SL"]["games"] == 1
    assert out["Hist_V13_5_SL"]["wins"] == 1
    assert out["Hist_V13_5_SL"]["win_rate"] == 100.0


def test_metrics_snapshot_appends_to_list():
    """`append_metrics_snapshot` appends valid entries; reads as a JSON list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "metrics.json")
        snap1 = {"games": 100, "updates": 5, "win_rate": 0.45,
                 "policy_entropy": 1.2, "avg_value_loss": 0.5,
                 "avg_policy_loss": -0.1, "timestamp": 12345.0,
                 "eval_win_rate": 70.0}
        snap2 = {"games": 200, "updates": 10, "win_rate": 0.55,
                 "policy_entropy": 1.0, "avg_value_loss": 0.4,
                 "avg_policy_loss": -0.2, "timestamp": 12400.0,
                 "eval_win_rate": 75.0}
        t.append_metrics_snapshot(path, snap1)
        t.append_metrics_snapshot(path, snap2)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2
        for d in data:
            missing = REQUIRED_METRICS_FIELDS - set(d.keys())
            assert not missing, f"missing metrics fields: {missing}"
        assert data[0]["games"] == 100
        assert data[1]["games"] == 200


def test_chain_status_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "chain.json")
        t.write_chain(path, "training", "v15_rich_test")
        with open(path) as f:
            data = json.load(f)
        assert data["stage"] == "RL"
        assert data["phase"] == "training"
        assert data["arch"] == "v15"
        assert data["run_name"] == "v15_rich_test"
        assert "ts" in data
