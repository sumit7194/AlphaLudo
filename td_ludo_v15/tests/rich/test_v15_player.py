"""Tests for V15RichPlayer — rollout shapes + opponent picking."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
# Also ensure legacy is on path
LEGACY = ROOT.parent / "td_ludo"
sys.path.insert(0, str(LEGACY))

from td_ludo_v15.rich.v15_player import (
    V15RichPlayer, HISTORY_LEN, TOTAL_FRAMES,
)


def _random_picker(state, legal):
    return random.choice(list(legal))


def test_player_construction_renormalizes_weights():
    p = V15RichPlayer(
        batch_size=4,
        opponents={"a": _random_picker, "b": _random_picker},
        opponent_probs={"a": 3.0, "b": 1.0},
        seed=42,
    )
    # Weights should be renormalized to [0.75, 0.25]
    assert abs(p.opp_probs.sum() - 1.0) < 1e-6
    assert len(p.opp_names) == 2


def test_player_rejects_empty_opponents():
    try:
        V15RichPlayer(batch_size=2, opponents={}, opponent_probs={})
    except ValueError:
        return
    assert False, "expected ValueError"


def test_player_rejects_no_shared_keys():
    try:
        V15RichPlayer(
            batch_size=2,
            opponents={"a": _random_picker},
            opponent_probs={"b": 1.0},
        )
    except ValueError:
        return
    assert False, "expected ValueError"


def test_decisions_have_correct_shape():
    p = V15RichPlayer(
        batch_size=4,
        opponents={"rand": _random_picker},
        opponent_probs={"rand": 1.0},
        seed=42,
    )
    decisions, finished = p.collect_student_decisions()
    # All games should be at a decision (no game finishes that fast)
    assert len(decisions) == 4
    assert len(finished) == 0
    for d in decisions:
        assert d["v15_x"].shape == (TOTAL_FRAMES, 15, 15, 3)
        assert d["v15_mask"].shape == (225,)
        assert d["v15_mask"].sum() >= 1  # at least 1 legal cell
        assert len(d["legal"]) >= 1
        assert 0 <= d["game_idx"] < 4


def test_apply_actions_records_trajectory_step():
    p = V15RichPlayer(
        batch_size=2,
        opponents={"rand": _random_picker},
        opponent_probs={"rand": 1.0},
        seed=42,
    )
    decisions, _ = p.collect_student_decisions()
    n = len(decisions)
    # Pick the first legal cell for each
    chosen_cells = np.array(
        [int(np.argwhere(d["v15_mask"] > 0.5)[0, 0]) for d in decisions],
        dtype=np.int64,
    )
    log_probs = np.full(n, -0.69, dtype=np.float32)
    temps = np.ones(n, dtype=np.float32)
    p.apply_student_actions(decisions, chosen_cells, log_probs, temps)
    # Trajectory should now have one entry for each decided game
    for d in decisions:
        traj = p.trajectory[d["game_idx"]]
        assert len(traj) >= 1
        last = traj[-1]
        assert last["v15_x"].shape == (TOTAL_FRAMES, 15, 15, 3)
        assert last["v15_mask"].shape == (225,)
        assert isinstance(last["action"], int)
        assert 0 <= last["action"] < 225


def test_games_progress_and_eventually_finish():
    """Run many decision cycles; some games should finish."""
    p = V15RichPlayer(
        batch_size=8,
        opponents={"rand": _random_picker},
        opponent_probs={"rand": 1.0},
        max_game_len=200,
        seed=1,
    )
    total_finished = 0
    for _ in range(2000):
        decisions, finished = p.collect_student_decisions()
        total_finished += len(finished)
        if total_finished >= 4:
            break
        if not decisions:
            continue
        chosen = np.array([int(np.argwhere(d["v15_mask"] > 0.5)[0, 0])
                           for d in decisions], dtype=np.int64)
        log_probs = np.full(len(decisions), -0.69, dtype=np.float32)
        temps = np.ones(len(decisions), dtype=np.float32)
        p.apply_student_actions(decisions, chosen, log_probs, temps)
    assert total_finished >= 4, "expected several games to finish in 2000 cycles"
    for game in []:
        pass


def test_opponent_mix_sampled_proportionally():
    """Over many games, mix should roughly track opp_probs."""
    p = V15RichPlayer(
        batch_size=16,
        opponents={"a": _random_picker, "b": _random_picker, "c": _random_picker},
        opponent_probs={"a": 7.0, "b": 2.0, "c": 1.0},
        max_game_len=200, seed=42,
    )
    # Force many resets by truncating quickly
    for _ in range(2000):
        decisions, _ = p.collect_student_decisions()
        if not decisions:
            continue
        chosen = np.array([int(np.argwhere(d["v15_mask"] > 0.5)[0, 0])
                           for d in decisions], dtype=np.int64)
        log_probs = np.full(len(decisions), -0.69, dtype=np.float32)
        temps = np.ones(len(decisions), dtype=np.float32)
        p.apply_student_actions(decisions, chosen, log_probs, temps)
        if p.games_played >= 100:
            break
    total = sum(p.opp_game_counts.values())
    assert total >= 80, f"too few games finished: {total}"
    frac_a = p.opp_game_counts.get("a", 0) / max(1, total)
    # Expect ~70%; allow generous tolerance for short runs
    assert frac_a > 0.5, f"opponent 'a' should dominate; got {frac_a:.2f}"
