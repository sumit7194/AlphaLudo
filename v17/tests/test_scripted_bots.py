"""Rust Heuristic/Expert bots pick exactly the Python bots' move (noise disabled)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "td_ludo"), str(ROOT / "td_ludo" / "td_ludo"), str(ROOT / "td_ludo" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import alphaludo_rs
td = pytest.importorskip("td_ludo_cpp")
hb = pytest.importorskip("td_ludo.game.heuristic_bot")


def random_state(rng, four):
    active = [True] * 4 if four else [True, False, True, False]
    seats = [s for s in range(4) if active[s]]
    pos = np.full((4, 4), -1, dtype=np.int8)
    for s in seats:
        for t in range(4):
            u = rng.random()
            pos[s, t] = -1 if u < 0.25 else (99 if u < 0.3 else rng.integers(0, 56))
    cp = int(rng.choice(seats))
    dice = int(rng.integers(1, 7))
    st = td.create_initial_state() if four else td.create_initial_state_2p()
    st.player_positions = pos.tolist()
    st.scores = [int((pos[s] == 99).sum()) for s in range(4)]
    st.current_player = cp
    st.current_dice_roll = dice
    return st, pos, cp, dice, active


@pytest.mark.parametrize("name,cls", [("heuristic", "HeuristicLudoBot"), ("expert", "ExpertBot")])
def test_rust_matches_python(monkeypatch, name, cls):
    monkeypatch.setattr(hb.random, "uniform", lambda a, b: 0.0)
    rng = np.random.default_rng(5)
    checked = 0
    for four in (False, True):
        for _ in range(4000):
            st, pos, cp, dice, active = random_state(rng, four)
            legal = list(td.get_legal_moves(st))
            if len(legal) < 2:
                continue
            py = getattr(hb, cls)(player_id=cp).select_move(st, legal)
            rs = alphaludo_rs.select_bot_move(name, pos, cp, dice, active, noise=False)
            assert py == rs, (name, pos.tolist(), cp, dice, legal, py, rs)
            checked += 1
    assert checked > 3000
