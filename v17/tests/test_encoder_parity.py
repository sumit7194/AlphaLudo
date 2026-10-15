"""Rust unified encoder == independent Python reference, on random configurations."""
import numpy as np
import alphaludo_rs

from v17 import geometry as g
from v17.reference_encoder import encode


def random_config(rng, four: bool):
    active = [True] * 4 if four else [True, False, True, False]
    seats = [s for s in range(4) if active[s]]
    positions = np.full((4, 4), -1, dtype=np.int8)
    for s in seats:
        for t in range(4):
            u = rng.random()
            positions[s, t] = -1 if u < 0.3 else (99 if u < 0.4 else rng.integers(0, 56))
    cp = int(rng.choice(seats))
    dice = int(rng.integers(1, 7))
    csix = [int(rng.integers(0, 3)) if active[s] else 0 for s in range(4)]
    return positions, cp, dice, active, csix


def test_track_cells_match_rust():
    for owner in range(4):
        for pov in range(4):
            for pos in range(56):
                assert alphaludo_rs.unified_track_cell(pos, owner, pov) == g.track_cell(pos, owner, pov)


def test_encoder_parity_random_configs():
    rng = np.random.default_rng(17)
    for four in (False, True):
        for _ in range(20_000):
            positions, cp, dice, active, csix = random_config(rng, four)
            xr, mr = alphaludo_rs.encode_unified(positions, cp, dice, active, csix)
            xp, mp = encode(positions.tolist(), cp, dice, active, csix)
            assert np.array_equal(xr, xp), (positions, cp, dice, active, csix)
            assert np.array_equal(mr, mp)
