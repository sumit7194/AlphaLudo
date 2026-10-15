"""SL stream records are consistent with the unified encoding and the rules."""
import time

import numpy as np
import alphaludo_rs


def test_records_consistent():
    s = alphaludo_rs.SlStream(threads=4, sims=200, keep_prob=0.5, capacity=5000, seed=1)
    got = {k: [] for k in ("x", "cells", "pi", "n_cells", "action", "winner_rel", "legal_tokens")}
    n = 0
    t0 = time.time()
    while n < 3000 and time.time() - t0 < 120:
        d = s.take(1000, 2000)
        for k in got:
            got[k].append(d[k])
        n += len(d["action"])
    s.stop()
    d = {k: np.concatenate(v) for k, v in got.items()}
    assert n >= 3000
    x = d["x"].reshape(-1, 225, 5)
    for i in range(len(d["action"])):
        k = d["n_cells"][i]
        cells = d["cells"][i, :k]
        assert 1 <= k <= 4
        assert abs(d["pi"][i, :k].sum() - 1.0) < 1e-4
        assert d["action"][i] in cells
        for c in cells:
            assert x[i, c, 0] >= 1                      # a legal cell holds a mover token
        assert x[i, 0, 0] >= 1 and x[i, 0, 4] >= 1      # mover active, dice shown in corner (0,0)
        assert x[i, 0 * 15 + 14, 1] == -1               # 2P: 'next' seat inactive
        assert d["winner_rel"][i] in (0, 2, 255)        # 2P: winner is me or opposite
    forced = d["legal_tokens"] == 1
    assert (d["n_cells"][forced] == 1).all()
    # Roughly balanced outcomes from the mover's view (symmetric self-play).
    fin = d["winner_rel"] != 255
    win = (d["winner_rel"][fin] == 0).mean()
    assert 0.35 < win < 0.65
