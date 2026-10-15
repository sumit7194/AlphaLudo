"""GameFarm mechanics with a stand-in network (uniform policy over legal cells)."""
import numpy as np
import alphaludo_rs


def run(farm, games):
    farm.set_game_limit(games)
    while farm.collect():
        for m in farm.models_pending():
            x, mask = farm.batch(m)
            pol = mask / mask.sum(1, keepdims=True)
            farm.submit(m, pol.astype(np.float32), np.full(len(mask), 0.5, np.float32))
    return farm.take_results(), farm.take_records()


def test_2p_records_and_results():
    farm = alphaludo_rs.Farm(["unified"], concurrency=32, seed=1)
    farm.set_matchups([
        {"players": [{"net": 0, "temp": 1.0, "record": True}, {"bot": "heuristic"}], "weight": 1},
        {"players": [{"net": 0, "temp": 1.0, "record": True}, {"net": 0, "temp": 1.0, "record": True}], "weight": 1},
    ])
    res, rec = run(farm, 200)
    assert len(res) == 200 and farm.in_flight() == 0
    assert all(w in (0, 2) for (_, _, _, w, _) in res)
    assert set(np.unique(rec["seat"]).tolist()) <= {0, 2}
    assert (rec["legal_tokens"] >= 2).all()                 # forced moves never recorded
    r = rec["reward"]
    assert set(np.unique(r[~np.isnan(r)]).tolist()) <= {0.0, 1.0}
    assert rec["x"].shape[1] == 225 * 5
    # One reward per (game, seat) trajectory, consistent with the winner.
    win = {g: w for (g, _, _, w, _) in res}
    for g, s, rw in zip(rec["game"], rec["seat"], rec["reward"]):
        assert rw == (1.0 if win[g] == s else 0.0)
    assert np.isfinite(rec["logp"]).all() and (rec["logp"] <= 0).all()


def test_4p_and_bots():
    farm = alphaludo_rs.Farm(["unified"], concurrency=16, seed=2)
    farm.set_matchups([{"players": [{"net": 0, "temp": 1.0, "record": True},
                                    {"bot": "expert"}, {"bot": "depth2"}, {"bot": "mcts", "sims": 200}], "weight": 1}])
    res, rec = run(farm, 40)
    assert len(res) == 40
    assert set(np.unique(rec["n_players"]).tolist()) == {4}
    assert set(np.unique(rec["seat"]).tolist()) <= {0, 1, 2, 3}


def test_bot_strength_sanity():
    # A uniform-random "network" should lose clearly to Expert (role 1).
    farm = alphaludo_rs.Farm(["unified"], concurrency=64, seed=3)
    farm.set_matchups([{"players": [{"net": 0, "temp": 1.0}, {"bot": "expert"}], "weight": 1}])
    res, _ = run(farm, 400)
    expert_wins = sum(1 for (_, _, role, w, _) in res if w >= 0 and role[w] == 1)
    assert expert_wins / len(res) > 0.75
