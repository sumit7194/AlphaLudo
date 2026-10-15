"""Fixed evaluation suite for V17 (and any model the farm can serve).

Every match is the evaluated model (greedy) against one fixed reference, seats
shuffled per game, a fixed seed (so successive checkpoints face the same dice
streams — paired comparisons are far less noisy). References are never trained
against by default; they measure progress only.

  python -m v17.evaluate checkpoints/v17/model_latest.pt --games 1000

4-player references are named "4p:<bot>": the model plays ONE seat against
three copies of the bot (fair share 25%). "4p:teacher" is the 4P SL teacher as
configured in v17/teacher4.json (the single source of truth for its settings).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List

import alphaludo_rs

from .inference import CHAMPION_PATH, Player, V15Player, V17Player, load_v17, serve

TEACHER4_PATH = Path(__file__).resolve().parent / "teacher4.json"


def teacher4_spec() -> dict:
    """The 4P teacher bot spec (v17/teacher4.json)."""
    import json
    return json.loads(TEACHER4_PATH.read_text())["spec"]

# name -> (player spec for the farm, needs the v15 champion model)
REFERENCES = {
    "v15.2": None,                                   # the champion network (model id 1)
    "mcts3000": {"bot": "mcts", "sims": 3000},
    "depth2": {"bot": "depth2"},
    "expert": {"bot": "expert"},
    "heuristic": {"bot": "heuristic"},
}
DEFAULT_SUITE = ["v15.2", "mcts3000", "depth2", "expert", "heuristic"]
# 4P: one model seat vs 3 copies (fair share 0.25). Expert == heuristic in 4P, so left out.
FOURP_REFERENCES = {
    "4p:heuristic": {"bot": "heuristic"},
    "4p:expectimax": {"bot": "expectimax"},
    "4p:depth2": {"bot": "depth2"},
}
FOURP_SUITE = ["4p:teacher", "4p:depth2", "4p:expectimax", "4p:heuristic"]


def reference_spec(name: str) -> dict:
    if name == "4p:teacher":
        return teacher4_spec()
    if name in FOURP_REFERENCES:
        return FOURP_REFERENCES[name]
    return REFERENCES[name]


def match(student: Player, opponent: str, games: int, seed: int, champion: Player | None = None,
          concurrency: int = 256, players: int = 2) -> dict:
    encoders = ["unified", "v15"]
    farm = alphaludo_rs.Farm(encoders, concurrency=concurrency, seed=seed)
    opp = {"net": 1, "temp": 0.0} if opponent == "v15.2" else reference_spec(opponent)
    n_seats = 4 if opponent.startswith("4p:") else 2
    farm.set_matchups([{"players": [{"net": 0, "temp": 0.0}] + [opp] * (n_seats - 1), "weight": 1.0}])
    farm.set_game_limit(games)
    t0 = time.time()
    serving = {0: student}
    if opponent == "v15.2":
        serving[1] = champion or V15Player()
    serve(farm, serving)
    res = farm.take_results()
    wins = sum(1 for (_, _, role, w, _) in res if w >= 0 and role[w] == 0)
    seat0 = [r for r in res if r[2][0] == 0]
    p = wins / max(1, len(res))
    return {
        "opponent": opponent, "players": n_seats, "fair": 1.0 / n_seats,
        "games": len(res), "wins": wins, "win_rate": p,
        "stderr": math.sqrt(max(p * (1 - p), 1e-12) / max(1, len(res))),
        "seat0_games": len(seat0),
        "unfinished": sum(1 for r in res if r[3] < 0),
        "seconds": time.time() - t0,
    }


def suite(student: Player, games: int = 1000, seed: int = 1700, opponents: List[str] = DEFAULT_SUITE,
          champion: Player | None = None) -> Dict[str, dict]:
    out = {}
    for i, o in enumerate(opponents):
        out[o] = match(student, o, games, seed + 1000 * i, champion=champion)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="V17 checkpoint, or 'v15.2' to evaluate the champion itself")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--opponents", default=",".join(DEFAULT_SUITE))
    ap.add_argument("--seed", type=int, default=1700)
    ap.add_argument("--json", default=None, help="also write results to this JSON file")
    ap.add_argument("--backend", default="mlx", choices=("mlx", "torch"))
    a = ap.parse_args()
    if a.checkpoint == "v15.2":
        student = V15Player()
        student.encoder = "v15"
        # Evaluate the champion: model 0 must use the v15 encoder.
        res = {}
        for i, o in enumerate(a.opponents.split(",")):
            if o == "v15.2":
                continue
            farm = alphaludo_rs.Farm(["v15"], concurrency=256, seed=a.seed + 1000 * i)
            farm.set_matchups([{"players": [{"net": 0, "temp": 0.0}, REFERENCES[o]], "weight": 1.0}])
            farm.set_game_limit(a.games)
            serve(farm, {0: student})
            r = farm.take_results()
            w = sum(1 for (_, _, role, wn, _) in r if wn >= 0 and role[wn] == 0)
            p = w / len(r)
            res[o] = {"win_rate": p, "stderr": math.sqrt(p * (1 - p) / len(r)), "games": len(r), "wins": w}
    else:
        student = V17Player(load_v17(a.checkpoint), backend=a.backend, device="mps" if a.backend == "torch" else "cpu")
        res = suite(student, a.games, a.seed, a.opponents.split(","))
    for o, r in res.items():
        print(f"{o:10s} {r['win_rate']:.3f} ±{r['stderr']:.3f}  ({r['wins']}/{r['games']})", flush=True)
    if a.json:
        import json, os
        tmp = a.json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=2)
        os.replace(tmp, a.json)


def run_isolated(checkpoint, games: int, opponents, seed: int, out_json, python: str | None = None) -> dict:
    """Runs the suite in a SEPARATE process (MLX on the GPU) and returns its results.
    Training processes use PyTorch-MPS; keeping MLX out of them avoids mixing two
    Metal frameworks in one long-running process (an intermittent hang was seen)."""
    import json, subprocess, sys
    py = python or sys.executable
    cmd = [py, "-m", "v17.evaluate", str(checkpoint), "--games", str(games), "--opponents", ",".join(opponents),
           "--seed", str(seed), "--json", str(out_json)]
    r = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"evaluation subprocess failed: {r.stderr[-2000:]}")
    return json.loads(Path(out_json).read_text())


if __name__ == "__main__":
    main()
