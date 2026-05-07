"""V13.4 chain Phase 3 H2H runner.

Round-robin: V13.2_latest vs V13.4_SL vs V13.4_RL. Greedy play, mirrored seeds.
Writes JSON output for the chain dashboard to consume.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS
from experiments.distillation_14ch.model_14ch import MinimalCNN14
from td_ludo.models.v13_3 import V133Temporal


HISTORY_K = 8
MAX_MOVES_PER_GAME = 400


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--teacher", required=True, help="V13.2 checkpoint path")
    p.add_argument("--sl", required=True, help="V13.4 SL checkpoint path")
    p.add_argument("--rl", required=True, help="V13.4 RL checkpoint path")
    p.add_argument("--output", required=True, help="Output JSON path")
    # V13.4 arch (must match training)
    p.add_argument("--cnn-blocks", type=int, default=10)
    p.add_argument("--cnn-channels", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_device(name):
    if name in ("cuda", "cpu", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Agent wrappers ────────────────────────────────────────────────────────
class V132Agent:
    def __init__(self, name, path, device):
        self.name = name
        self.device = device
        self.model = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=17)
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(device)

    def reset(self): pass
    def observe(self, state): pass

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        enc = encode_state_v17(state)
        x = torch.from_numpy(enc).unsqueeze(0).to(self.device, dtype=torch.float32)
        mask = np.zeros(4, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            policy, _, _ = self.model(x, m)
            action = int(policy.argmax(dim=1).item())
        return action if action in legal else legal[0]


class V134Agent:
    def __init__(self, name, path, device, **arch):
        self.name = name
        self.device = device
        self.model = V133Temporal(history_k=HISTORY_K, in_channels=V17_CHANNELS, **arch)
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(device)
        self.history = collections.deque(maxlen=HISTORY_K)

    def reset(self):
        self.history.clear()

    def observe(self, state):
        cur_frame = encode_state_v17(state)
        self.history.append(cur_frame)

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        hist = list(self.history)
        pad = HISTORY_K - len(hist)
        if pad > 0:
            zero = np.zeros((V17_CHANNELS, 15, 15), dtype=np.float32)
            stack = np.stack([zero] * pad + hist, axis=0)
            hmask = np.array([False] * pad + [True] * len(hist), dtype=bool)
        else:
            stack = np.stack(hist, axis=0)
            hmask = np.ones(HISTORY_K, dtype=bool)
        mask = np.zeros(4, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0
        with torch.no_grad():
            x = torch.from_numpy(stack).unsqueeze(0).to(self.device, dtype=torch.float32)
            h = torch.from_numpy(hmask).unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask).unsqueeze(0).to(self.device, dtype=torch.float32)
            policy, _, _ = self.model(x, m, h)
            action = int(policy.argmax(dim=1).item())
        return action if action in legal else legal[0]


# ── Game loop ─────────────────────────────────────────────────────────────
def play_one(agents, agent_for_player, seed):
    random.seed(seed); np.random.seed(seed)
    state = ludo_cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]; mc = 0
    for a in agents:
        a.reset()
    while not state.is_terminal and mc < MAX_MOVES_PER_GAME:
        cp = int(state.current_player)
        if not state.active_players[cp]:
            n = (cp + 1) % 4
            while not state.active_players[n]: n = (n + 1) % 4
            state.current_player = n
            continue
        if state.current_dice_roll == 0:
            state.current_dice_roll = random.randint(1, 6)
            if state.current_dice_roll == 6:
                csix[cp] += 1
            else:
                csix[cp] = 0
            if csix[cp] >= 3:
                n = (cp + 1) % 4
                while not state.active_players[n]: n = (n + 1) % 4
                state.current_player = n
                state.current_dice_roll = 0
                csix[cp] = 0
                continue
        legal = ludo_cpp.get_legal_moves(state)
        if not legal:
            n = (cp + 1) % 4
            while not state.active_players[n]: n = (n + 1) % 4
            state.current_player = n
            state.current_dice_roll = 0
            continue
        ag = agent_for_player[cp]
        ag.observe(state)
        action = ag.select(state, list(legal))
        state = ludo_cpp.apply_move(state, int(action))
        mc += 1
    return int(ludo_cpp.get_winner(state)) if state.is_terminal else -1


def head_to_head(a, b, n_games, seed_base, progress_cb=None):
    aw = bw = dr = 0
    for i in range(n_games):
        seed = seed_base + (i // 2)
        if i % 2 == 0:
            mapping = {0: a, 2: b}
        else:
            mapping = {0: b, 2: a}
        winner = play_one([a, b], mapping, seed)
        if winner == -1:
            dr += 1
        else:
            agent_won = mapping[winner]
            if agent_won is a:
                aw += 1
            else:
                bw += 1
        if progress_cb and (i + 1) % max(1, n_games // 20) == 0:
            progress_cb(i + 1, aw, bw, dr)
    return aw, bw, dr


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)

    print(f"[h2h_v134] device={device}, {args.games} games per pair")

    arch = dict(
        cnn_blocks=args.cnn_blocks, cnn_channels=args.cnn_channels,
        d_model=args.d_model, n_layers=args.n_layers,
        nhead=args.nhead, ffn_dim=args.ffn_dim,
    )

    print("[h2h_v134] loading agents...")
    agents = [
        V132Agent("V13.2_latest", args.teacher, device),
        V134Agent("V13.4_SL", args.sl, device, **arch),
        V134Agent("V13.4_RL", args.rl, device, **arch),
    ]

    results = []
    t0 = time.time()
    print(f"\n{'pair':<45}  W   L   D   WR%")
    print("-" * 70)
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            t_pair = time.time()
            aw, bw, dr = head_to_head(a, b, args.games, args.seed + i * 100 + j)
            n = max(1, aw + bw + dr)
            wr_a = 100 * aw / n
            wr_b = 100 * bw / n
            elapsed = time.time() - t_pair
            print(f"{a.name:>20} vs {b.name:<22}  {aw:>3} {bw:>3} {dr:>3}  "
                  f"{wr_a:5.1f}% / {wr_b:5.1f}%  ({elapsed:.0f}s)")
            results.append({
                "a": a.name, "b": b.name, "a_wins": aw, "b_wins": bw, "draws": dr,
                "wr_a": wr_a, "wr_b": wr_b, "elapsed_sec": elapsed,
            })
    total = time.time() - t0
    print(f"\n[h2h_v134] done in {total:.0f}s")

    # Standings
    standings = {a.name: {"wins": 0, "games": 0} for a in agents}
    for r in results:
        n = r["a_wins"] + r["b_wins"] + r["draws"]
        standings[r["a"]]["wins"] += r["a_wins"]
        standings[r["a"]]["games"] += n
        standings[r["b"]]["wins"] += r["b_wins"]
        standings[r["b"]]["games"] += n
    sorted_standings = sorted(
        [{"name": k, "wins": v["wins"], "games": v["games"],
          "wr": 100 * v["wins"] / max(1, v["games"])}
         for k, v in standings.items()],
        key=lambda d: -d["wr"],
    )
    print("\n=== Overall standings ===")
    for s in sorted_standings:
        print(f"  {s['name']:<25}  {s['wins']:>4}/{s['games']:>4}  {s['wr']:5.1f}%")

    out = {
        "games_per_pair": args.games,
        "total_sec": total,
        "ts": int(time.time()),
        "results": results,
        "standings": sorted_standings,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[h2h_v134] wrote {args.output}")


if __name__ == "__main__":
    main()
