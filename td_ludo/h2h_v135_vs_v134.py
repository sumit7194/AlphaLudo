"""H2H: V13.5 (best/latest) vs V13.4 (final). Greedy, mirrored-seed pairs.

V13.5 = V135ProductionAdapter, 21ch V18 production encoder, single-frame, 3M params.
V13.4 = V133Temporal, 17ch V17 encoder, K=8 history, 3.79M params.

Note: V13.4 was a failed branch — at its chain-end it lost 90/10 vs V13.2.
Expectation: V13.5 should win heavily. This is a sanity-check baseline more
than a competitive test.

Usage:
    td_env/bin/python h2h_v135_vs_v134.py --games 2000 --device mps
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_ludo_cpp as ludo_cpp
from td_ludo.models.v13_5_production import V135ProductionAdapter
from td_ludo.models.v13_3 import V133Temporal
from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS
from td_ludo.game.encoder_v18_production import encode_state_v18_production


HISTORY_K = 8
MAX_MOVES_PER_GAME = 400


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--v135-ckpt", default="play/model_weights/v13_5/model_latest.pt")
    p.add_argument("--v134-ckpt", default="checkpoints/v134/model_latest.pt")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_device(name):
    if name in ("cpu", "cuda", "mps"):
        return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _load_sd(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


class V135Agent:
    name = "V13.5"
    def __init__(self, ckpt, device):
        self.device = device
        self.model = V135ProductionAdapter()
        self.model.load_state_dict(_load_sd(ckpt, device))
        self.model.to(device).eval()

    def reset(self): pass
    def observe(self, state): pass

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        enc = np.array(encode_state_v18_production(state), dtype=np.float32)
        x = torch.from_numpy(enc).unsqueeze(0).to(self.device)
        mask = np.zeros(4, dtype=np.float32)
        for a in legal: mask[a] = 1.0
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, _, _, _ = self.model(x, m)
            action = int(policy.argmax(dim=1).item())
        return action if action in legal else legal[0]


class V134Agent:
    name = "V13.4"
    def __init__(self, ckpt, device, **arch):
        self.device = device
        self.model = V133Temporal(history_k=HISTORY_K, in_channels=V17_CHANNELS, **arch)
        self.model.load_state_dict(_load_sd(ckpt, device), strict=False)
        self.model.eval().to(device)
        self.history = collections.deque(maxlen=HISTORY_K)

    def reset(self):
        self.history.clear()

    def observe(self, state):
        self.history.append(encode_state_v17(state))

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
        for a in legal: mask[a] = 1.0
        with torch.no_grad():
            x = torch.from_numpy(stack).unsqueeze(0).to(self.device, dtype=torch.float32)
            h = torch.from_numpy(hmask).unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask).unsqueeze(0).to(self.device, dtype=torch.float32)
            policy, _, _ = self.model(x, m, h)
            action = int(policy.argmax(dim=1).item())
        return action if action in legal else legal[0]


def play_one(p0_agent, p2_agent, seed):
    """Mirrors evaluate_v11.py / h2h_v134.py game loop."""
    rng = random.Random(seed)
    state = ludo_cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    mc = 0
    p0_agent.reset(); p2_agent.reset()

    while not state.is_terminal and mc < MAX_MOVES_PER_GAME:
        cp = int(state.current_player)
        if not state.active_players[cp]:
            n = (cp + 1) % 4
            while not state.active_players[n]: n = (n + 1) % 4
            state.current_player = n
            continue
        if state.current_dice_roll == 0:
            state.current_dice_roll = rng.randint(1, 6)
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
        agent = p0_agent if cp == 0 else p2_agent
        agent.observe(state)  # update history (no-op for V13.5)
        action = agent.select(state, list(legal))
        state = ludo_cpp.apply_move(state, int(action))
        mc += 1

    winner = int(ludo_cpp.get_winner(state)) if state.is_terminal else -1
    if winner == 0: return p0_agent.name
    if winner == 2: return p2_agent.name
    return "draw"


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"[h2h] device={device}")
    print(f"[h2h] V13.5: {args.v135_ckpt}")
    print(f"[h2h] V13.4: {args.v134_ckpt}")

    v135 = V135Agent(args.v135_ckpt, device)
    v134 = V134Agent(
        args.v134_ckpt, device,
        cnn_blocks=10, cnn_channels=128, d_model=128,
        n_layers=4, nhead=4, ffn_dim=512,
    )

    pairs = max(1, args.games // 2)
    total = pairs * 2
    print(f"[h2h] running {total} games ({pairs} mirrored seed pairs)")

    wins = {"V13.5": 0, "V13.4": 0, "draw": 0}
    seat = {"V13.5_P0": [0, 0], "V13.5_P2": [0, 0]}

    t0 = time.time()
    last_report = t0
    for i in range(pairs):
        seed = args.seed + i

        # V13.5 = P0
        w1 = play_one(v135, v134, seed)
        wins[w1] += 1
        seat["V13.5_P0"][1] += 1
        if w1 == "V13.5":
            seat["V13.5_P0"][0] += 1

        # Mirror: V13.5 = P2
        w2 = play_one(v134, v135, seed)
        wins[w2] += 1
        seat["V13.5_P2"][1] += 1
        if w2 == "V13.5":
            seat["V13.5_P2"][0] += 1

        if time.time() - last_report > 10:
            done = (i + 1) * 2
            wr = 100 * wins["V13.5"] / done
            elapsed = time.time() - t0
            eta = elapsed * (total - done) / max(done, 1)
            print(f"  [{done:>5d}/{total}] V13.5 {wins['V13.5']:>4d} ({wr:5.1f}%)"
                  f" | V13.4 {wins['V13.4']:>4d} | draws {wins['draw']:>3d}"
                  f" | gpm={done/(elapsed/60):.0f} | eta={eta/60:.1f}min")
            last_report = time.time()

    elapsed = time.time() - t0
    total = sum(wins.values())
    print()
    print("=" * 60)
    print(f"H2H: V13.5 (model_latest) vs V13.4 ({total} games, {elapsed:.1f}s)")
    print("=" * 60)
    print(f"  V13.5 wins:   {wins['V13.5']:>5d}  ({100*wins['V13.5']/total:5.2f}%)")
    print(f"  V13.4 wins:   {wins['V13.4']:>5d}  ({100*wins['V13.4']/total:5.2f}%)")
    print(f"  Draws:        {wins['draw']:>5d}  ({100*wins['draw']/total:5.2f}%)")
    print()
    p0 = seat["V13.5_P0"]; p2 = seat["V13.5_P2"]
    print(f"  V13.5 as P0:  {p0[0]:>4d}/{p0[1]} ({100*p0[0]/max(p0[1],1):5.1f}%)")
    print(f"  V13.5 as P2:  {p2[0]:>4d}/{p2[1]} ({100*p2[0]/max(p2[1],1):5.1f}%)")
    p = wins["V13.5"] / total
    se = (p * (1 - p) / total) ** 0.5
    print()
    print(f"  Net WR: {100*p:.2f}% ± {100*1.96*se:.2f}pp (95% CI)")
    if p > 0.55:
        print("  Verdict: V13.5 is STRONGER than V13.4 (>55% with margin).")
    elif p > 0.50:
        print("  Verdict: V13.5 is marginally ahead.")
    elif p > 0.45:
        print("  Verdict: too close to call.")
    else:
        print("  Verdict: V13.5 is WEAKER than V13.4 — investigate.")


if __name__ == "__main__":
    main()
