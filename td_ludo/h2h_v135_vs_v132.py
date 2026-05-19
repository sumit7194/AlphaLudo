"""H2H: V13.5 (latest) vs V13.2 (deployed on alphaludo.in).

V13.5 = V135ProductionAdapter, 21ch V18 production encoder, single-frame, 3.0M params.
V13.2 = MinimalCNN14, 17ch V17 encoder, single-frame, ~3M params.

Both single-frame, so no history bookkeeping needed (unlike V13.4).
Mirrored seed pairs for seat fairness. Greedy argmax.

Usage:
    td_env/bin/python h2h_v135_vs_v132.py --games 3000 --device mps
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_ludo_cpp as ludo_cpp
from td_ludo.models.v13_5_production import V135ProductionAdapter
from experiments.distillation_14ch.model_14ch import MinimalCNN14
from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS
from td_ludo.game.encoder_v18_production import encode_state_v18_production


MAX_MOVES_PER_GAME = 400

# Same V13.2 ckpt the website was exported from.
V132_CKPT_DEFAULT = (
    "/Users/sumit/Github/AlphaLudo/checkpoint_backups/"
    "v132_20260506_015608/model_latest.pt"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=3000)
    p.add_argument("--v135-ckpt", default="play/model_weights/v13_5/model_latest.pt")
    p.add_argument("--v132-ckpt", default=V132_CKPT_DEFAULT)
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


class V132Agent:
    name = "V13.2"
    def __init__(self, ckpt, device):
        self.device = device
        self.model = MinimalCNN14(
            num_res_blocks=10, num_channels=128, in_channels=V17_CHANNELS,
        )
        # strict=False to drop aux heads if loading from MinimalCNN14Aux ckpt
        sd = _load_sd(ckpt, device)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            # No missing keys expected for MinimalCNN14 from a V13.2 ckpt
            print(f"[V13.2] WARN missing keys: {len(missing)} (e.g. {missing[:3]})")
        self.model.to(device).eval()

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        enc = encode_state_v17(state)
        x = torch.from_numpy(enc).unsqueeze(0).to(self.device, dtype=torch.float32)
        mask = np.zeros(4, dtype=np.float32)
        for a in legal: mask[a] = 1.0
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x, m)
            policy = out[0] if isinstance(out, tuple) else out
            action = int(policy.argmax(dim=1).item())
        return action if action in legal else legal[0]


def play_one(p0_agent, p2_agent, seed):
    rng = random.Random(seed)
    state = ludo_cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    mc = 0
    while not state.is_terminal and mc < MAX_MOVES_PER_GAME:
        cp = int(state.current_player)
        if not state.active_players[cp]:
            n = (cp + 1) % 4
            while not state.active_players[n]: n = (n + 1) % 4
            state.current_player = n
            continue
        if state.current_dice_roll == 0:
            state.current_dice_roll = rng.randint(1, 6)
            if state.current_dice_roll == 6: csix[cp] += 1
            else: csix[cp] = 0
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
    print(f"[h2h] V13.2: {args.v132_ckpt}")

    v135 = V135Agent(args.v135_ckpt, device)
    v132 = V132Agent(args.v132_ckpt, device)

    pairs = max(1, args.games // 2)
    total = pairs * 2
    print(f"[h2h] running {total} games ({pairs} mirrored seed pairs)", flush=True)

    wins = {"V13.5": 0, "V13.2": 0, "draw": 0}
    seat = {"V13.5_P0": [0, 0], "V13.5_P2": [0, 0]}

    t0 = time.time()
    last_report = t0
    for i in range(pairs):
        seed = args.seed + i

        w1 = play_one(v135, v132, seed)
        wins[w1] += 1
        seat["V13.5_P0"][1] += 1
        if w1 == "V13.5": seat["V13.5_P0"][0] += 1

        w2 = play_one(v132, v135, seed)
        wins[w2] += 1
        seat["V13.5_P2"][1] += 1
        if w2 == "V13.5": seat["V13.5_P2"][0] += 1

        if time.time() - last_report > 10:
            done = (i + 1) * 2
            wr = 100 * wins["V13.5"] / done
            elapsed = time.time() - t0
            eta = elapsed * (total - done) / max(done, 1)
            print(f"  [{done:>5d}/{total}] V13.5 {wins['V13.5']:>4d} ({wr:5.1f}%)"
                  f" | V13.2 {wins['V13.2']:>4d} | draws {wins['draw']:>3d}"
                  f" | gpm={done/(elapsed/60):.0f} | eta={eta/60:.1f}min", flush=True)
            last_report = time.time()

    elapsed = time.time() - t0
    total = sum(wins.values())
    print()
    print("=" * 60)
    print(f"H2H: V13.5 (model_latest) vs V13.2 ({total} games, {elapsed:.1f}s)")
    print("=" * 60)
    print(f"  V13.5 wins:   {wins['V13.5']:>5d}  ({100*wins['V13.5']/total:5.2f}%)")
    print(f"  V13.2 wins:   {wins['V13.2']:>5d}  ({100*wins['V13.2']/total:5.2f}%)")
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
        print("  Verdict: V13.5 is STRONGER than V13.2 (>55% with margin).")
    elif p > 0.50:
        print("  Verdict: V13.5 is marginally ahead.")
    elif p > 0.45:
        print("  Verdict: too close to call.")
    else:
        print("  Verdict: V13.5 is WEAKER than V13.2 — investigate.")


if __name__ == "__main__":
    main()
