"""H2H: V13.5 (best) vs V12.2 (production). Greedy 1000-game tournament.

Mirrors who-goes-first across seed pairs for fairness (each seed s plays once
with V135 starting and once with V122 starting).

Usage:
    td_env/bin/python h2h_v135_vs_v122.py --games 1000
    td_env/bin/python h2h_v135_vs_v122.py --games 1000 --device mps
    td_env/bin/python h2h_v135_vs_v122.py --games 1000 --v135-ckpt play/model_weights/v13_5/model_latest.pt
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
from td_ludo.models.v12 import AlphaLudoV12
from td_ludo.game.encoder_v18_production import encode_state_v18_production


MAX_MOVES_PER_GAME = 400


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--v135-ckpt", default="play/model_weights/v13_5/model_best.pt")
    p.add_argument("--v122-ckpt", default="play/model_weights/v12_2/model_latest.pt")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_device(name):
    if name in ("cpu", "cuda", "mps"):
        return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _load_state_dict(path, device):
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
        self.model.load_state_dict(_load_state_dict(ckpt, device))
        self.model.to(device).eval()

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        enc = np.array(encode_state_v18_production(state), dtype=np.float32)
        x = torch.from_numpy(enc).unsqueeze(0).to(self.device)
        mask = np.zeros(4, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, _, _, _ = self.model(x, m)
            action = int(policy.argmax(dim=1).item())
        return action if action in legal else legal[0]


class V122Agent:
    name = "V12.2"
    def __init__(self, ckpt, device):
        self.device = device
        self.model = AlphaLudoV12(
            num_res_blocks=3, num_channels=128,
            num_attn_layers=2, num_heads=4, ffn_ratio=4,
            dropout=0.0, in_channels=33,
        )
        self.model.load_state_dict(_load_state_dict(ckpt, device))
        self.model.to(device).eval()

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        enc = np.array(ludo_cpp.encode_state_v11(state), dtype=np.float32)
        x = torch.from_numpy(enc).unsqueeze(0).to(self.device)
        mask = np.zeros(4, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x, m)
            policy = out[0] if isinstance(out, tuple) else out
            action = int(policy.argmax(dim=1).item())
        return action if action in legal else legal[0]


def play_one_game(p0_agent, p2_agent, seed):
    """Play one 2-player game. p0_agent plays as P0, p2_agent plays as P2.
    Returns winner_agent_name or 'draw'. Mirrors evaluate_v11.py game loop."""
    rng = random.Random(seed)
    state = ludo_cpp.create_initial_state_2p()
    consecutive_sixes = [0, 0, 0, 0]
    move_count = 0

    while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
        cp = state.current_player

        # Skip inactive players (in 2P only P0 and P2 are active)
        if not state.active_players[cp]:
            next_p = (cp + 1) % 4
            while not state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            state.current_player = next_p
            continue

        # Roll dice if not already rolled
        if state.current_dice_roll == 0:
            state.current_dice_roll = rng.randint(1, 6)
            if state.current_dice_roll == 6:
                consecutive_sixes[cp] += 1
            else:
                consecutive_sixes[cp] = 0

            # 3 sixes in a row → forfeit turn
            if consecutive_sixes[cp] >= 3:
                next_p = (cp + 1) % 4
                while not state.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                state.current_player = next_p
                state.current_dice_roll = 0
                consecutive_sixes[cp] = 0
                continue

        legal = ludo_cpp.get_legal_moves(state)

        # No legal moves → pass turn
        if not legal:
            next_p = (state.current_player + 1) % 4
            while not state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            state.current_player = next_p
            state.current_dice_roll = 0
            continue

        # Pick agent for this seat
        agent = p0_agent if cp == 0 else p2_agent
        action = agent.select(state, legal)
        state = ludo_cpp.apply_move(state, action)
        move_count += 1

    if not state.is_terminal:
        return "draw"  # timeout
    winner = ludo_cpp.get_winner(state)
    if winner == 0:
        return p0_agent.name
    if winner == 2:
        return p2_agent.name
    return "draw"


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"[h2h] device={device}")
    print(f"[h2h] V13.5 ckpt: {args.v135_ckpt}")
    print(f"[h2h] V12.2 ckpt: {args.v122_ckpt}")

    v135 = V135Agent(args.v135_ckpt, device)
    v122 = V122Agent(args.v122_ckpt, device)

    pairs = args.games // 2
    if pairs < 1:
        pairs = 1
    total_games = pairs * 2
    print(f"[h2h] running {total_games} games ({pairs} mirrored seed pairs)")

    wins = {"V13.5": 0, "V12.2": 0, "draw": 0}
    seat_split = {"V13.5_as_P0": [0, 0], "V13.5_as_P2": [0, 0]}  # [wins, games]

    t0 = time.time()
    last_report = t0
    for i in range(pairs):
        seed = args.seed + i

        # Seed s: V13.5 = P0, V12.2 = P2
        w1 = play_one_game(v135, v122, seed)
        wins[w1] += 1
        seat_split["V13.5_as_P0"][1] += 1
        if w1 == "V13.5":
            seat_split["V13.5_as_P0"][0] += 1

        # Mirror: V12.2 = P0, V13.5 = P2 (same seed → same dice rolls)
        w2 = play_one_game(v122, v135, seed)
        wins[w2] += 1
        seat_split["V13.5_as_P2"][1] += 1
        if w2 == "V13.5":
            seat_split["V13.5_as_P2"][0] += 1

        # Progress report every 5s
        if time.time() - last_report > 5:
            done = (i + 1) * 2
            v135_wr = 100 * wins["V13.5"] / max(done, 1)
            print(f"  [{done:4d}/{total_games}] V13.5 wins {wins['V13.5']:>4d} "
                  f"({v135_wr:5.1f}%) | V12.2 {wins['V12.2']:>4d} | "
                  f"draws {wins['draw']:>3d} | "
                  f"GPM={done/((time.time()-t0)/60):.0f}")
            last_report = time.time()

    elapsed = time.time() - t0
    total = sum(wins.values())
    print()
    print("=" * 60)
    print(f"H2H result: V13.5 vs V12.2  ({total} games, {elapsed:.1f}s)")
    print("=" * 60)
    print(f"  V13.5 wins:   {wins['V13.5']:>5d}  ({100*wins['V13.5']/total:5.2f}%)")
    print(f"  V12.2 wins:   {wins['V12.2']:>5d}  ({100*wins['V12.2']/total:5.2f}%)")
    print(f"  Draws:        {wins['draw']:>5d}  ({100*wins['draw']/total:5.2f}%)")
    print()
    p0 = seat_split["V13.5_as_P0"]
    p2 = seat_split["V13.5_as_P2"]
    print(f"  V13.5 as P0:  {p0[0]:>4d}/{p0[1]} ({100*p0[0]/max(p0[1],1):5.1f}%)")
    print(f"  V13.5 as P2:  {p2[0]:>4d}/{p2[1]} ({100*p2[0]/max(p2[1],1):5.1f}%)")
    # Std error of net WR
    p = wins["V13.5"] / total
    se = (p * (1 - p) / total) ** 0.5
    print()
    print(f"  Net WR: {100*p:.2f}% ± {100*1.96*se:.2f}pp (95% CI)")
    if p > 0.55:
        print("  Verdict: V13.5 is STRONGER than V12.2 (>55% with margin).")
    elif p > 0.50:
        print("  Verdict: V13.5 is marginally ahead.")
    elif p > 0.45:
        print("  Verdict: too close to call (within margin of V12.2).")
    else:
        print("  Verdict: V13.5 is WEAKER than V12.2 — investigate.")


if __name__ == "__main__":
    main()
