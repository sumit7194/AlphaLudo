"""Quick round-robin H2H runner that supports both single-frame (V13.2) and
temporal (V13.3) models. Greedy play, alternating who-goes-first via mirrored
seeds for fairness.

Usage:
    ./td_env/bin/python quick_h2h.py --games 1000 [--device mps]
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
import os
import time
from typing import Callable, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS
from experiments.distillation_14ch.model_14ch import MinimalCNN14
from td_ludo.models.v13_3 import V133Temporal


HISTORY_K = 8
MAX_MOVES_PER_GAME = 400


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1000,
                   help="Games per pair (split 50/50 first-player by mirroring)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
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
    """Single-frame V13.2-style (MinimalCNN14)."""
    def __init__(self, name: str, path: str, device):
        self.name = name
        self.device = device
        self.model = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=17)
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[{name}] missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval().to(device)

    def reset(self):
        pass  # stateless

    def observe(self, state):
        pass  # stateless

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


class V133Agent:
    """Temporal V13.3 with per-game K-frame history."""
    def __init__(self, name: str, path: str, device,
                 cnn_blocks=4, cnn_channels=64, d_model=64,
                 nhead=4, n_layers=2, ffn_dim=256):
        self.name = name
        self.device = device
        self.model = V133Temporal(
            history_k=HISTORY_K, in_channels=V17_CHANNELS,
            cnn_blocks=cnn_blocks, cnn_channels=cnn_channels,
            d_model=d_model, nhead=nhead, n_layers=n_layers, ffn_dim=ffn_dim,
        )
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[{name}] missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval().to(device)
        self.history = collections.deque(maxlen=HISTORY_K)

    def reset(self):
        self.history.clear()

    def observe(self, state):
        # Push current frame into history each time it's THIS agent's turn-decision
        cur_frame = encode_state_v17(state)
        self.history.append(cur_frame)

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        # history was pushed in observe(); now build stack
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


# ── 2-player game loop ────────────────────────────────────────────────────
def play_one(agents, agent_for_player, seed):
    """agent_for_player: dict {0: agent, 2: agent}"""
    random.seed(seed)
    np.random.seed(seed)
    state = ludo_cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    mc = 0
    for a in agents:
        a.reset()
    while not state.is_terminal and mc < MAX_MOVES_PER_GAME:
        cp = int(state.current_player)
        if not state.active_players[cp]:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
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
                while not state.active_players[n]:
                    n = (n + 1) % 4
                state.current_player = n
                state.current_dice_roll = 0
                csix[cp] = 0
                continue
        legal = ludo_cpp.get_legal_moves(state)
        if not legal:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            state.current_dice_roll = 0
            continue
        ag = agent_for_player[cp]
        ag.observe(state)
        action = ag.select(state, list(legal))
        state = ludo_cpp.apply_move(state, int(action))
        mc += 1
    if state.is_terminal:
        w = int(ludo_cpp.get_winner(state))
        return w
    return -1  # draw / truncated


def head_to_head(a, b, n_games, seed_base):
    """Returns (a_wins, b_wins, draws). Mirrored seeds: each pair of games
    swaps who plays player 0 vs player 2 with the same RNG seed."""
    a_wins = b_wins = draws = 0
    for i in range(n_games):
        # Even i: a is player 0; odd i: b is player 0 (mirror)
        seed = seed_base + (i // 2)
        if i % 2 == 0:
            mapping = {0: a, 2: b}
        else:
            mapping = {0: b, 2: a}
        winner = play_one([a, b], mapping, seed)
        if winner == -1:
            draws += 1
        else:
            agent_won = mapping[winner]
            if agent_won is a:
                a_wins += 1
            else:
                b_wins += 1
    return a_wins, b_wins, draws


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    BACKUPS = "/Users/sumit/Github/AlphaLudo/checkpoint_backups"

    print(f"[h2h] device={device}, {args.games} games per pair (mirrored seeds)")

    print("[h2h] loading agents...")
    agents = [
        V132Agent("V13.2_latest",
                  f"{BACKUPS}/v132_20260506_015608/model_latest.pt", device),
        V133Agent("V13.3_SL_82pct",
                  f"{BACKUPS}/v133_sl_20260506_203425/model_latest.pt", device),
        V133Agent("V13.3_RL_v2_DEGRADED",
                  f"{BACKUPS}/v133_rl_v2_DEGRADED_20260506_203425/model_latest.pt", device),
        V132Agent("Step1_Distill",
                  f"{BACKUPS}/mcts_step1_distill_v2_20260506_203425/model_latest.pt", device),
    ]

    # Round-robin
    print(f"\n{'pair':<45}  W   L   D   WR%")
    print("-" * 70)
    results = []
    t0 = time.time()
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            t_pair = time.time()
            aw, bw, dr = head_to_head(a, b, args.games, args.seed + i * 100 + j)
            n = aw + bw + dr
            wr_a = 100 * aw / max(1, n)
            wr_b = 100 * bw / max(1, n)
            elapsed = time.time() - t_pair
            print(f"{a.name:>20} vs {b.name:<22}  {aw:>3} {bw:>3} {dr:>3}  "
                  f"{wr_a:5.1f}% / {wr_b:5.1f}%  ({elapsed:.0f}s)")
            results.append((a.name, b.name, aw, bw, dr, wr_a, wr_b))
    total = time.time() - t0
    print(f"\n[h2h] done in {total:.0f}s")

    # Score table: total WR across all pairs (excluding self)
    wr_total = {a.name: [0, 0] for a in agents}  # name → [wins, games]
    for a, b, aw, bw, dr, wra, wrb in results:
        n = aw + bw + dr
        wr_total[a][0] += aw; wr_total[a][1] += n
        wr_total[b][0] += bw; wr_total[b][1] += n
    print("\n=== Overall standings ===")
    scored = sorted(wr_total.items(), key=lambda kv: -kv[1][0] / max(1, kv[1][1]))
    for name, (w, n) in scored:
        wr = 100 * w / max(1, n)
        print(f"  {name:<25}  {w:>4}/{n:>4}  {wr:5.1f}%")


if __name__ == "__main__":
    main()
