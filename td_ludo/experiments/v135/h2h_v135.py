"""H2H runner for V13.5 POC.

Compares V13.2_latest vs V135_SL_POC head-to-head, greedy, mirrored seeds.
This is the actual decision gate — bot evals saturated at 80-82% across all
post-V13.2 architectures and cannot distinguish them.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17
from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric, V18_CHANNELS
from td_ludo.game.rank_mapping import (
    state_to_rank_mapping, legal_mask_per_rank, rank_to_token_id,
)
from td_ludo.models.v13_5 import V135Symmetric, compute_rank_masks
from experiments.distillation_14ch.model_14ch import MinimalCNN14


MAX_MOVES_PER_GAME = 400


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--v132", required=True)
    p.add_argument("--v135", required=True)
    p.add_argument("--num-res-blocks", type=int, default=6)
    p.add_argument("--num-channels", type=int, default=96)
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


class V132Agent:
    def __init__(self, name, path, device):
        self.name = name
        self.device = device
        m = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=17)
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        m.load_state_dict(sd, strict=False)
        m.eval().to(device)
        self.model = m

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


class V135Agent:
    def __init__(self, name, path, device, num_res_blocks=6, num_channels=96):
        self.name = name
        self.device = device
        m = V135Symmetric(num_res_blocks=num_res_blocks, num_channels=num_channels,
                          in_channels=V18_CHANNELS)
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        m.load_state_dict(sd, strict=False)
        m.eval().to(device)
        self.model = m

    def reset(self): pass
    def observe(self, state): pass

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        cp = int(state.current_player)
        pp = state.player_positions[cp]
        _, rank_tokens = state_to_rank_mapping(pp)
        rank_legal = legal_mask_per_rank(legal, rank_tokens)
        enc = encode_state_v18_symmetric(state)
        rm = compute_rank_masks(state)
        with torch.no_grad():
            x = torch.from_numpy(enc).unsqueeze(0).to(self.device, dtype=torch.float32)
            rmt = torch.from_numpy(rm).unsqueeze(0).to(self.device, dtype=torch.float32)
            lmt = torch.from_numpy(rank_legal).unsqueeze(0).to(self.device, dtype=torch.float32)
            logits = self.model.forward_policy_only(x, rmt, lmt)
            rank = int(logits.argmax(dim=1).item())
        action = rank_to_token_id(rank, legal, rank_tokens)
        return action if action in legal else legal[0]


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


def head_to_head(a, b, n_games, seed_base):
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
        if (i + 1) % 50 == 0:
            n = aw + bw + dr
            print(f"  [{i+1}/{n_games}] {a.name} {100*aw/n:.1f}% / {b.name} {100*bw/n:.1f}%", flush=True)
    return aw, bw, dr


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)

    print(f"[h2h_v135] device={device}, {args.games} games, mirrored seeds, greedy")
    print(f"[h2h_v135] V13.2: {args.v132}")
    print(f"[h2h_v135] V13.5: {args.v135}")

    a = V132Agent("V13.2_latest", args.v132, device)
    b = V135Agent("V13.5_SL_POC", args.v135, device,
                  num_res_blocks=args.num_res_blocks, num_channels=args.num_channels)

    print()
    t0 = time.time()
    aw, bw, dr = head_to_head(a, b, args.games, args.seed)
    elapsed = time.time() - t0
    n = aw + bw + dr
    wr_a = 100 * aw / max(1, n)
    wr_b = 100 * bw / max(1, n)
    se = 100 * (wr_a / 100 * (1 - wr_a / 100) / max(1, n)) ** 0.5
    print(f"\n=== H2H RESULT ({n} games, {elapsed:.0f}s) ===")
    print(f"  {a.name:>20}: {aw:>4}/{n} = {wr_a:5.1f}%  (SE ±{se:.1f})")
    print(f"  {b.name:>20}: {bw:>4}/{n} = {wr_b:5.1f}%  (SE ±{se:.1f})")
    print(f"  {'draws':>20}: {dr}")
    delta = wr_b - wr_a
    z = abs(delta) / (2 * se) if se > 0 else 0
    print(f"\n  V13.5 vs V13.2 delta: {delta:+.1f}pp (z = {z:.2f})")
    if delta > 2 * se:
        print(f"  V13.5 BEATS V13.2 — promising, recommend full-size run on VM.")
    elif delta < -2 * se:
        print(f"  V13.5 LOSES to V13.2 — symmetric arch alone doesn't help; reconsider.")
    else:
        print(f"  V13.5 STATISTICALLY TIED with V13.2 — not a clear win, but not a loss either.")


if __name__ == "__main__":
    main()
