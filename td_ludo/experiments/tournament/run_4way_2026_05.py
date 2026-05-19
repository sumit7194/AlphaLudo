"""4-way round-robin tournament runner: V13.2, V13.4, V13.5, V14_scalar.

Why a standalone runner instead of extending experiments/tournament/run.py?

The existing tournament infra (`agents.py` / `run.py`) assumes a stateless
agent interface: `select_move(state, legal, consec_sixes) -> token_id`.
That's incompatible with two of our newer architectures:

  - V13.4 (`V133Temporal`) needs a per-game K=8 history deque updated on
    every decision-state observation, plus rank-permutation-aware
    encoding. Stateful agent, doesn't fit the existing API.
  - V13.5 (`V135Symmetric`) outputs 4 logits indexed by canonical
    POSITION RANK (most-advanced first), not by token-ID. The existing
    select_move expects a token-ID action. Need rank→token-ID mapping
    at action time.

Rather than retrofit a more general API into the existing tournament
code (which would touch many call sites), this script defines a thin
local agent interface with `reset()` + `observe(state)` + `select(state,
legal)` and routes all four architectures through it.

Usage:
    ./td_env/bin/python experiments/tournament/run_4way_2026_05.py \\
        --games 1000 --device mps --output runs/4way_2026_05_<ts>.json

Default checkpoint paths can be overridden via CLI flags.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS
from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric, V18_CHANNELS
from td_ludo.game.encoder_v14_scalar import encode_state_v14_scalar_flat, FLAT_DIM
from td_ludo.game.rank_mapping import (
    state_to_rank_mapping, legal_mask_per_rank, rank_to_token_id,
)
from td_ludo.models.v13_3 import V133Temporal
from td_ludo.models.v13_5 import V135Symmetric, compute_rank_masks
from td_ludo.models.v14_scalar import V14ScalarDeepSets
from td_ludo.models.v12 import AlphaLudoV12
from experiments.distillation_14ch.model_14ch import MinimalCNN14


HISTORY_K = 8
MAX_MOVES_PER_GAME = 400


# ── Agent base ────────────────────────────────────────────────────────────
class Agent:
    name: str = "?"

    def reset(self) -> None:
        """Called at the start of each game. Stateless agents can no-op."""
        pass

    def observe(self, state) -> None:
        """Called BEFORE select() on every decision state where this agent
        is the player to move. Stateful agents (V13.4) update history here."""
        pass

    def select(self, state, legal: List[int]) -> int:
        raise NotImplementedError


# ── V13.2 / MinimalCNN14 ──────────────────────────────────────────────────
class V132Agent(Agent):
    def __init__(self, name: str, path: str, device: torch.device,
                 num_res_blocks: int = 10, num_channels: int = 128):
        self.name = name
        self.device = device
        m = MinimalCNN14(
            num_res_blocks=num_res_blocks, num_channels=num_channels, in_channels=17,
        )
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        miss, unx = m.load_state_dict(sd, strict=False)
        if miss or unx:
            print(f"  [{name}] missing={len(miss)} unexpected={len(unx)}")
        m.eval().to(device)
        self.model = m

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


# ── V13.4 / V133Temporal (per-player history) ─────────────────────────────
class V134Agent(Agent):
    def __init__(self, name: str, path: str, device: torch.device,
                 cnn_blocks: int = 10, cnn_channels: int = 128,
                 d_model: int = 128, n_layers: int = 4,
                 nhead: int = 4, ffn_dim: int = 512):
        self.name = name
        self.device = device
        m = V133Temporal(
            history_k=HISTORY_K, in_channels=V17_CHANNELS,
            cnn_blocks=cnn_blocks, cnn_channels=cnn_channels,
            d_model=d_model, nhead=nhead, n_layers=n_layers, ffn_dim=ffn_dim,
        )
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        miss, unx = m.load_state_dict(sd, strict=False)
        if miss or unx:
            print(f"  [{name}] missing={len(miss)} unexpected={len(unx)}")
        m.eval().to(device)
        self.model = m
        self.history: collections.deque = collections.deque(maxlen=HISTORY_K)

    def reset(self):
        self.history.clear()

    def observe(self, state):
        # Per-player history: this agent only observes its own decision states
        # (matches training, where per-player deques are populated only on
        # that player's decision turns).
        self.history.append(encode_state_v17(state))

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        hist = list(self.history)
        if not hist:  # safety: shouldn't happen because observe() runs first
            hist.append(encode_state_v17(state))
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
            mt = torch.from_numpy(mask).unsqueeze(0).to(self.device, dtype=torch.float32)
            policy, _, _ = self.model(x, mt, h)
            action = int(policy.argmax(dim=1).item())
        return action if action in legal else legal[0]


# ── V13.5 / V135Symmetric (rank-indexed) ──────────────────────────────────
class V135Agent(Agent):
    def __init__(self, name: str, path: str, device: torch.device,
                 num_res_blocks: int = 10, num_channels: int = 128):
        self.name = name
        self.device = device
        m = V135Symmetric(
            num_res_blocks=num_res_blocks, num_channels=num_channels,
            in_channels=V18_CHANNELS,
        )
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        miss, unx = m.load_state_dict(sd, strict=False)
        if miss or unx:
            print(f"  [{name}] missing={len(miss)} unexpected={len(unx)}")
        m.eval().to(device)
        self.model = m

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


# ── V12.2 / AlphaLudoV12 ──────────────────────────────────────────────────
class V122Agent(Agent):
    def __init__(self, name: str, path: str, device: torch.device,
                 num_res_blocks: int = 3, num_channels: int = 128,
                 num_attn_layers: int = 2, num_heads: int = 4, ffn_ratio: int = 4):
        self.name = name
        self.device = device
        m = AlphaLudoV12(
            num_res_blocks=num_res_blocks, num_channels=num_channels,
            num_attn_layers=num_attn_layers, num_heads=num_heads,
            ffn_ratio=ffn_ratio, dropout=0.0, in_channels=33,
        )
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        miss, unx = m.load_state_dict(sd, strict=False)
        if miss or unx:
            print(f"  [{name}] missing={len(miss)} unexpected={len(unx)}")
        m.eval().to(device)
        self.model = m

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        enc = ludo_cpp.encode_state_v11(state)  # 33ch
        x = torch.from_numpy(np.asarray(enc, dtype=np.float32)).unsqueeze(0).to(
            self.device, dtype=torch.float32
        )
        mask = np.zeros(4, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            policy, _, _ = self.model(x, m)
            action = int(policy.argmax(dim=1).item())
        return action if action in legal else legal[0]


# ── V14_scalar / V14ScalarDeepSets ────────────────────────────────────────
class V14ScalarAgent(Agent):
    def __init__(self, name: str, path: str, device: torch.device):
        self.name = name
        self.device = device
        m = V14ScalarDeepSets()
        sd = torch.load(path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        miss, unx = m.load_state_dict(sd, strict=False)
        if miss or unx:
            print(f"  [{name}] missing={len(miss)} unexpected={len(unx)}")
        m.eval().to(device)
        self.model = m

    def select(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        enc = encode_state_v14_scalar_flat(state)  # (FLAT_DIM, 1, 1)
        x = torch.from_numpy(enc).unsqueeze(0).to(self.device, dtype=torch.float32)
        mask = np.zeros(4, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            policy, _, _ = self.model(x, m)
            action = int(policy.argmax(dim=1).item())
        return action if action in legal else legal[0]


# ── Game loop ─────────────────────────────────────────────────────────────
def play_one(agents: List[Agent], agent_for_player: Dict[int, Agent], seed: int) -> int:
    """Returns winner player_id, or -1 for draw / truncation."""
    random.seed(seed); np.random.seed(seed)
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
    return int(ludo_cpp.get_winner(state)) if state.is_terminal else -1


def head_to_head(a: Agent, b: Agent, n_games: int, seed_base: int,
                 progress_interval: int = 100) -> Tuple[int, int, int]:
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
        if (i + 1) % progress_interval == 0:
            n = aw + bw + dr
            print(f"    [{i+1}/{n_games}] {a.name} {100*aw/n:.1f}% / "
                  f"{b.name} {100*bw/n:.1f}%", flush=True)
    return aw, bw, dr


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1000,
                   help="Games per pair (mirrored seeds: 50/50 first-player split)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None,
                   help="Optional JSON output path for results")

    # Default checkpoint paths (latest known good)
    p.add_argument("--v132", default="/Users/sumit/Github/AlphaLudo/checkpoint_backups/v132_20260506_015608/model_latest.pt")
    p.add_argument("--v134", default="/Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/v134/model_latest.pt",
                   help="V13.4 latest weight (post-RL chain end)")
    p.add_argument("--v135", default="/Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/v135_full/model_latest.pt",
                   help="V13.5 SL completion weight")
    p.add_argument("--v14scalar", default="/Users/sumit/Github/AlphaLudo/checkpoint_backups/v14_scalar_rl_20260507_081221/model_latest.pt",
                   help="V14_scalar RL latest weight (paused 2026-05-07)")
    p.add_argument("--v12_2", default="/Users/sumit/Github/AlphaLudo/td_ludo/play/model_weights/v12_2/model_latest.pt",
                   help="V12.2 latest weight (older lineage anchor)")
    p.add_argument("--v135_rl_vm", default="/Users/sumit/Github/AlphaLudo/checkpoint_backups/v135_rl_vm_20260508_173708/model_latest.pt",
                   help="V13.5 RL latest from VM (9.5M states of self-play REINFORCE on top of SL)")

    p.add_argument("--skip", default="",
                   help="Comma-separated list of agents to skip (v132,v134,v135,v14scalar,v12_2,v135_rl_vm)")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name in ("cuda", "cpu", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    print("=" * 70)
    print(f"4-way tournament — {args.games} games per pair, mirrored seeds, greedy")
    print(f"Device: {device}")
    print("=" * 70)

    agents: List[Agent] = []
    print("\n[loading]")
    if "v132" not in skip:
        agents.append(V132Agent("V13.2_latest", args.v132, device))
        print(f"  V13.2_latest    {args.v132}")
    if "v134" not in skip:
        agents.append(V134Agent("V13.4_RLfinal", args.v134, device))
        print(f"  V13.4_RLfinal   {args.v134}")
    if "v135" not in skip:
        agents.append(V135Agent("V13.5_SL", args.v135, device))
        print(f"  V13.5_SL        {args.v135}")
    if "v14scalar" not in skip:
        agents.append(V14ScalarAgent("V14_scalar_RL", args.v14scalar, device))
        print(f"  V14_scalar_RL   {args.v14scalar}")
    if "v12_2" not in skip:
        agents.append(V122Agent("V12.2_latest", args.v12_2, device))
        print(f"  V12.2_latest    {args.v12_2}")
    if "v135_rl_vm" not in skip:
        agents.append(V135Agent("V13.5_RL_VM", args.v135_rl_vm, device))
        print(f"  V13.5_RL_VM     {args.v135_rl_vm}")

    # Round-robin
    print(f"\n{'pair':<48}    W   L   D   WR%")
    print("-" * 75)
    results = []
    t_total = time.time()
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            t_pair = time.time()
            seed_base = args.seed + i * 1000 + j
            aw, bw, dr = head_to_head(a, b, args.games, seed_base, progress_interval=200)
            n = max(1, aw + bw + dr)
            wr_a = 100 * aw / n
            wr_b = 100 * bw / n
            elapsed = time.time() - t_pair
            print(f"  {a.name:>18} vs {b.name:<22}  {aw:>4} {bw:>4} {dr:>3}  "
                  f"{wr_a:5.1f}% / {wr_b:5.1f}%  ({elapsed:.0f}s)")
            results.append({
                "a": a.name, "b": b.name, "a_wins": aw, "b_wins": bw, "draws": dr,
                "wr_a": wr_a, "wr_b": wr_b, "elapsed_sec": elapsed,
            })
    total_sec = time.time() - t_total
    print(f"\n[done] tournament took {total_sec:.0f}s")

    # Standings (W / GAMES / WR)
    standings: Dict[str, Dict[str, int]] = {a.name: {"wins": 0, "games": 0} for a in agents}
    for r in results:
        n = r["a_wins"] + r["b_wins"] + r["draws"]
        standings[r["a"]]["wins"] += r["a_wins"]
        standings[r["a"]]["games"] += n
        standings[r["b"]]["wins"] += r["b_wins"]
        standings[r["b"]]["games"] += n
    rows = []
    for k, v in standings.items():
        wr = 100 * v["wins"] / max(1, v["games"])
        rows.append({"name": k, "wins": v["wins"], "games": v["games"], "wr": wr})
    rows.sort(key=lambda r: -r["wr"])

    print("\n=== Overall standings ===")
    print(f"  {'Rank':<5} {'Agent':<20} {'Wins':>5}/{'Games':<5}  WR")
    for i, r in enumerate(rows):
        print(f"  {i+1:<5} {r['name']:<20} {r['wins']:>5}/{r['games']:<5}  {r['wr']:5.1f}%")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "games_per_pair": args.games,
                "total_sec": total_sec,
                "device": str(device),
                "ts": int(time.time()),
                "agents": [{"name": a.name} for a in agents],
                "pairs": results,
                "standings": rows,
            }, f, indent=2)
        print(f"\n[output] wrote {args.output}")


if __name__ == "__main__":
    main()
