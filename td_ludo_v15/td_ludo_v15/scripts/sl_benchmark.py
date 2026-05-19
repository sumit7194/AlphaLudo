"""Real V13.5 → V15 SL distillation benchmark.

Unlike `poc_learn.py` (which uses synthetic targets), this script does the
honest cross-arch distillation that real training will use later:
    1. Load V13.5_SL teacher from a checkpoint backup.
    2. Generate states via random play on V15's engine.
    3. For each state: encode via both encoders, run V13.5 teacher → rank
       policy, project rank → per-cell target.
    4. Train V15 student to minimize KL(student || target).
    5. Report throughput (states/sec, games/min) + loss curve.

This is the gate that proves V15 can actually distill from V13.5 BEFORE
we kick off real training. ~30-60 seconds of runtime on Mac MPS.

Usage:
    python -m td_ludo_v15.scripts.sl_benchmark
    python -m td_ludo_v15.scripts.sl_benchmark --device mps --states 1000 --steps 200
    python -m td_ludo_v15.scripts.sl_benchmark --teacher /path/to/v135.pt
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

# ─── Bridge to the legacy code for V13.5 model + V18 encoder ───────────────
_LEGACY_ROOT = Path("/Users/sumit/Github/AlphaLudo/td_ludo")
if str(_LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGACY_ROOT))

from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric  # type: ignore
from td_ludo.game.rank_mapping import (  # type: ignore
    state_to_rank_mapping,
    legal_mask_per_rank,
)
from td_ludo.models.v13_5 import V135Symmetric, compute_rank_masks  # type: ignore

import td_ludo_cpp as _legacy_cpp
import td_ludo_v15_cpp as _v15_cpp

# Legacy `td_ludo_cpp` doesn't expose BASE_POS / HOME_POS at module level;
# V15 does. The values are identical (game.h constants); reuse V15's.
_BASE_POS: int = _v15_cpp.BASE_POS
_HOME_POS: int = _v15_cpp.HOME_POS

from ..game.cells import NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov
from ..game.encoder import encode_history
from ..game.state import V15GameWrapper
from ..models.v15 import V15GraphTransformer


def _probe_v135_arch(state_dict: dict) -> dict:
    """Auto-detect num_res_blocks + num_channels from a V13.5 state_dict."""
    conv_w = state_dict.get("conv_input.weight")
    if conv_w is None:
        raise RuntimeError("state_dict missing conv_input.weight — not a V13.5 checkpoint")
    num_channels = int(conv_w.shape[0])
    # Count res_blocks by looking for keys like "res_blocks.{N}.*"
    indices = set()
    for k in state_dict.keys():
        if k.startswith("res_blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                indices.add(int(parts[1]))
    num_res_blocks = max(indices) + 1 if indices else 0
    return {"num_res_blocks": num_res_blocks, "num_channels": num_channels}


def _load_v135_teacher(path: Path, device: torch.device) -> V135Symmetric:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    arch = _probe_v135_arch(sd)
    print(f"[teacher] auto-detected arch: {arch}", flush=True)
    model = V135Symmetric(**arch)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in model.parameters())
    print(f"[teacher] V13.5 params: {n:,}", flush=True)
    return model


# ──────────────────────────────────────────────────────────────────────────
# State collection (timed)
# ──────────────────────────────────────────────────────────────────────────

def collect_states(n_states: int, seed: int = 0):
    """Random-play decision-states from the LEGACY engine.

    Both engines are parity-tested (3000-game step-by-step equivalence), so
    using legacy here is safe and lets V13.5's V18 encoder (which expects
    `td_ludo_cpp.GameState`) work directly. V15's encoder is pure Python
    and works on legacy state objects too — the property API is identical.

    Returns list of dicts with:
        state: legacy GameState at this decision
        history: list of past legacy GameStates (7 frames, oldest first;
                 None for missing slots at game start)
        legal_cells: list of (row, col) legal source cells in V15 POV
        legal_token_ids: list of legal token-id slots
    """
    import collections as _coll
    rng = random.Random(seed)
    samples = []
    games_played = 0
    games_finished = 0
    t0 = time.time()
    while len(samples) < n_states:
        state = _legacy_cpp.create_initial_state_2p()
        history: _coll.deque = _coll.deque(maxlen=7)
        csix = [0, 0, 0, 0]
        moves = 0
        while not state.is_terminal and moves < 400 and len(samples) < n_states:
            cp = int(state.current_player)
            d = rng.randint(1, 6)
            if d == 6:
                csix[cp] += 1
                if csix[cp] >= 3:
                    csix[cp] = 0
                    nxt = (cp + 1) % 4
                    while not state.active_players[nxt]:
                        nxt = (nxt + 1) % 4
                    state.current_player = nxt
                    state.current_dice_roll = 0
                    continue
            else:
                csix[cp] = 0
            state.current_dice_roll = d
            legal_token_ids = _legacy_cpp.get_legal_moves(state)
            if not legal_token_ids:
                nxt = (cp + 1) % 4
                while not state.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                state.current_player = nxt
                state.current_dice_roll = 0
                continue
            # Compute V15-style legal source cells from legal token-ids
            legal_cells_set = set()
            for tid in legal_token_ids:
                pos = int(state.player_positions[cp][tid])
                if pos == _BASE_POS:
                    r, c = position_to_cell_in_pov(_BASE_POS, cp, cp)
                else:
                    r, c = position_to_cell_in_pov(pos, cp, cp)
                legal_cells_set.add((r, c))
            # Snapshot pre-decision state into history-padded list of 8
            past = list(history)
            pad = 7 - len(past)
            v15_history = [None] * pad + past + [state]
            samples.append({
                "state": state,
                "history": v15_history,
                "legal_cells": list(legal_cells_set),
                "legal_token_ids": list(legal_token_ids),
            })
            # Push current state into history before move
            history.append(state)
            slot = rng.choice(legal_token_ids)
            state = _legacy_cpp.apply_move(state, slot)
            moves += 1
        games_played += 1
        if state.is_terminal:
            games_finished += 1
    elapsed = time.time() - t0
    return samples, elapsed, games_played, games_finished


# ──────────────────────────────────────────────────────────────────────────
# Cross-arch projection: V13.5 rank policy → V15 per-cell target
# ──────────────────────────────────────────────────────────────────────────

def build_v15_target_from_v135(
    v15_state,
    legal_token_ids: List[int],
    v135_rank_policy: torch.Tensor,  # (4,) — P(rank_k)
    rank_token_ids: List[List[int]],
) -> np.ndarray:
    """Project V13.5 per-rank policy onto V15's per-cell space.

    Method:
        1. Expand rank_policy → per-token-id probability:
           For each rank r with prob P[r], split P[r] equally over the
           token-ids that share rank r (i.e., tokens at same position).
        2. For each token-id, find its current cell in V15 POV.
        3. Sum probabilities of token-ids landing on the same cell.
        4. Normalize to a clean probability distribution over the 225 cells.

    Returns: (225,) float32 — V15's policy target.
    """
    cp = int(v15_state.current_player)
    per_token = np.zeros(4, dtype=np.float32)
    for rank, tids in enumerate(rank_token_ids):
        n = len(tids)
        if n == 0:
            continue
        share = float(v135_rank_policy[rank]) / n
        for t in tids:
            per_token[t] += share

    target = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
    # Only consider token-ids that are LEGAL — illegal tokens shouldn't get
    # any policy mass even if V13.5 spuriously assigned some.
    for t in legal_token_ids:
        pos = int(v15_state.player_positions[cp][t])
        if pos == _BASE_POS:
            r, c = position_to_cell_in_pov(_BASE_POS, cp, cp)
        else:
            r, c = position_to_cell_in_pov(pos, cp, cp)
        target[cell_to_index(r, c)] += per_token[t]

    s = target.sum()
    if s > 1e-6:
        target /= s
    return target


# ──────────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=str,
                   default="/Users/sumit/Github/AlphaLudo/checkpoint_backups/v135_full_20260508_071609/model_sl.pt")
    p.add_argument("--states", type=int, default=500, help="number of states to collect")
    p.add_argument("--steps", type=int, default=200, help="training steps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    print(f"[bench] device={device}", flush=True)
    torch.manual_seed(args.seed)

    # ─── 1. Load V13.5 teacher ─────────────────────────────────────────
    teacher = _load_v135_teacher(Path(args.teacher), device)

    # ─── 2. Collect states ─────────────────────────────────────────────
    print(f"[bench] collecting {args.states} states...", flush=True)
    samples, gen_elapsed, games_played, games_finished = collect_states(
        args.states, seed=args.seed
    )
    states_per_sec = len(samples) / max(1e-6, gen_elapsed)
    games_per_min = games_played * 60 / max(1e-6, gen_elapsed)
    games_finished_per_min = games_finished * 60 / max(1e-6, gen_elapsed)
    print(f"[bench]   collected {len(samples)} states in {gen_elapsed:.2f}s "
          f"({states_per_sec:.0f} states/s, {games_per_min:.0f} games/min started, "
          f"{games_finished_per_min:.0f} games/min finished)", flush=True)

    # ─── 3. Compute targets via V13.5 teacher (timed) ─────────────────
    print(f"[bench] running V13.5 teacher on {len(samples)} states...", flush=True)
    t0 = time.time()
    student_inputs = []   # V15 inputs: (8, 15, 15, 3)
    student_masks = []    # legal-cell masks: (225,)
    student_targets = []  # projected target distributions: (225,)

    for sample in samples:
        state = sample["state"]
        cp = int(state.current_player)
        # V13.5 inputs (uses legacy GameState directly)
        v18_x = encode_state_v18_symmetric(state).astype(np.float32)  # (13, 15, 15)
        rmasks = compute_rank_masks(state).astype(np.float32)         # (4, 15, 15)
        pp = state.player_positions[cp]
        _, rank_token_ids = state_to_rank_mapping(pp)
        rank_legal = legal_mask_per_rank(sample["legal_token_ids"], rank_token_ids)
        # Teacher forward
        with torch.no_grad():
            x_t = torch.from_numpy(v18_x).unsqueeze(0).to(device)
            rm_t = torch.from_numpy(rmasks).unsqueeze(0).to(device)
            lm_t = torch.from_numpy(rank_legal.astype(np.float32)).unsqueeze(0).to(device)
            policy_4, _, _, _ = teacher(x_t, rm_t, lm_t)
            policy_4 = policy_4.squeeze(0).cpu().numpy()  # (4,)
        # Project to V15 target
        v15_target = build_v15_target_from_v135(
            state, sample["legal_token_ids"], policy_4, rank_token_ids
        )
        # V15 inputs — encode the history of legacy GameStates with V15's encoder.
        # V15 encoder is pure Python and reads `.player_positions`, `.scores`,
        # `.current_player`, `.current_dice_roll`, `.active_players` — all present
        # on the legacy GameState object too. So this works without conversion.
        v15_x = encode_history(sample["history"], pov_player=cp)  # (8,15,15,3)
        legal_mask = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
        for (r, c) in sample["legal_cells"]:
            legal_mask[cell_to_index(r, c)] = 1.0

        student_inputs.append(v15_x)
        student_masks.append(legal_mask)
        student_targets.append(v15_target)

    teacher_elapsed = time.time() - t0
    print(f"[bench]   teacher inference + projection: {teacher_elapsed:.2f}s "
          f"({len(samples)/teacher_elapsed:.0f} states/s)", flush=True)

    # ─── 4. Stack into tensors ─────────────────────────────────────────
    X = torch.from_numpy(np.stack(student_inputs).astype(np.float32)).to(device)
    M = torch.from_numpy(np.stack(student_masks)).to(device)
    T = torch.from_numpy(np.stack(student_targets)).to(device)
    print(f"[bench] dataset: X{tuple(X.shape)} M{tuple(M.shape)} T{tuple(T.shape)}",
          flush=True)

    # ─── 5. Build student + train ─────────────────────────────────────
    student = V15GraphTransformer().to(device)
    print(f"[bench] V15 student params: {student.count_parameters():,}", flush=True)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    student.train()

    N = X.shape[0]
    losses = []
    train_t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch_size,), device=device)
        xb = X[idx]
        mb = M[idx]
        tb = T[idx]
        policy, _ = student(xb, mb)
        log_p = torch.log(policy + 1e-9)
        loss = -(tb * log_p).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(f"[bench] step {step:>4}: loss={loss.item():.4f}", flush=True)
    train_elapsed = time.time() - train_t0
    train_steps_per_sec = args.steps / max(1e-6, train_elapsed)
    train_samples_per_sec = args.steps * args.batch_size / max(1e-6, train_elapsed)
    print(f"[bench] training: {args.steps} steps in {train_elapsed:.2f}s "
          f"({train_steps_per_sec:.1f} steps/s, {train_samples_per_sec:.0f} samples/s)",
          flush=True)

    # ─── 6. Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("V13.5 → V15 SL DISTILLATION BENCHMARK")
    print("=" * 60)
    print(f"  device:                   {device}")
    print(f"  states collected:         {len(samples):,}")
    print(f"  games started/finished:   {games_played} / {games_finished}")
    print(f"  state generation:         {states_per_sec:.0f} states/s "
          f"({games_per_min:.0f} games started/min, "
          f"{games_finished_per_min:.0f} games finished/min)")
    print(f"  teacher inference:        {len(samples)/teacher_elapsed:.0f} states/s")
    print(f"  training throughput:      {train_steps_per_sec:.1f} steps/s "
          f"({train_samples_per_sec:.0f} samples/s, batch={args.batch_size})")
    print()
    init_win = sum(losses[: max(1, len(losses) // 10)]) / max(1, len(losses) // 10)
    final_win = sum(losses[-max(1, len(losses) // 10):]) / max(1, len(losses) // 10)
    print(f"  initial-window avg loss:  {init_win:.4f}")
    print(f"  final-window avg loss:    {final_win:.4f}")
    drop = (init_win - final_win) / max(1e-8, init_win)
    print(f"  loss drop:                {drop:.1%}")
    if drop >= 0.30:
        print(f"  VERDICT: V15 IS LEARNING from V13.5 ✓ (drop ≥ 30%)")
    elif drop >= 0.10:
        print(f"  VERDICT: V15 is learning weakly (drop {drop:.1%}; ≥10% but <30%)")
    else:
        print(f"  VERDICT: ⚠ V15 may not be learning (drop {drop:.1%} < 10%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
