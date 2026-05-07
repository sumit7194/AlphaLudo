"""CKA (Centered Kernel Alignment) analysis on the distilled MinimalCNN14
student vs V12.2 production.

Hypothesis (from training_journal.md, Exp 25): with rich V12.2 inputs,
deep layers are redundant — V12 mech-interp showed CKA > 0.95 across all
4 ResBlocks. With minimal 14ch inputs, the student should re-engage
depth and show DIVERGENT CKA across blocks 1-7 (each block doing
distinct work).

Method:
  - Generate N mid-game states with a deterministic seed.
  - Forward through each model with hooks capturing per-ResBlock output.
  - Flatten activations to (N, C*H*W).
  - Compute pairwise linear CKA via the Hilbert-Schmidt formulation:
        K_X = X X^T  (centered)
        K_Y = Y Y^T  (centered)
        CKA(X, Y) = (K_X * K_Y).sum() / sqrt((K_X * K_X).sum() * (K_Y * K_Y).sum())
    This avoids materialising the (d × d) covariance for activations
    with d = 128 * 15 * 15 = 28,800.

Output: per-model heatmap of pairwise CKA, plus a one-line summary
(min off-diagonal CKA, mean off-diagonal, fraction of pairs > 0.95).

Run from td_ludo/ root:
  td_env/bin/python -m experiments.distillation_14ch.cka_analysis
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, List

import numpy as np
import torch

import td_ludo_cpp as cpp


N_STATES_DEFAULT = 2000


# ---------------------------------------------------------------------------
#  State generation — diverse mid-game positions
# ---------------------------------------------------------------------------

def make_mid_game_states(n: int, seed: int = 1234) -> List:
    """Generate `n` mid-game GameState objects with diverse dice + token
    distributions. Each is a 2-player Ludo game rolled forward 10–60
    moves with random play."""
    rng = np.random.RandomState(seed)
    states = []
    while len(states) < n:
        g = cpp.create_initial_state_2p()
        n_steps = int(rng.randint(10, 60))
        for _ in range(n_steps):
            if cpp.get_winner(g) >= 0:
                break
            if g.current_dice_roll == 0:
                g.current_dice_roll = int(rng.randint(1, 7))
            moves = cpp.get_legal_moves(g)
            if not moves:
                # advance to next active player
                cp = g.current_player
                nxt = (cp + 1) % 4
                while not g.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                g.current_player = nxt
                g.current_dice_roll = 0
                continue
            g = cpp.apply_move(g, int(rng.choice(moves)))
        if cpp.get_winner(g) >= 0:
            continue  # terminal state, no decision to make
        if g.current_dice_roll == 0:
            g.current_dice_roll = int(rng.randint(1, 7))
        if cpp.get_legal_moves(g):
            states.append(g)
    return states


# ---------------------------------------------------------------------------
#  Forward + per-block activation capture
# ---------------------------------------------------------------------------

def collect_block_activations(model, encoder_fn: Callable, states, device,
                              chunk: int = 256) -> List[np.ndarray]:
    """Run forward on `states`, return list of per-block activations
    [(N, C*H*W)] — one entry per ResBlock."""
    res_blocks = list(model.res_blocks)
    n_blocks = len(res_blocks)
    print(f"  collecting activations: {n_blocks} blocks × {len(states)} states "
          f"({chunk}-chunked)")

    acts_per_block = [[] for _ in range(n_blocks)]

    handles = []
    captured = [None] * n_blocks
    def make_hook(idx):
        def _hook(_module, _input, output):
            captured[idx] = output.detach()
        return _hook
    for i, blk in enumerate(res_blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), chunk):
            batch = states[start:start + chunk]
            encs = np.stack([encoder_fn(g) for g in batch])
            x = torch.from_numpy(encs).to(device, dtype=torch.float32)
            # forward — we only need the side effect of the hooks
            try:
                model(x)
            except TypeError:
                # some models require legal_mask
                mask = torch.ones(x.shape[0], 4, device=device)
                model(x, mask)
            for i in range(n_blocks):
                a = captured[i]
                acts_per_block[i].append(a.flatten(start_dim=1).cpu().numpy())

    for h in handles:
        h.remove()

    return [np.concatenate(parts, axis=0) for parts in acts_per_block]


# ---------------------------------------------------------------------------
#  Linear CKA — kernel form, avoids (d × d) materialisation
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA via centered Gram matrices.

    K = X X^T (centered) — (N, N).
    CKA = (K_X * K_Y).sum() / sqrt((K_X * K_X).sum() * (K_Y * K_Y).sum())
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    K_X = Xc @ Xc.T
    K_Y = Yc @ Yc.T
    num = float((K_X * K_Y).sum())
    den_x = float((K_X * K_X).sum())
    den_y = float((K_Y * K_Y).sum())
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / np.sqrt(den_x * den_y)


def cka_matrix(activations: List[np.ndarray]) -> np.ndarray:
    """Pairwise linear CKA between every pair of layer activations.
    Returns symmetric (n_blocks, n_blocks) matrix with diagonal = 1.0."""
    n = len(activations)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            v = linear_cka(activations[i], activations[j])
            M[i, j] = v
            M[j, i] = v
    return M


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------

def print_matrix(M: np.ndarray, label: str):
    n = M.shape[0]
    print(f"\n  {label} CKA matrix ({n}×{n}, diag=1):")
    header = "       " + " ".join(f"  blk{j}" for j in range(n))
    print(header)
    for i in range(n):
        row = " ".join(f" {M[i, j]:5.3f}" for j in range(n))
        print(f"   blk{i}  {row}")

    # Off-diagonal stats
    mask = ~np.eye(n, dtype=bool)
    off = M[mask]
    print(f"  off-diagonal: min={off.min():.3f} mean={off.mean():.3f} "
          f"max={off.max():.3f}")
    n_high = int((off > 0.95).sum() // 2)  # /2 because symmetric
    n_pairs = n * (n - 1) // 2
    print(f"  pairs with CKA > 0.95: {n_high}/{n_pairs}")
    n_low = int((off < 0.80).sum() // 2)
    print(f"  pairs with CKA < 0.80: {n_low}/{n_pairs} (depth diversity)")


# ---------------------------------------------------------------------------
#  Model loaders
# ---------------------------------------------------------------------------

def load_minimal_cnn14(ckpt_path: str, device):
    from experiments.distillation_14ch.model_14ch import MinimalCNN14
    model = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=14).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  loaded MinimalCNN14: missing={len(miss)} unexpected={len(unexp)}")
    return model


def load_v12_2(ckpt_path: str, device):
    from td_ludo.models.v12 import AlphaLudoV12
    model = AlphaLudoV12(
        num_res_blocks=3, num_channels=128,
        num_attn_layers=2, num_heads=4, ffn_ratio=4,
        dropout=0.0, in_channels=33,
    ).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  loaded V12.2: missing={len(miss)} unexpected={len(unexp)}")
    return model


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    n_states = int(os.environ.get("CKA_N_STATES", N_STATES_DEFAULT))
    print(f"[CKA] generating {n_states} mid-game states ...")
    states = make_mid_game_states(n_states)
    print(f"[CKA] got {len(states)} states")

    device = torch.device("cpu")

    student_path = "experiments/distillation_14ch/student_14ch_final.pt"
    v122_path = "play/model_weights/v12_2/model_latest.pt"

    results = {}

    if os.path.exists(student_path):
        print(f"\n[CKA] === MinimalCNN14 (14ch input, 10×128 deep CNN) ===")
        t0 = time.time()
        student = load_minimal_cnn14(student_path, device)
        acts = collect_block_activations(
            student, cpp.encode_state_v14_minimal, states, device,
        )
        M_student = cka_matrix(acts)
        print(f"  elapsed: {time.time() - t0:.1f}s")
        print_matrix(M_student, "MinimalCNN14 (14ch)")
        results["student_14ch"] = M_student
    else:
        print(f"[CKA] student weights not found at {student_path}, skipping")

    if os.path.exists(v122_path):
        print(f"\n[CKA] === V12.2 (33ch rich input, 3×128 + transformer) ===")
        t0 = time.time()
        v122 = load_v12_2(v122_path, device)
        acts = collect_block_activations(
            v122, cpp.encode_state_v11, states, device,
        )
        M_v122 = cka_matrix(acts)
        print(f"  elapsed: {time.time() - t0:.1f}s")
        print_matrix(M_v122, "V12.2 (33ch)")
        results["v12_2"] = M_v122
    else:
        print(f"[CKA] V12.2 weights not found at {v122_path}, skipping")

    # Save raw matrices for downstream plotting if wanted.
    out_dir = "experiments/distillation_14ch"
    np.savez(
        os.path.join(out_dir, "cka_matrices.npz"),
        **{k: v for k, v in results.items()},
    )
    print(f"\n[CKA] matrices saved to {out_dir}/cka_matrices.npz")


if __name__ == "__main__":
    main()
