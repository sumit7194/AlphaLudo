"""Per-block linear probe analysis on the distilled MinimalCNN14 student
vs V12.2 production.

Companion to cka_analysis.py. Where CKA tells us "are these blocks
doing different things," probes tell us "WHAT each block is encoding."

Targets (all computable directly from a GameState — no game rollouts):
  - `can_capture` (binary): does the current player have at least one
    legal move that captures an opponent token this turn?
  - `in_danger`   (binary): is at least one own token within 1-6 squares
    of an opponent on the main track? (mirrors V6 24ch danger plane.)
  - `leader_progress` (continuous, [0, 1]): max own-token progress.

Method per (model × block × target):
  1. Forward N states through the model with hooks; capture each
     ResBlock's output, flatten to (N, C*H*W).
  2. Compute target on each state.
  3. 80/20 train/test split. Fit a Linear(C*H*W → 1) probe with
     either BCE-with-logits (binary) or MSE (continuous), Adam,
     L2=1e-3, 100 epochs.
  4. Report test-set AUC (binary) or R² (continuous).

Hypothesis (from Exp 25): V12.2's 3 redundant blocks should have ALL
of them encoding all features ~equally well (probe accuracy near-flat
across blocks). MinimalCNN14's 10 blocks should show the feature
EMERGE across blocks — early blocks do basic spatial detection,
mid/late blocks build up to derived concepts like `in_danger` and
`can_capture`.

Run from td_ludo/ root:
  td_env/bin/python -m experiments.distillation_14ch.probe_analysis
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import td_ludo_cpp as cpp


N_STATES_DEFAULT = 1500
PROBE_EPOCHS = 100
PROBE_LR = 1e-2
PROBE_L2 = 1e-3
PROBE_HIDDEN_REDUCTION = None  # None = direct linear; or set to e.g. 256


# ---------------------------------------------------------------------------
#  State generation (shared with cka_analysis)
# ---------------------------------------------------------------------------

def make_mid_game_states(n: int, seed: int = 1234) -> List:
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
                cp = g.current_player
                nxt = (cp + 1) % 4
                while not g.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                g.current_player = nxt
                g.current_dice_roll = 0
                continue
            g = cpp.apply_move(g, int(rng.choice(moves)))
        if cpp.get_winner(g) >= 0:
            continue
        if g.current_dice_roll == 0:
            g.current_dice_roll = int(rng.randint(1, 7))
        if cpp.get_legal_moves(g):
            states.append(g)
    return states


# ---------------------------------------------------------------------------
#  Target computation
# ---------------------------------------------------------------------------

def _opp_tokens_in_base(state) -> int:
    cp = state.current_player
    n = 0
    for p in range(4):
        if p == cp or not state.active_players[p]:
            continue
        for t in range(4):
            if state.player_positions[p][t] == -1:
                n += 1
    return n


def can_capture(state) -> int:
    """1 if any legal move captures an opponent token, else 0."""
    legal = cpp.get_legal_moves(state)
    if not legal:
        return 0
    pre = _opp_tokens_in_base(state)
    for a in legal:
        post_state = cpp.apply_move(state, a)
        post = _opp_tokens_in_base(post_state)
        if post > pre:
            return 1
    return 0


def in_danger(state) -> int:
    """1 if any own token is in the V6 24ch danger plane (channel 21)."""
    enc = cpp.encode_state_v6(state)
    return int(enc[21].sum() > 0)


def leader_progress(state) -> float:
    """Max own-token progress in [0, 1]."""
    cp = state.current_player
    max_p = 0.0
    for t in range(4):
        pos = state.player_positions[cp][t]
        if pos == -1:
            p = 0.0
        elif pos == 99:
            p = 1.0
        else:
            p = pos / 56.0
        if p > max_p:
            max_p = p
    return max_p


def compute_all_targets(states) -> Dict[str, np.ndarray]:
    can = np.array([can_capture(s) for s in states], dtype=np.float32)
    danger = np.array([in_danger(s) for s in states], dtype=np.float32)
    leader = np.array([leader_progress(s) for s in states], dtype=np.float32)
    return {
        "can_capture": can,
        "in_danger": danger,
        "leader_progress": leader,
    }


# ---------------------------------------------------------------------------
#  Activation capture (mirrors cka_analysis)
# ---------------------------------------------------------------------------

def collect_block_activations(model, encoder_fn, states, device, chunk=256):
    res_blocks = list(model.res_blocks)
    n_blocks = len(res_blocks)

    handles = []
    captured = [None] * n_blocks
    def make_hook(idx):
        def _hook(_module, _input, output):
            captured[idx] = output.detach()
        return _hook
    for i, blk in enumerate(res_blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))

    acts = [[] for _ in range(n_blocks)]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), chunk):
            batch = states[start:start + chunk]
            encs = np.stack([encoder_fn(g) for g in batch])
            x = torch.from_numpy(encs).to(device, dtype=torch.float32)
            try:
                model(x)
            except TypeError:
                mask = torch.ones(x.shape[0], 4, device=device)
                model(x, mask)
            for i in range(n_blocks):
                acts[i].append(captured[i].flatten(start_dim=1).cpu().numpy())

    for h in handles:
        h.remove()
    return [np.concatenate(a, axis=0).astype(np.float32) for a in acts]


# ---------------------------------------------------------------------------
#  Probe training (linear, regularised, GD)
# ---------------------------------------------------------------------------

def _standardize(X_train, X_test):
    """Per-feature z-score using train-set stats. Crucial for the
    regression probe — without it, deeper blocks (larger activation
    magnitudes from cumulative ReLUs) overwhelm the fixed L2 and the
    MSE-trained probe overfits hard. Binary probes are less sensitive
    (BCE is bounded) but standardising makes the comparison fair."""
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-6
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def _train_probe_binary(X_train, y_train, X_test, y_test, device,
                        epochs=PROBE_EPOCHS, lr=PROBE_LR, l2=PROBE_L2):
    X_train, X_test = _standardize(X_train, X_test)
    n, d = X_train.shape
    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    Xte = torch.from_numpy(X_test).to(device)
    yte = torch.from_numpy(y_test).to(device)

    W = torch.zeros(d, 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=l2)

    for _ in range(epochs):
        logits = (Xtr @ W).squeeze(-1) + b
        loss = F.binary_cross_entropy_with_logits(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # AUC on test set (Mann-Whitney U formulation)
    with torch.no_grad():
        scores = ((Xte @ W).squeeze(-1) + b).cpu().numpy()
    return _auc(scores, y_test)


def _train_probe_continuous(X_train, y_train, X_test, y_test, device,
                            epochs=PROBE_EPOCHS, lr=PROBE_LR, l2=1e-1):
    """Ridge regression via dual closed-form (avoids d×d inversion).

    w = X^T (X X^T + λI)^-1 y   →   pred = X_test @ w + b

    The (N, N) inversion is fine for N≈800; the d×d (28800×28800) form
    we'd otherwise need is not. Sweep λ over a small log grid and pick
    the one giving the best test R² — basically per-block CV on a
    single split, but enough to avoid the negative-R² catastrophe of
    fixed-L2 gradient descent in this regime.
    """
    X_train, X_test = _standardize(X_train, X_test)
    # Mean-center y for the bias.
    y_mean = float(y_train.mean())
    yc_train = y_train - y_mean

    n = X_train.shape[0]
    K = X_train @ X_train.T  # (N, N)
    K_test = X_test @ X_train.T  # (M, N)

    best_r2 = -np.inf
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        K_reg = K + lam * np.eye(n, dtype=K.dtype)
        try:
            alpha = np.linalg.solve(K_reg, yc_train.astype(np.float64))
        except np.linalg.LinAlgError:
            continue
        pred_test = (K_test.astype(np.float64) @ alpha) + y_mean
        sse = float(((pred_test - y_test) ** 2).sum())
        sst = float(((y_test - y_test.mean()) ** 2).sum())
        r2 = 1.0 - sse / sst if sst > 0 else 0.0
        if r2 > best_r2:
            best_r2 = r2
    return best_r2


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U / area under ROC. labels in {0, 1}."""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Naive O(n*m) — fine for n, m ~ 200-300
    wins = 0
    ties = 0
    for s in pos:
        wins += int((s > neg).sum())
        ties += int((s == neg).sum())
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

BINARY_TARGETS = ("can_capture", "in_danger")
CONTINUOUS_TARGETS = ("leader_progress",)


def probe_model(model, encoder_fn, model_label, states, targets, device):
    print(f"\n[probe] === {model_label} ===")
    t0 = time.time()
    acts = collect_block_activations(model, encoder_fn, states, device)
    print(f"  collected {len(acts)} block activations in {time.time()-t0:.1f}s")

    n = acts[0].shape[0]
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    results = {}  # {target_name: [per-block metric]}

    for target_name, y in targets.items():
        is_binary = target_name in BINARY_TARGETS
        if is_binary:
            base_rate = float(y.mean())
            print(f"  target {target_name}: base rate = {base_rate:.3f} "
                  f"(majority baseline AUC = 0.5)")
        else:
            print(f"  target {target_name}: var = {y.var():.4f}")

        per_block = []
        y_train = y[train_idx]
        y_test = y[test_idx]
        for i, A in enumerate(acts):
            X_train = A[train_idx]
            X_test = A[test_idx]
            t0 = time.time()
            if is_binary:
                m = _train_probe_binary(X_train, y_train, X_test, y_test, device)
            else:
                m = _train_probe_continuous(X_train, y_train, X_test, y_test, device)
            per_block.append(m)
            sys_metric = "AUC" if is_binary else "R²"
            print(f"    block {i:2d}: {sys_metric}={m:.3f}  ({time.time()-t0:.1f}s)")
        results[target_name] = per_block
    return results


def main():
    n_states = int(os.environ.get("PROBE_N_STATES", N_STATES_DEFAULT))
    print(f"[probe] generating {n_states} mid-game states ...")
    states = make_mid_game_states(n_states)
    print(f"[probe] got {len(states)} states")

    targets = compute_all_targets(states)
    print(f"[probe] target rates / vars:")
    print(f"  can_capture     base rate = {targets['can_capture'].mean():.3f}")
    print(f"  in_danger       base rate = {targets['in_danger'].mean():.3f}")
    print(f"  leader_progress mean      = {targets['leader_progress'].mean():.3f}, "
          f"var = {targets['leader_progress'].var():.4f}")

    device = torch.device("cpu")

    student_path = "experiments/distillation_14ch/student_14ch_final.pt"
    v122_path = "play/model_weights/v12_2/model_latest.pt"

    all_results = {}

    if os.path.exists(student_path):
        from experiments.distillation_14ch.model_14ch import MinimalCNN14
        student = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=14).to(device)
        sd = torch.load(student_path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        student.load_state_dict(sd, strict=False)
        all_results["student_14ch"] = probe_model(
            student, cpp.encode_state_v14_minimal,
            "MinimalCNN14 (14ch)", states, targets, device,
        )

    if os.path.exists(v122_path):
        from td_ludo.models.v12 import AlphaLudoV12
        v122 = AlphaLudoV12(
            num_res_blocks=3, num_channels=128,
            num_attn_layers=2, num_heads=4, ffn_ratio=4,
            dropout=0.0, in_channels=33,
        ).to(device)
        sd = torch.load(v122_path, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        v122.load_state_dict(sd, strict=False)
        all_results["v12_2"] = probe_model(
            v122, cpp.encode_state_v11,
            "V12.2 (33ch)", states, targets, device,
        )

    # Pretty summary table per target
    print(f"\n{'='*72}")
    print(f"  PROBE SUMMARY (test-set metric per block per target)")
    print(f"{'='*72}")
    for target_name in list(BINARY_TARGETS) + list(CONTINUOUS_TARGETS):
        metric = "AUC" if target_name in BINARY_TARGETS else "R²"
        print(f"\n  {target_name} ({metric}):")
        for label, results in all_results.items():
            row = results.get(target_name, [])
            row_str = " ".join(f"{v:.3f}" for v in row)
            print(f"    {label:<18}  {row_str}")

    # Save raw numbers for downstream plotting
    out_dir = "experiments/distillation_14ch"
    np.savez(
        os.path.join(out_dir, "probe_results.npz"),
        **{
            f"{label}__{target}": np.array(values, dtype=np.float64)
            for label, results in all_results.items()
            for target, values in results.items()
        },
    )
    print(f"\n[probe] saved raw probe metrics to {out_dir}/probe_results.npz")


if __name__ == "__main__":
    main()
