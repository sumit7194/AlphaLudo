"""Diagnostic: does N=100-sim MCTS produce different policies than V13.5_exp's
intrinsic policy?

Purpose: before committing 2-3 weeks of compute to full iterated MCTS training,
verify that the search engine actually finds policy improvements over V13.5.

The metric: KL(π_mcts || π_v13.5) averaged across sampled states.
  - KL ≈ 0:   MCTS and V13.5 agree → search adds nothing
  - KL > 0.1: MCTS finds meaningfully different (presumably better) actions
  - KL > 0.5: MCTS strongly disagrees with V13.5

Also reports:
  - top1_agreement: fraction of states where MCTS and V13.5 pick same top action
  - mean entropy of each policy
  - max V13.5-prob action's relative MCTS ranking

Usage:
    python3 -m experiments.mcts_v1.mcts_diagnostic \
        --model checkpoints/v135_shaping_exp/model_latest.pt \
        --n-states 200 --n-sims 100 --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v18_production import encode_state_v18_production
from td_ludo.models.v13_5_production import V135ProductionAdapter

from experiments.mcts_v1.mcts_engine import MCTS, NetworkEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="V13.5 checkpoint path (e.g., v135_shaping_exp/model_latest.pt)")
    p.add_argument("--n-states", type=int, default=200,
                   help="Number of game states to sample + analyze")
    p.add_argument("--n-sims", type=int, default=100,
                   help="MCTS simulations per state")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-eps", type=float, default=0.0,
                   help="Dirichlet at root (0 for diagnostic — measure clean policy)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Visit-count temperature for π_mcts (1.0 = proportional)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/mcts_diagnostic_v135.json")
    return p.parse_args()


def pick_device(name: str):
    import torch
    if name in ("cpu", "mps", "cuda"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_v135(path, device):
    import torch
    model = V135ProductionAdapter(num_res_blocks=10, num_channels=128)
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _advance_to_decision(state, rng):
    """Spin a game forward until a decision is needed (or terminal)."""
    while not state.is_terminal:
        cp = int(state.current_player)
        if not state.active_players[cp]:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            continue
        if state.current_dice_roll == 0:
            state.current_dice_roll = rng.randint(1, 6)
        legal = ludo_cpp.get_legal_moves(state)
        if legal:
            return state, legal
        # pass turn
        n = (cp + 1) % 4
        while not state.active_players[n]:
            n = (n + 1) % 4
        state.current_player = n
        state.current_dice_roll = 0
    return state, []


def _v135_intrinsic_policy(model, device, state) -> np.ndarray:
    """Get V13.5's intrinsic policy distribution at this state (4-vec, masked)."""
    import torch
    enc = encode_state_v18_production(state).astype(np.float32)
    legal = ludo_cpp.get_legal_moves(state)
    mask = np.zeros(4, dtype=np.float32)
    for a in legal:
        mask[a] = 1.0
    with torch.no_grad():
        x = torch.from_numpy(enc).unsqueeze(0).to(device)
        m = torch.from_numpy(mask).unsqueeze(0).to(device)
        out = model(x, m)
        policy = out[0]
    return policy.cpu().numpy().reshape(-1).astype(np.float32)


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """KL(p || q) over discrete distribution. Both must sum to ~1, illegal=0."""
    mask = p > eps
    if not mask.any():
        return 0.0
    p_safe = p[mask]
    q_safe = np.clip(q[mask], eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def entropy(p: np.ndarray, eps: float = 1e-8) -> float:
    mask = p > eps
    if not mask.any():
        return 0.0
    p_safe = p[mask]
    return float(-np.sum(p_safe * np.log(p_safe)))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device(args.device)
    print(f"[Diag] device: {device}")
    print(f"[Diag] loading {args.model}...")
    model = load_v135(args.model, device)
    print(f"[Diag] loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    evaluator = NetworkEvaluator(model, device, encode_state_v18_production, root_player=0)
    mcts = MCTS(evaluator, c_puct=args.c_puct, n_sims=args.n_sims,
                dirichlet_eps=args.dirichlet_eps, rng=random.Random(args.seed))

    # Sample game states: play random games + capture states at decision points
    print(f"[Diag] sampling {args.n_states} decision states...")
    states = []
    state = ludo_cpp.create_initial_state_2p()
    moves_in_curr_game = 0
    max_moves = 400
    while len(states) < args.n_states:
        state, legal = _advance_to_decision(state, rng)
        if state.is_terminal or moves_in_curr_game >= max_moves:
            # reset
            state = ludo_cpp.create_initial_state_2p()
            moves_in_curr_game = 0
            continue
        if not legal:
            continue
        # Capture this state (deepcopy via _copy_state to detach)
        from experiments.mcts_v1.mcts_engine import _copy_state
        states.append(_copy_state(state))
        # Take a random action to advance
        action = rng.choice(list(legal))
        state = ludo_cpp.apply_move(state, int(action))
        state.current_dice_roll = 0
        moves_in_curr_game += 1
    print(f"[Diag] {len(states)} states sampled.")

    # For each state, compute (π_mcts, π_v13.5, KL)
    print(f"[Diag] running {args.n_sims}-sim MCTS on each state...")
    kls = []
    entropies_mcts = []
    entropies_v135 = []
    top1_agreements = 0
    skipped = 0
    t_start = time.time()
    last_log = t_start
    for i, s in enumerate(states):
        legal = list(ludo_cpp.get_legal_moves(s))
        if len(legal) < 2:
            skipped += 1
            continue
        # MCTS
        root = mcts.search(s, training=False)
        pi_mcts = root.visit_distribution(temperature=args.temperature)
        # V13.5 intrinsic
        pi_v135 = _v135_intrinsic_policy(model, device, s)

        kls.append(kl_div(pi_mcts, pi_v135))
        entropies_mcts.append(entropy(pi_mcts))
        entropies_v135.append(entropy(pi_v135))
        if np.argmax(pi_mcts) == np.argmax(pi_v135):
            top1_agreements += 1

        if time.time() - last_log > 10:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(states) - i - 1) / max(1, rate)
            print(f"  [{i + 1}/{len(states)}] mean_kl={np.mean(kls):.4f} "
                  f"top1_agree={100*top1_agreements/(i+1-skipped):.1f}% "
                  f"rate={rate:.1f}/s ETA={eta:.0f}s")
            last_log = time.time()

    n_evaluated = len(kls)
    mean_kl = float(np.mean(kls)) if kls else 0.0
    median_kl = float(np.median(kls)) if kls else 0.0
    max_kl = float(np.max(kls)) if kls else 0.0
    top1_pct = 100.0 * top1_agreements / max(1, n_evaluated)
    mean_h_mcts = float(np.mean(entropies_mcts)) if entropies_mcts else 0.0
    mean_h_v135 = float(np.mean(entropies_v135)) if entropies_v135 else 0.0

    print()
    print("=" * 70)
    print("  DIAGNOSTIC RESULTS")
    print("=" * 70)
    print(f"  States analyzed:       {n_evaluated} (skipped {skipped} single-legal)")
    print(f"  MCTS sims per state:   {args.n_sims}")
    print(f"  Mean KL(π_mcts || π_v13.5):    {mean_kl:.4f}")
    print(f"  Median KL:                      {median_kl:.4f}")
    print(f"  Max KL:                         {max_kl:.4f}")
    print(f"  Top-1 action agreement:         {top1_pct:.1f}%")
    print(f"  Mean entropy π_mcts:            {mean_h_mcts:.3f}")
    print(f"  Mean entropy π_v13.5:           {mean_h_v135:.3f}")
    print()
    # Verdict
    if mean_kl >= 0.10:
        verdict = "GREEN — search finds meaningfully different policies. Commit to Step 2."
    elif mean_kl >= 0.03:
        verdict = "YELLOW — search differs but only mildly. Step 2 might yield marginal lift."
    else:
        verdict = "RED — search and V13.5 agree. Step 2 unlikely to help."
    print(f"  VERDICT: {verdict}")
    print()

    # Save JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "n_states": n_evaluated,
            "n_sims": args.n_sims,
            "c_puct": args.c_puct,
            "temperature": args.temperature,
            "mean_kl": mean_kl,
            "median_kl": median_kl,
            "max_kl": max_kl,
            "top1_agreement_pct": top1_pct,
            "mean_entropy_mcts": mean_h_mcts,
            "mean_entropy_v135": mean_h_v135,
            "elapsed_sec": time.time() - t_start,
            "verdict": verdict,
        }, f, indent=2)
    print(f"[Diag] saved → {args.output}")


if __name__ == "__main__":
    main()
