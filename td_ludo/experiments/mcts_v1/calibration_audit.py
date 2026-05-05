"""V13.2 value-head calibration audit (Step 0 of MCTS plateau-break experiment).

Verifies that V13.2's `win_prob` head is calibrated enough to seed
AlphaZero-style MCTS leaf evaluation. If it isn't, MCTS will be
systematically misled by bad leaf estimates and we should retrain the
value head before committing to the bigger experiment.

Procedure:
  1. Load V13.2 checkpoint, freeze in eval mode.
  2. Run N self-play games of V13.2 vs V13.2 with stochastic policy
     (τ=1.0). Vectorized via VectorGameState (B=200 parallel).
  3. Record (V_pred from current player POV, current_player) at every
     decision state. Discard the first 10 turns of each game.
  4. After all games finish, walk back from each game's outcome and
     label every recorded state with `eventual_outcome` from the POV.
  5. Bin V_pred into 10 equal-frequency deciles. Compute per-bin
     empirical WR.
  6. Compute Brier score and Expected Calibration Error (ECE).
  7. Output JSON report + calibration plot.

Pass criteria:
  - ECE ≤ 5pp
  - No bin > 10pp deviation from y=x
  - Brier score ≤ 0.20

Usage:
    python -m experiments.mcts_v1.calibration_audit \
        --model checkpoints/v132/model_latest.pt \
        --num-games 5000 \
        --batch-size 200 \
        --output runs/mcts_v1_calibration.json
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
import torch

# Project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS

# V13.2 architecture: MinimalCNN14 with 17ch input, 10×128. Class lives in
# distillation_14ch (same as v13).
from experiments.distillation_14ch.model_14ch import MinimalCNN14

# Discard first N turns of each game when collecting calibration data —
# game-start states are too easy and would dominate the high-V bins.
DISCARD_FIRST_N_TURNS = 10
MAX_TURNS_PER_GAME = 400


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to V13.2 checkpoint (.pt)")
    p.add_argument("--num-games", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=200,
                   help="Parallel games per VectorGameState batch")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Stochastic move sampling temperature (1.0 = uncorrected softmax)")
    p.add_argument("--num-bins", type=int, default=10,
                   help="Number of calibration bins (deciles by default)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/mcts_v1_calibration.json")
    p.add_argument("--num-res-blocks", type=int, default=10)
    p.add_argument("--num-channels", type=int, default=128)
    p.add_argument("--no-plot", action="store_true",
                   help="Skip generating the matplotlib calibration plot.")
    return p.parse_args()


def pick_device(name):
    if name in ("cuda", "cpu", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_v132(path, device, num_res_blocks, num_channels):
    """Load V13.2 checkpoint into MinimalCNN14 (17ch input)."""
    print(f"[Audit] Loading V13.2 from {path}...")
    model = MinimalCNN14(
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        in_channels=V17_CHANNELS,
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"[Audit] Loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def _advance_to_decision(game, rng):
    """Spin a game forward until it reaches a decision state (legal moves
    available) or terminates. Mirrors train_v132_sl.py's pattern.

    Returns: True if game has a decision state ready, False if terminal.
    """
    while not game.is_terminal:
        if game.current_dice_roll == 0:
            game.current_dice_roll = int(rng.integers(1, 7))
        legal = ludo_cpp.get_legal_moves(game)
        if legal:
            return True
        # No legal moves — advance turn (skip this player's turn entirely)
        cp = int(game.current_player)
        nxt = (cp + 1) % 4
        while not game.active_players[nxt]:
            nxt = (nxt + 1) % 4
        game.current_player = nxt
        game.current_dice_roll = 0
    return False


def collect_calibration_data(model, device, num_games, batch_size, temperature, seed):
    """Run self-play (V13.2 vs V13.2 stochastic) and collect (V_pred, outcome).

    Each parallel game is spun forward to a decision state before batched
    inference fires. Mirrors the proven train_v132_sl.py collection pattern.

    Returns:
      v_preds:   np.ndarray (M,) — V13.2's win-prob prediction at each state
      outcomes:  np.ndarray (M,) — eventual outcome from POV (1=win, 0=loss)
      meta:      dict with collection stats
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = ludo_cpp.VectorGameState(batch_size=batch_size, two_player_mode=True)
    pending = [[] for _ in range(batch_size)]
    turn_count = np.zeros(batch_size, dtype=np.int32)

    finalized_v = []
    finalized_o = []
    finalized_cp = []
    games_done = 0

    t_start = time.time()
    last_log = t_start
    print(f"[Audit] Collecting from {num_games} games, batch={batch_size}, τ={temperature}...")

    while games_done < num_games:
        # Spin every game forward to its next decision state (or terminal)
        decision_idxs = []
        encoded_batch = []
        masks_batch = []
        cps_batch = []
        for i in range(batch_size):
            game = env.get_game(i)
            if game.is_terminal or turn_count[i] >= MAX_TURNS_PER_GAME:
                continue
            ok = _advance_to_decision(game, rng)
            if not ok:
                continue  # game became terminal during spin
            decision_idxs.append(i)
            encoded_batch.append(encode_state_v17(game))
            mask = np.zeros(4, dtype=np.float32)
            for m in ludo_cpp.get_legal_moves(game):
                mask[m] = 1.0
            masks_batch.append(mask)
            cps_batch.append(int(game.current_player))

        # Batched inference (only games at a live decision point)
        if encoded_batch:
            x = torch.from_numpy(np.stack(encoded_batch, axis=0)).to(device, dtype=torch.float32)
            m = torch.from_numpy(np.stack(masks_batch, axis=0)).to(device, dtype=torch.float32)
            with torch.no_grad():
                policy, win_prob, _ = model(x, m)
            policy_np = policy.cpu().numpy()
            v_pred_np = win_prob.cpu().numpy().reshape(-1)
        else:
            policy_np = np.zeros((0, 4), dtype=np.float32)
            v_pred_np = np.zeros((0,), dtype=np.float32)

        # Pick actions, record states past warmup
        actions = [-1] * batch_size
        for k, i in enumerate(decision_idxs):
            cp = cps_batch[k]
            v_pred = float(v_pred_np[k])

            p = policy_np[k]
            mask = masks_batch[k]
            p = np.where(mask > 0, p, 0.0)
            ps = p.sum()
            p = p / ps if ps > 0 else mask / mask.sum()

            if abs(temperature - 1.0) > 1e-6:
                p_t = np.power(p + 1e-12, 1.0 / temperature)
                p_t = np.where(mask > 0, p_t, 0.0)
                p = p_t / p_t.sum()

            action = int(rng.choice(4, p=p))
            if mask[action] == 0:
                legal_idx = np.where(mask > 0)[0]
                action = int(rng.choice(legal_idx))
            actions[i] = action

            if turn_count[i] >= DISCARD_FIRST_N_TURNS:
                pending[i].append((cp, v_pred))

        # Step env (advances each game by its chosen action, returns terminations)
        _, _, _, infos = env.step(actions)
        for i in range(batch_size):
            if actions[i] >= 0:
                turn_count[i] += 1

        # Handle terminations + max-turns timeout
        for i, info in enumerate(infos):
            timed_out = turn_count[i] >= MAX_TURNS_PER_GAME and not info["is_terminal"]
            if info["is_terminal"] or timed_out:
                if info["is_terminal"]:
                    winner = int(info["winner"])
                    for (cp, vp) in pending[i]:
                        outcome = 1.0 if cp == winner else 0.0
                        finalized_v.append(vp)
                        finalized_o.append(outcome)
                        finalized_cp.append(cp)
                pending[i].clear()
                env.reset_game(i)
                turn_count[i] = 0
                games_done += 1
                if games_done >= num_games:
                    break

        # Progress log every 5s
        now = time.time()
        if now - last_log > 5.0:
            elapsed = now - t_start
            gpm = (games_done / elapsed) * 60.0 if elapsed > 0 else 0
            states_per_game = (len(finalized_v) / max(1, games_done))
            pending_total = sum(len(p) for p in pending)
            print(f"  [{games_done:>5}/{num_games}]  done_states={len(finalized_v):>7} "
                  f"pending={pending_total:>5}  gpm={gpm:>6.1f}  "
                  f"states/game={states_per_game:.1f}  elapsed={elapsed:.0f}s")
            last_log = now

    print(f"[Audit] Collection done: {games_done} games, {len(finalized_v)} states, "
          f"{time.time() - t_start:.0f}s elapsed")

    return (
        np.array(finalized_v, dtype=np.float32),
        np.array(finalized_o, dtype=np.float32),
        {
            "num_games": games_done,
            "num_states": len(finalized_v),
            "states_per_game": float(len(finalized_v) / max(1, games_done)),
            "elapsed_sec": time.time() - t_start,
        },
    )


def compute_calibration(v_preds: np.ndarray, outcomes: np.ndarray, num_bins: int):
    """Bin V_pred into equal-frequency deciles, compute per-bin stats, ECE, Brier."""
    n = len(v_preds)
    if n == 0:
        raise RuntimeError("No states collected — bug?")

    # Sort by V_pred, split into roughly-equal chunks (equal-frequency binning)
    sort_idx = np.argsort(v_preds)
    v_sorted = v_preds[sort_idx]
    o_sorted = outcomes[sort_idx]

    bin_edges = np.linspace(0, n, num_bins + 1).astype(int)
    bins = []
    ece = 0.0
    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if hi <= lo:
            continue
        v_bin = v_sorted[lo:hi]
        o_bin = o_sorted[lo:hi]
        v_mean = float(v_bin.mean())
        v_lo = float(v_bin.min())
        v_hi = float(v_bin.max())
        wr = float(o_bin.mean())
        # Wilson 95% CI for empirical WR
        n_bin = len(o_bin)
        wr_se = (wr * (1 - wr) / n_bin) ** 0.5 if n_bin > 0 else 0.0
        deviation = abs(wr - v_mean)
        ece += deviation * (n_bin / n)
        bins.append({
            "bin_idx": b,
            "v_pred_mean": v_mean,
            "v_pred_range": [v_lo, v_hi],
            "empirical_wr": wr,
            "deviation_pp": 100.0 * deviation,
            "wr_stderr_pp": 100.0 * wr_se,
            "n_samples": int(n_bin),
        })

    brier = float(((v_preds - outcomes) ** 2).mean())
    max_dev = max((b["deviation_pp"] for b in bins), default=0.0)
    return {
        "num_bins": num_bins,
        "num_states": int(n),
        "brier_score": brier,
        "ece_pp": 100.0 * ece,
        "max_bin_deviation_pp": max_dev,
        "bins": bins,
    }


def verdict(calib: dict) -> tuple[str, str]:
    """Return (verdict, reasoning)."""
    ece = calib["ece_pp"]
    max_dev = calib["max_bin_deviation_pp"]
    brier = calib["brier_score"]

    # Pass: ECE ≤ 5pp, no bin > 10pp, Brier ≤ 0.20
    if ece <= 5.0 and max_dev <= 10.0 and brier <= 0.20:
        return "PASS", (
            f"ECE={ece:.2f}pp ≤ 5, max_bin_dev={max_dev:.2f}pp ≤ 10, "
            f"Brier={brier:.3f} ≤ 0.20. Value head is fit for MCTS use."
        )

    # Marginal: 5pp < ECE ≤ 10pp, etc.
    if ece <= 10.0 and max_dev <= 15.0 and brier <= 0.22:
        return "MARGINAL", (
            f"ECE={ece:.2f}pp, max_bin_dev={max_dev:.2f}pp, Brier={brier:.3f}. "
            f"Calibration is suboptimal but MCTS may still help. Proceed with caution."
        )

    # Fail
    return "FAIL", (
        f"ECE={ece:.2f}pp, max_bin_dev={max_dev:.2f}pp, Brier={brier:.3f}. "
        f"Value head is too poorly calibrated. Recommend retraining value head only "
        f"(freeze backbone, BCE on outcomes, 1-2 days), then re-audit."
    )


def maybe_plot(calib: dict, out_path: Path):
    """Generate calibration plot — empirical WR vs V_pred per bin, with y=x reference."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Audit] matplotlib not installed, skipping plot.")
        return

    bins = calib["bins"]
    v_means = [b["v_pred_mean"] for b in bins]
    wrs = [b["empirical_wr"] for b in bins]
    err_pp = [b["wr_stderr_pp"] / 100.0 for b in bins]
    n_samples = [b["n_samples"] for b in bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration plot
    ax1.errorbar(v_means, wrs, yerr=err_pp, fmt="o-", color="steelblue",
                 markersize=8, capsize=4, label="Empirical WR (±1σ)")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration (y=x)")
    ax1.fill_between([0, 1], [0.05, 1.05], [-0.05, 0.95], color="green", alpha=0.1,
                     label="±5pp band")
    ax1.set_xlabel("V_pred (binned, mean per bin)")
    ax1.set_ylabel("Empirical win-rate")
    ax1.set_title(
        f"V13.2 calibration ({calib['num_states']:,} states)\n"
        f"ECE={calib['ece_pp']:.2f}pp · Brier={calib['brier_score']:.3f} · "
        f"max_dev={calib['max_bin_deviation_pp']:.2f}pp"
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Sample-count histogram
    ax2.bar(range(len(n_samples)), n_samples, color="darkorange", alpha=0.7)
    ax2.set_xlabel("Bin index (V_pred decile)")
    ax2.set_ylabel("Number of samples")
    ax2.set_title("Sample count per bin")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Audit] Plot saved → {out_path}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"[Audit] Device: {device}")

    model = load_v132(args.model, device, args.num_res_blocks, args.num_channels)

    v_preds, outcomes, meta = collect_calibration_data(
        model, device,
        num_games=args.num_games,
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed=args.seed,
    )

    print(f"[Audit] Computing calibration metrics ({args.num_bins} bins)...")
    calib = compute_calibration(v_preds, outcomes, args.num_bins)

    verdict_str, reasoning = verdict(calib)
    print()
    print("=" * 70)
    print(f"  VERDICT: {verdict_str}")
    print("=" * 70)
    print(f"  {reasoning}")
    print()
    print("  Per-bin breakdown:")
    print(f"  {'bin':>3} {'v_range':>16} {'v_mean':>8} {'wr':>8} {'dev':>8} {'n':>8}")
    for b in calib["bins"]:
        v_lo, v_hi = b["v_pred_range"]
        print(f"  {b['bin_idx']:>3} [{v_lo:.3f},{v_hi:.3f}] {b['v_pred_mean']:>8.3f} "
              f"{b['empirical_wr']:>8.3f} {b['deviation_pp']:>7.2f}pp "
              f"{b['n_samples']:>8}")

    # Save report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_path": str(args.model),
        "num_games": args.num_games,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "device": str(device),
        "seed": args.seed,
        "verdict": verdict_str,
        "reasoning": reasoning,
        "collection_meta": meta,
        "calibration": calib,
        "ts": int(time.time()),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Audit] Report saved → {out_path}")

    # Plot
    if not args.no_plot:
        plot_path = out_path.with_suffix(".png")
        maybe_plot(calib, plot_path)

    return 0 if verdict_str == "PASS" else (1 if verdict_str == "MARGINAL" else 2)


if __name__ == "__main__":
    sys.exit(main())
