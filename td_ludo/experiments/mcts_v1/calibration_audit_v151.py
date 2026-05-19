"""V15.1 value-head calibration audit (Step 0 of the V15.1+MCTS experiment).

Same diagnostic as the V13.2/V13.5 audit, adapted for V15.1's
GraphTransformer architecture:
  - Input is `(B, 2, 15, 15, 3)` per-cell triplet × 2-frame history
  - Policy is `(B, 225)` source-cell softmax — projected to `(B, 4)` per
    token via `policy_cells_to_tokens` for action sampling
  - Value is sigmoid from CLS token

What we're testing
------------------
If V15.1's `win_prob` is already well-calibrated (low ECE, no bin
deviates from y=x by much), then MCTS leaf evaluations will agree with
the model's prior — no search signal → "coherent equilibrium" failure
(same mode that killed V13.5+MCTS).

If V15.1 is poorly calibrated, MCTS has room to correct the leaf
estimates and produce improved-policy targets for training.

Pass criteria (same as V13.x):
  - ECE ≤ 5pp
  - No bin > 10pp deviation from y=x
  - Brier ≤ 0.20

But note: for *this* experiment a PASS is BAD news (means MCTS won't
help). We want a FAIL (or at least MARGINAL) to justify the effort.

Usage:
    python -m experiments.mcts_v1.calibration_audit_v151 \\
        --model /Users/sumit/Github/AlphaLudo/td_ludo_v15/checkpoints/v151_sl/model_sl.pt \\
        --num-games 2000 \\
        --output runs/v151_sl_calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # td_ludo root
V15_ROOT = HERE.parent.parent.parent / "td_ludo_v15"
sys.path.insert(0, str(V15_ROOT))

import td_ludo_cpp as ludo_cpp
from td_ludo_v15.models.v15 import V15GraphTransformer
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov,
)

# Match the V13.x audit's discard window so the metric is comparable.
DISCARD_FIRST_N_TURNS = 10
MAX_TURNS_PER_GAME = 400
_BASE_POS = -1
_HOME_POS = 99


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--num-games", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    # V15.1 defaults
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=256)
    p.add_argument("--history-len", type=int, default=2)
    p.add_argument("--output", default="runs/v151_calibration.json")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--label", default="V15.1",
                   help="Display name for plots/output (e.g. 'V15.1 SL' or 'V15.1 RL')")
    return p.parse_args()


def pick_device(name):
    if name in ("cuda", "cpu", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_v151(path, device, *, d_model, n_heads, n_layers, ffn_dim, history_len):
    print(f"[Audit] Loading V15.1 from {path}...")
    model = V15GraphTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        ffn_dim=ffn_dim, history_len=history_len,
    )
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"[Audit] Loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


# ── State encoding ──────────────────────────────────────────────────────


def encode_pair(game, prev_game):
    """Build (2, 15, 15, 3) input tensor + (225,) cell legal mask."""
    cp = int(game.current_player)
    out = np.zeros((2, 15, 15, 3), dtype=np.float32)
    if prev_game is not None:
        out[0] = encode_frame(prev_game, pov_player=cp).astype(np.float32)
    out[1] = encode_frame(game, pov_player=cp).astype(np.float32)
    mask = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
    legal = ludo_cpp.get_legal_moves(game)
    cell_to_tokens: dict[int, list[int]] = {}
    for tok in legal:
        pos = int(game.player_positions[cp][tok])
        if pos == _HOME_POS:
            continue
        c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
        ci = cell_to_index(*c)
        mask[ci] = 1.0
        cell_to_tokens.setdefault(ci, []).append(int(tok))
    return out, mask, cell_to_tokens, legal


def cell_policy_to_tokens(cell_policy_225, cell_to_tokens, legal_tokens):
    """Project (225,) cell prob → (4,) per-token prob.

    Multiple tokens at the same cell split that cell's mass evenly.
    Illegal tokens get 0. Renormalize over legal tokens.
    """
    tok_p = np.zeros(4, dtype=np.float32)
    for ci, tokens in cell_to_tokens.items():
        share = cell_policy_225[ci] / max(1, len(tokens))
        for t in tokens:
            tok_p[t] += share
    s = tok_p.sum()
    return tok_p / s if s > 1e-9 else (
        np.array([1.0 if t in legal_tokens else 0.0 for t in range(4)], dtype=np.float32)
        / max(1, len(legal_tokens))
    )


# ── Game-advance helpers ────────────────────────────────────────────────


def _advance_to_decision(game, rng):
    while not game.is_terminal:
        if game.current_dice_roll == 0:
            game.current_dice_roll = int(rng.integers(1, 7))
        legal = ludo_cpp.get_legal_moves(game)
        if legal:
            return True
        cp = int(game.current_player)
        nxt = (cp + 1) % 4
        while not game.active_players[nxt]:
            nxt = (nxt + 1) % 4
        game.current_player = nxt
        game.current_dice_roll = 0
    return False


class _GameSnap:
    """Minimal snapshot of GameState fields the encoder reads.

    Avoids deepcopy of the C-bound GameState; used as `prev_game` for
    the 2-frame V15.1 encoder.
    """
    __slots__ = ("player_positions", "scores", "active_players",
                 "current_player", "current_dice_roll", "is_terminal")

    def __init__(self, game):
        self.player_positions = np.array(game.player_positions, dtype=np.int8).copy()
        self.scores = np.array(game.scores, dtype=np.int8).copy()
        self.active_players = np.array(game.active_players, dtype=bool).copy()
        self.current_player = int(game.current_player)
        self.current_dice_roll = int(game.current_dice_roll)
        self.is_terminal = bool(getattr(game, "is_terminal", False))


# ── Self-play data collection ───────────────────────────────────────────


def collect_calibration_data(model, device, num_games, batch_size, temperature, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = ludo_cpp.VectorGameState(batch_size=batch_size, two_player_mode=True)
    pending = [[] for _ in range(batch_size)]
    turn_count = np.zeros(batch_size, dtype=np.int32)
    prev_games: list[_GameSnap | None] = [None] * batch_size

    finalized_v: list[float] = []
    finalized_o: list[float] = []
    finalized_cp: list[int] = []
    games_done = 0

    t_start = time.time()
    last_log = t_start
    print(f"[Audit] Collecting from {num_games} games, batch={batch_size}, τ={temperature}...")

    while games_done < num_games:
        decision_idxs = []
        encoded_x = []
        masks225 = []
        cell_to_tokens_per = []
        legals_per = []
        cps_batch = []
        prevs_batch = []
        for i in range(batch_size):
            game = env.get_game(i)
            if game.is_terminal or turn_count[i] >= MAX_TURNS_PER_GAME:
                continue
            ok = _advance_to_decision(game, rng)
            if not ok:
                continue
            x, mask, c2t, legal = encode_pair(game, prev_games[i])
            decision_idxs.append(i)
            encoded_x.append(x)
            masks225.append(mask)
            cell_to_tokens_per.append(c2t)
            legals_per.append(legal)
            cps_batch.append(int(game.current_player))
            prevs_batch.append(prev_games[i])

        if encoded_x:
            xb = torch.from_numpy(np.stack(encoded_x, axis=0)).to(device)
            mb = torch.from_numpy(np.stack(masks225, axis=0)).to(device)
            with torch.no_grad():
                policy225, win_prob = model(xb, mb)
            policy225_np = policy225.cpu().numpy()
            v_pred_np = win_prob.cpu().numpy().reshape(-1)
        else:
            policy225_np = np.zeros((0, NUM_BOARD_CELLS), dtype=np.float32)
            v_pred_np = np.zeros((0,), dtype=np.float32)

        actions = [-1] * batch_size
        for k, i in enumerate(decision_idxs):
            cp = cps_batch[k]
            v_pred = float(v_pred_np[k])
            tok_p = cell_policy_to_tokens(
                policy225_np[k], cell_to_tokens_per[k], legals_per[k]
            )
            mask4 = np.zeros(4, dtype=np.float32)
            for t in legals_per[k]:
                mask4[t] = 1.0
            tok_p = tok_p * mask4
            s = tok_p.sum()
            tok_p = tok_p / s if s > 1e-9 else mask4 / mask4.sum()

            if abs(temperature - 1.0) > 1e-6:
                p_t = np.power(tok_p + 1e-12, 1.0 / temperature)
                p_t = np.where(mask4 > 0, p_t, 0.0)
                tok_p = p_t / max(p_t.sum(), 1e-9)

            action = int(rng.choice(4, p=tok_p))
            if mask4[action] == 0:
                legal_idx = np.where(mask4 > 0)[0]
                action = int(rng.choice(legal_idx))
            actions[i] = action

            if turn_count[i] >= DISCARD_FIRST_N_TURNS:
                pending[i].append((cp, v_pred))

            # Snapshot the pre-step state as the prev_game for the next decision
            # in this game. Done BEFORE env.step mutates the C-bound state.
            prev_games[i] = _GameSnap(env.get_game(i))

        _, _, _, infos = env.step(actions)
        for i in range(batch_size):
            if actions[i] >= 0:
                turn_count[i] += 1

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
                prev_games[i] = None
                games_done += 1
                if games_done >= num_games:
                    break

        now = time.time()
        if now - last_log > 5.0:
            elapsed = now - t_start
            gpm = (games_done / elapsed) * 60.0 if elapsed > 0 else 0
            states_per_game = (len(finalized_v) / max(1, games_done))
            print(f"  [{games_done:>5}/{num_games}]  done_states={len(finalized_v):>7} "
                  f"gpm={gpm:>6.1f}  states/game={states_per_game:.1f}  "
                  f"elapsed={elapsed:.0f}s")
            last_log = now

    print(f"[Audit] Done: {games_done} games, {len(finalized_v)} states, "
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


# ── Metrics + verdict (verbatim from V13.x audit) ───────────────────────


def compute_calibration(v_preds, outcomes, num_bins):
    n = len(v_preds)
    if n == 0:
        raise RuntimeError("No states collected — bug?")
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
        wr = float(o_bin.mean())
        n_bin = len(o_bin)
        wr_se = (wr * (1 - wr) / n_bin) ** 0.5 if n_bin > 0 else 0.0
        dev = abs(wr - v_mean)
        ece += dev * (n_bin / n)
        bins.append({
            "bin_idx": b,
            "v_pred_mean": v_mean,
            "v_pred_range": [float(v_bin.min()), float(v_bin.max())],
            "empirical_wr": wr,
            "deviation_pp": 100.0 * dev,
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


def verdict(calib):
    ece = calib["ece_pp"]
    max_dev = calib["max_bin_deviation_pp"]
    brier = calib["brier_score"]
    if ece <= 5.0 and max_dev <= 10.0 and brier <= 0.20:
        return "PASS", (f"ECE={ece:.2f}pp ≤ 5, max_bin_dev={max_dev:.2f}pp ≤ 10, "
                        f"Brier={brier:.3f} ≤ 0.20. Value head IS well-calibrated. "
                        f"For MCTS purposes: BAD news — search will agree with the prior. "
                        f"Same coherent-equilibrium failure mode as V13.5+MCTS expected.")
    if ece <= 10.0 and max_dev <= 15.0 and brier <= 0.22:
        return "MARGINAL", (f"ECE={ece:.2f}pp, max_bin_dev={max_dev:.2f}pp, "
                            f"Brier={brier:.3f}. Calibration is so-so — MCTS has SOME "
                            f"room to correct the value estimates. Proceed but expect "
                            f"modest gains.")
    return "FAIL", (f"ECE={ece:.2f}pp, max_bin_dev={max_dev:.2f}pp, Brier={brier:.3f}. "
                    f"Value head is poorly calibrated. For MCTS purposes: GOOD news — "
                    f"search has lots of room to correct the value estimates. Proceed.")


def maybe_plot(calib, out_path, label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    bins = calib["bins"]
    v_means = [b["v_pred_mean"] for b in bins]
    wrs = [b["empirical_wr"] for b in bins]
    err_pp = [b["wr_stderr_pp"] / 100.0 for b in bins]
    n_samples = [b["n_samples"] for b in bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.errorbar(v_means, wrs, yerr=err_pp, fmt="o-", color="steelblue",
                 markersize=8, capsize=4, label="Empirical WR (±1σ)")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration (y=x)")
    ax1.fill_between([0, 1], [0.05, 1.05], [-0.05, 0.95], color="green", alpha=0.1,
                     label="±5pp band")
    ax1.set_xlabel("V_pred (binned, mean per bin)")
    ax1.set_ylabel("Empirical win-rate")
    ax1.set_title(
        f"{label} calibration ({calib['num_states']:,} states)\n"
        f"ECE={calib['ece_pp']:.2f}pp · Brier={calib['brier_score']:.3f} · "
        f"max_dev={calib['max_bin_deviation_pp']:.2f}pp"
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax2.bar(range(len(n_samples)), n_samples, color="darkorange", alpha=0.7)
    ax2.set_xlabel("V_pred decile")
    ax2.set_ylabel("samples")
    ax2.set_title("Sample count per bin")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Audit] Plot → {out_path}")


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"[Audit] Device: {device}  Label: {args.label}")

    model = load_v151(
        args.model, device,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ffn_dim=args.ffn_dim, history_len=args.history_len,
    )

    v_preds, outcomes, meta = collect_calibration_data(
        model, device, args.num_games, args.batch_size,
        args.temperature, args.seed,
    )

    calib = compute_calibration(v_preds, outcomes, args.num_bins)
    v_name, v_reason = verdict(calib)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "label": args.label,
        "model_path": str(args.model),
        "arch": {
            "d_model": args.d_model, "n_heads": args.n_heads,
            "n_layers": args.n_layers, "ffn_dim": args.ffn_dim,
            "history_len": args.history_len,
        },
        "num_games_requested": args.num_games,
        "num_games_played": meta["num_games"],
        "num_states": meta["num_states"],
        "states_per_game": meta["states_per_game"],
        "elapsed_sec": meta["elapsed_sec"],
        "calibration": calib,
        "verdict": v_name,
        "verdict_reason": v_reason,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Audit] Report → {out_path}")
    print(f"[Audit] VERDICT: {v_name}")
    print(f"        {v_reason}")
    print(f"\n[Audit] Summary:  ECE={calib['ece_pp']:.2f}pp  "
          f"Brier={calib['brier_score']:.3f}  "
          f"max_bin_dev={calib['max_bin_deviation_pp']:.2f}pp  "
          f"({calib['num_states']:,} states)")

    if not args.no_plot:
        plot_path = out_path.with_suffix(".png")
        maybe_plot(calib, plot_path, args.label)


if __name__ == "__main__":
    main()
