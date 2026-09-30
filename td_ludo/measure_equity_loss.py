"""Per-decision EQUITY-LOSS metric for v15.2 (SQUEEZE_RESEARCH lever 1).

Quantifies "how much could it play better": over many real decisions, how much
win-probability does v15.2's greedy move leave on the table vs the rollout-best
move? This is backgammon's "PR / error-rate" idea — every unforced decision is a
datapoint, so a few hundred games yield thousands of samples (far more sensitive
than win-rate).

For each sampled decision position (deciding player to move, >=2 legal):
  - reference equity of each legal move m = win-rate of `deciding` from R greedy
    self-play rollouts after playing m (rollouts share nothing but the dice → the
    lookahead is what reveals suboptimality the flat value head misses).
  - model move = v15.2 policy argmax.
  - equity_loss = rolloutEq(best move) - rolloutEq(model move)   (>= 0)
Mean equity_loss / decision = the looseness. Rollouts are batched per position.

Usage: cd td_ludo && PYTHONPATH=.:../td_ludo_v15 ./td_env/bin/python -u measure_equity_loss.py --positions 120 --rollouts 40
"""
from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "td_ludo_v15"))
import td_ludo_cpp as cpp
from td_ludo.game.strong_bots import _clone_state
from inference_search import V15Eval

MAX_MOVES = 400
V152 = HERE.parent / "checkpoint_backups/v152_gaeterminal_2026-06-19/v152_gaeterminal_BEST_eval84.65pct_2026-06-19.pt"


def _advance_no_decision(s, rng, csix):
    """Roll/skip until current player has a real decision. Returns legal list or
    None if terminal. Mutates s (dice/current_player)."""
    while not s.is_terminal:
        cp = int(s.current_player)
        if not s.active_players[cp]:
            n = (cp + 1) % 4
            while not s.active_players[n]: n = (n + 1) % 4
            s.current_player = n; continue
        if s.current_dice_roll == 0:
            d = rng.randint(1, 6)
            if d == 6:
                csix[cp] += 1
                if csix[cp] >= 3:
                    csix[cp] = 0; n = (cp + 1) % 4
                    while not s.active_players[n]: n = (n + 1) % 4
                    s.current_player = n; s.current_dice_roll = 0; continue
            else: csix[cp] = 0
            s.current_dice_roll = d
        legal = [int(x) for x in cpp.get_legal_moves(s)]
        if not legal:
            n = (cp + 1) % 4
            while not s.active_players[n]: n = (n + 1) % 4
            s.current_player = n; s.current_dice_roll = 0; continue
        return legal
    return None


def batched_greedy_rollout(ev, jobs, decider):
    """jobs: list of [state, rng, csix, alive]. Play greedy self-play (v15 argmax)
    to terminal, batching policy calls across all alive jobs each step. Returns
    np.array of 1.0/0.0 = did `decider` win each job."""
    res = [None] * len(jobs)
    steps = 0
    while steps < MAX_MOVES * 2:
        steps += 1
        pending = []  # (job_idx, legal)
        for i, jb in enumerate(jobs):
            if res[i] is not None: continue
            s, rng, csix = jb
            legal = _advance_no_decision(s, rng, csix)
            if legal is None:
                w = cpp.get_winner(s)
                res[i] = 1.0 if w == decider else 0.0
                continue
            if len(legal) == 1:
                jb[0] = cpp.apply_move(s, legal[0])
                # carry rng/csix; re-loop next pass
                pending.append((i, None))
            else:
                pending.append((i, legal))
        decide = [(jobs[i][0], lg, int(jobs[i][0].current_player)) for i, lg in pending if lg]
        if decide:
            moves = ev.policy_argmax_batch(decide)
            k = 0
            for i, lg in pending:
                if lg:
                    jobs[i][0] = cpp.apply_move(jobs[i][0], int(moves[k])); k += 1
        if all(r is not None for r in res):
            break
    for i in range(len(res)):
        if res[i] is None: res[i] = 0.5  # unfinished (rare) → neutral
    return np.array(res, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=120)
    ap.add_argument("--rollouts", type=int, default=40)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ev = V15Eval(str(V152))
    rng0 = random.Random(args.seed)
    print(f"v15.2 | positions {args.positions} | rollouts/move {args.rollouts}\n", flush=True)

    # ── sample decision positions from greedy self-play ──
    positions = []  # (state_snapshot, decider, legal)
    g = 0
    while len(positions) < args.positions:
        g += 1
        s = cpp.create_initial_state_2p(); csix = [0, 0, 0, 0]; rng = random.Random(1000 + g); mc = 0
        while not s.is_terminal and mc < MAX_MOVES and len(positions) < args.positions:
            legal = _advance_no_decision(s, rng, csix)
            if legal is None: break
            if len(legal) >= 2 and rng0.random() < 0.15:   # sample ~15% of unforced decisions
                positions.append((_clone_state(s), int(s.current_player), list(legal)))
            mv = ev.policy_argmax(s, legal, int(s.current_player)) if len(legal) > 1 else legal[0]
            s = cpp.apply_move(s, int(mv)); mc += 1
    print(f"sampled {len(positions)} decisions from {g} games", flush=True)

    losses = []; t0 = time.time()
    for pi, (s0, decider, legal) in enumerate(positions):
        model_move = ev.policy_argmax(s0, legal, decider)
        # R rollouts per legal move, grouped move-then-rollout.
        jobs = []
        for m in legal:
            after = cpp.apply_move(s0, int(m))
            for r in range(args.rollouts):
                jobs.append([_clone_state(after), random.Random(7000 + pi * 1000 + int(m) * 100 + r), [0, 0, 0, 0]])
        wins = batched_greedy_rollout(ev, jobs, decider)
        # DOUBLE-SAMPLING: pick best on half A, score the gap on half B (independent
        # → no winner's-curse bias). Per-decision loss can be slightly negative
        # (noise); the MEAN is an unbiased estimate of true regret (>=0).
        R = args.rollouts; half = R // 2
        qA, qB = {}, {}
        for j, m in enumerate(legal):
            w = wins[j * R:(j + 1) * R]
            qA[m] = float(w[:half].mean()); qB[m] = float(w[half:].mean())
        best_A = max(legal, key=lambda m: qA[m])
        loss = qB[best_A] - qB[model_move]
        losses.append(loss)
        if (pi + 1) % 20 == 0:
            print(f"  {pi+1}/{len(positions)} | running mean equity-loss {np.mean(losses)*100:.2f}% "
                  f"| {(pi+1)/(time.time()-t0):.1f} pos/s", flush=True)

    losses = np.array(losses)
    se = losses.std() / max(1, len(losses)) ** 0.5
    clipped = np.maximum(losses, 0)  # per-decision regret floored (biased high, for context only)
    print("\n" + "=" * 56)
    print(f"  MEAN equity-loss / decision : {losses.mean()*100:.2f}%  ± {se*100:.2f}pp  (unbiased)")
    print(f"     (this = avg win-prob v15.2 leaves on the table per move)")
    print(f"  fraction of decisions where a clearly-better move existed (B-gap >3%): {(losses > 0.03).mean()*100:.1f}%")
    print(f"  [context] floored-mean (biased high): {clipped.mean()*100:.2f}%")
    print(f"  decisions: {len(losses)} | rollouts/move: {args.rollouts} (half pick / half score) | {time.time()-t0:.0f}s")
    print("=" * 56)


if __name__ == "__main__":
    main()
