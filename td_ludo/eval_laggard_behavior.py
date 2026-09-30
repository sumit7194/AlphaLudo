#!/usr/bin/env python3
"""Objective laggard-protection probe (Exp 52 validation step 2, automated).

The behavioral question the play-test asks, quantified: when the model faces a
position where it COULD rescue an endangered own token (move it from capture-
danger to safety) and has a real choice, how often does it? Measures the
tested model as P0 vs a FIXED opponent (so world-model and plain champion
face comparable positions), greedy play.

  rescue_rate          : of rescue-available decisions, fraction it rescues
  laggard_rescue_rate   : same, restricted to when the rescuable endangered
                          token is the TRAILING (lowest-pos) own token — the
                          exact flaw
  avg_post_move_risk    : mean own cells-at-risk after its move (lower = more
                          protective); secondary signal

A higher rescue/laggard rate for v13_6_wm than v13_6 ⇒ the consequence heads
changed how it PLAYS (acts on the learned risk), not just what it predicts.

Usage: python eval_laggard_behavior.py --model <wm.pt> --opponent <plain.pt> --games 200
"""
import argparse
import random

import numpy as np
import td_ludo_cpp as cpp

import gen_tournament as G
from td_ludo.game.consequence_targets import (
    capture_prob_next_turn, compute_per_token_targets,
)
from td_ludo.game.bias_penalties import _is_main_track

MAX_MOVES = 400
DANGER_THRESH = 2.0 / 6.0   # "endangered" = opp can capture with ≥2 dice values


def total_own_risk(state, player):
    cap, car, valid = compute_per_token_targets(state, player)
    return float((car * valid).sum())


def analyze_decision(state, player, legal):
    """Return (rescue_moves set, endangered_tokens, laggard_token) for the
    current (state, dice). A rescue move takes an endangered own token to
    safety (post-move capture_prob 0)."""
    own = state.player_positions[player]
    endangered = []
    for t in range(4):
        pos = int(own[t])
        if _is_main_track(pos) and capture_prob_next_turn(state, player, pos) >= DANGER_THRESH:
            endangered.append(t)
    # trailing (laggard) = lowest-position own token currently on the track
    on_track = [(int(own[t]), t) for t in range(4) if _is_main_track(int(own[t]))]
    laggard = min(on_track)[1] if on_track else None

    rescue_moves = set()
    for t in legal:
        if t not in endangered:
            continue
        nxt = cpp.apply_move(state, int(t))
        new_pos = int(nxt.player_positions[player][t])
        # safe if off-track (scored) or capture_prob now 0
        if not _is_main_track(new_pos) or capture_prob_next_turn(nxt, player, new_pos) == 0.0:
            rescue_moves.add(t)
    return rescue_moves, endangered, laggard


def run(model, opp, games, label):
    n_resc_avail = n_resc = 0
    n_lag_avail = n_lag = 0
    post_risk_sum = 0.0
    post_risk_n = 0
    random.seed(123); np.random.seed(123)
    for gi in range(games):
        state = cpp.create_initial_state_2p()
        mc = 0
        while not state.is_terminal and mc < MAX_MOVES:
            cp = int(state.current_player)
            if not state.active_players[cp]:
                n = (cp + 1) % 4
                while not state.active_players[n]:
                    n = (n + 1) % 4
                state.current_player = n
                continue
            if state.current_dice_roll == 0:
                state.current_dice_roll = random.randint(1, 6)
            legal = cpp.get_legal_moves(state)
            if not legal:
                state.current_dice_roll = 0
                n = (cp + 1) % 4
                while not state.active_players[n]:
                    n = (n + 1) % 4
                state.current_player = n
                continue
            picker = model if cp == 0 else opp
            move = G.pick(picker, state, list(legal))
            # Only analyze the TESTED model's (P0) genuine choices.
            if cp == 0 and len(legal) >= 2:
                rescue_moves, endangered, laggard = analyze_decision(state, cp, list(legal))
                if rescue_moves:
                    n_resc_avail += 1
                    if move in rescue_moves:
                        n_resc += 1
                    if laggard is not None and laggard in rescue_moves:
                        n_lag_avail += 1
                        if move == laggard:
                            n_lag += 1
                nxt = cpp.apply_move(state, int(move))
                post_risk_sum += total_own_risk(nxt, cp); post_risk_n += 1
            state = cpp.apply_move(state, int(move))
            mc += 1
    rr = 100 * n_resc / n_resc_avail if n_resc_avail else float("nan")
    lr = 100 * n_lag / n_lag_avail if n_lag_avail else float("nan")
    apr = post_risk_sum / post_risk_n if post_risk_n else float("nan")
    print(f"\n[{label}] {games} games vs fixed opponent")
    print(f"  rescue_rate         : {rr:.1f}%  ({n_resc}/{n_resc_avail} rescue-available)")
    print(f"  laggard_rescue_rate : {lr:.1f}%  ({n_lag}/{n_lag_avail})")
    print(f"  avg_post_move_risk  : {apr:.4f}  (lower = more protective)")
    return dict(rescue_rate=rr, laggard_rescue_rate=lr, avg_post_move_risk=apr,
                n_resc_avail=n_resc_avail, n_lag_avail=n_lag_avail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tested model (P0)")
    ap.add_argument("--opponent", required=True, help="fixed opponent (P2)")
    ap.add_argument("--label", default="model")
    ap.add_argument("--games", type=int, default=200)
    args = ap.parse_args()
    model = G.load_model(args.model)
    opp = G.load_model(args.opponent)
    run(model, opp, args.games, args.label)


if __name__ == "__main__":
    main()
