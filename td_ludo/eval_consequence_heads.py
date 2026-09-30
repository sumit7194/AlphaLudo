#!/usr/bin/env python3
"""Did the V13.6 world-model heads actually learn? (Exp 52 validation step 1)

Self-plays the world-model checkpoint, and on each decision state compares the
model's capture-prob / cells-at-risk head outputs against the engine ground
truth (consequence_targets.compute_per_token_targets) on valid token slots.
Reports MAE + Pearson correlation. High correlation ⇒ the trunk learned to
ENCODE capture risk (the prerequisite for acting on it — curing laggard
neglect). Low ⇒ the aux loss didn't take.

Usage: python eval_consequence_heads.py --ckpt <path> --games 60
"""
import argparse
import random

import numpy as np
import torch
import td_ludo_cpp as cpp

import gen_tournament as G  # proven arch-probing loader + neural pick
from td_ludo.game.encoder_v18_production import encode_state_v18_production
from td_ludo.game.consequence_targets import compute_per_token_targets

MAX_MOVES = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=60)
    args = ap.parse_args()

    model = G.load_model(args.ckpt)  # V135ProductionAdapter, eval mode

    cap_pred, cap_true, risk_pred, risk_true = [], [], [], []
    random.seed(7); np.random.seed(7)
    for gi in range(args.games):
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
            # Collect head outputs vs ground truth for the current player.
            enc = encode_state_v18_production(state).astype(np.float32)
            tl = np.zeros(4, dtype=np.float32)
            for a in legal:
                tl[a] = 1.0
            with torch.no_grad():
                out = model(torch.from_numpy(enc).unsqueeze(0),
                            torch.from_numpy(tl).unsqueeze(0))
            cap_h = out[4].squeeze(0).numpy()   # token-space capture pred
            risk_h = out[5].squeeze(0).numpy()
            ct, rt, valid = compute_per_token_targets(state, cp)
            for t in range(4):
                if valid[t] > 0:
                    cap_pred.append(float(cap_h[t])); cap_true.append(float(ct[t]))
                    risk_pred.append(float(risk_h[t])); risk_true.append(float(rt[t]))
            # greedy-ish move
            state = cpp.apply_move(state, int(G.pick(model, state, list(legal))))
            mc += 1

    cap_pred = np.array(cap_pred); cap_true = np.array(cap_true)
    risk_pred = np.array(risk_pred); risk_true = np.array(risk_true)

    def stats(pred, true, name):
        mae = float(np.mean(np.abs(pred - true)))
        if pred.std() > 1e-6 and true.std() > 1e-6:
            corr = float(np.corrcoef(pred, true)[0, 1])
        else:
            corr = float("nan")
        # baseline: predicting the mean target
        base_mae = float(np.mean(np.abs(true - true.mean())))
        print(f"  {name}: n={len(pred)}  MAE={mae:.4f}  corr={corr:.3f}  "
              f"(mean-baseline MAE={base_mae:.4f}, true_mean={true.mean():.3f})")

    print(f"\n=== consequence-head accuracy ({args.games} self-play games) ===")
    stats(cap_pred, cap_true, "capture_prob")
    stats(risk_pred, risk_true, "cells_at_risk")
    print("\nHigh corr + MAE well below mean-baseline ⇒ heads learned to "
          "predict risk (trunk now encodes it).")


if __name__ == "__main__":
    main()
