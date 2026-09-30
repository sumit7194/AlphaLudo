#!/usr/bin/env python3
"""Probe: is a STRONGER policy reachable? (target-B attack-angle step 1)

H2H the V13.6 champion (neural, greedy) vs strong SEARCH bots that use an
INDEPENDENT heuristic eval (not the net's own value head — that's why this
isn't the Exp-45 coherent-equilibrium wall). If a search bot clearly beats
the champion (>~55%), a stronger policy is REACHABLE and distillation is a
fast path. If ~50%, the net already matches the best search → the
world-model/understanding route is the necessary path, not distillation.

One bot per invocation (so 3 can run concurrently). Greedy neural play;
dice supply variance; mirrored seeds for seat/dice fairness.

Usage:
    python probe_champion_vs_search.py --bot MCTSExpectimaxPrior --games 300 \
        --champion checkpoints/.../model.pt --out probe_<bot>.json
"""
import argparse
import collections
import json
import math
import random
import time

import numpy as np
import td_ludo_cpp as cpp

import gen_tournament as G  # reuse the proven arch-probing loader + neural pick
from td_ludo.game.strong_bots import STRONG_BOT_REGISTRY

MAX_MOVES = 400


def make_bot_pick(bot):
    def pick(state, legal):
        bot.player_id = int(state.current_player)
        return bot.select_move(state, list(legal))
    return pick


def play_one(pick_p0, pick_p2, seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    state = cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]
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
            d = random.randint(1, 6)
            if d == 6:
                csix[cp] += 1
                if csix[cp] >= 3:
                    csix[cp] = 0
                    n = (cp + 1) % 4
                    while not state.active_players[n]:
                        n = (n + 1) % 4
                    state.current_player = n
                    state.current_dice_roll = 0
                    continue
            else:
                csix[cp] = 0
            state.current_dice_roll = d
        legal = cpp.get_legal_moves(state)
        if not legal:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            state.current_dice_roll = 0
            continue
        fn = pick_p0 if cp == 0 else pick_p2
        state = cpp.apply_move(state, int(fn(state, list(legal))))
        mc += 1
    return int(cpp.get_winner(state)) if state.is_terminal else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True)
    ap.add_argument("--champion", required=True)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    champ = G.load_model(args.champion)
    champ_pick = lambda s, l: G.pick(champ, s, l)
    bot = STRONG_BOT_REGISTRY[args.bot]()
    bot_pick = make_bot_pick(bot)

    half = args.games // 2
    champ_wins = bot_wins = draws = 0
    t0 = time.time()
    for g in range(half * 2):
        champ_p0 = (g % 2 == 0)
        seed = 1000 + (g // 2)
        if champ_p0:
            w = play_one(champ_pick, bot_pick, seed); cpid, bpid = 0, 2
        else:
            w = play_one(bot_pick, champ_pick, seed); cpid, bpid = 2, 0
        if w == cpid:
            champ_wins += 1
        elif w == bpid:
            bot_wins += 1
        else:
            draws += 1
        if (g + 1) % 50 == 0:
            t = champ_wins + bot_wins + draws
            print(f"  [{g+1}/{half*2}] champion {100*champ_wins/t:.1f}% vs {args.bot} "
                  f"({champ_wins}-{bot_wins}, d{draws})", flush=True)

    n = champ_wins + bot_wins
    p = champ_wins / n if n else 0.0
    se = math.sqrt(p * (1 - p) / n) * 100 if n else 0.0
    out = {
        "bot": args.bot, "games": args.games,
        "champion_winpct": round(100 * p, 1),
        "champion_winpct_se": round(se, 1),
        "champion_wins": champ_wins, "bot_wins": bot_wins, "draws": draws,
        "elapsed_s": round(time.time() - t0),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nRESULT champion vs {args.bot}: {out['champion_winpct']}% "
          f"+/- {out['champion_winpct_se']}pp  ({champ_wins}-{bot_wins}, "
          f"{out['elapsed_s']}s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
