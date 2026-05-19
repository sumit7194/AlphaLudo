"""Quick H2H test for ExpectimaxBot + MCTSPureBot.

Plays each new bot against the scripted bots (Heuristic + Expert) to verify:
  1. Both new bots play coherent Ludo (no crashes, games terminate)
  2. They're at least as strong as the heuristic bots
  3. They play QUALITATIVELY different (different argmax preferences)

Light load — 50 games per matchup, takes a few minutes total.

Usage:
    python test_strong_bots.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

# Add the repo root so we can import td_ludo
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import td_ludo_cpp as cpp
from td_ludo.game.heuristic_bot import HeuristicLudoBot, ExpertBot, RandomBot
from td_ludo.game.strong_bots import ExpectimaxBot, MCTSPureBot

MAX_MOVES = 400


def play_game(bot_p0, bot_p2, seed):
    """Play one 2-player game. Returns winner (0 or 2) or -1 for truncation."""
    random.seed(seed)
    state = cpp.create_initial_state_2p()
    consec_sixes = [0, 0, 0, 0]
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
                consec_sixes[cp] += 1
                if consec_sixes[cp] >= 3:
                    consec_sixes[cp] = 0
                    n = (cp + 1) % 4
                    while not state.active_players[n]:
                        n = (n + 1) % 4
                    state.current_player = n
                    state.current_dice_roll = 0
                    continue
            else:
                consec_sixes[cp] = 0
            state.current_dice_roll = d
        legal = cpp.get_legal_moves(state)
        if not legal:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            state.current_dice_roll = 0
            continue
        bot = bot_p0 if cp == 0 else bot_p2
        action = bot.select_move(state, list(legal))
        state = cpp.apply_move(state, int(action))
        mc += 1
    if state.is_terminal:
        return int(cpp.get_winner(state))
    return -1


def run_h2h(name_a, make_a, name_b, make_b, n_games=50, log_every=10):
    """Play `n_games`, half with A as P0, half as P2. Returns (a_wins, b_wins, draws)."""
    a_wins = b_wins = draws = 0
    t0 = time.time()
    for g in range(n_games):
        a_is_p0 = (g % 2 == 0)
        if a_is_p0:
            winner = play_game(make_a(player_id=0), make_b(player_id=2), seed=42 + g // 2)
            a_pid, b_pid = 0, 2
        else:
            winner = play_game(make_b(player_id=0), make_a(player_id=2), seed=42 + g // 2)
            a_pid, b_pid = 2, 0
        if winner == a_pid:
            a_wins += 1
        elif winner == b_pid:
            b_wins += 1
        else:
            draws += 1
        if (g + 1) % log_every == 0:
            total = a_wins + b_wins + draws
            print(f"    [{g + 1:>3}/{n_games}] {name_a} {100 * a_wins / total:5.1f}%  "
                  f"({a_wins}-{b_wins}, draws {draws})", flush=True)
    elapsed = time.time() - t0
    total = a_wins + b_wins + draws
    print(f"  FINAL: {name_a} {100 * a_wins / total:5.1f}%  vs  {name_b} {100 * b_wins / total:5.1f}%  "
          f"(draws {draws})  [{elapsed:.1f}s, {n_games/elapsed:.2f} g/s]")
    return a_wins, b_wins, draws


# Bot factories — match the `__init__(player_id=None)` signature
def make_heuristic(player_id=None):  return HeuristicLudoBot(player_id=player_id)
def make_expert(player_id=None):     return ExpertBot(player_id=player_id)
def make_random(player_id=None):     return RandomBot(player_id=player_id)
def make_expectimax(player_id=None): return ExpectimaxBot(player_id=player_id)
def make_mcts_pure_fast(player_id=None):
    return MCTSPureBot(player_id=player_id, n_sims=30, rollouts_per_leaf=4)
def make_mcts_pure_strong(player_id=None):
    return MCTSPureBot(player_id=player_id, n_sims=50, rollouts_per_leaf=8)


if __name__ == "__main__":
    print("=" * 60)
    print("Strong-bot smoke test — H2H vs scripted bots")
    print("=" * 60)

    matchups = [
        # Sanity: each new bot vs Random — should crush
        ("Expectimax",       make_expectimax,       "Random",     make_random, 20),
        ("MCTSPure(30/4)",   make_mcts_pure_fast,   "Random",     make_random, 20),

        # Strength: each new bot vs Heuristic
        ("Expectimax",       make_expectimax,       "Heuristic",  make_heuristic, 50),
        ("MCTSPure(30/4)",   make_mcts_pure_fast,   "Heuristic",  make_heuristic, 50),

        # Strength: each new bot vs Expert (the strongest scripted)
        ("Expectimax",       make_expectimax,       "Expert",     make_expert, 50),
        ("MCTSPure(30/4)",   make_mcts_pure_fast,   "Expert",     make_expert, 50),

        # Different-from-each-other? Direct H2H
        ("Expectimax",       make_expectimax,       "MCTSPure",   make_mcts_pure_fast, 50),

        # If time, strong MCTS sanity
        ("MCTSPure(50/8)",   make_mcts_pure_strong, "Expert",     make_expert, 20),
    ]

    for a_name, a_make, b_name, b_make, n in matchups:
        print(f"\n━━ {a_name} vs {b_name} ({n} games) ━━")
        run_h2h(a_name, a_make, b_name, b_make, n_games=n, log_every=max(5, n // 5))

    print("\n" + "=" * 60)
    print("Done.")
