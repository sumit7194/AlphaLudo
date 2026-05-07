"""Round-robin tournament runner.

Each unordered pair of competitors plays N games. Seats are rotated
between games to control for first-mover advantage. Outputs:

  - Live per-pair progress
  - Final per-pair WR matrix
  - Aggregate WR ranking
  - JSON dump of full results (optional)

CLI examples:

  # Just the four historicals (defaults via OpponentRegistry)
  python -m td_ludo.experiments.tournament.run \\
    --hist V6_big,V6_1,V6_3,V10 \\
    --games-per-pair 1000

  # Add bots
  python -m td_ludo.experiments.tournament.run \\
    --hist V6_big,V10 \\
    --bots Expert,Heuristic,Random \\
    --games-per-pair 500

  # Add the current V12.2 model
  python -m td_ludo.experiments.tournament.run \\
    --hist V6_3,V10 \\
    --add-model V12_2:v122:play/model_weights/v12_2/model_latest.pt \\
    --games-per-pair 1000

  # Multiple custom checkpoints (each as NAME:ARCH_PRESET:PATH)
  python -m td_ludo.experiments.tournament.run \\
    --add-model A:v122:path/to/a.pt \\
    --add-model B:v10:path/to/b.pt \\
    --games-per-pair 500

  # Add the 14ch distilled student
  python -m td_ludo.experiments.tournament.run \\
    --hist all --add-model V12_2:v122:play/model_weights/v12_2/model_latest.pt \\
    --add-model Distill14:v14_minimal:experiments/distillation_14ch/student_14ch_final.pt

  # Save results to JSON
  python -m td_ludo.experiments.tournament.run \\
    --hist all --bots Expert \\
    --output runs/tournament_001.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from typing import List, Tuple

import numpy as np
import torch

import td_ludo_cpp as cpp

from experiments.tournament.agents import (
    HistAgent, ModelAgent, BotAgent, list_arch_presets,
)
from td_ludo.game.players.opponent_registry import OpponentRegistry


MAX_MOVES_PER_GAME = 10000


# ---------------------------------------------------------------------------
#  2-player game loop (adapted from eval_teacher_tournament.py)
# ---------------------------------------------------------------------------

def play_game(agent_p0, agent_p2):
    """One 2-player Ludo game between agent_p0 (seat 0) and agent_p2
    (seat 2). Returns (winner_seat, total_moves). winner_seat is 0, 2,
    or -1 if the game timed out.
    """
    state = cpp.create_initial_state_2p()
    consec = {0: 0, 2: 0}
    moves = 0

    while not state.is_terminal and moves < MAX_MOVES_PER_GAME:
        cp = state.current_player

        if state.current_dice_roll == 0:
            state.current_dice_roll = random.randint(1, 6)
            if state.current_dice_roll == 6:
                consec[cp] += 1
            else:
                consec[cp] = 0
            if consec[cp] >= 3:
                # Triple-six: pass turn, reset counter.
                nxt = (cp + 1) % 4
                while not state.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                state.current_player = nxt
                state.current_dice_roll = 0
                consec[cp] = 0
                continue

        legal = cpp.get_legal_moves(state)
        if not legal:
            nxt = (cp + 1) % 4
            while not state.active_players[nxt]:
                nxt = (nxt + 1) % 4
            state.current_player = nxt
            state.current_dice_roll = 0
            continue

        agent = agent_p0 if cp == 0 else agent_p2
        action = agent.select_move(state, legal, consec[cp])
        if action < 0:
            action = legal[0]
        state = cpp.apply_move(state, action)
        moves += 1

    if state.is_terminal:
        return cpp.get_winner(state), moves
    return -1, moves


def run_pair(a, b, n_games: int, verbose: bool = True):
    """Play n_games between a and b with seat rotation. Returns dict
    with wins/draws/lengths/per-seat breakdown."""
    a_wins = 0; b_wins = 0; draws = 0
    lengths: List[int] = []
    seat = {'a_p0': [0, 0], 'a_p2': [0, 0]}  # [wins, total]
    t0 = time.time()

    for i in range(n_games):
        a_at_p0 = (i % 2 == 0)
        if a_at_p0:
            winner, m = play_game(a, b)
            if winner == 0:   a_wins += 1; seat['a_p0'][0] += 1
            elif winner == 2: b_wins += 1
            else:             draws += 1
            seat['a_p0'][1] += 1
        else:
            winner, m = play_game(b, a)
            if winner == 2:   a_wins += 1; seat['a_p2'][0] += 1
            elif winner == 0: b_wins += 1
            else:             draws += 1
            seat['a_p2'][1] += 1
        lengths.append(m)

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            gpm = (i + 1) / (elapsed / 60)
            wr = a_wins / (i + 1) * 100
            print(f"    [{i+1:>5}/{n_games}] {a.name} {wr:5.1f}% "
                  f"vs {b.name} | {gpm:5.0f} gpm", flush=True)

    return {
        'a': a.name, 'b': b.name, 'n_games': n_games,
        'a_wins': a_wins, 'b_wins': b_wins, 'draws': draws,
        'a_wr_pct': a_wins / max(1, n_games) * 100,
        'b_wr_pct': b_wins / max(1, n_games) * 100,
        'avg_length': float(np.mean(lengths)) if lengths else 0.0,
        'seat_a_p0': seat['a_p0'],
        'seat_a_p2': seat['a_p2'],
    }


# ---------------------------------------------------------------------------
#  CLI + competitor builder
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Round-robin tournament between Ludo agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--hist', default='',
        help="Comma-separated list of historical model tags (without the "
             "'Hist_' prefix). Use 'all' for every tag in OpponentRegistry. "
             "Examples: 'V6_big,V10' or 'all'.",
    )
    p.add_argument(
        '--bots', default='',
        help="Comma-separated bot names. Use 'all' for "
             "Expert,Heuristic,Aggressive,Defensive,Racing,Random.",
    )
    p.add_argument(
        '--add-model', action='append', default=[],
        metavar='NAME:ARCH:PATH',
        help="Add a custom checkpoint as a competitor. Format "
             "'NAME:ARCH:PATH'. ARCH must be one of: "
             f"{list_arch_presets()}. Repeatable.",
    )
    p.add_argument(
        '--games-per-pair', type=int, default=2000,
        help="Games per unordered pair (default 2000). Total games = "
             "C(N,2) * games_per_pair, where N is the number of competitors.",
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help="RNG seed for reproducibility (default 42).",
    )
    p.add_argument(
        '--device', type=str, default='cpu',
        help="Torch device (default cpu).",
    )
    p.add_argument(
        '--output', type=str, default=None,
        help="Optional path to write JSON results.",
    )
    p.add_argument(
        '--quiet', action='store_true',
        help="Suppress per-100-game progress output.",
    )
    return p.parse_args()


def build_competitors(args, device) -> List:
    """Construct the agent list from CLI flags."""
    competitors = []

    # Historicals via registry
    if args.hist:
        registry = OpponentRegistry(device=device)
        all_tags = registry.available_tags()
        if args.hist.strip() == 'all':
            wanted = all_tags
        else:
            wanted = []
            for short in args.hist.split(','):
                short = short.strip()
                # Accept both "V6_3" and "Hist_V6_3"
                full = short if short.startswith('Hist_') else f"Hist_{short}"
                if full not in all_tags:
                    raise ValueError(
                        f"Unknown historical tag '{short}'. "
                        f"Registry has: {all_tags}"
                    )
                wanted.append(full)
        for tag in wanted:
            short_name = tag.replace('Hist_', '')
            competitors.append(HistAgent(tag, registry, name=short_name))

    # Bots
    if args.bots:
        if args.bots.strip() == 'all':
            bot_names = ['Expert', 'Heuristic', 'Aggressive',
                         'Defensive', 'Racing', 'Random']
        else:
            bot_names = [b.strip() for b in args.bots.split(',') if b.strip()]
        for bn in bot_names:
            competitors.append(BotAgent(bn))

    # Custom models via --add-model
    for spec in args.add_model:
        parts = spec.split(':', 2)
        if len(parts) != 3:
            raise ValueError(
                f"--add-model spec '{spec}' invalid; expected NAME:ARCH:PATH"
            )
        name, arch, path = parts
        if arch not in list_arch_presets():
            raise ValueError(
                f"--add-model arch '{arch}' invalid; "
                f"available presets: {list_arch_presets()}"
            )
        competitors.append(ModelAgent(name=name, ckpt_path=path,
                                      arch_preset=arch, device=device))

    if len(competitors) < 2:
        raise SystemExit(
            "Need at least 2 competitors. Use --hist, --bots, or --add-model."
        )

    return competitors


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------

def print_matrix(competitors, results: dict):
    """Print head-to-head WR matrix (row beats col)."""
    names = [c.name for c in competitors]
    name_w = max(8, max(len(n) for n in names))
    print(f"\n{'='*60}")
    print(f"  HEAD-TO-HEAD MATRIX (row's win-rate against col)")
    print(f"{'='*60}")
    print(f"  {'':<{name_w}}", end='')
    for c in names:
        print(f" {c:>8}", end='')
    print()
    for row in names:
        print(f"  {row:<{name_w}}", end='')
        for col in names:
            if row == col:
                print(f" {'—':>8}", end='')
                continue
            key = (row, col)
            rev = (col, row)
            if key in results:
                print(f" {results[key]['a_wr_pct']:>7.1f}%", end='')
            elif rev in results:
                print(f" {results[rev]['b_wr_pct']:>7.1f}%", end='')
            else:
                print(f" {'?':>8}", end='')
        print()


def print_leaderboard(competitors, results: dict):
    """Aggregate WR over all pairs."""
    scores = {c.name: 0 for c in competitors}
    totals = {c.name: 0 for c in competitors}
    for (a_name, b_name), r in results.items():
        scores[a_name] += r['a_wins']
        scores[b_name] += r['b_wins']
        totals[a_name] += r['n_games']
        totals[b_name] += r['n_games']
    rows = []
    for name in sorted(scores.keys()):
        wr = scores[name] / max(1, totals[name]) * 100
        rows.append((name, scores[name], totals[name], wr))
    rows.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'='*60}")
    print(f"  LEADERBOARD (aggregate WR across all pairs)")
    print(f"{'='*60}")
    print(f"  {'#':<3} {'name':<14} {'wins':>8} {'games':>8} {'WR%':>8}")
    for i, (name, w, g, wr) in enumerate(rows, 1):
        print(f"  {i:<3} {name:<14} {w:>8} {g:>8} {wr:>7.1f}%")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"[Tournament] device={device}  seed={args.seed}  "
          f"games_per_pair={args.games_per_pair}\n")

    competitors = build_competitors(args, device)

    print(f"[Tournament] {len(competitors)} competitors:")
    for c in competitors:
        print(f"  - {c.name}  ({type(c).__name__})")
    pairs = list(itertools.combinations(competitors, 2))
    total_games = len(pairs) * args.games_per_pair
    print(f"[Tournament] {len(pairs)} pairs × {args.games_per_pair} games "
          f"= {total_games:,} total games\n")

    results = {}
    t_start = time.time()
    for idx, (a, b) in enumerate(pairs, 1):
        print(f"{'='*60}")
        print(f"  Pair {idx}/{len(pairs)}: {a.name} vs {b.name}")
        print(f"{'='*60}")
        r = run_pair(a, b, args.games_per_pair, verbose=not args.quiet)
        results[(a.name, b.name)] = r
        print(f"  → {a.name}: {r['a_wins']}/{r['n_games']} "
              f"({r['a_wr_pct']:.1f}%)  |  "
              f"{b.name}: {r['b_wins']}/{r['n_games']} "
              f"({r['b_wr_pct']:.1f}%)  |  "
              f"draws: {r['draws']}  |  avg len: {r['avg_length']:.1f}\n")

    elapsed = time.time() - t_start
    gpm = total_games / (elapsed / 60) if elapsed > 0 else 0
    print(f"\n[Tournament] Complete in {elapsed/60:.1f} min "
          f"({gpm:.0f} games/min total)")

    print_leaderboard(competitors, results)
    print_matrix(competitors, results)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.',
                    exist_ok=True)
        # JSON-friendly serialisation: stringify the (a, b) tuple keys.
        out = {
            'meta': {
                'games_per_pair': args.games_per_pair,
                'seed': args.seed,
                'competitors': [c.name for c in competitors],
                'elapsed_min': round(elapsed / 60, 2),
                'total_games': total_games,
            },
            'pairs': [
                {**r, 'pair': [a, b]}
                for (a, b), r in results.items()
            ],
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n[Tournament] Results written to {args.output}")


if __name__ == '__main__':
    main()
