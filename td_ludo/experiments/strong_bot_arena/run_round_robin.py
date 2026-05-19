"""Strong-bot arena round-robin runner — Phase 2 of STRONG_BOTS_PLAN.

Unified bot registry over scripted bots (heuristic_bot.py), expectimax
+ MCTS (strong_bots.py), and expectimax personality variants
(strong_bots_v2.py). Plays N games per unordered pair, seats rotated
to control first-mover advantage, JSON-checkpointed per pair so a
killed run resumes cleanly.

Outputs:
  - Per-pair WR rows (live, suppressible with --quiet)
  - Final aggregate WR leaderboard
  - Head-to-head matrix
  - Bradley-Terry ELO derived from the matrix
  - JSON dump of full results (always written incrementally)

CLI:
  # Default suite — all expectimax variants + base + Expert
  python -m experiments.strong_bot_arena.run_round_robin \\
    --games-per-pair 200 \\
    --output runs/strong_bot_phase1.json

  # Custom bot list
  python -m experiments.strong_bot_arena.run_round_robin \\
    --bots Expert,Expectimax,AggressiveExpectimax,RacingExpectimax \\
    --games-per-pair 500

  # Resume an interrupted run (just re-run with the same --output path;
  # already-completed pairs are skipped)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import td_ludo_cpp as cpp


MAX_MOVES_PER_GAME = 10000


# ─── Unified bot registry ────────────────────────────────────────────────


def _build_bot_registry() -> Dict[str, callable]:
    """Map bot_name → factory callable returning a bot instance.

    Lazy imports inside functions so module load stays cheap even if
    e.g. mcts_v1 engine is missing.
    """
    def _heuristic_factory(name):
        from td_ludo.game.heuristic_bot import (
            HeuristicLudoBot, AggressiveBot, DefensiveBot,
            RacingBot, RandomBot, ExpertBot,
        )
        cls = {
            "Heuristic":  HeuristicLudoBot,
            "Aggressive": AggressiveBot,
            "Defensive":  DefensiveBot,
            "Racing":     RacingBot,
            "Random":     RandomBot,
            "Expert":     ExpertBot,
        }[name]
        return cls()  # heuristic bots ignore player_id

    def _strong_factory(name):
        from td_ludo.game.strong_bots import ExpectimaxBot, MCTSPureBot
        if name == "MCTSHighSim":
            # Bonus variant: MCTSPureBot at 2× the default sims (100 vs
            # 50). ~130ms/move (vs 67ms for MCTSPure default). Tests
            # whether more search alone (no informed prior) closes the
            # gap to Expectimax — a useful counterfactual for the
            # MCTSExpertPrior / MCTSExpectimaxPrior comparison.
            return MCTSPureBot(n_sims=100, rollouts_per_leaf=8)
        cls = {"Expectimax": ExpectimaxBot, "MCTSPure": MCTSPureBot}[name]
        return cls()

    def _strong_v2_factory(name):
        from td_ludo.game.strong_bots_v2 import EXPECTIMAX_V2_REGISTRY
        return EXPECTIMAX_V2_REGISTRY[name]()

    def _depth2_factory(name):
        from td_ludo.game.strong_bots_depth2 import DEPTH2_REGISTRY
        return DEPTH2_REGISTRY[name]()

    def _mcts_prior_factory(name):
        from td_ludo.game.strong_bots_mcts_prior import MCTS_PRIOR_REGISTRY
        return MCTS_PRIOR_REGISTRY[name]()

    def _adaptive_factory(name):
        from td_ludo.game.strong_bots_adaptive import ADAPTIVE_REGISTRY
        return ADAPTIVE_REGISTRY[name]()

    def _rule_factory(name):
        from td_ludo.game.strong_bots_rule import RULE_BOT_REGISTRY
        return RULE_BOT_REGISTRY[name]()

    return {
        # Scripted family
        "Heuristic":  lambda: _heuristic_factory("Heuristic"),
        "Aggressive": lambda: _heuristic_factory("Aggressive"),
        "Defensive":  lambda: _heuristic_factory("Defensive"),
        "Racing":     lambda: _heuristic_factory("Racing"),
        "Random":     lambda: _heuristic_factory("Random"),
        "Expert":     lambda: _heuristic_factory("Expert"),
        # Strong family (existing)
        "Expectimax": lambda: _strong_factory("Expectimax"),
        "MCTSPure":   lambda: _strong_factory("MCTSPure"),
        "MCTSHighSim": lambda: _strong_factory("MCTSHighSim"),
        # Expectimax personalities (Phase 1 of plan)
        "AggressiveExpectimax": lambda: _strong_v2_factory("AggressiveExpectimax"),
        "DefensiveExpectimax":  lambda: _strong_v2_factory("DefensiveExpectimax"),
        "RacingExpectimax":     lambda: _strong_v2_factory("RacingExpectimax"),
        "MinimaxExpectimax":    lambda: _strong_v2_factory("MinimaxExpectimax"),
        "BlockadeExpectimax":   lambda: _strong_v2_factory("BlockadeExpectimax"),
        # Depth-2 expectimax variants (Phase 3 of plan)
        "Depth2Expectimax":             lambda: _depth2_factory("Depth2Expectimax"),
        "Depth2AggressiveExpectimax":   lambda: _depth2_factory("Depth2AggressiveExpectimax"),
        "Depth2DefensiveExpectimax":    lambda: _depth2_factory("Depth2DefensiveExpectimax"),
        # MCTS with informed prior (Phase 4 of plan)
        "MCTSExpertPrior":     lambda: _mcts_prior_factory("MCTSExpertPrior"),
        "MCTSExpectimaxPrior": lambda: _mcts_prior_factory("MCTSExpectimaxPrior"),
        # Adaptive + ensemble (Phase 6 of plan)
        "AdaptiveExpectimax":  lambda: _adaptive_factory("AdaptiveExpectimax"),
        "VoteExpectimax":      lambda: _adaptive_factory("VoteExpectimax"),
        # Rule-based variants (Phase 6 extension — pure heuristics, no search)
        "MaxCapture":     lambda: _rule_factory("MaxCapture"),
        "TwoStack":       lambda: _rule_factory("TwoStack"),
        "HomeRush":       lambda: _rule_factory("HomeRush"),
        "StackHomeRush":  lambda: _rule_factory("StackHomeRush"),
    }


DEFAULT_BOTS = [
    "Expert",
    "Expectimax",
    "AggressiveExpectimax",
    "DefensiveExpectimax",
    "RacingExpectimax",
    "MinimaxExpectimax",
]


# ─── 2-player game loop (adapted from experiments/tournament/run.py) ────


def play_game(bot_p0, bot_p2) -> Tuple[int, int]:
    """One 2P Ludo game. Returns (winner_seat ∈ {0, 2, -1}, total_moves)."""
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

        bot = bot_p0 if cp == 0 else bot_p2
        action = bot.select_move(state, legal)
        if action is None or action < 0:
            action = legal[0]
        state = cpp.apply_move(state, action)
        moves += 1

    if state.is_terminal:
        return cpp.get_winner(state), moves
    return -1, moves


def run_pair(a_name, b_name, a_factory, b_factory,
             n_games: int, verbose: bool = True) -> Dict:
    """Play n_games between a and b with seat rotation. Each bot is
    re-instantiated with the correct player_id so internal state (e.g.
    MCTS tree) doesn't bleed between games.

    Returns dict matching the existing tournament JSON schema.
    """
    a_wins = 0
    b_wins = 0
    draws = 0
    lengths: List[int] = []
    t0 = time.time()

    for i in range(n_games):
        a_at_p0 = (i % 2 == 0)
        # Re-instantiate per game to clear any internal bot state.
        a = a_factory(); b = b_factory()
        if hasattr(a, "player_id"):
            a.player_id = 0 if a_at_p0 else 2
        if hasattr(b, "player_id"):
            b.player_id = 2 if a_at_p0 else 0

        if a_at_p0:
            winner, m = play_game(a, b)
            if winner == 0:   a_wins += 1
            elif winner == 2: b_wins += 1
            else:             draws += 1
        else:
            winner, m = play_game(b, a)
            if winner == 2:   a_wins += 1
            elif winner == 0: b_wins += 1
            else:             draws += 1
        lengths.append(m)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            gpm = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            wr = a_wins / (i + 1) * 100
            print(f"    [{i+1:>5}/{n_games}] {a_name} {wr:5.1f}% "
                  f"vs {b_name} | {gpm:6.1f} gpm", flush=True)

    return {
        'a': a_name, 'b': b_name, 'n_games': n_games,
        'a_wins': a_wins, 'b_wins': b_wins, 'draws': draws,
        'a_wr_pct': a_wins / max(1, n_games) * 100,
        'b_wr_pct': b_wins / max(1, n_games) * 100,
        'avg_length': float(np.mean(lengths)) if lengths else 0.0,
        'elapsed_sec': time.time() - t0,
    }


# ─── Bradley-Terry ELO from WR matrix ────────────────────────────────────


def bradley_terry_elo(
    wins: Dict[Tuple[str, str], int],
    games: Dict[Tuple[str, str], int],
    iterations: int = 200,
) -> Dict[str, float]:
    """Iterative MM update for Bradley-Terry, returns ELO (scale 400).

    π_i ∝ Σ_j w_{ij} / Σ_j n_{ij} / (π_i + π_j)
    ELO_i = 400 * log10(π_i / mean(π)) + 1500
    """
    # Collect all unique bot names
    names = set()
    for (a, b) in wins:
        names.add(a); names.add(b)
    names = sorted(names)
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}

    # Build wins / games matrices
    W = np.zeros((n, n), dtype=np.float64)
    N = np.zeros((n, n), dtype=np.float64)
    for (a, b), w in wins.items():
        i, j = idx[a], idx[b]
        N[i, j] += games[(a, b)]
        N[j, i] += games[(a, b)]
        W[i, j] += w
        W[j, i] += games[(a, b)] - w

    pi = np.ones(n)
    for _ in range(iterations):
        new_pi = np.zeros(n)
        for i in range(n):
            num = float(W[i, :].sum())
            den = 0.0
            for j in range(n):
                if i == j or N[i, j] == 0:
                    continue
                den += N[i, j] / (pi[i] + pi[j])
            new_pi[i] = num / den if den > 0 else pi[i]
        # Normalize to prevent drift
        mean = float(new_pi.mean())
        new_pi /= mean if mean > 0 else 1.0
        if np.allclose(pi, new_pi, atol=1e-6):
            pi = new_pi
            break
        pi = new_pi

    pi = np.clip(pi, 1e-9, None)
    elo = {names[i]: 1500.0 + 400.0 * math.log10(pi[i]) for i in range(n)}
    return elo


# ─── Reporting ───────────────────────────────────────────────────────────


def print_leaderboard(results: Dict[Tuple[str, str], Dict],
                      bot_names: List[str]) -> Dict[str, Dict]:
    """Aggregate WR + ELO. Returns leaderboard dict."""
    scores = {n: 0 for n in bot_names}
    totals = {n: 0 for n in bot_names}
    for (a, b), r in results.items():
        scores[a] += r['a_wins']
        scores[b] += r['b_wins']
        totals[a] += r['n_games']
        totals[b] += r['n_games']

    wins = {(a, b): r['a_wins'] for (a, b), r in results.items()}
    games = {(a, b): r['n_games'] for (a, b), r in results.items()}
    elos = bradley_terry_elo(wins, games)

    rows = []
    for n in scores:
        wr = scores[n] / max(1, totals[n]) * 100
        rows.append((n, scores[n], totals[n], wr, elos.get(n, 1500)))
    rows.sort(key=lambda x: x[4], reverse=True)

    print(f"\n{'='*72}")
    print(f"  LEADERBOARD (Bradley-Terry ELO + aggregate WR)")
    print(f"{'='*72}")
    print(f"  {'#':<3} {'name':<25} {'wins':>8} {'games':>8} {'WR%':>7} {'ELO':>8}")
    leaderboard = {}
    for i, (n, w, g, wr, elo) in enumerate(rows, 1):
        print(f"  {i:<3} {n:<25} {w:>8} {g:>8} {wr:>6.1f}% {elo:>8.0f}")
        leaderboard[n] = {'rank': i, 'wins': w, 'games': g,
                          'wr_pct': wr, 'elo': elo}
    return leaderboard


def print_matrix(results: Dict[Tuple[str, str], Dict], bot_names: List[str]):
    name_w = max(8, max(len(n) for n in bot_names))
    print(f"\n{'='*72}")
    print(f"  HEAD-TO-HEAD MATRIX (row's WR against col)")
    print(f"{'='*72}")
    print(f"  {'':<{name_w}}", end='')
    for n in bot_names:
        print(f" {n[:10]:>10}", end='')
    print()
    for row in bot_names:
        print(f"  {row:<{name_w}}", end='')
        for col in bot_names:
            if row == col:
                print(f" {'—':>10}", end='')
                continue
            if (row, col) in results:
                print(f" {results[(row, col)]['a_wr_pct']:>9.1f}%", end='')
            elif (col, row) in results:
                print(f" {results[(col, row)]['b_wr_pct']:>9.1f}%", end='')
            else:
                print(f" {'?':>10}", end='')
        print()


# ─── Persistence ─────────────────────────────────────────────────────────


def load_results(output_path: str) -> Dict[Tuple[str, str], Dict]:
    """Restore previous run's per-pair results from JSON, if any."""
    if not os.path.exists(output_path):
        return {}
    try:
        with open(output_path) as f:
            data = json.load(f)
        return {tuple(p['pair']): p for p in data.get('pairs', [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_results(output_path: str, results: Dict[Tuple[str, str], Dict],
                 meta: Dict):
    """Write current per-pair results atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.',
                exist_ok=True)
    pairs = []
    for (a, b), r in results.items():
        rec = dict(r)
        rec['pair'] = [a, b]
        pairs.append(rec)
    out = {'meta': meta, 'pairs': pairs}
    tmp = output_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, output_path)


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bots', default=','.join(DEFAULT_BOTS),
                   help=f"Comma-separated bot names. Default: {','.join(DEFAULT_BOTS)}.")
    p.add_argument('--games-per-pair', type=int, default=200,
                   help="Games per unordered pair (default 200).")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, required=True,
                   help="JSON file to write/append per-pair results.")
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    registry = _build_bot_registry()
    bot_names = [b.strip() for b in args.bots.split(',') if b.strip()]
    unknown = [b for b in bot_names if b not in registry]
    if unknown:
        print(f"[arena] unknown bots: {unknown}")
        print(f"[arena] known: {sorted(registry.keys())}")
        sys.exit(1)

    pairs = list(itertools.combinations(bot_names, 2))
    total = len(pairs) * args.games_per_pair
    print(f"[arena] {len(bot_names)} bots × {len(pairs)} pairs × "
          f"{args.games_per_pair} games = {total:,} total")
    print(f"[arena] output: {args.output}")
    print(f"[arena] seed: {args.seed}")

    results = load_results(args.output)
    if results:
        done = {tuple(sorted([a, b])) for (a, b) in results}
        print(f"[arena] resuming: {len(results)}/{len(pairs)} pairs already complete")

    meta = {
        'games_per_pair': args.games_per_pair,
        'seed': args.seed,
        'bot_names': bot_names,
    }

    t_start = time.time()
    for idx, (a_name, b_name) in enumerate(pairs, 1):
        key = tuple(sorted([a_name, b_name]))
        already_have = (
            (a_name, b_name) in results or (b_name, a_name) in results
        )
        if already_have:
            print(f"[arena] {idx}/{len(pairs)} {a_name} vs {b_name} — skipped (cached)")
            continue

        print(f"\n{'='*72}")
        print(f"  Pair {idx}/{len(pairs)}: {a_name} vs {b_name}")
        print(f"{'='*72}")
        a_factory = registry[a_name]
        b_factory = registry[b_name]
        r = run_pair(a_name, b_name, a_factory, b_factory,
                     args.games_per_pair, verbose=not args.quiet)
        results[(a_name, b_name)] = r
        print(f"  → {a_name}: {r['a_wins']}/{r['n_games']} "
              f"({r['a_wr_pct']:.1f}%)  |  "
              f"{b_name}: {r['b_wins']}/{r['n_games']} "
              f"({r['b_wr_pct']:.1f}%)  |  "
              f"draws: {r['draws']}  |  avg len: {r['avg_length']:.1f}  |  "
              f"{r['elapsed_sec']:.1f}s")

        # Persist after every pair
        save_results(args.output, results, meta)

    elapsed_min = (time.time() - t_start) / 60
    print(f"\n[arena] Complete in {elapsed_min:.1f} min")

    leaderboard = print_leaderboard(results, bot_names)
    print_matrix(results, bot_names)

    # Final save with leaderboard included
    meta['elapsed_min'] = round(elapsed_min, 2)
    meta['leaderboard'] = leaderboard
    save_results(args.output, results, meta)
    print(f"\n[arena] Final results → {args.output}")


if __name__ == "__main__":
    main()
