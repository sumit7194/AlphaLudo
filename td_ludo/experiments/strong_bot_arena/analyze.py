"""Analyze a strong-bot arena results JSON.

Beyond the leaderboard + matrix the runner prints at end, this:
  - Computes diversity metric per bot (how different are its WRs vs
    the rest of the field)
  - Reports games-per-second per bot
  - Proposes a recommended training-mix composition based on:
      * include all bots that beat the median by ≥5 ELO
      * include "low-but-different" bots (high diversity, low ELO) for
        playstyle variety
      * weights inversely proportional to wallclock cost (so faster
        bots get more games for the same compute budget)

Usage:
  python -m experiments.strong_bot_arena.analyze runs/arena_phase1_200g.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


def load_arena(path: str):
    with open(path) as f:
        data = json.load(f)
    meta = data['meta']
    pairs = data['pairs']
    # Reconstruct WR matrix and games-per-second per bot
    wr_matrix = {}
    cost_per_bot = {}
    games_per_bot = {}
    wins_per_bot = {}
    for r in pairs:
        a, b = r['pair']
        n = r['n_games']
        a_wins = r['a_wins']
        b_wins = r['b_wins']
        wr_matrix[(a, b)] = a_wins / n
        wr_matrix[(b, a)] = b_wins / n
        elapsed = r.get('elapsed_sec', 1.0)
        gps = n / elapsed if elapsed > 0 else 0
        cost_per_bot.setdefault(a, []).append(gps)
        cost_per_bot.setdefault(b, []).append(gps)
        wins_per_bot[a] = wins_per_bot.get(a, 0) + a_wins
        wins_per_bot[b] = wins_per_bot.get(b, 0) + b_wins
        games_per_bot[a] = games_per_bot.get(a, 0) + n
        games_per_bot[b] = games_per_bot.get(b, 0) + n

    bot_names = sorted({b for pair in [r['pair'] for r in pairs] for b in pair})
    # Average games-per-second per bot (across its pairs)
    avg_gps = {b: float(np.mean(cost_per_bot[b])) for b in bot_names}

    return meta, bot_names, wr_matrix, avg_gps, wins_per_bot, games_per_bot


def diversity_score(bot: str, others: List[str], wr_matrix: Dict) -> float:
    """How far from 50/50 are this bot's WRs vs the field on average?

    diversity = mean(|wr_i - 0.5|) across all others — high = pronounced
    asymmetries, low = mid-field generalist.

    Also returns count of opponents where this bot's WR differs from
    the bot's overall mean WR by ≥5pp, which is a proxy for
    'play-style asymmetry' (this bot is differently-good against
    different opponents)."""
    wrs = []
    for other in others:
        if other == bot:
            continue
        if (bot, other) not in wr_matrix:
            continue
        wrs.append(wr_matrix[(bot, other)])
    if not wrs:
        return 0.0, 0.0
    wrs = np.array(wrs)
    mean_wr = wrs.mean()
    spread_from_mean = float(np.mean(np.abs(wrs - mean_wr)))
    return mean_wr, spread_from_mean


def get_leaderboard_from_meta(meta) -> Dict[str, Dict]:
    return meta.get('leaderboard', {}) or {}


def propose_mix(bot_names, leaderboard, diversity, avg_gps,
                self_play_fraction: float = 0.45) -> Dict[str, float]:
    """Build a recommended training-mix composition.

    Strategy: each bot gets a score from ELO + diversity bonus.
    Top-K bots by score get the remaining (1 - self_play_fraction)
    weight, distributed inversely-proportionally to wallclock cost
    (so fast bots get more games, slow ones get fewer).
    """
    if not leaderboard:
        return {"SelfPlay": self_play_fraction}

    scored = []
    elos = [leaderboard[b]['elo'] for b in bot_names if b in leaderboard]
    median_elo = float(np.median(elos))

    for b in bot_names:
        if b not in leaderboard:
            continue
        elo = leaderboard[b]['elo']
        mean_wr, spread = diversity.get(b, (0.5, 0.0))
        # Diversity bonus: each 5pp of spread = +20 ELO points equivalent
        diversity_bonus = spread * 400
        composite = elo + diversity_bonus
        if elo >= median_elo - 30:  # include all bots near or above median
            scored.append((b, composite, avg_gps.get(b, 1.0)))
        elif spread >= 0.05:        # include low-elo but high-diversity bots
            scored.append((b, composite * 0.7, avg_gps.get(b, 1.0)))

    if not scored:
        return {"SelfPlay": self_play_fraction}

    # Weight inversely proportional to cost (faster = more weight)
    total_inv_cost = sum(1.0 / max(0.01, gps) for _, _, gps in scored)
    raw_weights = {b: (1.0 / max(0.01, gps)) / total_inv_cost
                   for b, _, gps in scored}
    # Normalize to (1 - self_play_fraction)
    target = 1.0 - self_play_fraction
    mix = {b: w * target for b, w in raw_weights.items()}
    mix["SelfPlay"] = self_play_fraction
    return mix


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('json_path', help="Path to arena JSON results")
    p.add_argument('--self-play', type=float, default=0.45,
                   help="Fraction reserved for SelfPlay in proposed mix")
    args = p.parse_args()

    if not os.path.exists(args.json_path):
        print(f"file not found: {args.json_path}")
        sys.exit(1)

    meta, bot_names, wr_matrix, avg_gps, wins, games = load_arena(args.json_path)
    leaderboard = get_leaderboard_from_meta(meta)

    print(f"\n{'='*78}")
    print(f"  ARENA ANALYSIS — {args.json_path}")
    print(f"{'='*78}")
    print(f"  {len(bot_names)} bots, {meta['games_per_pair']} games/pair, "
          f"{sum(games.values())//2:,} total games")
    print(f"  elapsed: {meta.get('elapsed_min', '?')} min")

    # Per-bot diversity + cost table
    diversity = {}
    print(f"\n  {'Bot':<28} {'Mean WR':>9} {'Diversity':>10} {'g/s':>8} {'ELO':>8}")
    rows = []
    for b in bot_names:
        mean_wr, spread = diversity_score(b, bot_names, wr_matrix)
        diversity[b] = (mean_wr, spread)
        elo = leaderboard.get(b, {}).get('elo', 1500)
        rows.append((b, mean_wr, spread, avg_gps.get(b, 0), elo))
    rows.sort(key=lambda x: x[4], reverse=True)  # by ELO desc
    for b, wr, sp, gps, elo in rows:
        print(f"  {b:<28} {wr*100:>8.1f}% {sp*100:>9.1f}pp {gps:>7.1f} {elo:>8.0f}")

    # Recommended training mix
    mix = propose_mix(bot_names, leaderboard, diversity, avg_gps,
                      self_play_fraction=args.self_play)
    mix_sorted = sorted(mix.items(), key=lambda x: -x[1])

    print(f"\n  Recommended training mix (sum = {sum(mix.values()):.2f}):")
    for name, w in mix_sorted:
        print(f"    {name:<28} {w*100:>5.1f}%")

    # Diversity champions (high diversity score, regardless of ELO)
    div_rows = [(b, sp) for b, (_, sp) in diversity.items()]
    div_rows.sort(key=lambda x: -x[1])
    print(f"\n  Top-5 diversity (most-asymmetric WRs across the field):")
    for b, sp in div_rows[:5]:
        print(f"    {b:<28} spread = {sp*100:.1f}pp")


if __name__ == "__main__":
    main()
