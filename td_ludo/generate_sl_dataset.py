"""Generate a large SL dataset from weighted bot-vs-bot games.

For each game:
  - Sample two bots from a weighted pool (Tier-S strongest down to Tier-C
    rule-bots), each playing one of the 2 active players in a 2P game.
  - Play the game to completion.
  - Record the WINNER's per-decision (state, action) pairs.

Output: sharded .npz files (default 5000 games / shard) under
`checkpoints/sl_dataset_v1/`. Each shard stores compact arrays:
  - player_positions:  (N, 4, 4)  int8   — token positions
  - current_player:    (N,)       int8   — whose turn
  - dice_roll:         (N,)       int8   — current dice (1..6)
  - action:            (N,)       int8   — token id chosen by winner
  - meta:              dict-as-array     — bot names, winner id, game idx

At train time, downstream trainers reconstruct the game state from these
arrays and re-encode it however their model needs (V18-symmetric for V13.6,
per-cell triplet for V15.2, V11/V12 engineered for V12.3).

Resumable: skips shards that already exist on disk.

Parallelism: multiprocessing.Pool with --workers (default = cpu count).
Each worker plays a slice of games and returns lists of (state, action)
tuples + game-level metadata; main process aggregates into shards.

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=. ./td_env/bin/python -u generate_sl_dataset.py \
        --target-games 500000 \
        --shard-size 5000 \
        --workers 8 \
        --out-dir checkpoints/sl_dataset_v1
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ── Weighted bot pool ───────────────────────────────────────────────────
# Tier-S = strongest known (slow, ~1s/move). Tier-A = strong + fast.
# Tier-B = mid. Tier-C = easy/varied.
DEFAULT_BOT_POOL = {
    # Tier-S (30%) — strongest, slowest
    "Depth2Expectimax":        15.0,
    "MCTSExpectimaxPrior":     15.0,
    # Tier-A (45%) — hard + fast
    "Expectimax":              12.0,
    "MinimaxExpectimax":       11.0,
    "AggressiveExpectimax":    11.0,
    "DefensiveExpectimax":     11.0,
    # Tier-B (15%) — mid, style variety
    "BlockadeExpectimax":       4.0,
    "VoteExpectimax":           4.0,
    "RacingExpectimax":         3.0,
    "Expert":                   4.0,
    # Tier-C (10%) — easy / variety floor
    "Aggressive":               2.0,
    "Defensive":                2.0,
    "Racing":                   2.0,
    "Heuristic":                2.0,
    "Random":                   2.0,
}

MAX_MOVES = 400


# ── Worker code (runs in subprocess) ────────────────────────────────────

def _worker_init():
    """Import heavy modules once per worker process."""
    global _ludo_cpp, _get_unified_bot, _np
    import td_ludo_cpp as _ludo_cpp_mod
    _ludo_cpp = _ludo_cpp_mod
    # Reuse the unified bot factory from train_v135_rl (handles all variants).
    from train_v135_rl import get_unified_bot
    _get_unified_bot = get_unified_bot


def _sample_bot(bot_pool: Dict[str, float], rng: random.Random) -> str:
    """Weighted sample of one bot name."""
    items = list(bot_pool.items())
    names = [n for n, _ in items]
    weights = [w for _, w in items]
    return rng.choices(names, weights=weights, k=1)[0]


def _play_one_game(args_tuple):
    """Play one bot-vs-bot game, return winner's (state, action) pairs.

    Returns dict:
      states:  list of (player_positions (4,4) int8, current_player int,
                        dice_roll int)
      actions: list of int  (token_id chosen by winner)
      bots:    (p0_name, p2_name)
      winner:  int (0 or 2, -1 if truncated)
      moves:   int (total game length)
    """
    (game_idx, seed, bot_pool) = args_tuple
    rng = random.Random(seed)

    p0_name = _sample_bot(bot_pool, rng)
    p2_name = _sample_bot(bot_pool, rng)
    # Bot instances. Player_id=0 / 2.
    bot0 = _get_unified_bot(p0_name, player_id=0)
    bot2 = _get_unified_bot(p2_name, player_id=2)

    state = _ludo_cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]

    # Per-step trace: for each (state, action), remember which player moved.
    # We filter to winner's moves at the end.
    trace_pp = []     # list of (4,4) int8 arrays
    trace_cp = []     # list of int (current_player)
    trace_dice = []   # list of int (dice roll)
    trace_action = [] # list of int (token_id)
    trace_actor = []  # list of int (player who moved — 0 or 2)

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
            d = rng.randint(1, 6)
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
        legal = _ludo_cpp.get_legal_moves(state)
        if not legal:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            state.current_dice_roll = 0
            continue

        # Snapshot state BEFORE the move (this is what we'll save).
        pp_snapshot = np.array(state.player_positions, dtype=np.int8).copy()
        dice_now = int(state.current_dice_roll)

        # Bot decides.
        bot = bot0 if cp == 0 else bot2
        action = int(bot.select_move(state, list(legal)))
        # Defensive: ensure action is legal.
        if action not in legal:
            action = int(legal[0])

        trace_pp.append(pp_snapshot)
        trace_cp.append(int(cp))
        trace_dice.append(dice_now)
        trace_action.append(int(action))
        trace_actor.append(int(cp))

        state = _ludo_cpp.apply_move(state, action)
        mc += 1

    if state.is_terminal:
        winner = int(_ludo_cpp.get_winner(state))
    else:
        winner = -1  # truncated as draw — skip this game entirely

    if winner < 0:
        return {
            "winner": -1, "moves": mc,
            "bots": (p0_name, p2_name),
            "n_winner_decisions": 0,
            "winner_pp": np.zeros((0, 4, 4), dtype=np.int8),
            "winner_cp": np.zeros(0, dtype=np.int8),
            "winner_dice": np.zeros(0, dtype=np.int8),
            "winner_action": np.zeros(0, dtype=np.int8),
        }

    # Filter to winner's moves only.
    mask = [i for i, a in enumerate(trace_actor) if a == winner]
    if not mask:
        # Should not happen but be defensive.
        return {
            "winner": winner, "moves": mc,
            "bots": (p0_name, p2_name),
            "n_winner_decisions": 0,
            "winner_pp": np.zeros((0, 4, 4), dtype=np.int8),
            "winner_cp": np.zeros(0, dtype=np.int8),
            "winner_dice": np.zeros(0, dtype=np.int8),
            "winner_action": np.zeros(0, dtype=np.int8),
        }

    winner_pp = np.stack([trace_pp[i] for i in mask], axis=0)
    winner_cp = np.array([trace_cp[i] for i in mask], dtype=np.int8)
    winner_dice = np.array([trace_dice[i] for i in mask], dtype=np.int8)
    winner_action = np.array([trace_action[i] for i in mask], dtype=np.int8)

    return {
        "winner": winner,
        "moves": mc,
        "bots": (p0_name, p2_name),
        "n_winner_decisions": len(mask),
        "winner_pp": winner_pp,
        "winner_cp": winner_cp,
        "winner_dice": winner_dice,
        "winner_action": winner_action,
    }


# ── Main / aggregator ──────────────────────────────────────────────────

def _save_shard(shard_path: Path, shard_data: dict):
    """Write one shard to .npz file (compressed)."""
    np.savez_compressed(
        shard_path,
        player_positions=shard_data["pp"],
        current_player=shard_data["cp"],
        dice_roll=shard_data["dice"],
        action=shard_data["action"],
        game_idx=shard_data["game_idx"],   # which game in shard each row belongs to
    )
    # Sidecar JSON with metadata (bot pairings, winners, etc.)
    meta_path = shard_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(shard_data["meta"], f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-games", type=int, default=500_000,
                    help="Total number of SUCCESSFUL games to generate (truncated games are re-rolled).")
    ap.add_argument("--shard-size", type=int, default=200,
                    help="Games per shard file. Small shards = small RAM "
                         "footprint AND small data loss on interrupt. "
                         "Default 200 → ~30s of work per shard, ~16K rows "
                         "(150 decisions × 200 games / 2 sides recorded). "
                         "2500 shards for 500K games total.")
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of worker processes. 0 = cpu_count.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=HERE / "checkpoints" / "sl_dataset_v1")
    ap.add_argument("--bot-pool-json", type=str, default=None,
                    help="Override the bot weights via a JSON file.")
    args = ap.parse_args()

    bot_pool = DEFAULT_BOT_POOL.copy()
    if args.bot_pool_json:
        with open(args.bot_pool_json) as f:
            bot_pool = json.load(f)
    print(f"Bot pool ({len(bot_pool)} bots, total weight = "
          f"{sum(bot_pool.values()):.1f}):")
    total_w = sum(bot_pool.values())
    for n, w in sorted(bot_pool.items(), key=lambda kv: -kv[1]):
        print(f"  {n:30s}  w={w:5.1f}  ({100*w/total_w:.1f}%)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers > 0 else os.cpu_count()
    print(f"\nWorkers: {workers}")
    print(f"Target: {args.target_games:,} games  /  shard: {args.shard_size:,}")
    print(f"Out:    {args.out_dir}")

    # Resume: find highest existing shard index.
    existing_shards = sorted(args.out_dir.glob("shard_*.npz"))
    start_shard = 0
    completed_games = 0
    if existing_shards:
        for sp in existing_shards:
            idx = int(sp.stem.split("_")[1])
            start_shard = max(start_shard, idx + 1)
            # Count games in shard via metadata JSON.
            meta = sp.with_suffix(".meta.json")
            if meta.exists():
                with open(meta) as f:
                    m = json.load(f)
                completed_games += m.get("n_games", 0)
        print(f"\n[resume] {start_shard} shards already on disk, "
              f"{completed_games:,} games completed.")

    n_shards_total = (args.target_games + args.shard_size - 1) // args.shard_size
    print(f"Total shards: {n_shards_total}  (start from shard {start_shard})")

    if completed_games >= args.target_games:
        print("Target already met. Nothing to do.")
        return

    pool = mp.Pool(processes=workers, initializer=_worker_init)
    t_start = time.time()

    # ── Continuous-feed pattern ───────────────────────────────────────
    # The PROBLEM with the old "submit a batch, iterate, then re-batch"
    # approach: when a batch contained a slow Tier-S × Tier-S game (~4
    # min each), 7 of 8 workers would finish their fast games quickly
    # and SLEEP — main wouldn't dispatch the next batch until the slow
    # one finished. Workers idled ~80% of the time.
    #
    # Fix: dispatch a single large stream of tasks via a generator.
    # imap_unordered consumes from the generator, keeps every worker
    # continuously fed regardless of per-task variance. We over-request
    # by 1.5x to absorb truncated games (winner=-1, discarded).
    target_remaining = args.target_games - completed_games
    n_tasks = int(target_remaining * 1.5) + 100
    seed_base = args.seed + completed_games

    def gen_tasks():
        for i in range(n_tasks):
            yield (seed_base + i, seed_base + i, bot_pool)

    shard_pp = []
    shard_cp = []
    shard_dice = []
    shard_action = []
    shard_game_idx = []
    shard_meta_games = []
    games_in_shard = 0
    t_shard_start = time.time()
    shard_idx = start_shard

    try:
        for res in pool.imap_unordered(_play_one_game, gen_tasks(), chunksize=4):
            if res["winner"] < 0:
                continue   # truncated game — discard
            n = res["n_winner_decisions"]
            if n == 0:
                continue
            shard_pp.append(res["winner_pp"])
            shard_cp.append(res["winner_cp"])
            shard_dice.append(res["winner_dice"])
            shard_action.append(res["winner_action"])
            shard_game_idx.append(np.full(n, games_in_shard, dtype=np.int32))
            shard_meta_games.append({
                "p0": res["bots"][0], "p2": res["bots"][1],
                "winner": res["winner"], "moves": res["moves"],
                "n_winner_decisions": n,
            })
            games_in_shard += 1

            if games_in_shard >= args.shard_size:
                # Flush shard to disk.
                shard_data = {
                    "pp":      np.concatenate(shard_pp, axis=0),
                    "cp":      np.concatenate(shard_cp, axis=0),
                    "dice":    np.concatenate(shard_dice, axis=0),
                    "action":  np.concatenate(shard_action, axis=0),
                    "game_idx": np.concatenate(shard_game_idx, axis=0),
                    "meta": {
                        "shard_idx": shard_idx,
                        "n_games": games_in_shard,
                        "n_rows": int(np.concatenate(shard_action, axis=0).shape[0]),
                        "games": shard_meta_games,
                    },
                }
                shard_path = args.out_dir / f"shard_{shard_idx:05d}.npz"
                _save_shard(shard_path, shard_data)

                completed_games += games_in_shard
                shard_elapsed = time.time() - t_shard_start
                total_elapsed = time.time() - t_start
                shard_rate = games_in_shard / shard_elapsed
                row_count = shard_data["meta"]["n_rows"]
                eta_min = (args.target_games - completed_games) / max(0.01, shard_rate) / 60.0
                print(f"[shard {shard_idx:>5d}] {games_in_shard} games "
                      f"({row_count} rows) in {shard_elapsed:>5.1f}s "
                      f"= {shard_rate:.1f} g/s  | total: {completed_games:,}"
                      f"/{args.target_games:,}  ({total_elapsed/60:.1f} min "
                      f"elapsed, ETA {eta_min:.0f} min)",
                      flush=True)

                # Reset shard buffers.
                shard_pp = []
                shard_cp = []
                shard_dice = []
                shard_action = []
                shard_game_idx = []
                shard_meta_games = []
                games_in_shard = 0
                t_shard_start = time.time()
                shard_idx += 1

                if completed_games >= args.target_games:
                    break  # done
    finally:
        pool.terminate()  # stop in-flight workers — we have what we need
        pool.join()

    print(f"\nDone. {completed_games:,} games written to {args.out_dir}.")


if __name__ == "__main__":
    main()
