"""
TD-Ludo V9 — SL Data Generator (Bot Behavioral Cloning)

Plays fast games between heuristic bots, recording V9's 14-channel encoding.
No model inference needed — pure CPU, extremely fast.

Saves per decision:
  - states: (14, 15, 15) float32 — V9 encoding
  - actions: int — bot's chosen action
  - masks: (4,) float32 — legal move mask
  - values: float — +1 win, -1 loss (terminal outcome)

Supports graceful shutdown (Ctrl+C) and resume (re-run to continue).
"""

import os
import sys
import signal
import argparse
import random
import glob as glob_module
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Graceful shutdown
STOP_REQUESTED = False

def signal_handler(sig, frame):
    global STOP_REQUESTED
    if STOP_REQUESTED:
        print("\n[V9 SL Gen] Force exit.")
        sys.exit(1)
    STOP_REQUESTED = True
    print("\n[V9 SL Gen] Graceful shutdown. Will save current buffer and exit...")

signal.signal(signal.SIGINT, signal_handler)

import td_ludo_cpp as ludo_cpp
from src.heuristic_bot import HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot, ExpertBot
from src.config import MAX_MOVES_PER_GAME

DATA_DIR = os.path.join("checkpoints", "sl_data_v9")
os.makedirs(DATA_DIR, exist_ok=True)

BATCH_SIZE = 200  # More parallel games since no GPU bottleneck
CHUNK_SIZE = 100_000


def generate_data(num_games):
    bots = {
        'Heuristic': HeuristicLudoBot(),
        'Aggressive': AggressiveBot(),
        'Defensive': DefensiveBot(),
        'Racing': RacingBot(),
        'Expert': ExpertBot(),
    }
    bot_names = list(bots.keys())

    env = ludo_cpp.VectorGameState(BATCH_SIZE, True)  # 2P mode

    trajectories = {i: {0: [], 2: []} for i in range(BATCH_SIZE)}
    assignments = {}
    for i in range(BATCH_SIZE):
        assignments[i] = {0: random.choice(bot_names), 2: random.choice(bot_names)}

    consecutive_sixes = np.zeros((BATCH_SIZE, 4), dtype=int)
    move_counts = np.zeros(BATCH_SIZE, dtype=int)

    # Auto-detect existing chunks for resume
    existing = glob_module.glob(os.path.join(DATA_DIR, "chunk_*.npz"))
    if existing:
        max_idx = max(int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing)
        chunk_idx = max_idx + 1
        print(f"Found {len(existing)} existing chunks. Continuing from chunk_{chunk_idx:04d}.")
    else:
        chunk_idx = 0

    buffer_states = []
    buffer_actions = []
    buffer_masks = []
    buffer_values = []

    games_completed = 0
    pbar = tqdm(total=num_games, desc="Generating V9 SL Data (Bots)")

    def save_chunk():
        nonlocal buffer_states, buffer_actions, buffer_masks, buffer_values, chunk_idx
        if not buffer_states:
            return

        chunk_path = os.path.join(DATA_DIR, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            chunk_path,
            states=np.stack(buffer_states).astype(np.float32),
            actions=np.array(buffer_actions, dtype=np.int64),
            masks=np.stack(buffer_masks).astype(np.float32),
            values=np.array(buffer_values, dtype=np.float32),
        )
        print(f"\nSaved {chunk_path} with {len(buffer_states)} samples.")

        chunk_idx += 1
        buffer_states, buffer_actions, buffer_masks, buffer_values = [], [], [], []

    while games_completed < num_games and not STOP_REQUESTED:
        actions = []
        for i in range(BATCH_SIZE):
            game = env.get_game(i)

            if game.is_terminal:
                actions.append(-1)
                continue

            cp = game.current_player

            if move_counts[i] >= MAX_MOVES_PER_GAME:
                game.is_terminal = True
                actions.append(-1)
                continue

            if game.current_dice_roll == 0:
                roll = random.randint(1, 6)
                game.current_dice_roll = roll
                if roll == 6:
                    consecutive_sixes[i, cp] += 1
                else:
                    consecutive_sixes[i, cp] = 0

                if consecutive_sixes[i, cp] >= 3:
                    next_p = (cp + 1) % 4
                    while not game.active_players[next_p]:
                        next_p = (next_p + 1) % 4
                    game.current_player = next_p
                    game.current_dice_roll = 0
                    consecutive_sixes[i, cp] = 0
                    actions.append(-1)
                    continue

            legal_moves = ludo_cpp.get_legal_moves(game)
            if not legal_moves:
                next_p = (cp + 1) % 4
                while not game.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                game.current_player = next_p
                game.current_dice_roll = 0
                actions.append(-1)
                continue

            # Bot picks action
            bot_name = assignments[i][cp]
            bot = bots[bot_name]
            action = bot.select_move(game, legal_moves)
            actions.append(action)

            # Record: V9 14-channel encoding
            state_v9 = ludo_cpp.encode_state_v9(game)
            legal_mask = np.zeros(4, dtype=np.float32)
            for m in legal_moves:
                legal_mask[m] = 1.0

            trajectories[i][cp].append((state_v9, action, legal_mask))

        # Step env
        env.step(actions)

        # Check completions
        for i in range(BATCH_SIZE):
            game = env.get_game(i)
            if game.is_terminal:
                move_counts[i] += 1

                winner = ludo_cpp.get_winner(game)

                if winner != -1:
                    for pid in (0, 2):
                        traj = trajectories[i][pid]
                        z = 1.0 if pid == winner else -1.0

                        for step in traj:
                            buffer_states.append(step[0])
                            buffer_actions.append(step[1])
                            buffer_masks.append(step[2])
                            buffer_values.append(z)

                games_completed += 1
                pbar.update(1)

                if len(buffer_states) >= CHUNK_SIZE:
                    save_chunk()

                # Reset
                env.reset_game(i)
                move_counts[i] = 0
                consecutive_sixes[i] = 0
                trajectories[i] = {0: [], 2: []}
                assignments[i] = {0: random.choice(bot_names), 2: random.choice(bot_names)}
            else:
                if actions[i] >= 0:
                    move_counts[i] += 1

    pbar.close()

    # Save remaining buffer
    if buffer_states:
        save_chunk()

    print(f"\nDone! Generated {games_completed} games.")
    print(f"Data saved in: {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate V9 SL Data (Bot Behavioral Cloning)")
    parser.add_argument("--games", type=int, default=300000, help="Number of games")
    args = parser.parse_args()

    print(f"[V9 SL Gen] Target games: {args.games:,}")
    print(f"[V9 SL Gen] Batch size: {BATCH_SIZE} (pure CPU, no model)")

    generate_data(args.games)
