"""
TD-Ludo V7 — SL Data Generator (Behavioral Cloning for Transformer)

Plays thousands of 2-player games between heuristic bots and records
1D state data compatible with the V7 Sequence Transformer.

Per-step data saved:
  - token_positions: (8,) int64  — 4 self + 4 opponent token positions
  - continuous: (9,) float32     — locked fracs, score diff, dice one-hot
  - action: int (0-3)            — which token the bot moved
  - legal_mask: (4,) float32     — legal moves mask
  - value: float (+1 win, -1 loss)

Context windows (K=16 turn history) are reconstructed at training time
to avoid storing redundant data.
"""

import os
import sys
import argparse
import time
import random
import glob
import numpy as np
from tqdm import tqdm

# Add project roots to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import td_ludo_cpp as ludo_cpp
from src.heuristic_bot import HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot, ExpertBot
from src.state_encoder_1d import encode_state_1d
from src.config import MAX_MOVES_PER_GAME

DATA_DIR = os.path.join("checkpoints", "sl_data_v7")
os.makedirs(DATA_DIR, exist_ok=True)

BATCH_SIZE = 100
CHUNK_SIZE = 100_000  # Save a chunk every 100k steps


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

    # Per-game trajectory: game_idx -> player_id -> list of (tok, cont, action, mask)
    trajectories = {i: {0: [], 2: []} for i in range(BATCH_SIZE)}

    # Randomly assign bots to P0 and P2
    assignments = {}
    for i in range(BATCH_SIZE):
        assignments[i] = {0: random.choice(bot_names), 2: random.choice(bot_names)}

    consecutive_sixes = np.zeros((BATCH_SIZE, 4), dtype=int)
    move_counts = np.zeros(BATCH_SIZE, dtype=int)

    # Auto-detect existing chunks
    existing = glob.glob(os.path.join(DATA_DIR, "chunk_*.npz"))
    if existing:
        max_idx = max(int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing)
        chunk_idx = max_idx + 1
        print(f"Found {len(existing)} existing chunks. Continuing from chunk_{chunk_idx:04d}.")
    else:
        chunk_idx = 0

    # Buffers
    buffer_tok = []
    buffer_cont = []
    buffer_actions = []
    buffer_masks = []
    buffer_values = []
    buffer_game_ids = []     # To group steps into games for context window reconstruction
    buffer_step_ids = []     # Step index within the game

    games_completed = 0
    game_id_counter = 0
    pbar = tqdm(total=num_games, desc="Generating V7 SL Data")

    def save_chunk():
        nonlocal buffer_tok, buffer_cont, buffer_actions, buffer_masks, buffer_values
        nonlocal buffer_game_ids, buffer_step_ids, chunk_idx
        if not buffer_tok:
            return

        chunk_path = os.path.join(DATA_DIR, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            chunk_path,
            token_positions=np.stack(buffer_tok).astype(np.int64),
            continuous=np.stack(buffer_cont).astype(np.float32),
            actions=np.array(buffer_actions, dtype=np.int64),
            masks=np.stack(buffer_masks).astype(np.float32),
            values=np.array(buffer_values, dtype=np.float32),
            game_ids=np.array(buffer_game_ids, dtype=np.int64),
            step_ids=np.array(buffer_step_ids, dtype=np.int64),
        )
        print(f"\nSaved {chunk_path} with {len(buffer_tok)} samples.")

        chunk_idx += 1
        buffer_tok, buffer_cont, buffer_actions, buffer_masks = [], [], [], []
        buffer_values, buffer_game_ids, buffer_step_ids = [], [], []

    while games_completed < num_games:
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

            # Ask the assigned bot for a move
            bot_name = assignments[i][cp]
            bot = bots[bot_name]
            action = bot.select_move(game, legal_moves)
            actions.append(action)

            # Record 1D state
            tok, cont = encode_state_1d(game)
            legal_mask = np.zeros(4, dtype=np.float32)
            for m in legal_moves:
                legal_mask[m] = 1.0

            step_idx = len(trajectories[i][cp])
            trajectories[i][cp].append((tok, cont, action, legal_mask, step_idx))

        # Step env
        env.step(actions)

        # Check for completed games
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
                            buffer_tok.append(step[0])
                            buffer_cont.append(step[1])
                            buffer_actions.append(step[2])
                            buffer_masks.append(step[3])
                            buffer_values.append(z)
                            buffer_game_ids.append(game_id_counter)
                            buffer_step_ids.append(step[4])

                        game_id_counter += 1

                games_completed += 1
                pbar.update(1)

                if len(buffer_tok) >= CHUNK_SIZE:
                    save_chunk()

                # Reset game
                env.reset_game(i)
                move_counts[i] = 0
                consecutive_sixes[i] = 0
                trajectories[i] = {0: [], 2: []}
                assignments[i] = {0: random.choice(bot_names), 2: random.choice(bot_names)}
            else:
                if actions[i] >= 0:
                    move_counts[i] += 1

    pbar.close()

    # Save remaining
    if buffer_tok:
        save_chunk()

    print(f"\nDone! Generated {games_completed} games.")
    print(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate V7 Ludo Behavioral Cloning Data")
    parser.add_argument("--games", type=int, default=50000, help="Number of games to simulate")
    args = parser.parse_args()

    generate_data(args.games)
