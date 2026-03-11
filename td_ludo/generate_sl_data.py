"""
TD-Ludo — SL Data Generator (Behavioral Cloning)

This script plays thousands of fast 2-player games between our best bots
(Heuristic and Aggressive). It canonicalizes the state for EVERY move
and records:
  - state: (11, 15, 15) float32
  - action: int (0 to 3)
  - prob_mask: (4,) float32 (legal moves)
  - value: float (+1 win, -1 loss)

Data is saved in compressed chunk files (.npz) to be used for
Behavioral Cloning supervised pre-training.
"""

import os
import sys
import argparse
import time
import random
import numpy as np
from tqdm import tqdm

# Add the project root to the path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import td_ludo_cpp as ludo_cpp
from src.heuristic_bot import HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot
from src.config import MAX_MOVES_PER_GAME

DATA_DIR = os.path.join("checkpoints", "sl_data")
os.makedirs(DATA_DIR, exist_ok=True)

BATCH_SIZE = 100
CHUNK_SIZE = 100_000   # Save a chunk every 100k steps to avoid RAM bloat

def generate_data(num_games):
    bots = {
        'Heuristic': HeuristicLudoBot(),
        'Aggressive': AggressiveBot(),
        'Defensive': DefensiveBot(),
        'Racing': RacingBot(),
    }
    bot_names = list(bots.keys())
    
    env = ludo_cpp.VectorGameState(BATCH_SIZE, True)  # True = 2P mode
    
    # Store trajectories per game: game_idx -> player_id -> list of (state, action, mask)
    trajectories = {i: {0: [], 2: []} for i in range(BATCH_SIZE)}
    
    # Randomly assign bots to P0 and P2 for each game
    assignments = {}
    for i in range(BATCH_SIZE):
        assignments[i] = {0: random.choice(bot_names), 2: random.choice(bot_names)}
        
    consecutive_sixes = np.zeros((BATCH_SIZE, 4), dtype=int)
    move_counts = np.zeros(BATCH_SIZE, dtype=int)
    
    # Auto-detect existing chunks to avoid overwriting
    import glob
    existing = glob.glob(os.path.join(DATA_DIR, "chunk_*.npz"))
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
    pbar = tqdm(total=num_games, desc="Generating SL Data")
    
    def save_chunk():
        nonlocal buffer_states, buffer_actions, buffer_masks, buffer_values, chunk_idx
        if not buffer_states: return
        
        chunk_path = os.path.join(DATA_DIR, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            chunk_path,
            states=np.stack(buffer_states).astype(np.float32),
            actions=np.array(buffer_actions, dtype=np.int64),
            masks=np.stack(buffer_masks).astype(np.float32),
            values=np.array(buffer_values, dtype=np.float32)
        )
        print(f"\nSaved {chunk_path} with {len(buffer_states)} samples.")
        
        chunk_idx += 1
        buffer_states, buffer_actions, buffer_masks, buffer_values = [], [], [], []

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
            
            # Record trajectory step
            state_tensor = ludo_cpp.encode_state(game)
            legal_mask = np.zeros(4, dtype=np.float32)
            for m in legal_moves:
                legal_mask[m] = 1.0
                
            trajectories[i][cp].append((state_tensor, action, legal_mask))
            
        # Step env
        env.step(actions)
        
        # Check for completed games
        for i in range(BATCH_SIZE):
            game = env.get_game(i)
            if game.is_terminal:
                move_counts[i] += 1  # For completeness
                
                winner = ludo_cpp.get_winner(game)
                
                # If valid win (not draw/timeout limit)
                if winner != -1:
                    for pid in (0, 2):  # 2-player mode
                        traj = trajectories[i][pid]
                        z = 1.0 if pid == winner else -1.0
                        
                        for step in traj:
                            buffer_states.append(step[0])
                            buffer_actions.append(step[1])
                            buffer_masks.append(step[2])
                            buffer_values.append(z)
                
                games_completed += 1
                pbar.update(1)
                
                # Check buffer size
                if len(buffer_states) >= CHUNK_SIZE:
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
    
    # Save any remaining data
    if buffer_states:
        save_chunk()
        
    total_samples = chunk_idx * CHUNK_SIZE + len(buffer_states)
    print(f"\nDone! Generated {games_completed} games.")
    print(f"Total samples recorded: ~{total_samples:,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ludo Behavioral Cloning Data")
    parser.add_argument("--games", type=int, default=50000, help="Number of games to simulate")
    args = parser.parse_args()
    
    generate_data(args.games)
