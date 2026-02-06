
import os
import sys
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
import pickle

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import ludo_cpp
from src.tensor_utils_mastery import state_to_tensor_mastery, get_board_coords, BOARD_SIZE
from src.training_utils import rotate_token_indices
from src.heuristic_bot import HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot
from src.config import GAME_COMPOSITION

import glob

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Generate Kickstart Training Data")
parser.add_argument("--output", type=str, default=None,
                    help="Output path (base name for shards or directory)")
parser.add_argument("--size_gb", type=float, default=None,
                    help="Target size in GB")
args, _ = parser.parse_known_args()

# Configuration - CLI args take priority over env vars
if args.output is not None:
    BUFFER_PATH = args.output
else:
    BUFFER_PATH = os.environ.get("KICKSTART_BUFFER_PATH", "data/kickstart_buffer.pkl")

if args.size_gb is not None:
    TARGET_SIZE_GB = float(args.size_gb)
else:
    TARGET_SIZE_GB = float(os.environ.get("KICKSTART_TARGET_GB", "10.0"))

BATCH_SIZE = 512  # Run distinct games in parallel for speed

def get_dir_size(path):
    # Check for shards - support both file patterns and directory
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.pkl"))
    else:
        files = glob.glob(f"{path}*")
    total_bytes = sum(os.path.getsize(f) for f in files)
    return total_bytes / (1024**3)  # GB

def generate_kickstart_data():
    print(f"🚀 Starting Kickstart Data Generation...")
    print(f"   Target: {TARGET_SIZE_GB} GB")
    print(f"   Buffer: {BUFFER_PATH}")
    
    # Ensure dir
    os.makedirs(os.path.dirname(BUFFER_PATH), exist_ok=True)
    
    # Initialize Bots
    bots = {
        'Heuristic': HeuristicLudoBot(),
        'Aggressive': AggressiveBot(),
        'Defensive': DefensiveBot(),
        'Racing': RacingBot()
    }
    
    bot_types = ['Heuristic', 'Aggressive', 'Defensive', 'Racing']
    # Probabilities
    probs = [0.4, 0.3, 0.2, 0.1]
    
    # Metrics
    games_played = 0
    start_time = time.time()
    
    # We will accumulate data in memory chunks and append to disk
    # This is safer than keeping 10GB in RAM
    CHUNK_SIZE = 50000 # samples
    current_chunk = []
    
    # States for BATCH_SIZE games
    states = [ludo_cpp.create_initial_state() for _ in range(BATCH_SIZE)]
    move_counts = [0] * BATCH_SIZE
    
    # Assign Bots per Game per Player
    # game_bots[game_idx][player_idx] = 'BotName'
    game_bots = []
    for _ in range(BATCH_SIZE):
        g_bots = np.random.choice(bot_types, size=4, p=probs)
        game_bots.append(g_bots)
        
    game_histories = [[] for _ in range(BATCH_SIZE)]
    
    # Loop indefinitely until size limit
    pbar = tqdm(total=int(TARGET_SIZE_GB * 1000), unit='MB')
    
    while True:
        # Check size every N loops
        if games_played % 100 == 0:
            current_size = get_dir_size(BUFFER_PATH)
            pbar.n = int(current_size * 1000)
            pbar.refresh()
            if current_size >= TARGET_SIZE_GB:
                print(f"\n✅ Target size reached: {current_size:.2f} GB")
                break
        
        # --- BATCH STEP ---
        # 1. Roll Dice (if needed)
        for i in range(BATCH_SIZE):
            if states[i].is_terminal: continue
            
            if states[i].current_dice_roll == 0:
                roll = np.random.randint(1, 7)
                states[i].current_dice_roll = roll
        
        # 2. Process Moves
        for i in range(BATCH_SIZE):
            state = states[i]
            if state.is_terminal:
                # Reset game
                winner = ludo_cpp.get_winner(state)
                
                # Save History
                for record in game_histories[i]:
                    p = record['player']
                    val = 1.0 if winner == p else -1.0
                     # Create sample tuple: (state, policy, value)
                    sample = (record['state'], record['policy'], torch.tensor([val], dtype=torch.float32))
                    current_chunk.append(sample)
                
                # Check Chunk Flush
                if len(current_chunk) >= CHUNK_SIZE:
                    # Save shard - support both directory and file-prefix modes
                    shard_id = int(time.time() * 1000)
                    if os.path.isdir(BUFFER_PATH):
                        # Directory mode: output inside the directory
                        shard_path = os.path.join(BUFFER_PATH, f"shard_{shard_id}.pkl")
                    else:
                        # File-prefix mode: append .part_XXX
                        shard_path = f"{BUFFER_PATH}.part_{shard_id}"
                    with open(shard_path, 'wb') as f:
                        pickle.dump(current_chunk, f)
                    current_chunk = []
                
                # Reset
                states[i] = ludo_cpp.create_initial_state()
                game_bots[i] = np.random.choice(bot_types, size=4, p=probs)
                game_histories[i] = []
                move_counts[i] = 0
                games_played += 1
                continue
            
            # Not Terminal
            current_p = state.current_player
            current_bot_type = game_bots[i][current_p]
            bot = bots[current_bot_type]
            
            legal_moves = ludo_cpp.get_legal_moves(state)
            
            if not legal_moves:
                # No moves
                state.current_player = (current_p + 1) % 4
                state.current_dice_roll = 0
                continue
                
            # Bot chooses move
            # Note: HeuristicBot sometimes returns -1 if no moves (but we checked legal)
            # or if it passes. We trust it picks from legal.
            try:
                action = bot.select_move(state, legal_moves)
            except:
                action = legal_moves[0] # Fallback
            
            # Store Training Data
            # Policy is One-Hot
            policy = np.zeros(4, dtype=np.float32)
            policy[action] = 1.0
            
            state_tensor = state_to_tensor_mastery(state)
            
            game_histories[i].append({
                'state': state_tensor.clone(), # Clone to be safe
                'policy': torch.tensor(policy),
                'player': current_p
            })
            
            # Apply
            states[i] = ludo_cpp.apply_move(state, action)
            move_counts[i] += 1
            
            # Timeout (stalemate prevention)
            if move_counts[i] > 1000:
                # Force reset, discard data
                states[i] = ludo_cpp.create_initial_state()
                game_histories[i] = []
                move_counts[i] = 0
    
    pbar.close()
    print("Generation Complete.")

if __name__ == "__main__":
    generate_kickstart_data()
