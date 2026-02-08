
"""
Multi-Model Evaluation Tournament
---------------------------------
Evaluates 3 different model checkpoints against a Heuristic Bot in the same game.
Each game features:
- Model A (Cycle 75)
- Model B (Cycle 95)
- Model C (Cycle 100)
- Heuristic Bot

Positions are randomized for every game.
"""

import os
import sys
import random
import time
import argparse
import torch
import numpy as np
import glob

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ludo_cpp
from src.model_v3 import AlphaLudoV3
from src.heuristic_bot import HeuristicLudoBot
from src.tensor_utils_mastery import state_to_tensor_mastery

MODELS_DIR = "experiments/kickstart/models_to_test"

def load_model(path, device):
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    model.to(device)
    model.eval()
    try:
        ckpt = torch.load(path, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"✅ Loaded: {os.path.basename(path)}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        return None

def get_model_action(model, state, legal_moves, device):
    state_tensor = state_to_tensor_mastery(state)
    input_tensor = state_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        policy, _ = model.forward_policy_value(input_tensor)
    
    probs = policy[0].cpu().numpy()
    masked_probs = np.zeros(4)
    for m in legal_moves:
        masked_probs[m] = probs[m]
    
    if masked_probs.sum() > 0:
        return np.argmax(masked_probs) # Greedy
    return random.choice(legal_moves)

def run_tournament(num_games=100):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Models
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    if len(model_files) < 3:
        print(f"❌ Need at least 3 models in {MODELS_DIR}, found {len(model_files)}")
        return

    # Identify models by filename for tracking
    models = {}
    for f in model_files:
        name = os.path.basename(f).replace("model_kickstart_", "").replace("_cycle.pt", "")
        models[name] = load_model(f, device)
        
    model_names = list(models.keys())
    print(f"Competing Models: {model_names}")
    
    # Bot
    bot = HeuristicLudoBot()
    
    # Stats
    wins = {name: 0 for name in model_names}
    wins['Bot'] = 0
    total_moves_history = []
    
    print(f"\n🚀 Starting {num_games} Games Tournament...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for game_idx in range(num_games):
        # Assign players randomly
        # We need 4 slots: 3 models + 1 bot.
        # If we have < 3 models, strictly this logic fails as per user request.
        # User said "3 of the model weights".
        
        # Create deck: [ModelA, ModelB, ModelC, Bot]
        participants = [(name, models[name]) for name in model_names[:3]] 
        participants.append(('Bot', bot))
        
        random.shuffle(participants)
        
        # Map player_idx -> (name, agent)
        players = {i: participants[i] for i in range(4)}
        
        # Run Game
        state = ludo_cpp.GameState()
        moves = 0
        winner = -1
        
        while not state.is_terminal and moves < 2000:
            current_player = state.current_player
            name, agent = players[current_player]
            
            # Roll Dice (Simplified logic from test_pure_model.py needed here? 
            # Actually ludo_cpp handles turns somewhat, but we need to handle dice rolling if it's 0)
            if state.current_dice_roll == 0:
                state.current_dice_roll = np.random.randint(1, 7)
                # Note: Logic for 3 sixes is in C++ or Python? 
                # In `test_pure_model.py` we handled it manually. Let's replicate that minimal logic.
                # Actually, strictly standard Ludo rules might be complex. 
                # Let's trust `ludo_cpp` to handle state updates but we inject dice.
                # Wait, `actions` validation depends on dice.
            
            legal_moves = ludo_cpp.get_legal_moves(state)
            
            if not legal_moves:
                # Pass turn
                # Manual turn passing if C++ doesn't auto-pass on no moves?
                # `apply_move` usually returns next state. 
                # If no moves, we might need a "pass" move or just increment player?
                # test_pure_model handled this by:
                # state.current_player = (state.current_player + 1) % 4
                # state.current_dice_roll = 0
                state.current_player = (state.current_player + 1) % 4
                state.current_dice_roll = 0
                continue
                
            # Select Move
            if name == 'Bot':
                action = agent.select_move(state, legal_moves)
            else:
                # It's a model
                action = get_model_action(agent, state, legal_moves, device)
            
            # Apply
            state = ludo_cpp.apply_move(state, action)
            moves += 1
            
            # Check win (manual check from previous script is safer)
            if state.scores[current_player] == 4:
                winner = current_player
                break
        
        # Determine winner if max moves reached (highest score)
        if winner == -1:
            scores = [state.scores[i] for i in range(4)]
            winner = np.argmax(scores)
            
        winner_name = players[winner][0]
        wins[winner_name] += 1
        total_moves_history.append(moves)
        
        if (game_idx + 1) % 10 == 0:
            print(f"Game {game_idx+1}: Winner {winner_name} (Moves: {moves})")

    # Final Report
    print(f"\n{'='*60}")
    print(f"🏆 TOURNAMENT RESULTS ({num_games} Games)")
    print(f"{'='*60}")
    
    sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, count) in enumerate(sorted_wins, 1):
        rate = count / num_games * 100
        print(f"{rank}. {name:15s}: {count} wins ({rate:.1f}%)")
        
    print(f"\nAvg Moves: {np.mean(total_moves_history):.1f}")
    print(f"Total Time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    run_tournament()
