
import time
import random
import torch
import numpy as np
import ludo_cpp
from src.model_v3 import AlphaLudoV3
from src.config import MAIN_CKPT_PATH
from src.visualizer import visualizer
from src.tensor_utils_mastery import state_to_tensor_mastery

# Bots
class RandomBot:
    def select_move(self, state, legal_moves):
        if not legal_moves: return -1
        return random.choice(legal_moves)

def run_debug_session(games=10):
    print("Initializing Visualizer...")
    visualizer.start_server(port=8765)
    time.sleep(2)  # Wait for server
    
    # Send dummy config to activate dashboard
    visualizer.broadcast_training_config(1.0, 0.0, False)
    
    print("Loading Model...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    model.to(device)
    model.eval()
    
    try:
        ckpt = torch.load("checkpoints_mastery/mastery_v3_prod/model_latest.pt", map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Model Loaded!")
    except:
        print("Using Random Weights (Checkpoint not found)")

    print(f"\nStarting {games} Debug Games (Model vs 3 Random Bots)...")
    print("Open http://localhost:8765 to watch!\n")
    
    # Setup identities
    identities = ["Model (v3)", "Random Bot", "Random Bot", "Random Bot"]
    
    for game_idx in range(games):
        print(f"Starting Game {game_idx+1}...")
        
        # Init Game
        state = ludo_cpp.GameState()
        visualizer.broadcast_batch_init(batch_size=1)
        visualizer.broadcast_identities(identities, game_id=0)
        
        step = 0
        consecutive_sixes = [0]*4
        
        while not state.is_terminal and step < 1000:
            # Broadcast State
            visualizer.broadcast_state(state, game_id=0)
            time.sleep(0.5)  # Slow down for watching
            
            pid = state.current_player
            
            # Dice Roll
            if state.current_dice_roll == 0:
                roll = random.randint(1, 6)
                state.current_dice_roll = roll
                # Logic for 6s
                if roll == 6:
                    consecutive_sixes[pid] += 1
                else:
                    consecutive_sixes[pid] = 0
                
                if consecutive_sixes[pid] >= 3:
                    print(f"P{pid} rolled three 6s -> Skip")
                    state.current_player = (pid + 1) % 4
                    state.current_dice_roll = 0
                    consecutive_sixes[pid] = 0
                    continue
                
                # Re-broadcast with dice
                visualizer.broadcast_state(state, game_id=0)
                time.sleep(0.3)
            
            moves = ludo_cpp.get_legal_moves(state)
            
            if not moves:
                state.current_player = (pid + 1) % 4
                state.current_dice_roll = 0
                continue
                
            # Select Move
            if pid == 0:  # Model
                # Prepare input
                tens = state_to_tensor_mastery(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    policy, values = model.forward_policy_value(tens)
                
                # Mask & Greedy
                safe_policy = policy[0].cpu().numpy()
                mask = np.zeros(4)
                for m in moves: mask[m] = 1
                
                masked_probs = safe_policy * mask
                if masked_probs.sum() > 0:
                    action = np.argmax(masked_probs)
                    val = values[0].item()
                    print(f"Model plays {action} (Val: {val:.2f})")
                else:
                    action = random.choice(moves)
                    print("Model fallback random")
            else:
                action = random.choice(moves)
            
            # Apply
            visualizer.broadcast_move(pid, action, state.current_dice_roll, game_id=0)
            state = ludo_cpp.apply_move(state, action)
            step += 1
            
        # End
        winner = -1
        for p in range(4):
            if state.scores[p] == 4: winner = p
        
        print(f"Game Over! Winner: {winner}")
        visualizer.broadcast_game_result(0, winner)
        time.sleep(5) # Pause before next game

if __name__ == "__main__":
    run_debug_session()
