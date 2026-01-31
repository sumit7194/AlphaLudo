
import random
import torch
import numpy as np
import ludo_cpp
from src.model_v3 import AlphaLudoV3
from src.tensor_utils_mastery import state_to_tensor_mastery

def log_state(f, state, step, dice, current_player):
    f.write(f"\n--- Step {step} | Player {current_player} | Dice {dice} ---\n")
    # Log positions
    for p in range(4):
        pos = state.player_positions[p]
        f.write(f"  P{p}: {pos}\n")
    f.write(f"  Scores: {state.scores}\n")

def run_logged_game(game_id, model, device):
    filename = f"game_debug_{game_id}.txt"
    with open(filename, 'w') as f:
        f.write(f"DEBUG LOG FOR GAME {game_id}\n")
        f.write("==========================\n")
        
        state = ludo_cpp.GameState()
        step = 0
        consecutive_sixes = [0]*4
        
        while not state.is_terminal and step < 1000:
            pid = state.current_player
            
            # Dice
            if state.current_dice_roll == 0:
                roll = random.randint(1, 6)
                state.current_dice_roll = roll
                if roll == 6:
                    consecutive_sixes[pid] += 1
                else:
                    consecutive_sixes[pid] = 0
                
                if consecutive_sixes[pid] >= 3:
                    f.write(f"[System] P{pid} rolled three 6s -> Skip Turn\n")
                    state.current_player = (pid + 1) % 4
                    state.current_dice_roll = 0
                    consecutive_sixes[pid] = 0
                    continue
            
            dice = state.current_dice_roll
            log_state(f, state, step, dice, pid)
            
            moves = ludo_cpp.get_legal_moves(state)
            if not moves:
                f.write(f"  [System] No legal moves for P{pid} with dice {dice}\n")
                state.current_player = (pid + 1) % 4
                state.current_dice_roll = 0
                continue
            
            f.write(f"  Legal Moves: {moves}\n")
            
            # Select Action
            if pid == 0: # Model
                tens = state_to_tensor_mastery(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    policy, values = model.forward_policy_value(tens)
                
                probs = policy[0].cpu().numpy()
                val = values[0].item()
                
                f.write(f"  Model Value Estimate: {val:.3f}\n")
                f.write(f"  Raw Policy: {['%.3f'%p for p in probs]}\n")
                
                # Mask
                masked_probs = np.zeros(4)
                for m in moves:
                    masked_probs[m] = probs[m]
                
                if masked_probs.sum() > 0:
                    masked_probs /= masked_probs.sum()
                    action = np.argmax(masked_probs)
                    f.write(f"  -> Model chose {action} (prob {masked_probs[action]:.2f})\n")
                else:
                    action = random.choice(moves)
                    f.write(f"  -> Model forced random (zero prob on legal moves)\n")
            else:
                action = random.choice(moves)
                f.write(f"  -> RandomBot chose {action}\n")
            
            # Execute
            state = ludo_cpp.apply_move(state, action)
            step += 1
            
        # Result
        winner = -1
        for p in range(4):
            if state.scores[p] == 4: winner = p
        f.write(f"\nGAME OVER. Winner: {winner}\n")
        return winner

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    model.to(device) 
    model.eval()
    
    try:
        ckpt = torch.load("checkpoints_mastery/mastery_v3_prod/model_latest.pt", map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Model loaded.")
    except:
        print("Using random weights (checkpoint not found)")
        
    print("Running 5 debug games...")
    for i in range(5):
        w = run_logged_game(i+1, model, device)
        print(f"Game {i+1} Winner: {w}")
