import os
import sys
import torch
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV5
from src.heuristic_bot import RandomBot

def test_model_gameplay(model_path, num_moves=50):
    device = torch.device('cpu')
    model = AlphaLudoV5(num_res_blocks=5, num_channels=64)
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print(f"Loaded model from {model_path}")
    
    state = ludo_cpp.create_initial_state_2p()
    model_player = 0
    opp_player = 2
    bot = RandomBot()
    
    consecutive_sixes = [0, 0, 0, 0]
    moves_logged = 0
    
    while not state.is_terminal and moves_logged < num_moves:
        current_player = state.current_player
        
        if not state.active_players[current_player]:
            state.current_player = (current_player + 1) % 4
            if not state.active_players[state.current_player]:
                 state.current_player = (state.current_player + 1) % 4
            continue
            
        if state.current_dice_roll == 0:
            import random
            state.current_dice_roll = random.randint(1, 6)
            cp = state.current_player
            if state.current_dice_roll == 6:
                consecutive_sixes[cp] += 1
            else:
                consecutive_sixes[cp] = 0
                
            if consecutive_sixes[cp] >= 3:
                next_p = (cp + 1) % 4
                while not state.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                state.current_player = next_p
                state.current_dice_roll = 0
                consecutive_sixes[cp] = 0
                continue
                
        legal_moves = ludo_cpp.get_legal_moves(state)
        
        if len(legal_moves) == 0:
            next_p = (state.current_player + 1) % 4
            while not state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            state.current_player = next_p
            state.current_dice_roll = 0
            continue
            
        if current_player == model_player:
            # Model's turn
            print(f"\n--- Model's Turn (P{model_player}) ---")
            print(f"Dice Roll: {state.current_dice_roll}")
            print(f"Legal Moves (Tokens): {legal_moves}")
            
            # Print physical positions of tokens
            positions = state.player_positions[model_player]
            for t in range(4):
                if t in legal_moves:
                    status = "Base" if positions[t] == -1 else f"Pos {positions[t]}"
                    print(f"  Token {t} is at {status}")
            
            # Encode state
            state_tensor = ludo_cpp.encode_state(state)
            legal_mask = np.zeros(4, dtype=np.float32)
            for m in legal_moves:
                legal_mask[m] = 1.0
                
            with torch.no_grad():
                s_t = torch.from_numpy(state_tensor).unsqueeze(0).to(device, dtype=torch.float32)
                m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device, dtype=torch.float32)
                # Ensure the model uses softmax to print probabilities
                policy_logits, _ = model(s_t, m_t)
                policy_probs = torch.exp(policy_logits)[0].numpy() # using torch.exp because if we applied log softmax or similar
                
                # if model.forward does softmax:
                # wait, AlphaLudoV5's forward does softmax
                policy = policy_logits[0].numpy()
                action = int(np.argmax(policy))
                
            print("Model Probabilities:")
            for m in legal_moves:
                print(f"  Token {m}: {policy[m]:.4f}")
                
            print(f"Model chose Token: {action}")
            moves_logged += 1
            
        else:
            action = bot.select_move(state, legal_moves)
            
        state = ludo_cpp.apply_move(state, action)

if __name__ == "__main__":
    test_model_gameplay(os.path.join(os.path.dirname(__file__), "checkpoints", "ac_v5", "model_latest.pt"))
