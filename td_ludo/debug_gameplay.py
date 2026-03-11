import os
import sys
import torch
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV4
from src.heuristic_bot import HeuristicLudoBot

def debug_gameplay():
    device = torch.device('cpu')
    print("Loading model...")
    model = AlphaLudoV4(num_res_blocks=3, num_channels=32, in_channels=11)
    
    weights_path = "/Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/td_v3_small/model_latest.pt"
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.")
    
    # Setup 2-player game
    state = ludo_cpp.create_initial_state_2p()
    model_player = 0
    opp_player = 2
    bot = HeuristicLudoBot(player_id=opp_player)
    
    move_count = 0
    consecutive_sixes = [0, 0, 0, 0]
    
    print("\n" + "="*50)
    print("STARTING DEBUG GAME: Model (P0) vs Heuristic (P2)")
    print("="*50 + "\n")
    
    while not state.is_terminal and move_count < 1000:
        cp = state.current_player
        
        # Skip inactive players
        if not state.active_players[cp]:
            next_p = (cp + 1) % 4
            while not state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            state.current_player = next_p
            continue
            
        # Roll dice
        if state.current_dice_roll == 0:
            import random
            roll = random.randint(1, 6)
            # Force a 6 on the first turn so we can actually get out of base
            if move_count < 2 and cp == model_player and state.player_positions[cp][0] == -1:
                roll = 6
                
            state.current_dice_roll = roll
            
            if roll == 6:
                consecutive_sixes[cp] += 1
            else:
                consecutive_sixes[cp] = 0
                
            if consecutive_sixes[cp] >= 3:
                print(f"[{move_count:03d}] P{cp} rolled three 6s in a row! Turn ends.")
                next_p = (cp + 1) % 4
                while not state.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                state.current_player = next_p
                state.current_dice_roll = 0
                consecutive_sixes[cp] = 0
                continue
                
        roll = state.current_dice_roll
        legal_moves = ludo_cpp.get_legal_moves(state)
        
        if len(legal_moves) == 0:
            print(f"[{move_count:03d}] P{cp} rolled {roll}. No legal moves. Turn ends.")
            next_p = (cp + 1) % 4
            while not state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            state.current_player = next_p
            state.current_dice_roll = 0
            continue
            
        print(f"\n--- Move {move_count:03d} ---")
        print(f"Player: {'P0 (Model)' if cp == model_player else 'P2 (Bot)'}")
        print(f"Roll: {roll}")
        print(f"Scores: P0={state.scores[0]}, P2={state.scores[2]}")
        print(f"Positions: P0={list(state.player_positions[0])}, P2={list(state.player_positions[2])}")
        
        action = -1
        if cp == model_player:
            # Evaluate all legal moves with the model
            print(f"Legal Moves (Token IDs): {legal_moves}")
            
            if len(legal_moves) == 1:
                action = legal_moves[0]
                print(f"Only 1 legal move: Token {action}")
            else:
                next_tensors = []
                next_players = []
                for move in legal_moves:
                    ns = ludo_cpp.apply_move(state, move)
                    next_tensors.append(ludo_cpp.encode_state(ns))
                    next_players.append(ns.current_player)
                
                with torch.no_grad():
                    batch = torch.from_numpy(np.stack(next_tensors)).to(device, dtype=torch.float32)
                    _, values, _ = model(batch)
                    values = values.squeeze(-1).numpy()
                    
                    # Perspective flip: negate V(s') when it's from opponent's view
                    for i in range(len(values)):
                        if next_players[i] != model_player:
                            values[i] = -values[i]
                            
                print("Model Evaluation:")
                for i, move in enumerate(legal_moves):
                    print(f"  Token {move} -> Q-value: {values[i]:.4f} (Next Player: P{next_players[i]})")
                    
                best_idx = np.argmax(values)
                action = legal_moves[best_idx]
                print(f"Model selects move: Token {action} (max Q: {values[best_idx]:.4f})")
        else:
            action = bot.select_move(state, legal_moves)
            print(f"Bot selects move: Token {action}")
            
        state = ludo_cpp.apply_move(state, action)
        move_count += 1
        
    print("\n" + "="*50)
    winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
    print(f"GAME OVER. Winner: {'P0 (Model)' if winner == model_player else 'P2 (Bot)' if winner == opp_player else 'None'}")
    print(f"Total Moves: {move_count}")
    print("="*50)

if __name__ == "__main__":
    debug_gameplay()
