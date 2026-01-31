
import torch
import numpy as np
import ludo_cpp
from src.model_v3 import AlphaLudoV3
from src.tensor_utils_mastery import state_to_tensor_mastery
from src.model_v3 import AlphaLudoV3
from src.tensor_utils_mastery import state_to_tensor_mastery

def run_mcts_debug():
    print("Loading Model...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    model.to(device)
    model.eval()
    
    try:
        ckpt = torch.load("checkpoints_mastery/mastery_v3_prod/model_latest.pt", map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Model state loaded.")
    except Exception as e:
        print(f"Using random weights ({e})")

    # params match PROD config
    # We will run TWO tests:
    # 1. Standard: C_PUCT=3.0, EPS=0.25
    # 2. High Exp: C_PUCT=4.0, EPS=0.50
    
    configs = [
        {"name": "Standard (PROD)", "cpuct": 3.0, "eps": 0.25},
        {"name": "High Exploration", "cpuct": 4.0, "eps": 0.50}
    ]
    
    # Create a state where Token 0 is safe but Token 1 could move out
    state = ludo_cpp.GameState()
    state.current_dice_roll = 6 # Force a 6 so we can move out!
    # P0 T0 is at home (0). P0 T1 is at Base.
    # Default state has all at Base (-1).
    # Move T0 to 0.
    state.player_positions[0][0] = 0 
    
    print(f"\n--- DEBUG STATE ---")
    print(f"Dice: {state.current_dice_roll}")
    print(f"P0 Pos: {state.player_positions[0]}")
    print(f"Legal Moves should be: T0 (move 0->6), T1 (Base->0), T2 (Base->0), T3 (Base->0)")
    
    legal_moves = ludo_cpp.get_legal_moves(state)
    print(f"Legal Moves: {legal_moves}")
    
    # Trace model first
    tens = state_to_tensor_mastery(state).unsqueeze(0).to(device)
    with torch.no_grad():
        p, v = model.forward_policy_value(tens)
    raw_p = p[0].cpu().numpy()
    print(f"\nRaw Network Policy (No Noise):")
    for m in legal_moves:
        print(f"  Move {m}: {raw_p[m]:.4f}")

    print("\nRunning MCTS Comparisons...")
    
    for cfg in configs:
        print(f"\n[{cfg['name']}] Settings: C_PUCT={cfg['cpuct']}, EPS={cfg['eps']}")
        
        # Simulating MCTS Math

        
        # Run search
        # We need to bridge Python model to C++ MCTS
        # Using simple python loop for simulation since C++ callback is hard to mock here efficiently without full wrapper
        # Actually, let's use the actual C++ MCTS bound in ludo_cpp if possible, 
        # but providing the evaluator is tricky in a standalone script without the Actor infrastructure.
        
        # ... Wait, I can't easily inject the Python model into the C++ MCTS from a standalone script 
        # unless I use the 'BatchedMCTS' or similar wrapper used in actors.
        
        # Checking how actors do it:
        # they use `self.mcts = ludo_cpp.MCTS(...)` and `self.mcts.search(state, ...)`
        # But `search` requires an evaluator callback or batch processing.
        
        # Simplified approach: We will manually calculate the Root Noise effect here in Python
        # to demonstrate the math, since running full MCTS requires the C++ callback infrastructure.
        
        # 1. Calculate Noised Prior
        noise = np.random.dirichlet([0.3] * len(legal_moves))
        
        print("  Root Priors (P_noised):")
        for i, m in enumerate(legal_moves):
            # P_noised = (1-eps)*P + eps*Noise
            p_net = raw_p[m]
            # Normalization over legal moves
            p_net_norm = p_net / sum(raw_p[lm] for lm in legal_moves)
            
            p_final = (1 - cfg['eps']) * p_net_norm + cfg['eps'] * noise[i]
            print(f"    Move {m}: Net={p_net_norm:.3f} | Noise={noise[i]:.3f} | Final={p_final:.3f}")
            
        # 2. Estimate Impact on UCB
        # U = C * P_final * sqrt(N_total) / (1 + N)
        # Assume N_total=1 (start of search)
        print("  Initial UCB Scores (N=0):")
        for i, m in enumerate(legal_moves):
             p_net = raw_p[m] / sum(raw_p[lm] for lm in legal_moves)
             p_final = (1 - cfg['eps']) * p_net_norm + cfg['eps'] * noise[i]
             u = cfg['cpuct'] * p_final # * sqrt(1)/(1+0) = 1
             print(f"    Move {m}: U={u:.3f}")

if __name__ == "__main__":
    run_mcts_debug()
