import torch
import sys
import os
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV5
from evaluate import evaluate_model

def run():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = AlphaLudoV5().to(device)
    sl_path = os.path.join("checkpoints", "ac_v5", "model_sl.pt")
    
    if not os.path.exists(sl_path):
        print(f"ERROR: {sl_path} not found!")
        sys.exit(1)
        
    print(f"Loading SL weights from {sl_path}...")
    model_state = torch.load(sl_path, map_location=device)
    
    # Handle both full checkpoint dicts and raw state_dicts
    if isinstance(model_state, dict) and 'model_state_dict' in model_state:
        model.load_state_dict(model_state['model_state_dict'])
    else:
        model.load_state_dict(model_state)
        
    print(f"Model parameters: {model.count_parameters():,}")
    print("Evaluating SL Model against Heuristic Bots (1000 games)...")
    
    # Run evaluation
    results = evaluate_model(model, device, num_games=1000, verbose=True)
    
if __name__ == "__main__":
    run()
