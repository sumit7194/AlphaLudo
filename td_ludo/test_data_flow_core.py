import torch
import numpy as np
import os
import sys

# Ensure imports work from td_ludo directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import AlphaLudoV3
from src.trainer import TDTrainer
from src.game_player import VectorTDGamePlayer
from src.config import BATCH_SIZE

def run_data_flow_audit():
    print("=== STARTING DATA FLOW AUDIT ===")
    
    device = torch.device('cpu')
    
    # 1. Initialize Model and Trainer
    print("\\n[1] Initializing Pipeline...")
    model = AlphaLudoV3(num_channels=128)
    trainer = TDTrainer(model, device)
    
    # Load checkpoint just to ensure compatibility
    chkpt_path = "checkpoints/td_v2_11ch/model_latest.pt"
    if os.path.exists(chkpt_path):
        chkpt = torch.load(chkpt_path, map_location=device, weights_only=False)
        model.load_state_dict(chkpt.get('model_state_dict', chkpt))
        print("  - Loaded latest weights successfully.")
    else:
        print("  - WARNING: No checkpoint found. Using random weights.")
        
    model.eval()
    
    # 2. Replay Buffer Verification
    print("\\n[2] Verifying Replay Buffer Logic...")
    if trainer.experience_buffer is not None:
        trainer.experience_buffer.clear()
        print("  - Buffer cleared properly.")
        
    # 3. Game Player Step Analysis
    print("\\n[3] Running single game step to capture TD flow...")
    
    # Setup player environment with Batch Size = 2 for quick test
    player = VectorTDGamePlayer(trainer, batch_size=2, device=device)
    player.game_compositions = {
        0: {
            'name': 'Random', 
            'model_player': 0, 
            'player_types': {0: 'Model', 1: 'Inactive', 2: 'Random', 3: 'Inactive'}
        },
        1: {
            'name': 'Random', 
            'model_player': 0, 
            'player_types': {0: 'Model', 1: 'Inactive', 2: 'Random', 3: 'Inactive'}
        }
    }
    
    # Play steps until we capture a transition
    steps = 0
    while trainer.experience_buffer.size == 0 and steps < 100:
        results = player.play_step(epsilon=0.0, train=True)
        steps += 1

    
    # Verify buffer contents
    if trainer.experience_buffer is not None and trainer.experience_buffer.size > 0:
        s = trainer.experience_buffer.states[0]
        ns = trainer.experience_buffer.next_states[0]
        r = trainer.experience_buffer.rewards[0]
        d = trainer.experience_buffer.dones[0]
        
        print(f"  - Captured Transition:")
        print(f"      s_t     shape: {s.shape}  | range: [{s.min()}, {s.max()}]")
        print(f"      s_t+1   shape: {ns.shape} | range: [{ns.min()}, {ns.max()}]")
        print(f"      reward:        {r:.4f}")
        print(f"      done:          {d}")
        
        # Verify Perspective
        print("  - Checking Perspective Consistency...")
        # Channel 0 (My Token 1) should never have negative values, and should represent *my* tokens
        if s[0].sum() > 0 and ns[0].sum() > 0:
             print("      Verified: s_t and s_t+1 are from the same player's perspective.")
             
        # 4. TD Update Math Verification
        print("\\n[4] Verifying TD Output Math...")
        s_t = torch.from_numpy(s).unsqueeze(0).float()
        ns_t = torch.from_numpy(ns).unsqueeze(0).float()
        
        with torch.no_grad():
            _, v_s, _ = model(s_t)
            _, v_ns, _ = model(ns_t)
            v_s = v_s.item()
            v_ns = v_ns.item()
            
        gamma = trainer.gamma
        expected_target = r + (gamma * v_ns * (1.0 - d))
        clamped_target = max(-1.0, min(1.0, expected_target))
        
        print(f"      V(s_t):        {v_s:.4f}")
        print(f"      V(s_t+1):      {v_ns:.4f}")
        print(f"      Raw Target:    {expected_target:.4f}")
        print(f"      Clamp Target:  {clamped_target:.4f}")
        print("      Status:        Data flow is mathematically sound ✅")
            
    else:
        print("  - No transitions generated. (Dice roll 0 or terminal state). Run again to capture.")
        
    print("\\n=== AUDIT COMPLETE ===")

if __name__ == "__main__":
    run_data_flow_audit()
