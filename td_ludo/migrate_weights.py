"""
Migrates 21-channel AlphaLudo weights to the new 11-channel architecture.
This allows starting a new 11-channel training run while retaining 
the spatial features learned in the 21-channel version.

Usage: td_env/bin/python migrate_weights.py <input_pt> <output_pt>
"""
import torch
import sys
import os

# Ensure we can import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model import AlphaLudoV3

def migrate(input_path, output_path):
    print(f"Migrating {input_path} -> {output_path}...")
    
    # 1. Load old checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        old_sd = checkpoint['model_state_dict']
    else:
        old_sd = checkpoint
    
    # 2. Initialize new 11-channel model
    new_model = AlphaLudoV3(in_channels=11)
    new_sd = new_model.state_dict()
    
    # 3. Map weights
    # The ResNet blocks and heads are identical. Only conv_input (the first layer) changes shape.
    for key in old_sd:
        if key == 'conv_input.weight':
            print("  Mapping conv_input.weight (21 -> 11)...")
            val = old_sd[key] # [128, 21, 3, 3]
            new_val = torch.zeros((128, 11, 3, 3))
            
            # 0-3: My Tokens (unchanged)
            new_val[:, 0:4] = val[:, 0:4]
            
            # 4: Opponent density (mean of old 4, 5, 6 - Next, Team, Prev)
            new_val[:, 4] = val[:, 4:7].mean(dim=1)
            
            # 5: Safe zones (old 7)
            new_val[:, 5] = val[:, 7]
            
            # 6: My home path (old 8)
            new_val[:, 6] = val[:, 8]
            
            # 7: Opp home paths (mean of old 9, 10, 11)
            new_val[:, 7] = val[:, 9:12].mean(dim=1)
            
            # 8: Score diff (old 18)
            new_val[:, 8] = val[:, 18]
            
            # 9: My locked (old 19)
            new_val[:, 9] = val[:, 19]
            
            # 10: Opp locked (old 20)
            new_val[:, 10] = val[:, 20]
            
            new_sd[key] = new_val
        elif key in new_sd:
            if old_sd[key].shape == new_sd[key].shape:
                new_sd[key] = old_sd[key]
            else:
                print(f"  ⚠️ Skipping {key}: Shape mismatch {old_sd[key].shape} vs {new_sd[key].shape}")
        else:
            print(f"  ℹ️ Skipping {key}: Not in new model")

    # 4. Save new checkpoint
    # We strip the optimizer and metrics to start a fresh training run
    torch.save({'model_state_dict': new_sd}, output_path)
    print("Done! Model migrated successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: migrate_weights.py <input_pt> <output_pt>")
        sys.exit(1)
    
    migrate(sys.argv[1], sys.argv[2])
