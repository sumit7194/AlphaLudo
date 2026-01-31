import torch
import torch.optim as optim
import os
import shutil
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_mastery import AlphaLudoTopNet
from train_mastery import TrainerMastery

CHECKPOINT_DIR = "checkpoints_mastery/mastery_v1"
MAIN_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
GHOSTS_DIR = os.path.join(CHECKPOINT_DIR, "ghosts")

def reset_training():
    print(">>> RESETTING TRAINING STATE <<<")
    
    if not os.path.exists(MAIN_CKPT_PATH):
        print("Error: No checkpoint found to preserve!")
        return

    # 1. Load Existing Weights
    print(f"Loading weights from {MAIN_CKPT_PATH}...")
    checkpoint = torch.load(MAIN_CKPT_PATH, map_location='cpu')
    model_weights = checkpoint['model_state_dict']
    
    # 2. Create Fresh Optimization State
    print("Initializing fresh optimizer state...")
    model = AlphaLudoTopNet()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    fresh_optimizer_state = optimizer.state_dict()
    
    # 3. Save "Clean" Checkpoint
    new_checkpoint = {
        'model_state_dict': model_weights,
        'optimizer_state_dict': fresh_optimizer_state,
        'total_epochs': 0  # Reset Iteration
    }
    torch.save(new_checkpoint, MAIN_CKPT_PATH)
    print(f"Saved clean checkpoint to {MAIN_CKPT_PATH} (Epoch=0).")
    
    # 4. Clean Directories
    if os.path.exists(GHOSTS_DIR):
        print(f"Deleting ghosts dir: {GHOSTS_DIR}")
        shutil.rmtree(GHOSTS_DIR)
        
    stats_files = [
        "training_stats.json",
        "training_metrics.json",
        "elo_ratings.json",
        "wc_stats.json" 
    ]
    
    for f_name in stats_files:
        path = os.path.join(CHECKPOINT_DIR, f_name)
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted {f_name}")
            
    # Also delete buffer if it persists (it implies memory only, but check for pkl)
    buffer_path = os.path.join(CHECKPOINT_DIR, "replay_buffer.pkl")
    if os.path.exists(buffer_path):
        os.remove(buffer_path)
        print("Deleted replay_buffer.pkl")

    print("\n>>> RESET COMPLETE. READY FOR PROD. <<<")

if __name__ == "__main__":
    reset_training()
