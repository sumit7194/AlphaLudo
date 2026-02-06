
import os
import sys
import glob
import pickle
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model_v3 import AlphaLudoV3
from src.config import CONFIGS

# Force PROD Config
CONF = CONFIGS["PROD"]
LEARNING_RATE = CONF["LEARNING_RATE"]
BATCH_SIZE = CONF["TRAINING_BATCH_SIZE"]
AUX_WEIGHT = CONF["AUX_LOSS_WEIGHT"]

# Configuration
SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SANDBOX_DIR, "model_kickstart.pt")
STATS_PATH = os.path.join(SANDBOX_DIR, "kickstart_stats.json")
BUFFER_PATTERN = "data/kickstart_buffer.pkl.part_*"
EPOCHS = 100 
SAVE_INTERVAL = 1000 
BATCH_SIZE = 1024 # Reduced from 2048 for Stability

class ShardedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_pattern):
        self.files = glob.glob(file_pattern)
        np.random.shuffle(self.files) 
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_files = self.files
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            my_files = [f for i, f in enumerate(self.files) if i % num_workers == worker_id]
            
        for f_path in my_files:
            try:
                with open(f_path, 'rb') as f:
                    samples = pickle.load(f)
                    np.random.shuffle(samples)
                    for s in samples:
                         yield s
            except: pass

def get_buffer_stats():
    try:
        files = glob.glob(BUFFER_PATTERN)
        count = len(files)
        total_bytes = sum(os.path.getsize(f) for f in files)
        gb = total_bytes / (1024**3)
        # EST: 50k samples per shard (from generate_data.py)
        est_samples = count * 50000
        return count, gb, est_samples
    except:
        return 0, 0.0, 0

def write_stats(step, p_loss, v_loss, sps, duration):
    count, gb, samples = get_buffer_stats()
    stats = {
        'step': step,
        'policy_loss': p_loss,
        'value_loss': v_loss,
        'samples_per_sec': sps,
        'duration_sec': duration,
        'buffer_files': count,
        'buffer_gb': gb,
        'buffer_samples': samples,
        'timestamp': time.time()
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f)

def train_kickstart():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device} with LR={LEARNING_RATE}, BS={BATCH_SIZE}")
    
    # Load Model (Same)
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128) 
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}")
        ckpt = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    else:
        print("Starting from scratch (no model found).")
        
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    step = 0
    start_time = time.time()
    
    try:
        for epoch in range(EPOCHS):
            print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
            
            # Re-init dataset to discover new files
            dataset = ShardedIterableDataset(BUFFER_PATTERN)
            
            # OPTIMIZATION: 4 Workers (Safe for 16GB RAM)
            loader = DataLoader(
                dataset, 
                batch_size=BATCH_SIZE, 
                num_workers=4, 
                persistent_workers=True,
                prefetch_factor=2
            )
            
            total_policy_loss = 0
            total_value_loss = 0
            samples_seen = 0
            
            for batch in loader:
                states, target_pis, target_vs = batch
                states, target_pis, target_vs = states.to(device), target_pis.to(device), target_vs.to(device)
                
                # Forward
                pred_pis, pred_vs, aux = model(states) 
                
                # Losses
                log_pis = torch.log(pred_pis + 1e-8)
                policy_loss = -torch.sum(target_pis * log_pis, dim=1).mean()
                value_loss = nn.MSELoss()(pred_vs, target_vs)
                
                # Total Loss (TODO: Add Aux Loss if needed, but targets absent in current generator)
                loss = policy_loss + value_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Stats
                p_val = policy_loss.item()
                v_val = value_loss.item()
                total_policy_loss += p_val
                total_value_loss += v_val
                samples_seen += 1
                step += 1
                
                if step % 50 == 0:
                    avg_p = total_policy_loss / samples_seen
                    avg_v = total_value_loss / samples_seen
                    
                    duration = time.time() - start_time
                    sps = step * BATCH_SIZE / duration
                    
                    # Update JSON
                    write_stats(step, avg_p, avg_v, sps, duration)
                    print(f"Step {step} | P: {avg_p:.4f} | V: {avg_v:.4f} | {sps:.0f}/s")
                    
                    # Reset accumulators for rolling average? 
                    # Or keep epoch average. Let's reset for sharper feedback.
                    total_policy_loss = 0
                    total_value_loss = 0
                    samples_seen = 0
                    
                if step % SAVE_INTERVAL == 0:
                    torch.save({'model_state_dict': model.state_dict(), 'step': step}, MODEL_PATH)
                    
    except KeyboardInterrupt:
        print("Interrupted")
        
    torch.save({'model_state_dict': model.state_dict(), 'step': step}, MODEL_PATH)
    print("Training Complete.")

if __name__ == "__main__":
    train_kickstart()
