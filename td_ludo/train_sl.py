"""
TD-Ludo — Supervised Learning (Behavioral Cloning)

Trains the AlphaLudoV5 Actor-Critic model on the generated bot-vs-bot 
dataset to bootstrap its policy and value networks before self-play RL.

Losses:
- Policy: Cross-Entropy (predict what the bot did)
- Value: MSE (predict whether the bot won)
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import AlphaLudoV5
from src.config import LEARNING_RATE, WEIGHT_DECAY

DATA_DIR = os.path.join("checkpoints", "sl_data")
SAVE_PATH = os.path.join("checkpoints", "ac_v5", "model_sl.pt")
CHECKPOINT_PATH = os.path.join("checkpoints", "ac_v5", "checkpoint_sl.pt")

BATCH_SIZE = 1536
EPOCHS = 1
VAL_SPLIT = 0.05
SAVE_INTERVAL = 5000 # Save every 5000 batches (~5M samples)

class SLIterableDataset(IterableDataset):
    def __init__(self, data_files):
        self.data_files = data_files
        self.total_samples = 0
        # Quick pass to count total samples for progress bar sizing
        print(f"Scanning {len(data_files)} chunks to count samples...")
        for f in tqdm(data_files, desc="Counting"):
            try:
                # np.load with mmap_mode='r' only reads metadata, very fast
                with np.load(f, mmap_mode='r') as data:
                    self.total_samples += len(data['values'])
            except Exception as e:
                print(f"Error reading {f}: {e}")
        print(f"Total samples to process: {self.total_samples:,}")

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        # We shuffle the order of files per epoch
        files = list(self.data_files)
        np.random.shuffle(files)
        
        for f in files:
            try:
                data = np.load(f)
                states = data['states']
                actions = data['actions']
                masks = data['masks']
                values = data['values']
                
                # Shuffle within the chunk
                idx = np.arange(len(states))
                np.random.shuffle(idx)
                
                for i in idx:
                    yield (
                        torch.from_numpy(states[i]),
                        torch.tensor(actions[i], dtype=torch.long),
                        torch.from_numpy(masks[i]),
                        torch.tensor(values[i], dtype=torch.float32)
                    )
            except Exception as e:
                print(f"Error yielding from {f}: {e}")
                continue

def train_sl():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not data_files:
        print(f"No .npz files found in {DATA_DIR}. Run generate_sl_data.py first.")
        return
    # Split train/val at the file-level
    np.random.shuffle(data_files)
    val_count = max(1, int(len(data_files) * VAL_SPLIT))
    val_files = data_files[:val_count]
    train_files = data_files[val_count:]
    
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    train_ds = SLIterableDataset(train_files)
    val_ds = SLIterableDataset(val_files)
    
    train_size = len(train_ds)
    val_size = len(val_ds)
    
    # DataLoader for IterableDataset cannot use shuffle=True
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    # Initialize model — V6 Big Brain (10 blocks × 128 channels)
    model = AlphaLudoV5(num_res_blocks=10, num_channels=128).to(device)
    print(f"[SL] Architecture: V6-Big (128ch, 10res, {model.count_parameters():,} params)")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    
    criterion_policy = nn.NLLLoss()
    criterion_value = nn.MSELoss()
    
    start_epoch = 0
    start_batch = 0
    best_val_acc = 0.0

    # Resume from checkpoint if compatible V6 weights exist
    loaded_from = None
    if os.path.exists(CHECKPOINT_PATH):
        loaded_from = CHECKPOINT_PATH
    elif os.path.exists(SAVE_PATH):
        loaded_from = SAVE_PATH

    if loaded_from:
        print(f"Attempting to load weights from: {loaded_from}...")
        try:
            checkpoint = torch.load(loaded_from, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 1)
                start_batch = checkpoint.get('batch', 0)
                best_val_acc = checkpoint.get('val_acc', 0.0)
                print(f"Resuming from Epoch {start_epoch+1}, Batch {start_batch}")
            else:
                model.load_state_dict(checkpoint)
                start_epoch = 1
                print("Loaded raw state dict, starting from Epoch 2")
        except RuntimeError as e:
            print(f"[SL] Cannot load checkpoint (architecture mismatch?): {e}")
            print("[SL] Starting fresh with random weights.")
            start_epoch = 0
            start_batch = 0
            best_val_acc = 0.0
    
    print("\nStarting Supervised Training...")
    
    try:
      for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_correct = 0
        
        batch_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", total=train_size//BATCH_SIZE)
        
        for states, actions, masks, values in train_loader:
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_count < start_batch:
                batch_count += 1
                pbar.update(1)
                continue
                
            states = states.to(device)
            actions = actions.to(device)
            masks = masks.to(device)
            values = values.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_logits, value_preds = model(states, masks)
            value_preds = value_preds.squeeze(-1)
            
            # Policy loss (NLLLoss over log-probabilities since model returns probabilities)
            loss_p = criterion_policy(torch.log(policy_logits + 1e-8), actions)
            # Value loss (MSE against terminal outcome +1/-1)
            loss_v = criterion_value(value_preds, values)
            
            loss = loss_p + 0.5 * loss_v
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * states.size(0)
            train_policy_loss += loss_p.item() * states.size(0)
            train_value_loss += loss_v.item() * states.size(0)
            
            # Accuracy
            preds = torch.argmax(policy_logits, dim=1)
            train_correct += (preds == actions).sum().item()
            
            batch_count += 1
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item(), 'acc': (preds == actions).float().mean().item()})
            
            # Periodic Mid-Epoch Checkpoint
            if batch_count % SAVE_INTERVAL == 0:
                torch.save({
                    'epoch': epoch,
                    'batch': batch_count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc
                }, CHECKPOINT_PATH)
                # Also update model_sl.pt as a backup
                torch.save(model.state_dict(), SAVE_PATH)
        
        # Reset start_batch for next epoch
        start_batch = 0
        
        # Calculate epoch averages (approximate if we skip)
        processed_samples = (batch_count) * BATCH_SIZE
        if processed_samples > 0:
            avg_train_loss = train_loss / processed_samples
            avg_train_acc = train_correct / processed_samples
        else:
            avg_train_loss = 0
            avg_train_acc = 0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for states, actions, masks, values in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                states = states.to(device)
                actions = actions.to(device)
                masks = masks.to(device)
                values = values.to(device)
                
                policy_logits, value_preds = model(states, masks)
                value_preds = value_preds.squeeze(-1)
                
                loss_p = criterion_policy(policy_logits, actions)
                loss_v = criterion_value(value_preds, values)
                loss = loss_p + 0.5 * loss_v
                
                val_loss += loss.item() * states.size(0)
                preds = torch.argmax(policy_logits, dim=1)
                val_correct += (preds == actions).sum().item()
                
        val_loss /= val_size
        val_acc = val_correct / val_size
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.1f}%")
        print(f"                - Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.1f}%")
        
        # Save best model and full checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'batch': 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, CHECKPOINT_PATH)
            
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"*** New Best Val Acc: {val_acc*100:.2f}% - Saved model. ***")
            
    except KeyboardInterrupt:
        print("\n[SL] Interrupted! Saving checkpoint...")
        torch.save({
            'epoch': epoch if 'epoch' in dir() else 0,
            'batch': batch_count if 'batch_count' in dir() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': best_val_acc
        }, CHECKPOINT_PATH)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"[SL] Saved to {CHECKPOINT_PATH} and {SAVE_PATH}")
    
    print(f"\nTraining Complete! Best Val Accuracy: {best_val_acc*100:.1f}%")
    print(f"Saved pre-trained model to {SAVE_PATH}")
    print("\nTo use this with RL self-play:")
    print("  td_env/bin/python train.py --fresh")

if __name__ == "__main__":
    train_sl()
