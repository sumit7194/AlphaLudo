"""
TD-Ludo V9 — Supervised Learning (Behavioral Cloning from Bots)

Trains AlphaLudoV9 on bot game data.

Losses:
- Policy: Cross-Entropy (predict bot's action)
- Value: 0.5 × MSE (predict win/loss outcome)

Features:
- Graceful shutdown (Ctrl+C saves checkpoint, resume with --resume)
- Evaluation after each epoch against bots
- Saves best + latest checkpoints
"""

import os
import sys
import signal
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_v9 import AlphaLudoV9
from src.config import WEIGHT_DECAY

DATA_DIR = os.path.join("checkpoints", "sl_data_v9")
SAVE_DIR = os.path.join("checkpoints", "ac_v9")
SAVE_PATH = os.path.join(SAVE_DIR, "model_sl_v9.pt")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_sl_v9.pt")
BEST_CKPT_PATH = os.path.join(SAVE_DIR, "model_sl_v9_best.pt")

BATCH_SIZE = 1536
EPOCHS = 3
VAL_SPLIT = 0.05
SAVE_INTERVAL = 3000
LEARNING_RATE = 1e-3

# Graceful shutdown
STOP_REQUESTED = False

def signal_handler(sig, frame):
    global STOP_REQUESTED
    if STOP_REQUESTED:
        print("\n[SL V9] Force exit.")
        sys.exit(1)
    STOP_REQUESTED = True
    print("\n[SL V9] Graceful shutdown requested. Will save after current batch...")

signal.signal(signal.SIGINT, signal_handler)


class SLBotDataset(IterableDataset):
    """Streams V9 bot behavioral cloning data from .npz chunks."""

    def __init__(self, data_files):
        self.data_files = data_files
        self.total_samples = 0
        print(f"Scanning {len(data_files)} chunks...")
        for f in tqdm(data_files, desc="Counting"):
            try:
                with np.load(f, mmap_mode='r') as data:
                    self.total_samples += len(data['actions'])
            except Exception as e:
                print(f"Error reading {f}: {e}")
        print(f"Total samples: {self.total_samples:,}")

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        files = list(self.data_files)
        np.random.shuffle(files)

        for f in files:
            if STOP_REQUESTED:
                return
            try:
                data = np.load(f)
                states = data['states']    # (N, 14, 15, 15)
                actions = data['actions']  # (N,)
                masks = data['masks']      # (N, 4)
                values = data['values']    # (N,)

                idx = np.arange(len(states))
                np.random.shuffle(idx)

                for i in idx:
                    if STOP_REQUESTED:
                        return
                    yield (
                        torch.from_numpy(states[i]),
                        torch.tensor(actions[i], dtype=torch.long),
                        torch.from_numpy(masks[i]),
                        torch.tensor(values[i], dtype=torch.float32),
                    )
            except Exception as e:
                print(f"Error yielding from {f}: {e}")
                continue


def train_sl():
    global STOP_REQUESTED
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[SL V9] Device: {device}")

    # Load data
    data_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not data_files:
        print(f"No .npz files found in {DATA_DIR}. Run generate_sl_data_v9.py first.")
        return

    np.random.shuffle(data_files)
    val_count = max(1, int(len(data_files) * VAL_SPLIT))
    val_files = data_files[:val_count]
    train_files = data_files[val_count:]

    print(f"[SL V9] Train files: {len(train_files)}, Val files: {len(val_files)}")

    train_ds = SLBotDataset(train_files)
    val_ds = SLBotDataset(val_files)

    train_size = len(train_ds)
    val_size = len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    # Initialize V9 model
    model = AlphaLudoV9().to(device)
    print(f"[SL V9] Architecture: V9 (14ch, 5res, 80ch, 4TF) — {model.count_parameters():,} params")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion_policy = nn.NLLLoss()
    criterion_value = nn.MSELoss()

    start_epoch = 0
    start_batch = 0
    best_val_loss = float('inf')
    best_eval_wr = 0.0

    # Resume from checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[SL V9] Loading checkpoint from {CHECKPOINT_PATH}...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                start_batch = checkpoint.get('batch', 0)
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                best_eval_wr = checkpoint.get('best_eval_wr', 0.0)
                print(f"[SL V9] Resuming from Epoch {start_epoch+1}, Batch {start_batch}")
            else:
                model.load_state_dict(checkpoint)
                print("[SL V9] Loaded raw state dict, starting fresh epochs.")
        except Exception as e:
            print(f"[SL V9] Cannot load checkpoint: {e}. Starting fresh.")

    def save_checkpoint(epoch, batch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'batch': batch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_eval_wr': best_eval_wr,
        }
        tmp = CHECKPOINT_PATH + '.tmp'
        torch.save(ckpt, tmp)
        os.replace(tmp, CHECKPOINT_PATH)

        if is_best:
            torch.save({'model_state_dict': model.state_dict()}, BEST_CKPT_PATH)
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"\n[SL V9] Starting Behavioral Cloning Training")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Ctrl+C to save and exit\n")

    try:
        for epoch in range(start_epoch, EPOCHS):
            if STOP_REQUESTED:
                break

            model.train()
            train_policy_loss = 0.0
            train_value_loss = 0.0
            train_correct = 0
            processed = 0
            batch_count = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
                        total=train_size // BATCH_SIZE)

            for states, actions, masks, values in train_loader:
                if STOP_REQUESTED:
                    break

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

                # V9 forward (single-step, K=1 context for SL)
                B = states.shape[0]
                grids = states.unsqueeze(1)  # (B, 1, 14, 15, 15)
                prev_acts = torch.full((B, 1), 4, dtype=torch.long, device=device)
                seq_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

                policy, value = model(grids, prev_acts, seq_mask, masks)
                value = value.squeeze(-1)

                # Policy loss (NLL on log-probs)
                loss_p = criterion_policy(torch.log(policy + 1e-8), actions)
                # Value loss
                loss_v = criterion_value(value, values)

                loss = loss_p + 0.5 * loss_v

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                bs = states.size(0)
                train_policy_loss += loss_p.item() * bs
                train_value_loss += loss_v.item() * bs
                processed += bs

                preds = torch.argmax(policy, dim=1)
                train_correct += (preds == actions).sum().item()

                batch_count += 1
                pbar.update(1)
                pbar.set_postfix({
                    'ploss': f'{loss_p.item():.4f}',
                    'vloss': f'{loss_v.item():.4f}',
                    'acc': f'{(preds == actions).float().mean().item():.3f}',
                })

                if batch_count % SAVE_INTERVAL == 0:
                    save_checkpoint(epoch, batch_count)
                    print(f"\n  [Mid-epoch save] Epoch {epoch+1}, Batch {batch_count}")

            start_batch = 0  # Reset for next epoch
            pbar.close()

            if STOP_REQUESTED:
                break

            # Epoch stats
            if processed > 0:
                avg_pl = train_policy_loss / processed
                avg_vl = train_value_loss / processed
                avg_acc = train_correct / processed
                print(f"\n  Epoch {epoch+1} Train — Policy: {avg_pl:.4f}, Value MSE: {avg_vl:.4f}, "
                      f"Action Acc: {avg_acc*100:.1f}%")

            # Validation
            model.eval()
            val_pl = 0.0
            val_vl = 0.0
            val_correct = 0
            val_processed = 0

            with torch.no_grad():
                for states, actions, masks, values in tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"
                ):
                    if STOP_REQUESTED:
                        break

                    states = states.to(device)
                    actions = actions.to(device)
                    masks = masks.to(device)
                    values = values.to(device)

                    B = states.shape[0]
                    grids = states.unsqueeze(1)
                    prev_acts = torch.full((B, 1), 4, dtype=torch.long, device=device)
                    seq_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

                    policy, value = model(grids, prev_acts, seq_mask, masks)
                    value = value.squeeze(-1)

                    loss_p = criterion_policy(torch.log(policy + 1e-8), actions)
                    loss_v = criterion_value(value, values)

                    bs = states.size(0)
                    val_pl += loss_p.item() * bs
                    val_vl += loss_v.item() * bs
                    val_processed += bs

                    preds = torch.argmax(policy, dim=1)
                    val_correct += (preds == actions).sum().item()

            if val_processed > 0:
                avg_val_pl = val_pl / val_processed
                avg_val_vl = val_vl / val_processed
                val_acc = val_correct / val_processed
                val_total_loss = avg_val_pl + 0.5 * avg_val_vl

                print(f"  Epoch {epoch+1} Val — Policy: {avg_val_pl:.4f}, Value MSE: {avg_val_vl:.4f}, "
                      f"Action Acc: {val_acc*100:.1f}%")

                is_best = val_total_loss < best_val_loss
                if is_best:
                    best_val_loss = val_total_loss
                    print(f"  * New best val loss: {val_total_loss:.4f}")

                save_checkpoint(epoch + 1, 0, is_best=is_best)

            # Eval against bots after each epoch
            if not STOP_REQUESTED:
                print(f"\n  Running eval (200 games)...")
                from evaluate_v9 import evaluate_v9_model
                eval_results = evaluate_v9_model(model, device, num_games=200, verbose=False)
                eval_wr = eval_results['win_rate']
                print(f"  Epoch {epoch+1} Eval Win Rate: {eval_results['win_rate_percent']}%")

                if eval_wr > best_eval_wr:
                    best_eval_wr = eval_wr
                    print(f"  * New best eval WR: {eval_results['win_rate_percent']}%")
                    save_checkpoint(epoch + 1, 0, is_best=True)

    except KeyboardInterrupt:
        print("\n[SL V9] Keyboard interrupt.")
    except Exception as e:
        print(f"\n[SL V9] Error: {e}")
        import traceback
        traceback.print_exc()

    # Final save
    print("[SL V9] Final save...")
    save_checkpoint(
        epoch if 'epoch' in dir() else 0,
        batch_count if 'batch_count' in dir() else 0,
    )

    print(f"\n[SL V9] Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best eval WR: {best_eval_wr*100:.1f}%")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Best model: {SAVE_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V9 Behavioral Cloning SL Training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--resume", action='store_true', help="Resume from checkpoint")
    args = parser.parse_args()

    EPOCHS = args.epochs

    if not args.resume and os.path.exists(CHECKPOINT_PATH):
        print(f"[SL V9] Found existing checkpoint. Use --resume to continue, or delete {CHECKPOINT_PATH} to start fresh.")
        response = input("Continue from checkpoint? [y/N] ").strip().lower()
        if response != 'y':
            os.remove(CHECKPOINT_PATH)
            print("[SL V9] Starting fresh.")

    train_sl()
