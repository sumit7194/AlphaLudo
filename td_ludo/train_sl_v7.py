"""
TD-Ludo V7 — Supervised Learning (Behavioral Cloning) for Sequence Transformer

Trains the AlphaLudoV7 transformer model on bot-vs-bot game data to bootstrap
its policy and value networks before self-play RL.

Key design:
- Loads per-step 1D data and reconstructs K=16 context windows on-the-fly
- Groups steps by game_id to maintain temporal ordering within each game
- Uses CrossEntropy for policy and MSE for value

Losses:
- Policy: CrossEntropy (predict what the bot did)
- Value: MSE (predict whether the bot won)
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_v7 import AlphaLudoV7
from src.state_encoder_1d import NUM_ACTION_CLASSES

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = os.path.join("checkpoints", "sl_data_v7")
V7_CKPT_DIR = os.path.join("checkpoints", "ac_v7_transformer")
SAVE_PATH = os.path.join(V7_CKPT_DIR, "model_sl.pt")
CHECKPOINT_PATH = os.path.join(V7_CKPT_DIR, "checkpoint_sl.pt")

# =============================================================================
# Hyperparameters
# =============================================================================
CONTEXT_LENGTH = 16
BATCH_SIZE = 512
EPOCHS = 3
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.05
SAVE_INTERVAL = 2000  # Save every N batches
VALUE_LOSS_WEIGHT = 0.5


class V7SLDataset(Dataset):
    """
    Loads V7 1D state data and reconstructs context windows.

    Each sample returns:
    - token_positions: (K, 8) int64
    - continuous: (K, 9) float32
    - actions_seq: (K,) int64 — previous actions in context
    - seq_mask: (K,) bool — True = padding
    - target_action: int
    - legal_mask: (4,) float32
    - value: float
    """

    def __init__(self, data_files, context_length=16):
        self.K = context_length

        print(f"Loading {len(data_files)} data files...")

        # Load all data into memory
        all_tok = []
        all_cont = []
        all_actions = []
        all_masks = []
        all_values = []
        all_game_ids = []
        all_step_ids = []

        for f in tqdm(data_files, desc="Loading"):
            try:
                data = np.load(f)
                all_tok.append(data['token_positions'])
                all_cont.append(data['continuous'])
                all_actions.append(data['actions'])
                all_masks.append(data['masks'])
                all_values.append(data['values'])
                all_game_ids.append(data['game_ids'])
                all_step_ids.append(data['step_ids'])
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue

        self.token_positions = np.concatenate(all_tok)
        self.continuous = np.concatenate(all_cont)
        self.actions = np.concatenate(all_actions)
        self.masks = np.concatenate(all_masks)
        self.values = np.concatenate(all_values)
        self.game_ids = np.concatenate(all_game_ids)
        self.step_ids = np.concatenate(all_step_ids)

        # Build game index: game_id -> sorted list of global indices
        print("Building game index for context window reconstruction...")
        self.game_index = {}
        for idx in tqdm(range(len(self.game_ids)), desc="Indexing", mininterval=2.0):
            gid = int(self.game_ids[idx])
            if gid not in self.game_index:
                self.game_index[gid] = []
            self.game_index[gid].append(idx)

        # Sort each game's steps by step_id
        for gid in self.game_index:
            self.game_index[gid].sort(key=lambda i: self.step_ids[i])

        # Build a flat lookup: global_idx -> (game_id, position_in_game)
        self.idx_to_game_pos = np.zeros((len(self.game_ids), 2), dtype=np.int64)
        for gid, indices in self.game_index.items():
            for pos, global_idx in enumerate(indices):
                self.idx_to_game_pos[global_idx] = [gid, pos]

        print(f"Total samples: {len(self.actions):,} | Games: {len(self.game_index):,}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        gid, pos = self.idx_to_game_pos[idx]
        game_indices = self.game_index[int(gid)]

        # Get the context window: up to K steps ending at current position (inclusive)
        start = max(0, pos - self.K + 1)
        context_indices = game_indices[start:pos + 1]
        n_valid = len(context_indices)
        n_pad = self.K - n_valid

        # Build sequence tensors
        tok = np.zeros((self.K, 8), dtype=np.int64)
        cont = np.zeros((self.K, 9), dtype=np.float32)
        acts_seq = np.full(self.K, NUM_ACTION_CLASSES - 1, dtype=np.int64)  # 4 = pass/none
        seq_mask = np.ones(self.K, dtype=bool)  # True = padding

        # Fill valid turns (padding at start, valid at end)
        for j, ci in enumerate(context_indices):
            out_idx = n_pad + j
            tok[out_idx] = self.token_positions[ci]
            cont[out_idx] = self.continuous[ci]
            seq_mask[out_idx] = False

            # Action from previous step (for context)
            if j > 0:
                prev_ci = context_indices[j - 1]
                acts_seq[out_idx] = self.actions[prev_ci]
            # First turn in context: action = 4 (none)

        # Target
        target_action = self.actions[idx]
        legal_mask = self.masks[idx]
        value = self.values[idx]

        return (
            torch.from_numpy(tok),
            torch.from_numpy(cont),
            torch.from_numpy(acts_seq),
            torch.from_numpy(seq_mask),
            torch.tensor(target_action, dtype=torch.long),
            torch.from_numpy(legal_mask),
            torch.tensor(value, dtype=torch.float32),
        )


def train_sl(args):
    os.makedirs(V7_CKPT_DIR, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not data_files:
        print(f"No .npz files found in {DATA_DIR}. Run generate_sl_data_v7.py first.")
        return

    # Split train/val at file level
    np.random.shuffle(data_files)
    val_count = max(1, int(len(data_files) * VAL_SPLIT))
    val_files = data_files[:val_count]
    train_files = data_files[val_count:]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    train_ds = V7SLDataset(train_files, context_length=CONTEXT_LENGTH)
    val_ds = V7SLDataset(val_files, context_length=CONTEXT_LENGTH)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # Initialize model
    model = AlphaLudoV7(
        context_length=CONTEXT_LENGTH, embed_dim=128,
        num_heads=4, num_layers=4
    ).to(device)
    print(f"[SL] Architecture: AlphaLudoV7 Transformer ({model.count_parameters():,} params)")

    # Load starting weights (current PPO model)
    start_from = args.init_weights
    if start_from and os.path.exists(start_from):
        print(f"[SL] Loading initial weights from: {start_from}")
        try:
            checkpoint = torch.load(start_from, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            # Clean any compiled model prefixes
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print("[SL] Loaded initial weights successfully.")
        except Exception as e:
            print(f"[SL] Warning: Could not load initial weights: {e}")
            print("[SL] Starting from random weights.")
    else:
        print("[SL] No initial weights specified. Starting from random.")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # LR scheduler: cosine annealing
    total_steps = EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    best_val_acc = 0.0
    start_epoch = 0

    # Resume from SL checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH) and not args.fresh:
        print(f"[SL] Attempting to resume from {CHECKPOINT_PATH}...")
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            best_val_acc = ckpt.get('val_acc', 0.0)
            print(f"[SL] Resumed from epoch {start_epoch + 1}, best acc: {best_val_acc*100:.1f}%")
        except Exception as e:
            print(f"[SL] Could not resume: {e}. Starting fresh.")

    print(f"\n{'='*60}")
    print(f"  V7 Supervised Learning Training")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"  Context: K={CONTEXT_LENGTH} | Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"{'='*60}\n")

    try:
        for epoch in range(start_epoch, EPOCHS):
            # ---- Training ----
            model.train()
            train_loss = 0.0
            train_policy_loss = 0.0
            train_value_loss = 0.0
            train_correct = 0
            train_total = 0
            batch_count = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            for tok, cont, acts_seq, seq_mask, target_action, legal_mask, value in pbar:
                tok = tok.to(device)
                cont = cont.to(device)
                acts_seq = acts_seq.to(device)
                seq_mask = seq_mask.to(device)
                target_action = target_action.to(device)
                legal_mask = legal_mask.to(device)
                value = value.to(device)

                optimizer.zero_grad()

                # Forward: model returns (policy_probs, value)
                policy_probs, value_preds = model(tok, cont, acts_seq, seq_mask, legal_mask)
                value_preds = value_preds.squeeze(-1)

                # Policy loss: CrossEntropy needs logits, but model returns probs
                # Use NLLLoss on log-probs instead
                log_probs = torch.log(policy_probs + 1e-8)
                loss_p = nn.functional.nll_loss(log_probs, target_action)

                # Value loss
                loss_v = criterion_value(value_preds, value)

                loss = loss_p + VALUE_LOSS_WEIGHT * loss_v

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Stats
                train_loss += loss.item() * tok.size(0)
                train_policy_loss += loss_p.item() * tok.size(0)
                train_value_loss += loss_v.item() * tok.size(0)

                preds = torch.argmax(policy_probs, dim=1)
                train_correct += (preds == target_action).sum().item()
                train_total += tok.size(0)

                batch_count += 1
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(preds == target_action).float().mean().item()*100:.1f}%',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                })

                # Periodic save
                if batch_count % SAVE_INTERVAL == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': best_val_acc,
                    }, CHECKPOINT_PATH)

            # Epoch stats
            avg_train_loss = train_loss / max(1, train_total)
            avg_train_acc = train_correct / max(1, train_total)

            # ---- Validation ----
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for tok, cont, acts_seq, seq_mask, target_action, legal_mask, value in tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"
                ):
                    tok = tok.to(device)
                    cont = cont.to(device)
                    acts_seq = acts_seq.to(device)
                    seq_mask = seq_mask.to(device)
                    target_action = target_action.to(device)
                    legal_mask = legal_mask.to(device)
                    value = value.to(device)

                    policy_probs, value_preds = model(tok, cont, acts_seq, seq_mask, legal_mask)
                    value_preds = value_preds.squeeze(-1)

                    log_probs = torch.log(policy_probs + 1e-8)
                    loss_p = nn.functional.nll_loss(log_probs, target_action)
                    loss_v = criterion_value(value_preds, value)
                    loss = loss_p + VALUE_LOSS_WEIGHT * loss_v

                    val_loss += loss.item() * tok.size(0)
                    preds = torch.argmax(policy_probs, dim=1)
                    val_correct += (preds == target_action).sum().item()
                    val_total += tok.size(0)

            avg_val_loss = val_loss / max(1, val_total)
            val_acc = val_correct / max(1, val_total)

            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.1f}%")
            print(f"  Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_acc*100:.1f}%")

            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), SAVE_PATH)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, CHECKPOINT_PATH)
                print(f"  ★ New Best Val Acc: {val_acc*100:.2f}% — Saved to {SAVE_PATH}")
            else:
                # Still save checkpoint for resumability
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                }, CHECKPOINT_PATH)

    except KeyboardInterrupt:
        print("\n[SL] Interrupted! Saving checkpoint...")
        torch.save({
            'epoch': epoch if 'epoch' in dir() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': best_val_acc,
        }, CHECKPOINT_PATH)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"[SL] Saved to {CHECKPOINT_PATH} and {SAVE_PATH}")

    print(f"\n{'='*60}")
    print(f"  SL Training Complete!")
    print(f"  Best Val Accuracy: {best_val_acc*100:.1f}%")
    print(f"  Model saved to: {SAVE_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V7 Supervised Learning Training")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Path to initial model weights (e.g. current PPO checkpoint)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Learning rate (default: {LR})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--fresh", action='store_true',
                        help="Ignore existing SL checkpoint and start fresh")
    args = parser.parse_args()

    # Override globals with args
    EPOCHS = args.epochs
    LR = args.lr
    BATCH_SIZE = args.batch_size

    train_sl(args)
