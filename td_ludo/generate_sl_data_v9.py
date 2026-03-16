"""
TD-Ludo V9 — SL Data Generator (Knowledge Distillation from V6)

Plays games using V6 model as both players. For each decision state, records:
  - state_v9: (14, 15, 15) float32 — V9's 14-channel encoding
  - teacher_policy: (4,) float32 — V6's softmax policy (soft targets)
  - teacher_value: float32 — V6's value estimate
  - legal_mask: (4,) float32
  - action: int — V6's sampled action (for BC loss backup)

V6 uses the OLD 17-channel encoder internally; we encode the same game state
with BOTH encoders so V9 learns from V6's knowledge via the new encoding.
"""

import os
import sys
import signal
import argparse
import time
import random
import glob as glob_module
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Graceful shutdown
STOP_REQUESTED = False

def signal_handler(sig, frame):
    global STOP_REQUESTED
    if STOP_REQUESTED:
        print("\n[V9 SL Gen] Force exit.")
        sys.exit(1)
    STOP_REQUESTED = True
    print("\n[V9 SL Gen] Graceful shutdown. Will save current buffer and exit...")

signal.signal(signal.SIGINT, signal_handler)

import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV5
from src.config import MAX_MOVES_PER_GAME

DATA_DIR = os.path.join("checkpoints", "sl_data_v9")
os.makedirs(DATA_DIR, exist_ok=True)

BATCH_SIZE = 100
CHUNK_SIZE = 100_000

# V6 model path
V6_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'checkpoints', 'ac_v6_big', 'backups', 'model_final_v6_382k_70pct.pt'
)


def load_v6_model(device):
    """Load the V6 teacher model."""
    model = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=17)
    checkpoint = torch.load(V6_WEIGHTS_PATH, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[V9 SL] Loaded V6 teacher model ({model.count_parameters():,} params)")
    return model


def generate_data(num_games, device):
    teacher = load_v6_model(device)

    env = ludo_cpp.VectorGameState(BATCH_SIZE, True)  # 2P mode

    # Per-game trajectory: game_idx -> player_id -> list of (state_v9, teacher_policy, teacher_value, legal_mask, action)
    trajectories = {i: {0: [], 2: []} for i in range(BATCH_SIZE)}

    consecutive_sixes = np.zeros((BATCH_SIZE, 4), dtype=int)
    move_counts = np.zeros(BATCH_SIZE, dtype=int)

    # Auto-detect existing chunks
    existing = glob_module.glob(os.path.join(DATA_DIR, "chunk_*.npz"))
    if existing:
        max_idx = max(int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing)
        chunk_idx = max_idx + 1
        print(f"Found {len(existing)} existing chunks. Continuing from chunk_{chunk_idx:04d}.")
    else:
        chunk_idx = 0

    buffer_states = []
    buffer_teacher_policies = []
    buffer_teacher_values = []
    buffer_actions = []
    buffer_masks = []

    games_completed = 0
    pbar = tqdm(total=num_games, desc="Generating V9 SL Data (V6 Teacher)")

    def save_chunk():
        nonlocal buffer_states, buffer_teacher_policies, buffer_teacher_values
        nonlocal buffer_actions, buffer_masks, chunk_idx
        if not buffer_states:
            return

        chunk_path = os.path.join(DATA_DIR, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            chunk_path,
            states=np.stack(buffer_states).astype(np.float32),
            teacher_policies=np.stack(buffer_teacher_policies).astype(np.float32),
            teacher_values=np.array(buffer_teacher_values, dtype=np.float32),
            actions=np.array(buffer_actions, dtype=np.int64),
            masks=np.stack(buffer_masks).astype(np.float32),
        )
        print(f"\nSaved {chunk_path} with {len(buffer_states)} samples.")

        chunk_idx += 1
        buffer_states = []
        buffer_teacher_policies = []
        buffer_teacher_values = []
        buffer_actions = []
        buffer_masks = []

    # Batch inference buffers
    pending_decisions = []  # list of (game_idx, cp, legal_moves, state_v17, state_v9, legal_mask)

    while games_completed < num_games and not STOP_REQUESTED:
        pending_decisions.clear()
        actions = [-1] * BATCH_SIZE

        for i in range(BATCH_SIZE):
            game = env.get_game(i)

            if game.is_terminal:
                continue

            cp = game.current_player

            if move_counts[i] >= MAX_MOVES_PER_GAME:
                game.is_terminal = True
                continue

            if game.current_dice_roll == 0:
                roll = random.randint(1, 6)
                game.current_dice_roll = roll
                if roll == 6:
                    consecutive_sixes[i, cp] += 1
                else:
                    consecutive_sixes[i, cp] = 0

                if consecutive_sixes[i, cp] >= 3:
                    next_p = (cp + 1) % 4
                    while not game.active_players[next_p]:
                        next_p = (next_p + 1) % 4
                    game.current_player = next_p
                    game.current_dice_roll = 0
                    consecutive_sixes[i, cp] = 0
                    continue

            legal_moves = ludo_cpp.get_legal_moves(game)
            if not legal_moves:
                next_p = (cp + 1) % 4
                while not game.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                game.current_player = next_p
                game.current_dice_roll = 0
                continue

            # Encode state with both encoders
            state_v17 = ludo_cpp.encode_state(game)    # (17, 15, 15) for V6
            state_v9 = ludo_cpp.encode_state_v9(game)  # (14, 15, 15) for V9

            legal_mask = np.zeros(4, dtype=np.float32)
            for m in legal_moves:
                legal_mask[m] = 1.0

            pending_decisions.append((i, cp, legal_moves, state_v17, state_v9, legal_mask))

        # Batched V6 teacher inference
        if pending_decisions:
            batch_v17 = np.stack([d[3] for d in pending_decisions])
            batch_masks = np.stack([d[5] for d in pending_decisions])

            with torch.no_grad():
                t_states = torch.from_numpy(batch_v17).to(device, dtype=torch.float32)
                t_masks = torch.from_numpy(batch_masks).to(device, dtype=torch.float32)

                policy_probs, values = teacher(t_states, t_masks)

                policy_np = policy_probs.cpu().numpy()   # (B, 4) softmax probs
                values_np = values.squeeze(-1).cpu().numpy()  # (B,)

            for j, (game_idx, cp, legal_moves, state_v17, state_v9, legal_mask) in enumerate(pending_decisions):
                # Sample action from V6's policy
                probs = policy_np[j]
                probs = probs * legal_mask
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs = probs / prob_sum
                else:
                    probs = legal_mask / legal_mask.sum()

                action = np.random.choice(4, p=probs)
                if action not in legal_moves:
                    action = random.choice(legal_moves)

                actions[game_idx] = action

                # Record trajectory
                trajectories[game_idx][cp].append((
                    state_v9,           # V9 encoding
                    policy_np[j],       # V6 soft targets (full distribution)
                    float(values_np[j]),  # V6 value estimate
                    legal_mask,
                    action,
                ))

        # Step env
        env.step(actions)

        # Check completions
        for i in range(BATCH_SIZE):
            game = env.get_game(i)
            if game.is_terminal:
                move_counts[i] += 1

                winner = ludo_cpp.get_winner(game)

                if winner != -1:
                    for pid in (0, 2):
                        traj = trajectories[i][pid]
                        for step in traj:
                            buffer_states.append(step[0])
                            buffer_teacher_policies.append(step[1])
                            buffer_teacher_values.append(step[2])
                            buffer_masks.append(step[3])
                            buffer_actions.append(step[4])

                games_completed += 1
                pbar.update(1)

                if len(buffer_states) >= CHUNK_SIZE:
                    save_chunk()

                # Reset game
                env.reset_game(i)
                move_counts[i] = 0
                consecutive_sixes[i] = 0
                trajectories[i] = {0: [], 2: []}
            else:
                if actions[i] >= 0:
                    move_counts[i] += 1

    pbar.close()

    # Save remaining
    if buffer_states:
        save_chunk()

    total_samples = chunk_idx * CHUNK_SIZE + len(buffer_states)
    print(f"\nDone! Generated {games_completed} games.")
    print(f"Total samples recorded: ~{total_samples:,}")
    print(f"Data saved in: {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate V9 SL Data (V6 Teacher Distillation)")
    parser.add_argument("--games", type=int, default=300000, help="Number of games")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/mps/cuda)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"[V9 SL] Device: {device}")
    print(f"[V9 SL] V6 teacher weights: {V6_WEIGHTS_PATH}")
    print(f"[V9 SL] Target games: {args.games:,}")

    generate_data(args.games, device)
