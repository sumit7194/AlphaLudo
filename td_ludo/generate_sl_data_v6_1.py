"""
Generate SL training data for V6.1 using V6 model as teacher.

Knowledge distillation: V6 (17ch) plays games, V6.1 (24ch) encodes the same
states. V6's soft policy distribution becomes the training label.

Output: NPZ chunks in checkpoints/sl_data_v6_1/ with:
  - states: (N, 24, 15, 15) float32 — V6.1 encoding
  - policies: (N, 4) float32 — V6 teacher's soft policy (after temperature)
  - values: (N,) float32 — V6 teacher's value prediction
  - legal_masks: (N, 4) float32 — legal move masks
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV5
from src.heuristic_bot import ExpertBot, HeuristicLudoBot, AggressiveBot, DefensiveBot

# =========================================================================
# Config
# =========================================================================
TEACHER_PATH = "checkpoints/ac_v6_big/model_latest.pt"
OUTPUT_DIR = "checkpoints/sl_data_v6_1"
CHUNK_SIZE = 10000       # states per NPZ chunk
TARGET_STATES = 500000   # total states to generate
TEACHER_TEMP = 1.0       # temperature for soft labels (1.0 = exact policy)
BATCH_SIZE = 512         # parallel games

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================================
# Load V6 teacher
# =========================================================================
device = torch.device('cpu')  # CPU is fine for data generation
teacher = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=17)
ckpt = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
teacher.load_state_dict(ckpt['model_state_dict'])
teacher.eval()
print(f"[SL Data] Loaded V6 teacher from {TEACHER_PATH}")
print(f"[SL Data] Teacher: {teacher.count_parameters():,} params")
print(f"[SL Data] Target: {TARGET_STATES:,} states in {OUTPUT_DIR}/")

# =========================================================================
# Bot opponents (teacher plays against these)
# =========================================================================
bots = {
    'Expert': ExpertBot(player_id=2),
    'Heuristic': HeuristicLudoBot(player_id=2),
    'Aggressive': AggressiveBot(player_id=2),
    'Defensive': DefensiveBot(player_id=2),
}
bot_names = list(bots.keys())
bot_weights = [0.40, 0.25, 0.20, 0.15]  # sampling weights

# =========================================================================
# Data collection
# =========================================================================
all_states = []      # (24, 15, 15)
all_policies = []    # (4,) soft labels
all_values = []      # scalar
all_masks = []       # (4,)

chunk_idx = 0
total_states = 0
total_games = 0
t0 = time.time()

while total_states < TARGET_STATES:
    # Pick opponent
    bot_name = random.choices(bot_names, weights=bot_weights, k=1)[0]
    bot = bots[bot_name]

    # Random seat assignment
    model_player = random.choice([0, 2])
    bot_player = 2 if model_player == 0 else 0
    bot.player_id = bot_player

    state = ludo_cpp.create_initial_state_2p()
    consec = [0] * 4

    for step in range(500):
        if state.is_terminal:
            break
        cp = state.current_player

        # Roll dice
        if state.current_dice_roll == 0:
            roll = random.randint(1, 6)
            state.current_dice_roll = roll
            if roll == 6:
                consec[cp] += 1
            else:
                consec[cp] = 0
            if consec[cp] >= 3:
                nxt = (cp + 1) % 4
                while not state.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                state.current_player = nxt
                state.current_dice_roll = 0
                consec[cp] = 0
                continue

        lmoves = ludo_cpp.get_legal_moves(state)
        if not lmoves:
            nxt = (cp + 1) % 4
            while not state.active_players[nxt]:
                nxt = (nxt + 1) % 4
            state.current_player = nxt
            state.current_dice_roll = 0
            continue

        if cp == model_player:
            # Teacher inference (17ch)
            enc17 = ludo_cpp.encode_state(state)
            mask = np.zeros(4, dtype=np.float32)
            for m in lmoves:
                mask[m] = 1.0

            with torch.no_grad():
                st = torch.from_numpy(np.array(enc17, dtype=np.float32)).unsqueeze(0)
                mt = torch.from_numpy(mask).unsqueeze(0)
                policy, value = teacher(st, mt)

                # Soft labels with temperature
                if TEACHER_TEMP != 1.0:
                    logits = torch.log(policy + 1e-8) / TEACHER_TEMP
                    soft_policy = F.softmax(logits, dim=1)
                else:
                    soft_policy = policy

            # V6.1 encoding (24ch) for the same state
            enc24 = ludo_cpp.encode_state_v6(state)

            all_states.append(np.array(enc24, dtype=np.float32))
            all_policies.append(soft_policy.squeeze(0).numpy())
            all_values.append(value.item())
            all_masks.append(mask)
            total_states += 1

            # Teacher picks action (greedy for consistent play)
            action = policy.argmax(dim=1).item()
            if action not in lmoves:
                action = random.choice(lmoves)
        else:
            # Bot plays
            action = bot.select_move(state, lmoves)

        state = ludo_cpp.apply_move(state, action)

    total_games += 1

    # Save chunk
    if len(all_states) >= CHUNK_SIZE:
        chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            chunk_path,
            states=np.stack(all_states[:CHUNK_SIZE]),
            policies=np.stack(all_policies[:CHUNK_SIZE]),
            values=np.array(all_values[:CHUNK_SIZE], dtype=np.float32),
            legal_masks=np.stack(all_masks[:CHUNK_SIZE]),
        )
        print(f"  Chunk {chunk_idx}: {CHUNK_SIZE} states saved ({total_states:,}/{TARGET_STATES:,}, "
              f"{total_games} games, {time.time()-t0:.0f}s)", flush=True)
        all_states = all_states[CHUNK_SIZE:]
        all_policies = all_policies[CHUNK_SIZE:]
        all_values = all_values[CHUNK_SIZE:]
        all_masks = all_masks[CHUNK_SIZE:]
        chunk_idx += 1

# Save remainder
if all_states:
    chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        chunk_path,
        states=np.stack(all_states),
        policies=np.stack(all_policies),
        values=np.array(all_values, dtype=np.float32),
        legal_masks=np.stack(all_masks),
    )
    chunk_idx += 1

elapsed = time.time() - t0
print(f"\n[SL Data] Done: {total_states:,} states from {total_games:,} games "
      f"in {chunk_idx} chunks ({elapsed/60:.1f} min)")
