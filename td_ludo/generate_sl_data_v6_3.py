"""
Generate SL training data for V6.3 using V6.1 as teacher.

Pure V6.1 self-play (no bots). V6.1 plays against itself with 24ch encoding.
For each decision point we save:
  - state_27ch: V6.3's encoding of the same state (target architecture)
  - policy:     V6.1's full softmax policy distribution (soft distillation label)
  - value:      final game outcome from this decision-point's perspective (+1 / -1)
  - legal_mask: legal move mask

Output: NPZ chunks in checkpoints/sl_data_v6_3/

Design notes:
  - Batched: 512 parallel games on GPU for speed (~200 GPM expected)
  - Temperature 1.0 for natural V6.1 behavior (matches its RL training temp late-stage)
  - We save actual game outcome for value, NOT V6.1's value prediction — more accurate label
  - consecutive_sixes tracked per-player and passed to v6_3 encoder
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

# =========================================================================
# Config
# =========================================================================
TEACHER_PATH = "checkpoints/ac_v6_1_strategic/model_best.pt"
OUTPUT_DIR = "checkpoints/sl_data_v6_3"
CHUNK_SIZE = 10000       # states per NPZ chunk
TARGET_STATES = 1_000_000  # ~10K games
BATCH_SIZE = 512         # parallel games (GPU batched inference)
TEMPERATURE = 1.0        # teacher action sampling temperature

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================================
# Load V6.1 teacher (24 channels)
# =========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
ckpt = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
teacher.load_state_dict(ckpt['model_state_dict'])
teacher.to(device).eval()
print(f"[SL Data] Loaded V6.1 teacher from {TEACHER_PATH}")
print(f"[SL Data] Teacher: {teacher.count_parameters():,} params on {device}")
print(f"[SL Data] Target: {TARGET_STATES:,} states in {OUTPUT_DIR}/")
print(f"[SL Data] Batch size: {BATCH_SIZE} parallel games, temp={TEMPERATURE}")


# =========================================================================
# Batched game environment (raw C++ states, no wrapper)
# =========================================================================
class BatchedSelfPlay:
    """Runs N parallel games of V6.1 self-play, collects (state, policy, outcome)."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        # consecutive_sixes[game_idx][player_id]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        # pending trajectories per game: list of (state_27ch, policy, legal_mask, player)
        # we backfill "value" at game end using the game outcome
        self.pending = [[] for _ in range(batch_size)]
        # step counter for timeout
        self.step_count = np.zeros(batch_size, dtype=np.int32)
        self.MAX_STEPS = 500

    def play_step(self):
        """Advance all games by one decision. Returns list of completed trajectories (post-game)."""
        # Build batch of states that need model decisions
        decision_idxs = []   # game indices that need model action
        states_24ch = []     # for teacher inference
        states_27ch = []     # for saving
        legal_masks = []
        legal_moves_list = []

        for i, game in enumerate(self.games):
            if game.is_terminal or self.step_count[i] >= self.MAX_STEPS:
                continue

            cp = game.current_player

            # Roll dice if needed
            if game.current_dice_roll == 0:
                roll = random.randint(1, 6)
                game.current_dice_roll = roll
                if roll == 6:
                    self.consec_sixes[i, cp] += 1
                else:
                    self.consec_sixes[i, cp] = 0
                # Triple-6 penalty
                if self.consec_sixes[i, cp] >= 3:
                    nxt = (cp + 1) % 4
                    while not game.active_players[nxt]:
                        nxt = (nxt + 1) % 4
                    game.current_player = nxt
                    game.current_dice_roll = 0
                    self.consec_sixes[i, cp] = 0
                    continue

            lmoves = ludo_cpp.get_legal_moves(game)
            if not lmoves:
                # No legal moves: advance to next player
                nxt = (cp + 1) % 4
                while not game.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                game.current_player = nxt
                game.current_dice_roll = 0
                continue

            # Need a model decision
            mask = np.zeros(4, dtype=np.float32)
            for m in lmoves:
                mask[m] = 1.0
            enc24 = np.array(ludo_cpp.encode_state_v6(game), dtype=np.float32)
            enc27 = np.array(
                ludo_cpp.encode_state_v6_3(game, int(self.consec_sixes[i, cp])),
                dtype=np.float32,
            )

            decision_idxs.append(i)
            states_24ch.append(enc24)
            states_27ch.append(enc27)
            legal_masks.append(mask)
            legal_moves_list.append(lmoves)

        # If no decisions needed (all games terminal), return empty
        if not decision_idxs:
            return []

        # Batched teacher inference
        batch_states = torch.from_numpy(np.stack(states_24ch)).to(device)
        batch_masks = torch.from_numpy(np.stack(legal_masks)).to(device)

        with torch.no_grad():
            policies, _values = teacher(batch_states, batch_masks)
            # Sample action at temperature 1.0 (natural behavior)
            if TEMPERATURE != 1.0:
                logits = torch.log(policies + 1e-8) / TEMPERATURE
                sample_probs = F.softmax(logits, dim=1)
            else:
                sample_probs = policies
            actions = torch.multinomial(sample_probs, num_samples=1).squeeze(1)
            actions_np = actions.cpu().numpy()
            policies_np = policies.cpu().numpy()

        # Apply actions and record trajectories
        completed = []
        for j, i in enumerate(decision_idxs):
            game = self.games[i]
            cp = game.current_player
            action = int(actions_np[j])
            # If sampled action is illegal (shouldn't happen due to mask), fallback
            if legal_masks[j][action] == 0:
                action = legal_moves_list[j][0]

            # Record this decision point
            self.pending[i].append({
                'state_27ch': states_27ch[j],
                'policy': policies_np[j].copy(),
                'legal_mask': legal_masks[j].copy(),
                'player': cp,
            })

            # Apply move
            self.games[i] = ludo_cpp.apply_move(game, action)
            self.step_count[i] += 1

            # Check game end
            if self.games[i].is_terminal or self.step_count[i] >= self.MAX_STEPS:
                winner = -1
                if self.games[i].is_terminal:
                    # scored 4 wins
                    for p in range(4):
                        if self.games[i].scores[p] >= 4:
                            winner = p
                            break

                # Backfill values for this game's trajectory
                traj = self.pending[i]
                if winner >= 0 and traj:
                    # Outcome from each decision-maker's perspective
                    for step in traj:
                        step['value'] = 1.0 if step['player'] == winner else -1.0
                    completed.append(traj)
                # else: timeout, discard

                # Reset game
                self.games[i] = ludo_cpp.create_initial_state_2p()
                self.consec_sixes[i] = 0
                self.pending[i] = []
                self.step_count[i] = 0

        return completed


# =========================================================================
# Collection loop
# =========================================================================
env = BatchedSelfPlay(BATCH_SIZE)

all_states = []
all_policies = []
all_values = []
all_masks = []

chunk_idx = 0
total_states = 0
total_games = 0
t0 = time.time()
last_report = t0

print(f"[SL Data] Starting collection...", flush=True)

while total_states < TARGET_STATES:
    completed_trajs = env.play_step()

    for traj in completed_trajs:
        total_games += 1
        for step in traj:
            all_states.append(step['state_27ch'])
            all_policies.append(step['policy'])
            all_values.append(step['value'])
            all_masks.append(step['legal_mask'])
            total_states += 1

    # Save chunk
    while len(all_states) >= CHUNK_SIZE:
        chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            chunk_path,
            states=np.stack(all_states[:CHUNK_SIZE]).astype(np.float32),
            policies=np.stack(all_policies[:CHUNK_SIZE]).astype(np.float32),
            values=np.array(all_values[:CHUNK_SIZE], dtype=np.float32),
            legal_masks=np.stack(all_masks[:CHUNK_SIZE]).astype(np.float32),
        )
        all_states = all_states[CHUNK_SIZE:]
        all_policies = all_policies[CHUNK_SIZE:]
        all_values = all_values[CHUNK_SIZE:]
        all_masks = all_masks[CHUNK_SIZE:]
        chunk_idx += 1

    # Progress report every 20s
    now = time.time()
    if now - last_report > 20:
        elapsed = now - t0
        rate = total_states / elapsed if elapsed > 0 else 0
        gpm = total_games / (elapsed / 60) if elapsed > 0 else 0
        eta_s = (TARGET_STATES - total_states) / rate if rate > 0 else 0
        print(
            f"  [{elapsed:.0f}s] {total_states:,}/{TARGET_STATES:,} states | "
            f"{total_games:,} games | {rate:.0f} states/s | {gpm:.0f} GPM | ETA {eta_s:.0f}s",
            flush=True,
        )
        last_report = now

# Save remainder
if all_states:
    chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        chunk_path,
        states=np.stack(all_states).astype(np.float32),
        policies=np.stack(all_policies).astype(np.float32),
        values=np.array(all_values, dtype=np.float32),
        legal_masks=np.stack(all_masks).astype(np.float32),
    )
    chunk_idx += 1

elapsed = time.time() - t0
print(
    f"[SL Data] Done: {total_states:,} states, {total_games:,} games, "
    f"{chunk_idx} chunks, {elapsed:.0f}s",
    flush=True,
)
