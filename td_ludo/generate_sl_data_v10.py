"""
Generate joint SL training data for V10.

For each V6.1-self-play decision state, record:
  - state (28ch V10 encoding)       — input features
  - policy (4,)                     — V6.1 teacher's softmax output  (→ policy head target)
  - won (0/1)                       — game outcome from this player  (→ win_prob target)
  - moves_remaining                 — own-turns from this state to game end  (→ moves head target)
  - legal_mask (4,)                 — which tokens were legal

Trains all 3 heads jointly from scratch.
"""

import os, sys, time, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV5  # V6.1 teacher

TEACHER_PATH = "checkpoints/ac_v6_1_strategic/model_best.pt"
OUTPUT_DIR = "checkpoints/sl_data_v10"
CHUNK_SIZE = 10000
TARGET_STATES = int(os.environ.get('V10_TARGET_STATES', 150_000))
BATCH_SIZE = int(os.environ.get('V10_DATA_BATCH', 512))
TEMPERATURE = 1.0
MAX_TURNS = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prefer MPS (Apple Silicon) > CUDA > CPU. MPS ~20x faster than CPU for this model.
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
teacher = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
ckpt = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
teacher.load_state_dict(ckpt['model_state_dict'])
teacher.to(device).eval()
print(f"[V10 Data] Teacher: {TEACHER_PATH} on {device}")
print(f"[V10 Data] Target: {TARGET_STATES:,} states in {OUTPUT_DIR}")


class BatchedSelfPlay:
    """V6.1 self-play with V10-encoded state collection + moves-remaining tracking."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        self.model_players = [random.choice([0, 2]) for _ in range(batch_size)]
        self.trajectories = [[] for _ in range(batch_size)]
        self.own_move_counts = [0 for _ in range(batch_size)]
        self.step_count = np.zeros(batch_size, dtype=np.int32)

    def _reset(self, i):
        self.games[i] = ludo_cpp.create_initial_state_2p()
        self.consec_sixes[i] = 0
        self.model_players[i] = random.choice([0, 2])
        self.trajectories[i] = []
        self.own_move_counts[i] = 0
        self.step_count[i] = 0

    def play_step(self):
        """One decision across all active games. Returns completed game samples."""
        completed = []

        # Advance dice / handle no-ops / enqueue model decisions
        decision_idxs = []
        batch24_states = []   # V6.1 24ch for teacher inference
        batch28_states = []   # V10 28ch for student training
        batch_masks = []
        batch_legal = []

        for i, game in enumerate(self.games):
            if game.is_terminal or self.step_count[i] >= MAX_TURNS:
                continue

            cp = game.current_player
            mp = self.model_players[i]

            if game.current_dice_roll == 0:
                roll = random.randint(1, 6)
                game.current_dice_roll = roll
                if roll == 6:
                    self.consec_sixes[i, cp] += 1
                else:
                    self.consec_sixes[i, cp] = 0
                if self.consec_sixes[i, cp] >= 3:
                    nxt = (cp + 1) % 4
                    while not game.active_players[nxt]:
                        nxt = (nxt + 1) % 4
                    game.current_player = nxt
                    game.current_dice_roll = 0
                    self.consec_sixes[i, cp] = 0
                    continue

            legal = ludo_cpp.get_legal_moves(game)
            if not legal:
                nxt = (cp + 1) % 4
                while not game.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                game.current_player = nxt
                game.current_dice_roll = 0
                continue

            mask = np.zeros(4, dtype=np.float32)
            for m in legal:
                mask[m] = 1.0
            enc24 = np.array(ludo_cpp.encode_state_v6(game), dtype=np.float32)
            enc28 = np.array(ludo_cpp.encode_state_v10(game), dtype=np.float32)

            decision_idxs.append(i)
            batch24_states.append(enc24)
            batch28_states.append(enc28)
            batch_masks.append(mask)
            batch_legal.append(legal)

        if not decision_idxs:
            return completed

        # Batched teacher inference
        st24 = torch.from_numpy(np.stack(batch24_states)).to(device)
        st_mask = torch.from_numpy(np.stack(batch_masks)).to(device)
        with torch.no_grad():
            teacher_policy, _ = teacher(st24, st_mask)
            if TEMPERATURE != 1.0:
                logits = torch.log(teacher_policy + 1e-8) / TEMPERATURE
                sample_probs = torch.softmax(logits, dim=1)
            else:
                sample_probs = teacher_policy
            actions = torch.multinomial(sample_probs, num_samples=1).squeeze(1).cpu().numpy()
            policies_np = teacher_policy.cpu().numpy()

        # Apply and record
        for j, i in enumerate(decision_idxs):
            game = self.games[i]
            cp = game.current_player
            mp = self.model_players[i]
            action = int(actions[j])
            if batch_masks[j][action] == 0:
                action = batch_legal[j][0]

            # Only record states where OUR tracked player is to move — but since
            # both players are V6.1 self-play, either side's states are valid
            # training data. We key own_moves_remaining to the player actually
            # moving here so the "won" label below matches.
            self.own_move_counts[i] += 1
            self.trajectories[i].append({
                'state28': batch28_states[j],
                'policy': policies_np[j].copy(),
                'legal_mask': batch_masks[j].copy(),
                'own_move_idx': self.own_move_counts[i],
                'player_moving': cp,
            })

            self.games[i] = ludo_cpp.apply_move(game, action)
            self.step_count[i] += 1

            if self.games[i].is_terminal:
                completed.extend(self._finalize(i))
                self._reset(i)

        # Handle timeouts
        for i in range(self.batch_size):
            if self.step_count[i] >= MAX_TURNS and not self.games[i].is_terminal:
                self._reset(i)

        return completed

    def _finalize(self, i):
        """Build training samples with won + own_moves_remaining labels."""
        winner = ludo_cpp.get_winner(self.games[i])
        if winner < 0:
            return []

        # Count each player's total own moves from trajectory
        player_total_moves = {}
        for step in self.trajectories[i]:
            p = step['player_moving']
            player_total_moves[p] = player_total_moves.get(p, 0) + 1

        # Separate per-player own-move counters to compute remaining correctly
        player_move_cursor = {p: 0 for p in player_total_moves}
        samples = []
        for step in self.trajectories[i]:
            p = step['player_moving']
            player_move_cursor[p] += 1
            own_remaining = player_total_moves[p] - player_move_cursor[p]
            won = 1 if p == winner else 0
            samples.append({
                'state28': step['state28'],
                'policy': step['policy'],
                'legal_mask': step['legal_mask'],
                'won': won,
                'moves_remaining': own_remaining,
            })
        return samples


env = BatchedSelfPlay(BATCH_SIZE)
all_states, all_policies, all_masks = [], [], []
all_won, all_moves = [], []
chunk_idx = 0
total_samples = 0
t0 = time.time()
last_report = t0
print(f"[V10 Data] Collecting...")

while total_samples < TARGET_STATES:
    completed = env.play_step()
    for s in completed:
        all_states.append(s['state28'])
        all_policies.append(s['policy'])
        all_masks.append(s['legal_mask'])
        all_won.append(s['won'])
        all_moves.append(s['moves_remaining'])
        total_samples += 1

    while len(all_states) >= CHUNK_SIZE:
        path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            path,
            states=np.stack(all_states[:CHUNK_SIZE]).astype(np.float32),
            policies=np.stack(all_policies[:CHUNK_SIZE]).astype(np.float32),
            legal_masks=np.stack(all_masks[:CHUNK_SIZE]).astype(np.float32),
            won=np.array(all_won[:CHUNK_SIZE], dtype=np.int8),
            moves_remaining=np.array(all_moves[:CHUNK_SIZE], dtype=np.int32),
        )
        all_states = all_states[CHUNK_SIZE:]
        all_policies = all_policies[CHUNK_SIZE:]
        all_masks = all_masks[CHUNK_SIZE:]
        all_won = all_won[CHUNK_SIZE:]
        all_moves = all_moves[CHUNK_SIZE:]
        chunk_idx += 1

    now = time.time()
    if now - last_report > 15:
        elapsed = now - t0
        rate = total_samples / elapsed if elapsed > 0 else 0
        eta = (TARGET_STATES - total_samples) / rate if rate > 0 else 0
        avg_won = np.mean(all_won) if all_won else 0
        avg_mr = np.mean(all_moves) if all_moves else 0
        print(f"  [{elapsed:.0f}s] {total_samples:,}/{TARGET_STATES:,} | "
              f"{rate:.0f} states/s | ETA {eta:.0f}s | "
              f"wr {avg_won:.2f} avg_mr {avg_mr:.1f}", flush=True)
        last_report = now

# Final flush
if all_states:
    path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        path,
        states=np.stack(all_states).astype(np.float32),
        policies=np.stack(all_policies).astype(np.float32),
        legal_masks=np.stack(all_masks).astype(np.float32),
        won=np.array(all_won, dtype=np.int8),
        moves_remaining=np.array(all_moves, dtype=np.int32),
    )
    chunk_idx += 1

elapsed = time.time() - t0
print(f"[V10 Data] Done: {total_samples:,} samples, {chunk_idx} chunks, {elapsed:.0f}s "
      f"({total_samples/elapsed:.0f} states/s)")
