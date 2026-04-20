"""
Generate joint SL training data for V10, MIXED teachers (V6.1 + V6.3).

For each decision state from teacher self-play, record:
  - state (28ch V10 encoding)       — input features
  - policy (4,)                     — the teacher-of-that-player's softmax output
  - won (0/1)                       — game outcome from this player
  - moves_remaining                 — own-turns from this state to game end
  - legal_mask (4,)                 — which tokens were legal
  - teacher_id (0=V6.1, 1=V6.3)     — for diagnostics only

Each game independently picks a teacher (V6.1 or V6.3) that plays BOTH sides.
Half the games use V6.1, half use V6.3 (controlled by V10_TEACHER_MIX).

Trains all 3 heads jointly from scratch.
"""

import os, sys, time, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV5  # V6.1 teacher
from td_ludo.models.v6_3 import AlphaLudoV63  # V6.3 teacher

V61_PATH = "checkpoints/ac_v6_1_strategic/model_best.pt"
V63_PATH = "checkpoints/ac_v6_3_capture/model_best.pt"
OUTPUT_DIR = "checkpoints/sl_data_v10"
CHUNK_SIZE = 10000
TARGET_STATES = int(os.environ.get('V10_TARGET_STATES', 500_000))
BATCH_SIZE = int(os.environ.get('V10_DATA_BATCH', 512))
# Fraction of games assigned to V6.3 (rest go to V6.1). 0.5 = 50/50 mix.
V63_FRACTION = float(os.environ.get('V10_TEACHER_MIX', 0.5))
TEMPERATURE = 1.0
MAX_TURNS = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

teacher_v61 = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
ckpt = torch.load(V61_PATH, map_location=device, weights_only=False)
teacher_v61.load_state_dict(ckpt['model_state_dict'])
teacher_v61.to(device).eval()

teacher_v63 = AlphaLudoV63(num_res_blocks=10, num_channels=128, in_channels=27)
ckpt63 = torch.load(V63_PATH, map_location=device, weights_only=False)
teacher_v63.load_state_dict(ckpt63['model_state_dict'])
teacher_v63.to(device).eval()

TEACHER_NAMES = {0: 'V6.1', 1: 'V6.3'}
print(f"[V10 Data] Teachers:  V6.1={V61_PATH}  V6.3={V63_PATH}")
print(f"[V10 Data] Device: {device}  |  V6.3 game share: {V63_FRACTION:.0%}")
print(f"[V10 Data] Target: {TARGET_STATES:,} states in {OUTPUT_DIR}")


def pick_teacher():
    """0=V6.1, 1=V6.3 — each game independently."""
    return 1 if random.random() < V63_FRACTION else 0


class BatchedSelfPlay:
    """Self-play with mixed V6.1/V6.3 teachers. Collects V10-encoded states."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        self.teacher_id = [pick_teacher() for _ in range(batch_size)]
        self.trajectories = [[] for _ in range(batch_size)]
        self.own_move_counts = [0 for _ in range(batch_size)]
        self.step_count = np.zeros(batch_size, dtype=np.int32)

    def _reset(self, i):
        self.games[i] = ludo_cpp.create_initial_state_2p()
        self.consec_sixes[i] = 0
        self.teacher_id[i] = pick_teacher()
        self.trajectories[i] = []
        self.own_move_counts[i] = 0
        self.step_count[i] = 0

    def play_step(self):
        """One decision across all active games. Returns completed game samples."""
        completed = []

        # Advance dice / handle no-ops / gather decisions per teacher
        decision_idxs = []
        batch28_states = []   # always V10 (28ch) — student training input
        batch24_states = []   # V6.1 native (24ch) — for V6.1 teacher inference
        batch27_states = []   # V6.3 native (27ch) — for V6.3 teacher inference
        teacher_ids = []      # parallel to decision_idxs
        batch_masks = []
        batch_legal = []

        for i, game in enumerate(self.games):
            if game.is_terminal or self.step_count[i] >= MAX_TURNS:
                continue

            cp = game.current_player

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

            enc28 = np.array(ludo_cpp.encode_state_v10(game), dtype=np.float32)
            tid = self.teacher_id[i]
            if tid == 0:
                enc24 = np.array(ludo_cpp.encode_state_v6(game), dtype=np.float32)
                batch24_states.append(enc24)
                batch27_states.append(None)  # placeholder for alignment
            else:
                enc27 = np.array(
                    ludo_cpp.encode_state_v6_3(game, int(self.consec_sixes[i, cp])),
                    dtype=np.float32,
                )
                batch24_states.append(None)
                batch27_states.append(enc27)

            decision_idxs.append(i)
            teacher_ids.append(tid)
            batch28_states.append(enc28)
            batch_masks.append(mask)
            batch_legal.append(legal)

        if not decision_idxs:
            return completed

        # Split decisions by teacher, run one batched inference per teacher
        idx_v61 = [k for k, t in enumerate(teacher_ids) if t == 0]
        idx_v63 = [k for k, t in enumerate(teacher_ids) if t == 1]

        policies_np = np.empty((len(decision_idxs), 4), dtype=np.float32)
        actions = np.empty(len(decision_idxs), dtype=np.int64)

        if idx_v61:
            st24 = torch.from_numpy(
                np.stack([batch24_states[k] for k in idx_v61])
            ).to(device)
            mk = torch.from_numpy(
                np.stack([batch_masks[k] for k in idx_v61])
            ).to(device)
            with torch.no_grad():
                pol, _ = teacher_v61(st24, mk)
                if TEMPERATURE != 1.0:
                    logits = torch.log(pol + 1e-8) / TEMPERATURE
                    sample_probs = torch.softmax(logits, dim=1)
                else:
                    sample_probs = pol
                acts = torch.multinomial(sample_probs, num_samples=1).squeeze(1).cpu().numpy()
                pnp = pol.cpu().numpy()
            for local, k in enumerate(idx_v61):
                policies_np[k] = pnp[local]
                actions[k] = acts[local]

        if idx_v63:
            st27 = torch.from_numpy(
                np.stack([batch27_states[k] for k in idx_v63])
            ).to(device)
            mk = torch.from_numpy(
                np.stack([batch_masks[k] for k in idx_v63])
            ).to(device)
            with torch.no_grad():
                out = teacher_v63(st27, mk)
                pol = out[0]  # V6.3 returns (policy, value, aux_capture)
                if TEMPERATURE != 1.0:
                    logits = torch.log(pol + 1e-8) / TEMPERATURE
                    sample_probs = torch.softmax(logits, dim=1)
                else:
                    sample_probs = pol
                acts = torch.multinomial(sample_probs, num_samples=1).squeeze(1).cpu().numpy()
                pnp = pol.cpu().numpy()
            for local, k in enumerate(idx_v63):
                policies_np[k] = pnp[local]
                actions[k] = acts[local]

        # Apply and record
        for k, i in enumerate(decision_idxs):
            game = self.games[i]
            cp = game.current_player
            action = int(actions[k])
            if batch_masks[k][action] == 0:
                action = batch_legal[k][0]

            self.own_move_counts[i] += 1
            self.trajectories[i].append({
                'state28': batch28_states[k],
                'policy': policies_np[k].copy(),
                'legal_mask': batch_masks[k].copy(),
                'own_move_idx': self.own_move_counts[i],
                'player_moving': cp,
                'teacher_id': teacher_ids[k],
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

        player_total_moves = {}
        for step in self.trajectories[i]:
            p = step['player_moving']
            player_total_moves[p] = player_total_moves.get(p, 0) + 1

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
                'teacher_id': step['teacher_id'],
            })
        return samples


env = BatchedSelfPlay(BATCH_SIZE)
all_states, all_policies, all_masks = [], [], []
all_won, all_moves, all_teacher = [], [], []
chunk_idx = 0
total_samples = 0
teacher_counts = {0: 0, 1: 0}
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
        all_teacher.append(s['teacher_id'])
        teacher_counts[s['teacher_id']] += 1
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
            teacher_id=np.array(all_teacher[:CHUNK_SIZE], dtype=np.int8),
        )
        all_states = all_states[CHUNK_SIZE:]
        all_policies = all_policies[CHUNK_SIZE:]
        all_masks = all_masks[CHUNK_SIZE:]
        all_won = all_won[CHUNK_SIZE:]
        all_moves = all_moves[CHUNK_SIZE:]
        all_teacher = all_teacher[CHUNK_SIZE:]
        chunk_idx += 1

    now = time.time()
    if now - last_report > 15:
        elapsed = now - t0
        rate = total_samples / elapsed if elapsed > 0 else 0
        eta = (TARGET_STATES - total_samples) / rate if rate > 0 else 0
        avg_won = np.mean(all_won) if all_won else 0
        avg_mr = np.mean(all_moves) if all_moves else 0
        n_v61 = teacher_counts[0]; n_v63 = teacher_counts[1]
        mix = (n_v63 / max(1, n_v61 + n_v63)) * 100
        print(f"  [{elapsed:.0f}s] {total_samples:,}/{TARGET_STATES:,} | "
              f"{rate:.0f} states/s | ETA {eta:.0f}s | "
              f"wr {avg_won:.2f} avg_mr {avg_mr:.1f} | "
              f"mix V6.3 {mix:.0f}%", flush=True)
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
        teacher_id=np.array(all_teacher, dtype=np.int8),
    )
    chunk_idx += 1

elapsed = time.time() - t0
print(f"[V10 Data] Done: {total_samples:,} samples, {chunk_idx} chunks, "
      f"{elapsed:.0f}s ({total_samples/elapsed:.0f} states/s)")
print(f"[V10 Data] Teacher mix: V6.1={teacher_counts[0]:,}  V6.3={teacher_counts[1]:,}")
