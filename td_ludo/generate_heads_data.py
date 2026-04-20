"""
Generate training data for calibrated heads (win_prob + moves_remaining).

For each decision state, records:
  - state_tensor (27ch V6.3 encoding)
  - won: 1 if model won this game, 0 if lost
  - own_moves_remaining: number of OWN turns from this state until game end

The own_moves_remaining target is computed retroactively at game end.

V6.3 plays against diverse opponents (self-play + bots) to cover the
state distribution we care about at inference.
"""

import os
import sys
import time
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from td_ludo.models.v6_3 import AlphaLudoV63
from src.heuristic_bot import get_bot, BOT_REGISTRY

# =========================================================================
# Config
# =========================================================================
MODEL_PATH = "checkpoints/ac_v6_3_capture/model_latest.pt"
OUTPUT_DIR = "checkpoints/heads_data_v6_3"
CHUNK_SIZE = 10000
TARGET_STATES = 200_000
BATCH_SIZE = 512           # parallel games
TEMPERATURE = 1.0
# Opponent mix for diversity
OPPONENT_POOL = ['SelfPlay', 'Expert', 'Heuristic', 'Aggressive', 'Defensive']
OPPONENT_WEIGHTS = [0.35, 0.25, 0.15, 0.15, 0.10]
MAX_TURNS = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================================
# Load V6.3
# =========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlphaLudoV63(num_res_blocks=10, num_channels=128, in_channels=27)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device).eval()
print(f"[Heads Data] Model: {MODEL_PATH} ({sum(p.numel() for p in model.parameters()):,} params) on {device}")
print(f"[Heads Data] Target: {TARGET_STATES:,} states in {OUTPUT_DIR}")
print(f"[Heads Data] Opponents: {dict(zip(OPPONENT_POOL, OPPONENT_WEIGHTS))}")


# =========================================================================
# Batched self-play with trajectory collection
# =========================================================================
class BatchedCollector:
    """Runs BATCH_SIZE parallel games, collecting (state, own_move_count)
    tuples per game. At game end, backfills moves_remaining and won."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)
        # Per-game state
        self.opponent_bots = [self._pick_opponent() for _ in range(batch_size)]
        self.model_players = [random.choice([0, 2]) for _ in range(batch_size)]
        # Trajectory for each game: list of (state_27ch, own_move_idx)
        self.trajectories = [[] for _ in range(batch_size)]
        # Own-move counters for the model's player
        self.own_move_counts = [0 for _ in range(batch_size)]

    def _pick_opponent(self):
        name = random.choices(OPPONENT_POOL, weights=OPPONENT_WEIGHTS, k=1)[0]
        return name

    def _reset_game(self, i):
        self.games[i] = ludo_cpp.create_initial_state_2p()
        self.consec_sixes[i] = 0
        self.step_count[i] = 0
        self.opponent_bots[i] = self._pick_opponent()
        self.model_players[i] = random.choice([0, 2])
        self.trajectories[i] = []
        self.own_move_counts[i] = 0

    def play_step(self):
        """Advance each game by one decision. Return list of (state, won, moves_remaining)
        triples from games that just completed."""
        completed = []

        # Collect decisions that need model inference (model's turn)
        model_decision_idxs = []
        model_states = []
        model_masks = []
        model_legal = []

        for i, game in enumerate(self.games):
            if game.is_terminal or self.step_count[i] >= MAX_TURNS:
                continue

            cp = game.current_player
            mp = self.model_players[i]

            # Roll dice if needed
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

            if cp == mp:
                # Model decision — batch for inference
                state_tensor = np.array(
                    ludo_cpp.encode_state_v6_3(game, int(self.consec_sixes[i, cp])),
                    dtype=np.float32,
                )
                mask = np.zeros(4, dtype=np.float32)
                for m in legal:
                    mask[m] = 1.0
                model_decision_idxs.append(i)
                model_states.append(state_tensor)
                model_masks.append(mask)
                model_legal.append(legal)
            else:
                # Bot decision — inline
                bot_name = self.opponent_bots[i]
                if bot_name == 'SelfPlay':
                    # Use model for opponent as well
                    state_tensor = np.array(
                        ludo_cpp.encode_state_v6_3(game, int(self.consec_sixes[i, cp])),
                        dtype=np.float32,
                    )
                    mask = np.zeros(4, dtype=np.float32)
                    for m in legal:
                        mask[m] = 1.0
                    with torch.no_grad():
                        s_t = torch.tensor(state_tensor).unsqueeze(0).to(device)
                        m_t = torch.tensor(mask).unsqueeze(0).to(device)
                        logits = model.forward_policy_only(s_t, m_t)
                        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    action = int(np.random.choice(4, p=probs / probs.sum()))
                    if action not in legal:
                        action = random.choice(legal)
                else:
                    bot = get_bot(bot_name, player_id=cp)
                    action = bot.select_move(game, legal)

                self.games[i] = ludo_cpp.apply_move(game, action)
                self.step_count[i] += 1

                if self.games[i].is_terminal:
                    completed.extend(self._finalize(i))
                    self._reset_game(i)

        # Batched model inference
        if model_decision_idxs:
            batch_s = torch.from_numpy(np.stack(model_states)).to(device)
            batch_m = torch.from_numpy(np.stack(model_masks)).to(device)
            with torch.no_grad():
                logits = model.forward_policy_only(batch_s, batch_m)
                if TEMPERATURE != 1.0:
                    logits = logits / TEMPERATURE
                probs = torch.softmax(logits, dim=1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()

            for j, i in enumerate(model_decision_idxs):
                a = int(actions[j])
                if a not in model_legal[j]:
                    a = random.choice(model_legal[j])

                # Record this as a trajectory step (own turn)
                self.own_move_counts[i] += 1
                self.trajectories[i].append((
                    model_states[j], self.own_move_counts[i]  # own-move index (1-based)
                ))

                self.games[i] = ludo_cpp.apply_move(self.games[i], a)
                self.step_count[i] += 1

                if self.games[i].is_terminal:
                    completed.extend(self._finalize(i))
                    self._reset_game(i)

        # Handle timeouts
        for i in range(self.batch_size):
            if self.step_count[i] >= MAX_TURNS and not self.games[i].is_terminal:
                # Discard trajectory (no reliable outcome)
                self._reset_game(i)

        return completed

    def _finalize(self, i):
        """Build training samples from a completed game's trajectory."""
        winner = ludo_cpp.get_winner(self.games[i])
        mp = self.model_players[i]
        won = 1 if winner == mp else 0
        total_own = self.own_move_counts[i]
        samples = []
        for state_tensor, own_idx in self.trajectories[i]:
            own_remaining = total_own - own_idx  # 0 at last move before end
            samples.append((state_tensor, won, own_remaining))
        return samples


# =========================================================================
# Collection loop
# =========================================================================
collector = BatchedCollector(BATCH_SIZE)
all_states = []
all_won = []
all_moves_remaining = []
chunk_idx = 0
total_samples = 0
total_games = 0
t0 = time.time()
last_report = t0

print(f"[Heads Data] Collecting...")
while total_samples < TARGET_STATES:
    completed = collector.play_step()
    if completed:
        total_games += len(set(range(len(completed))))  # approximate
    for state, won, mr in completed:
        all_states.append(state)
        all_won.append(won)
        all_moves_remaining.append(mr)
        total_samples += 1

    # Save chunks
    while len(all_states) >= CHUNK_SIZE:
        chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            chunk_path,
            states=np.stack(all_states[:CHUNK_SIZE]).astype(np.float32),
            won=np.array(all_won[:CHUNK_SIZE], dtype=np.int8),
            moves_remaining=np.array(all_moves_remaining[:CHUNK_SIZE], dtype=np.int32),
        )
        all_states = all_states[CHUNK_SIZE:]
        all_won = all_won[CHUNK_SIZE:]
        all_moves_remaining = all_moves_remaining[CHUNK_SIZE:]
        chunk_idx += 1

    now = time.time()
    if now - last_report > 15:
        elapsed = now - t0
        rate = total_samples / elapsed if elapsed > 0 else 0
        eta = (TARGET_STATES - total_samples) / rate if rate > 0 else 0
        avg_won = np.mean(all_won) if all_won else 0
        avg_mr = np.mean(all_moves_remaining) if all_moves_remaining else 0
        print(f"  [{elapsed:.0f}s] {total_samples:,}/{TARGET_STATES:,} states | "
              f"{rate:.0f} states/s | ETA {eta:.0f}s | "
              f"batch win-rate {avg_won:.2f}, avg mr {avg_mr:.1f}", flush=True)
        last_report = now

# Final flush
if all_states:
    chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        chunk_path,
        states=np.stack(all_states).astype(np.float32),
        won=np.array(all_won, dtype=np.int8),
        moves_remaining=np.array(all_moves_remaining, dtype=np.int32),
    )
    chunk_idx += 1

elapsed = time.time() - t0
print(f"\n[Heads Data] Done: {total_samples:,} samples, {chunk_idx} chunks, "
      f"{elapsed:.0f}s ({total_samples/elapsed:.0f} states/s)")
