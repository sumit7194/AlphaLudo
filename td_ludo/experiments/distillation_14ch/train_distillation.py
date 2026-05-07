import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import td_ludo_cpp as ludo_cpp
from td_ludo.models.v12 import AlphaLudoV12
from experiments.distillation_14ch.model_14ch import MinimalCNN14

# --- CONFIG ---
BATCH_SIZE = 1024
LR = 1e-3
MAX_STEPS = 1000000  # Will train infinitely until manual stop
TEACHER_CKPT = "/Users/sumit/Github/AlphaLudo/gcp_snapshots/v122_exp24_pre_opt/model_latest.pt"

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Device: {device}")

# --- INIT MODELS ---
print("Loading Teacher V12.2...")
teacher = AlphaLudoV12(num_res_blocks=3, num_channels=128, num_attn_layers=2, in_channels=33)
ckpt = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
teacher.load_state_dict(ckpt['model_state_dict'])
teacher.to(device)
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

print("Initializing Student MinimalCNN14...")
student = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=14)
student.to(device)
student.train()

optimizer = torch.optim.Adam(student.parameters(), lr=LR)

class OnTheFlyDistillationEnv:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)

    def _reset(self, i):
        self.games[i] = ludo_cpp.create_initial_state_2p()
        self.consec_sixes[i] = 0
        self.step_count[i] = 0

    def get_batch(self):
        """Advances games to next decision state and returns a batch of features."""
        decision_idxs = []
        batch33_states = []
        batch14_states = []
        batch_masks = []
        batch_legal = []

        # Advance games until everyone has a decision
        for i in range(self.batch_size):
            while True:
                game = self.games[i]
                if game.is_terminal or self.step_count[i] >= 400:
                    self._reset(i)
                    game = self.games[i]

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

                # Decision state reached!
                mask = np.zeros(4, dtype=np.float32)
                for m in legal:
                    mask[m] = 1.0

                enc33 = np.array(ludo_cpp.encode_state_v11(game), dtype=np.float32)
                enc14 = np.array(ludo_cpp.encode_state_v14_minimal(game), dtype=np.float32)

                decision_idxs.append(i)
                batch33_states.append(enc33)
                batch14_states.append(enc14)
                batch_masks.append(mask)
                batch_legal.append(legal)
                break

        return (
            decision_idxs,
            torch.from_numpy(np.stack(batch33_states)).to(device),
            torch.from_numpy(np.stack(batch14_states)).to(device),
            torch.from_numpy(np.stack(batch_masks)).to(device),
            batch_legal
        )

    def apply_actions(self, decision_idxs, actions, batch_masks, batch_legal):
        for k, i in enumerate(decision_idxs):
            action = int(actions[k])
            if batch_masks[k][action] == 0:
                action = batch_legal[k][0]
            self.games[i] = ludo_cpp.apply_move(self.games[i], action)
            self.step_count[i] += 1

env = OnTheFlyDistillationEnv(BATCH_SIZE)

TARGET_STATES = 5000000
print(f"Starting on-the-fly distillation loop... Target: {TARGET_STATES:,} states")
t_start = time.time()
total_states_processed = 0

step = 0
while total_states_processed < TARGET_STATES:
    step_start = time.time()
    
    # 1. Gather Data
    decision_idxs, t_states33, s_states14, masks, legals = env.get_batch()
    
    # 2. Teacher Inference (Targets)
    with torch.no_grad():
        t_policy, t_win, t_moves = teacher(t_states33, masks)
        
        # Sample actions from teacher policy to advance games
        actions = torch.multinomial(t_policy, num_samples=1).squeeze(1).cpu().numpy()
    
    # 3. Apply actions
    env.apply_actions(decision_idxs, actions, masks.cpu().numpy(), legals)
    
    # 4. Student Training
    s_policy, s_win, s_moves = student(s_states14, masks)
    
    # Policy KL Divergence Loss
    s_log_policy = torch.log(s_policy + 1e-8)
    loss_policy = F.kl_div(s_log_policy, t_policy, reduction='batchmean', log_target=False)
    
    # Value MSE Loss
    loss_win = F.mse_loss(s_win, t_win)
    
    # Moves SmoothL1 Loss
    loss_moves = F.smooth_l1_loss(s_moves, t_moves)
    
    total_loss = loss_policy + loss_win + 0.01 * loss_moves
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    step += 1
    total_states_processed += BATCH_SIZE
    
    if step % 10 == 0:
        elapsed = time.time() - t_start
        fps = total_states_processed / elapsed
        print(f"Step {step:05d} | States: {total_states_processed:,}/{TARGET_STATES:,} | FPS: {fps:.1f} | Loss: {total_loss.item():.4f} "
              f"(Pol: {loss_policy.item():.4f}, Win: {loss_win.item():.4f}, Mov: {loss_moves.item():.4f})")
    
    # Periodic checkpoint save
    if total_states_processed % 1_000_000 < BATCH_SIZE and total_states_processed > BATCH_SIZE:
        save_path = f"experiments/distillation_14ch/student_14ch_{total_states_processed//1_000_000}M.pt"
        torch.save(student.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")

# Final save
final_path = f"experiments/distillation_14ch/student_14ch_final.pt"
torch.save(student.state_dict(), final_path)
print(f"Training complete! Final model saved to {final_path}")
