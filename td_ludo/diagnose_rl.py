"""
Diagnostic: Compare SL baseline vs RL model policy entropy on identical states.
This tells us if the RL training is collapsing the policy distribution.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import random
import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV5
from src.config import MAX_MOVES_PER_GAME

device = torch.device('cpu')

# Load both models
sl_model = AlphaLudoV5(num_res_blocks=5, num_channels=64)
sl_model.load_state_dict(torch.load('checkpoints/ac_v5/model_sl.pt', map_location='cpu'))
sl_model.eval()

rl_model = AlphaLudoV5(num_res_blocks=5, num_channels=64)
rl_ckpt = torch.load('checkpoints/ac_v5/model_latest.pt', map_location='cpu', weights_only=False)
if 'model_state_dict' in rl_ckpt:
    rl_model.load_state_dict(rl_ckpt['model_state_dict'])
else:
    rl_model.load_state_dict(rl_ckpt)
rl_model.eval()

# Play 10 random games, collect states
print("Collecting game states...")
states_collected = []
masks_collected = []

for game_idx in range(10):
    state = ludo_cpp.create_initial_state_2p()
    moves = 0
    while not state.is_terminal and moves < 500:
        cp = state.current_player
        if not state.active_players[cp]:
            next_p = (cp + 1) % 4
            while not state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            state.current_player = next_p
            continue
            
        if state.current_dice_roll == 0:
            state.current_dice_roll = random.randint(1, 6)
        
        legal_moves = ludo_cpp.get_legal_moves(state)
        if not legal_moves:
            next_p = (cp + 1) % 4
            while not state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            state.current_player = next_p
            state.current_dice_roll = 0
            continue
        
        # Collect this state
        s = ludo_cpp.encode_state(state)
        m = np.zeros(4, dtype=np.float32)
        for mv in legal_moves:
            m[mv] = 1.0
        states_collected.append(s)
        masks_collected.append(m)
        
        # Random action to advance
        action = random.choice(legal_moves)
        state = ludo_cpp.apply_move(state, action)
        moves += 1

print(f"Collected {len(states_collected)} states from 10 games")

# Compare policy distributions
states_t = torch.from_numpy(np.stack(states_collected)).float()
masks_t = torch.from_numpy(np.stack(masks_collected)).float()

with torch.no_grad():
    sl_policy, sl_value = sl_model(states_t, masks_t)
    rl_policy, rl_value = rl_model(states_t, masks_t)

# Entropy comparison
sl_entropy = -(sl_policy * torch.log(sl_policy + 1e-8)).sum(dim=1)
rl_entropy = -(rl_policy * torch.log(rl_policy + 1e-8)).sum(dim=1)

print(f"\n{'='*60}")
print(f"  Policy Entropy Comparison")
print(f"{'='*60}")
print(f"  SL Baseline:  mean={sl_entropy.mean():.4f}, std={sl_entropy.std():.4f}")
print(f"  RL Current:   mean={rl_entropy.mean():.4f}, std={rl_entropy.std():.4f}")
print(f"  Max possible: {np.log(4):.4f} (uniform over 4 tokens)")

# Value comparison
print(f"\n  Value Head Comparison")
print(f"  SL Values:  mean={sl_value.mean():.4f}, std={sl_value.std():.4f}")
print(f"  RL Values:  mean={rl_value.mean():.4f}, std={rl_value.std():.4f}")

# Look at some example policies
print(f"\n  Example Policy Distributions (first 10 states):")
print(f"  {'SL Policy':>40s} | {'RL Policy':>40s}")
print(f"  {'-'*40} | {'-'*40}")
for i in range(min(10, len(states_collected))):
    sl_p = sl_policy[i].numpy()
    rl_p = rl_policy[i].numpy()
    mask = masks_collected[i]
    legal = [j for j in range(4) if mask[j] > 0]
    sl_str = " ".join([f"T{j}:{sl_p[j]:.3f}" for j in range(4)])
    rl_str = " ".join([f"T{j}:{rl_p[j]:.3f}" for j in range(4)])
    print(f"  {sl_str:>40s} | {rl_str:>40s}  legal={legal}")

# Check how often RL model is >90% confident in one action
sl_max_probs = sl_policy.max(dim=1)[0]
rl_max_probs = rl_policy.max(dim=1)[0]
print(f"\n  Determinism Analysis:")
print(f"  SL: {(sl_max_probs > 0.9).float().mean()*100:.1f}% of states have >90% confidence")
print(f"  RL: {(rl_max_probs > 0.9).float().mean()*100:.1f}% of states have >90% confidence")
print(f"  SL: {(sl_max_probs > 0.7).float().mean()*100:.1f}% of states have >70% confidence")
print(f"  RL: {(rl_max_probs > 0.7).float().mean()*100:.1f}% of states have >70% confidence")
print(f"{'='*60}")
