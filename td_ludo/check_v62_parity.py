"""
Compare V6.1 forward vs V6.2 forward (alpha forced to 0) on real game states.
If they produce identical logits, then alpha drift / transformer noise is the
root cause of V6.2's eval underperformance.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV5
from src.model_v6_2 import AlphaLudoV62
from src.fast_actor_v62 import TurnHistory
from src.config import MAX_MOVES_PER_GAME

device = 'cpu'
random.seed(0); np.random.seed(0); torch.manual_seed(0)

# --- Load V6.1 ---
v61_ckpt = torch.load('checkpoints/ac_v6_1_strategic/model_latest.pt', map_location=device, weights_only=False)
v61 = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
v61.load_state_dict(v61_ckpt['model_state_dict'])
v61.eval()

# --- Build V6.2 with same V6.1 weights, force alpha=0 exactly ---
v62 = AlphaLudoV62(context_length=8, num_res_blocks=10, in_channels=24)
v62.load_v61_weights(v61_ckpt['model_state_dict'])
with torch.no_grad():
    v62.transformer_alpha.zero_()  # tanh(0) = 0
v62.eval()

print(f"alpha gate: {torch.tanh(v62.transformer_alpha).item()}")

# --- Sample real game states by playing a few games ---
states_to_test = []  # list of (grid_np, history_list, last_action, legal_mask_np)
n_games_to_collect = 5
for g in range(n_games_to_collect):
    state = ludo_cpp.create_initial_state_2p()
    history = TurnHistory(8, 128)
    last_action = 4
    move_count = 0
    while not state.is_terminal and move_count < 200 and len(states_to_test) < 30:
        if not state.active_players[state.current_player]:
            state.current_player = (state.current_player + 1) % 4
            continue
        if state.current_dice_roll == 0:
            state.current_dice_roll = random.randint(1, 6)
        legal = ludo_cpp.get_legal_moves(state)
        if len(legal) == 0:
            state.current_player = (state.current_player + 1) % 4
            state.current_dice_roll = 0
            continue
        if state.current_player == 0 and len(legal) > 1:
            grid = ludo_cpp.encode_state_v6(state)
            history.add_turn(grid.copy(), action=last_action, cnn_feature=None)
            with torch.no_grad():
                gt = torch.from_numpy(grid).unsqueeze(0).float()
                cnn_feat = v62.compute_single_cnn_features(gt)
                history._cnn_features[-1] = cnn_feat.cpu().numpy()[0]
            legal_mask = np.zeros(4, dtype=np.float32)
            for m in legal: legal_mask[m] = 1.0
            states_to_test.append((grid.copy(), history.get_cached_sequence(), legal_mask))
            # Pick random legal action to advance
            action = random.choice(legal)
            last_action = action
            state = ludo_cpp.apply_move(state, action)
            state.current_dice_roll = 0
            move_count += 1
        else:
            action = random.choice(legal)
            state = ludo_cpp.apply_move(state, action)
            state.current_dice_roll = 0
            move_count += 1

print(f"Collected {len(states_to_test)} test states")

# --- Compare V6.1 vs V6.2 on each state ---
mismatches = 0
max_diff = 0.0
total_logit_l2 = 0.0
total_argmax_diff = 0
for i, (grid, (cached_cnn, seq_acts, seq_mask), legal_mask) in enumerate(states_to_test):
    with torch.no_grad():
        # V6.1: just CNN + heads
        gt = torch.from_numpy(grid).unsqueeze(0).float()
        lm = torch.from_numpy(legal_mask).unsqueeze(0).float()
        v61_policy, v61_value = v61(gt, lm)

        # V6.2: full forward via cached path (matches eval code)
        t_cached = torch.from_numpy(cached_cnn).unsqueeze(0).float()
        t_acts = torch.from_numpy(seq_acts).unsqueeze(0).long()
        t_mask = torch.from_numpy(seq_mask).unsqueeze(0).bool()
        v62_policy, v62_value = v62.forward_cached(t_cached, t_acts, t_mask, lm)

    diff = (v61_policy - v62_policy).abs().max().item()
    max_diff = max(max_diff, diff)
    total_logit_l2 += (v61_policy - v62_policy).pow(2).sum().item()

    a61 = v61_policy.argmax(dim=1).item()
    a62 = v62_policy.argmax(dim=1).item()
    if a61 != a62:
        total_argmax_diff += 1
        print(f"  state {i}: argmax differs! v61={a61} v62={a62} | "
              f"v61_p={v61_policy[0].numpy().round(4)} v62_p={v62_policy[0].numpy().round(4)}")
    if diff > 1e-5:
        mismatches += 1

print(f"\nResults over {len(states_to_test)} states:")
print(f"  max policy abs diff: {max_diff:.6e}")
print(f"  states with diff > 1e-5: {mismatches}")
print(f"  states with different argmax: {total_argmax_diff}")
print(f"  total L2 sum: {total_logit_l2:.6e}")
