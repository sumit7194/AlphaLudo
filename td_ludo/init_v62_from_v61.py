"""Initialize V6.2 model from V6.1 weights. Transfers:
  - CNN backbone + policy head + value head (via model.load_v61_weights)
  - Return normalization running stats (critical: prevents value head blow-up)
  - Adam optimizer state for shared parameters (CNN + heads only; transformer params stay fresh)
  - Bookkeeping counters (total_games, total_updates, best_win_rate)

Without the return stats + optimizer state, V6.2's first PPO updates see
value_loss ~2.1 (vs V6.1 steady-state 0.26), which corrupts the CNN trunk
via backprop before the running EMA can catch up. See training_journal.md
Experiment 11 for the original discovery of the value-head drift failure mode.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import AlphaLudoV5
from src.model_v6_2 import AlphaLudoV62

v61_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ac_v6_1_strategic/model_latest.pt"
out_dir = "checkpoints/ac_v6_2_transformer"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "ghosts"), exist_ok=True)

print(f"Loading V6.1 from {v61_path}...")
ckpt = torch.load(v61_path, map_location='cpu', weights_only=False)
v61_sd = ckpt.get('model_state_dict', ckpt)

print("Creating V6.2 (128ch, 10res, 24in, K=8 transformer)...")
model = AlphaLudoV62(context_length=8, num_res_blocks=10, in_channels=24)
transferred = model.load_v61_weights(v61_sd)
print(f"  Transferred {transferred if isinstance(transferred, int) else '?'} weights via load_v61_weights()")

# --- Build the V6.2 checkpoint payload ---
payload = {'model_state_dict': model.state_dict()}

# 1. Return normalization stats (CRITICAL — see docstring)
if 'return_running_mean' in ckpt and 'return_running_std' in ckpt:
    payload['return_running_mean'] = ckpt['return_running_mean']
    payload['return_running_std'] = ckpt['return_running_std']
    print(f"  ✓ return_running_mean = {ckpt['return_running_mean']:.4f}")
    print(f"  ✓ return_running_std  = {ckpt['return_running_std']:.4f}")
else:
    print("  ✗ WARNING: V6.1 checkpoint has no return_running_mean/std — value head WILL blow up")

# 2. Bookkeeping counters (reset games/updates — this is a fresh RL run for V6.2,
#    but preserve best_win_rate so the eval-improvement check has a meaningful baseline)
payload['total_games']   = 0
payload['total_updates'] = 0
payload['best_win_rate'] = ckpt.get('best_win_rate', 0.0)
payload['last_ghost_game'] = 0
payload['metrics_history'] = []
payload['last_eval_wr'] = ckpt.get('last_eval_wr', None)
print(f"  ✓ best_win_rate carried over = {payload['best_win_rate']}")

# 3. Optimizer state — remap Adam momentum for parameters that exist in both models.
#    Torch Adam state is keyed by parameter *id* (integer), not name, so we rebuild
#    it by matching parameter names between the two state_dicts.
#
#    Strategy:
#      - Build V6.2 param_name -> param_id map by walking model.named_parameters()
#        in the order they were added to the optimizer.
#      - For each V6.2 param name that has a matching V6.1 name (via load_v61_weights'
#        rename map), copy the Adam state entry from V6.1's optimizer.
#      - V6.2-only params (transformer, alpha gate, etc.) get no entry → fresh Adam state.
if 'optimizer_state_dict' in ckpt:
    v61_opt = ckpt['optimizer_state_dict']
    v61_model_sd_keys = list(v61_sd.keys())

    # Rebuild V6.1's name->id map. V6.1 optimizer was built from AlphaLudoV5's
    # parameters() in order, so param_id i corresponds to the i-th entry in
    # v61_model.named_parameters(). We approximate that ordering via state_dict keys
    # (which preserve insertion order for nn.Module in modern PyTorch).
    # Filter out buffers (BatchNorm running stats etc.) — only true parameters get
    # optimizer state. We detect params by checking if they appear in v61_opt['state'].
    v61_param_names_ordered = [k for k in v61_model_sd_keys
                                if not (k.endswith('running_mean') or
                                        k.endswith('running_var') or
                                        k.endswith('num_batches_tracked'))]
    v61_name_to_id = {name: i for i, name in enumerate(v61_param_names_ordered)}

    # Name remapping V6.1 -> V6.2 (mirrors load_v61_weights)
    # Stem: conv_input -> stem.0, bn_input -> stem.1
    # Heads: policy_fc1/fc2 -> policy_head.0/2, value_fc1/fc2 -> value_head.0/2
    # ResBlocks: identical names
    def v62_to_v61(v62_name: str) -> str | None:
        if v62_name.startswith('stem.0.'):
            return v62_name.replace('stem.0.', 'conv_input.')
        if v62_name.startswith('stem.1.'):
            return v62_name.replace('stem.1.', 'bn_input.')
        if v62_name.startswith('policy_head.0.'):
            return v62_name.replace('policy_head.0.', 'policy_fc1.')
        if v62_name.startswith('policy_head.2.'):
            return v62_name.replace('policy_head.2.', 'policy_fc2.')
        if v62_name.startswith('value_head.0.'):
            return v62_name.replace('value_head.0.', 'value_fc1.')
        if v62_name.startswith('value_head.2.'):
            return v62_name.replace('value_head.2.', 'value_fc2.')
        if v62_name.startswith('res_blocks.'):
            return v62_name  # identical
        return None  # transformer, cnn_proj, alpha gate, etc. — no V6.1 counterpart

    # Build V6.2 optimizer state by matching params in order
    v62_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    v62_opt_state = {}
    matched = 0
    skipped = 0
    for v62_idx, (v62_name, _p) in enumerate(v62_named_params):
        v61_name = v62_to_v61(v62_name)
        if v61_name is not None and v61_name in v61_name_to_id:
            v61_idx = v61_name_to_id[v61_name]
            if v61_idx in v61_opt['state']:
                v62_opt_state[v62_idx] = v61_opt['state'][v61_idx]
                matched += 1
            else:
                skipped += 1
        else:
            skipped += 1

    # Rebuild param_groups — V6.2 has more params than V6.1, so we can't reuse
    # V6.1's group structure directly. We construct a single group using V6.1's
    # hyperparameters (lr, betas, eps, weight_decay) applied to all V6.2 params.
    v61_groups = v61_opt.get('param_groups', [{}])
    g0 = v61_groups[0]
    v62_group = {
        'lr': g0.get('lr', 3e-4),
        'betas': g0.get('betas', (0.9, 0.999)),
        'eps': g0.get('eps', 1e-8),
        'weight_decay': g0.get('weight_decay', 0.0),
        'amsgrad': g0.get('amsgrad', False),
        'params': list(range(len(v62_named_params))),
    }
    payload['optimizer_state_dict'] = {
        'state': v62_opt_state,
        'param_groups': [v62_group],
    }
    print(f"  ✓ optimizer state: {matched} params matched, {skipped} params fresh (transformer/new)")
else:
    print("  ✗ WARNING: V6.1 checkpoint has no optimizer_state_dict")

out_path = os.path.join(out_dir, "model_sl.pt")
torch.save(payload, out_path)
print(f"\nSaved to {out_path}")
print(f"V6.2 params: {model.count_parameters():,}")
print(f"Payload keys: {sorted(payload.keys())}")
