#!/usr/bin/env python3
"""
Conv surgery: convert a V12 checkpoint (28-channel input) into a V12.1
checkpoint (33-channel input) with all transferable weights preserved.

What this does:
  1. Load V12 state_dict.
  2. For conv_input.weight: shape (out, 28, 3, 3) → pad to (out, 33, 3, 3)
     with zeros along input-channel dim. The new slices (channels 28..32 =
     idle counters + streak) start with no influence on the network output;
     PPO/SL will gradient-train them up.
  3. Drop entries that don't exist in the V12.1 model:
       - token_idx_emb.weight   (V12.1 commit 2 removed this)
  4. Drop entries with mismatched shapes (V12.1 commit 2 changed policy head):
       - policy_fc1.weight (64, 192) → (64, 96)
       - policy_fc2.weight (4, 64)   → (1, 64)
       - policy_fc2.bias   (4,)      → (1,)
     These layers will be reinitialized; SL warm-up restores their quality.
  5. Save the surgically-modified state_dict to the output path, ready for
     `train_sl_v12.py --resume` (or `train_v12.py --resume` after SL).

Usage:
  python scripts/surgery_v12_to_v12_1.py \\
      --in   play/model_weights/v12_final/model_latest.pt \\
      --out  checkpoints/ac_v12/model_sl.pt

After running, your typical sequence is:
  python train_sl_v12.py --resume   # warms the new policy head + leaves
                                    # idle/streak conv slices at zero
  python train_v12.py --resume      # RL: PPO trains the new conv slices
                                    # and adapts policy/value heads
"""
import argparse
import os
import sys

# Make td_ludo.* importable when running from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)  # td_ludo/
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch


def surgery(state_dict: dict, target_in_channels: int = 33) -> dict:
    """Return a new state_dict ready to load into a V12.1 model (strict=False)."""
    # Strip torch.compile prefixes if present
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    out = {}
    notes = []

    # Keys we know V12.1 won't have or will reshape — drop them.
    DROP_KEYS = {
        'token_idx_emb.weight',          # commit 2: dropped
        'policy_fc1.weight',             # commit 2: shape changed (64, 192) → (64, 96)
        'policy_fc1.bias',               # commit 2: same module, regenerated
        'policy_fc2.weight',             # commit 2: shape changed (4, 64) → (1, 64)
        'policy_fc2.bias',               # commit 2: shape changed (4,) → (1,)
    }

    for k, v in state_dict.items():
        if k in DROP_KEYS:
            notes.append(f'  drop {k!r:40s} {tuple(v.shape)} '
                         f'(removed/reshaped in V12.1)')
            continue

        if k == 'conv_input.weight':
            # Expand input channels with zeros for the new V11-encoder slices.
            cur_in = v.shape[1]
            if cur_in == target_in_channels:
                out[k] = v
                notes.append(f'  keep {k!r:40s} {tuple(v.shape)} '
                             f'(already {target_in_channels}ch — no surgery)')
            elif cur_in < target_in_channels:
                pad = torch.zeros(
                    (v.shape[0], target_in_channels - cur_in, v.shape[2], v.shape[3]),
                    dtype=v.dtype, device=v.device,
                )
                v_new = torch.cat([v, pad], dim=1)
                out[k] = v_new
                notes.append(
                    f'  pad  {k!r:40s} {tuple(v.shape)} → {tuple(v_new.shape)} '
                    f'(zero-init for {target_in_channels - cur_in} new input slices)'
                )
            else:
                raise ValueError(
                    f'V12 conv_input has more input channels ({cur_in}) than '
                    f'target ({target_in_channels}). Cannot truncate safely.'
                )
            continue

        out[k] = v

    print('Surgery summary:')
    for n in notes:
        print(n)
    print(f'  kept {len(out)} tensors total')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True,
                    help='V12 checkpoint (.pt) — accepts either raw state_dict '
                         'or a dict with "model_state_dict" key.')
    ap.add_argument('--out', dest='out_path', required=True,
                    help='Output V12.1 checkpoint path.')
    ap.add_argument('--target-in-channels', type=int, default=33,
                    help='V12.1 input channel count (default 33 for V11 encoder).')
    args = ap.parse_args()

    print(f'Loading V12 ckpt from {args.in_path}...')
    ckpt = torch.load(args.in_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
        print(f'  detected full-checkpoint dict; extracting model_state_dict')
    else:
        sd = ckpt
        print(f'  detected raw state_dict')
    print(f'  {len(sd)} tensors loaded')

    new_sd = surgery(sd, target_in_channels=args.target_in_channels)

    # Try a sanity load against the V12.1 model
    try:
        from td_ludo.models.v12 import AlphaLudoV12
        m = AlphaLudoV12()  # default in_channels=33
        result = m.load_state_dict(new_sd, strict=False)
        print(f'\nSanity load (strict=False) into AlphaLudoV12():')
        print(f'  missing keys (need reinit / will train from scratch): '
              f'{result.missing_keys}')
        print(f'  unexpected keys (in surgery output but not in model): '
              f'{result.unexpected_keys}')
    except Exception as e:
        print(f'\nWARN: sanity load skipped: {e}')

    # Save in the same wrapper format the trainer expects.
    out_obj = {'model_state_dict': new_sd}
    if isinstance(ckpt, dict):
        # Carry over training metadata (game count, optimizer state) for
        # informational purposes — but don't try to resume the optimizer state
        # because the parameter shapes differ now.
        for k in ('total_games', 'total_updates', 'best_win_rate'):
            if k in ckpt:
                out_obj[k] = ckpt[k]
        # Note: we deliberately drop optimizer_state_dict — the LR/momentum
        # buffers refer to the old parameter shapes and will fail to load.

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    torch.save(out_obj, args.out_path)
    print(f'\nSaved V12.1-ready ckpt: {args.out_path}')
    print(f'  next step: python train_sl_v12.py --resume '
          f'--output {args.out_path}')


if __name__ == '__main__':
    main()
