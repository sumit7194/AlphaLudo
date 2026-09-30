#!/usr/bin/env python3
"""Async eval worker for train_v12.py (2026-06-11 throughput work).

Loads a frozen checkpoint snapshot, runs the exact same
evaluate_v11.evaluate_model call the in-loop eval used, and writes a JSON
result. Spawned nice'd by train_v12.py so the 2000-game eval no longer
blocks the training loop (~6.5 min ≈ 13% of wall time at 15K-game cadence).

Only --model-arch v13_5 is wired up (the only arch with active runs).
Stdout/stderr are discarded by the parent; the JSON file is the contract:
    {"win_rate": float, "win_rate_percent": str, "games": int}
Written atomically (tmp + os.replace) so the parent never reads a partial
file. Non-zero exit = eval failed; the parent skips that eval and moves on.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True,
                    help='Frozen checkpoint file to evaluate.')
    ap.add_argument('--out', required=True, help='Result JSON path.')
    ap.add_argument('--games', type=int, default=2000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--model-arch', default='v13_5')
    # Arch params mirror train_v12.py's v13_5 branch exactly (including the
    # 4→10 blocks and ≤0→128 channels sentinels).
    ap.add_argument('--num-res-blocks', type=int, default=6)
    ap.add_argument('--v135-num-channels', type=int, default=96)
    ap.add_argument('--head-hidden', type=int, default=64)
    args = ap.parse_args()

    if args.model_arch != 'v13_5':
        print(f"[eval-worker] unsupported arch: {args.model_arch}")
        sys.exit(2)

    import torch
    from td_ludo.models.v13_5_production import V135ProductionAdapter
    from td_ludo.game.encoder_v18_production import encode_state_v18_production
    from evaluate_v11 import evaluate_model

    blocks = args.num_res_blocks if args.num_res_blocks != 4 else 10
    channels = args.v135_num_channels if args.v135_num_channels > 0 else 128
    model = V135ProductionAdapter(
        num_res_blocks=blocks,
        num_channels=channels,
        head_hidden=args.head_hidden,
    )

    ckpt = torch.load(args.snapshot, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)

    want_cuda = str(args.device).startswith('cuda')
    device = torch.device(args.device) if (not want_cuda or torch.cuda.is_available()) \
        else torch.device('cpu')
    model.to(device)
    model.eval()

    results = evaluate_model(
        model, device, num_games=args.games, verbose=False,
        encoder_fn=encode_state_v18_production,
    )

    tmp = args.out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({
            'win_rate': results['win_rate'],
            'win_rate_percent': results.get('win_rate_percent'),
            'games': args.games,
            'snapshot': os.path.basename(args.snapshot),
        }, f)
    os.replace(tmp, args.out)


if __name__ == '__main__':
    main()
