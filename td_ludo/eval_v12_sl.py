"""V12 SL evaluation — reuses eval_v10_sl's eval_v10() loop."""
import os, sys, random, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from td_ludo.models.v12 import AlphaLudoV12
from eval_v10_sl import eval_v10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/ac_v12/model_sl.pt')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[V12 Eval] Device: {device}  Loading: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = ckpt.get('arch', {
        'num_res_blocks': 4, 'num_channels': 96,
        'num_attn_layers': 2, 'num_heads': 4, 'ffn_ratio': 4,
        'dropout': 0.1, 'in_channels': 28,
    })
    eval_arch = {**arch, 'dropout': 0.0}
    eval_arch.pop('attn_dim', None)  # legacy V11 ckpts may have this; V12 doesn't accept it
    model = AlphaLudoV12(**eval_arch).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[V12 Eval] Model: {model.count_parameters():,} params  ({arch['num_res_blocks']} ResBlocks × {arch['num_channels']}ch + token attention)")
    print(f"[V12 Eval] Running {args.games} games vs random bot mix...")

    res = eval_v10(model, device, num_games=args.games, verbose=True)

    print(f"\n{'='*60}")
    print(f"  V12 SL Evaluation Results")
    print(f"{'='*60}")
    print(f"  Win rate:        {res['win_rate_pct']}% ({res['wins']}/{res['total']})")
    print(f"  Avg game len:    {res['avg_game_length']:.1f} moves")
    print(f"  Brier score:     {res['brier_score']}")
    print(f"  Moves MAE:       {res['moves_mae']}")
    print(f"\n  Per-bot breakdown:")
    for bt, s in res['per_bot'].items():
        print(f"    {bt:14s}: {s['wr']}% ({s['games']} games)")
    print(f"\n  Calibration:")
    print(f"    {'Range':<14} {'N':>7} {'Predicted':>12} {'Actual':>10}")
    for b in res['calibration_buckets']:
        print(f"    {b['range']:<14} {b['n']:>7} {b['mean_pred']:>12.3f} {b['mean_actual']:>10.3f}")
    print(f"{'='*60}")

    if res['win_rate'] >= 0.73:
        delta = (res['win_rate'] - 0.735) * 100
        print(f"  ★ PARITY GATE PASSED: WR {res['win_rate_pct']}% (V10 SL ref 73.5%, Δ {'+' if delta>=0 else ''}{delta:.1f}pp)")
        print(f"  → Ready to build train_v12.py for RL phase")
    elif res['win_rate'] >= 0.70:
        print(f"  ? PARITY GATE BORDERLINE: WR {res['win_rate_pct']}%")
    else:
        print(f"  ✗ PARITY GATE FAILED: WR {res['win_rate_pct']}% — investigate before RL")


if __name__ == '__main__':
    main()
