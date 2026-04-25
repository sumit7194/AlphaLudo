"""
V11 SL evaluation — reuses eval_v10_sl's eval_v10() function (same model API).

The only V11-specific bits are: which class to instantiate and which arch
keys to pass. encode_state_v10, forward(), forward_policy_only() — all
identical between V10 and V11.
"""
import os, sys, random, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from td_ludo.models.v11 import AlphaLudoV11
from eval_v10_sl import eval_v10  # reuse the eval loop verbatim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/ac_v11/model_sl.pt')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[V11 Eval] Device: {device}")
    print(f"[V11 Eval] Loading: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = ckpt.get('arch', {
        'num_res_blocks': 4, 'num_channels': 96,
        'num_attn_layers': 2, 'num_heads': 4, 'ffn_ratio': 4,
        'dropout': 0.1, 'in_channels': 28,
    })
    # Set dropout to 0 for eval (deterministic)
    eval_arch = {**arch, 'dropout': 0.0}
    model = AlphaLudoV11(**eval_arch).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[V11 Eval] Model: {model.count_parameters():,} params")
    print(f"[V11 Eval]   {arch['num_res_blocks']} ResBlocks × {arch['num_channels']}ch + "
          f"{arch['num_attn_layers']} Attn layers")
    print(f"[V11 Eval] Running {args.games} games vs random bot mix...")

    res = eval_v10(model, device, num_games=args.games, verbose=True)

    print(f"\n{'='*60}")
    print(f"  V11 SL Evaluation Results")
    print(f"{'='*60}")
    print(f"  Win rate:        {res['win_rate_pct']}% ({res['wins']}/{res['total']})")
    print(f"  Avg game len:    {res['avg_game_length']:.1f} moves")
    print(f"  Brier score:     {res['brier_score']}   (lower=better, baseline 0.25)")
    print(f"  Moves MAE:       {res['moves_mae']}    (avg own-turns-to-end error)")
    print(f"\n  Per-bot breakdown:")
    for bt, s in res['per_bot'].items():
        print(f"    {bt:14s}: {s['wr']}% ({s['games']} games)")
    print(f"\n  Win-prob calibration (is prediction == reality?):")
    print(f"    {'Range':<14} {'N':>7} {'Predicted':>12} {'Actual':>10}")
    for b in res['calibration_buckets']:
        print(f"    {b['range']:<14} {b['n']:>7} {b['mean_pred']:>12.3f} {b['mean_actual']:>10.3f}")
    print(f"{'='*60}")

    # Parity gate verdict
    print()
    if res['win_rate'] >= 0.73:
        delta = (res['win_rate'] - 0.735) * 100
        print(f"  ★ PARITY GATE PASSED: WR {res['win_rate_pct']}% (V10 SL ref 73.5%, "
              f"Δ {'+' if delta>=0 else ''}{delta:.1f}pp)")
        print(f"  → Ready to build train_v11.py for RL phase")
    elif res['win_rate'] >= 0.70:
        print(f"  ? PARITY GATE BORDERLINE: WR {res['win_rate_pct']}% "
              f"(target ≥73%, V10 ref 73.5%)")
        print(f"  → Consider re-evaluating with more games or training another epoch")
    else:
        print(f"  ✗ PARITY GATE FAILED: WR {res['win_rate_pct']}% < 70%")
        print(f"  → Architecture may be hurting in this regime; debug before RL")


if __name__ == '__main__':
    main()
