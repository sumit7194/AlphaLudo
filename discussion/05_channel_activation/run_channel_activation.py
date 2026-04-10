"""
Experiment 05 — Channel Activation Analysis

Measure which of the 128 channels actually fire meaningfully across many game
states after each ResBlock. If many channels are near-zero on average, those
channels can be pruned to shrink the model (~50% fewer params in conv layers).

Outputs:
  - Per-block mean/std activation magnitude for each channel
  - Histogram of channel utilization
  - Count of "dead" channels (mean activation < threshold)
  - Pruning recommendation: how many channels could be removed

Usage:
    python run_channel_activation.py
    python run_channel_activation.py --num-states 2000 --threshold 0.01
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.common import advance_stuck_turn, load_checkpoint_model
import td_ludo_cpp as ludo_cpp


def parse_args():
    parser = argparse.ArgumentParser(description="Channel activation analysis.")
    parser.add_argument(
        "--weights",
        default="../../weights/model_latest_323k_shaped.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument("--num-states", type=int, default=1000, help="Number of game states to collect.")
    parser.add_argument("--num-games", type=int, default=100, help="Parallel games for state collection.")
    parser.add_argument("--max-loops", type=int, default=3000, help="Max collection loops.")
    parser.add_argument("--threshold", type=float, default=0.01, help="Mean activation below this = near-dead channel.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def resolve_weights_path(raw_path):
    path = Path(raw_path)
    if path.exists():
        return str(path)
    return str((Path(__file__).resolve().parent / raw_path).resolve())


def collect_states(num_games, num_states, max_loops, seed):
    """Collect game state tensors by running random games."""
    rng = np.random.default_rng(seed)
    env = ludo_cpp.VectorGameState(batch_size=num_games, two_player_mode=True)

    collected = []
    for loop in range(max_loops):
        if len(collected) >= num_states:
            break

        for i in range(num_games):
            game = env.get_game(i)
            if not game.is_terminal and game.current_dice_roll == 0:
                game.current_dice_roll = int(rng.integers(1, 7))

        legal_moves_batch = env.get_legal_moves()
        states_tensor = env.get_state_tensor()

        actions = []
        for i in range(num_games):
            game = env.get_game(i)
            moves = legal_moves_batch[i]

            if game.is_terminal:
                actions.append(-1)
                continue
            if not moves:
                advance_stuck_turn(game)
                actions.append(-1)
                continue

            action = int(rng.choice(moves))
            actions.append(action)

            if len(collected) < num_states:
                collected.append(states_tensor[i].copy())

        _, _, _, infos = env.step(actions)
        for i, info in enumerate(infos):
            if info["is_terminal"]:
                env.reset_game(i)

    print(f"Collected {len(collected)} game states.")
    return torch.tensor(np.array(collected[:num_states]), dtype=torch.float32)


def extract_activations(model, X, batch_size=256):
    """
    Run forward pass and capture the output activation of every ResBlock.

    Returns:
        activations: list of (num_states, 128, H, W) tensors, one per block
        stem_activations: (num_states, 128, H, W) tensor from the stem conv
    """
    num_blocks = len(model.res_blocks)
    # Accumulate per-block activations
    block_activations = [[] for _ in range(num_blocks)]
    stem_acts = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]

            # Stem
            out = F.relu(model.bn_input(model.conv_input(batch)))
            stem_acts.append(out.clone())

            # Each ResBlock
            for i, block in enumerate(model.res_blocks):
                out = block(out)
                block_activations[i].append(out.clone())

    stem_activations = torch.cat(stem_acts, dim=0)
    activations = [torch.cat(ba, dim=0) for ba in block_activations]
    return activations, stem_activations


def analyze_activations(activations, threshold):
    """
    For each block, compute per-channel statistics.

    Returns:
        block_stats: list of dicts with per-channel mean, std, fraction_zero
    """
    block_stats = []
    for block_idx, act in enumerate(activations):
        # act shape: (N, C, H, W)
        # Per-channel mean magnitude (averaged over spatial dims and samples)
        channel_mean = act.abs().mean(dim=(0, 2, 3)).numpy()  # (C,)
        channel_std = act.abs().std(dim=(0, 2, 3)).numpy()  # (C,)

        # Fraction of (sample, spatial) entries that are exactly zero (post-ReLU)
        channel_zero_frac = (act == 0).float().mean(dim=(0, 2, 3)).numpy()  # (C,)

        # Max activation per channel (across all samples and spatial)
        channel_max = act.amax(dim=(0, 2, 3)).numpy()  # (C,)

        near_dead = int((channel_mean < threshold).sum())
        block_stats.append({
            "block": block_idx,
            "channel_mean": channel_mean,
            "channel_std": channel_std,
            "channel_zero_frac": channel_zero_frac,
            "channel_max": channel_max,
            "near_dead_count": near_dead,
            "total_channels": len(channel_mean),
        })

    return block_stats


def visualize(block_stats, threshold, save_dir):
    """Generate visualizations for channel activation analysis."""
    num_blocks = len(block_stats)
    num_channels = block_stats[0]["total_channels"]

    # --- Plot 1: Heatmap of mean channel activations across blocks ---
    fig, ax = plt.subplots(figsize=(16, 6))
    heatmap_data = np.array([s["channel_mean"] for s in block_stats])
    im = ax.imshow(heatmap_data, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("ResBlock Index")
    ax.set_title(f"Mean |Activation| per Channel per Block ({num_channels} channels)")
    ax.set_yticks(range(num_blocks))
    plt.colorbar(im, ax=ax, label="Mean |Activation|")
    plt.tight_layout()
    plt.savefig(save_dir / "channel_activation_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Dead channel count per block ---
    fig, ax = plt.subplots(figsize=(10, 5))
    dead_counts = [s["near_dead_count"] for s in block_stats]
    bars = ax.bar(range(num_blocks), dead_counts, color="#e74c3c", edgecolor="black")
    ax.set_xlabel("ResBlock Index")
    ax.set_ylabel(f"Near-Dead Channels (mean < {threshold})")
    ax.set_title("Near-Dead Channels per Block")
    ax.set_xticks(range(num_blocks))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    for bar, count in zip(bars, dead_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_dir / "channel_dead_count.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Distribution of channel mean activations (all blocks combined) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    all_means = np.concatenate([s["channel_mean"] for s in block_stats])
    ax.hist(all_means, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold={threshold}")
    ax.set_xlabel("Mean |Activation|")
    ax.set_ylabel("Count (channels x blocks)")
    ax.set_title("Distribution of Channel Mean Activations (All Blocks)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_dir / "channel_activation_distribution.png", dpi=150)
    plt.close(fig)

    # --- Plot 4: Sorted channel importance (mean across all blocks) ---
    fig, ax = plt.subplots(figsize=(14, 5))
    avg_across_blocks = np.mean([s["channel_mean"] for s in block_stats], axis=0)
    sorted_idx = np.argsort(avg_across_blocks)
    ax.bar(range(num_channels), avg_across_blocks[sorted_idx], color="steelblue", width=1.0)
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={threshold}")
    ax.set_xlabel("Channel (sorted by mean activation)")
    ax.set_ylabel("Mean |Activation| (averaged across blocks)")
    ax.set_title("Channel Importance Ranking")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_dir / "channel_importance_ranking.png", dpi=150)
    plt.close(fig)

    print(f"Saved 4 visualizations to {save_dir}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    weights_path = resolve_weights_path(args.weights)
    model = load_checkpoint_model(weights_path)

    X = collect_states(args.num_games, args.num_states, args.max_loops, args.seed)

    print(f"Extracting activations from {len(model.res_blocks)} ResBlocks...")
    activations, stem_activations = extract_activations(model, X)

    print(f"Analyzing channel activations (threshold={args.threshold})...")
    block_stats = analyze_activations(activations, args.threshold)

    # Summary
    output_dir = Path(__file__).resolve().parent
    num_channels = block_stats[0]["total_channels"]

    print(f"\n=== CHANNEL ACTIVATION SUMMARY ===")
    for s in block_stats:
        print(f"  Block {s['block']:2d}: {s['near_dead_count']:3d}/{num_channels} near-dead channels, "
              f"mean activation range [{s['channel_mean'].min():.4f}, {s['channel_mean'].max():.4f}]")

    # Global pruning recommendation
    avg_across_blocks = np.mean([s["channel_mean"] for s in block_stats], axis=0)
    globally_dead = int((avg_across_blocks < args.threshold).sum())
    low_activity = int((avg_across_blocks < args.threshold * 5).sum())

    print(f"\nGlobal analysis (averaged across all blocks):")
    print(f"  Near-dead channels (mean < {args.threshold}): {globally_dead}/{num_channels}")
    print(f"  Low-activity channels (mean < {args.threshold * 5}): {low_activity}/{num_channels}")
    print(f"  Active channels: {num_channels - globally_dead}/{num_channels}")
    if globally_dead > 0:
        reduced = num_channels - globally_dead
        param_reduction = 1 - (reduced / num_channels) ** 2  # rough: params ~ channels^2
        print(f"  Pruning to {reduced} channels would reduce conv params by ~{param_reduction * 100:.0f}%")

    # Save metrics
    metrics = {
        "num_states": int(len(X)),
        "threshold": args.threshold,
        "num_channels": num_channels,
        "num_blocks": len(block_stats),
        "globally_dead_channels": globally_dead,
        "low_activity_channels": low_activity,
        "per_block": [],
    }
    for s in block_stats:
        metrics["per_block"].append({
            "block": s["block"],
            "near_dead_count": s["near_dead_count"],
            "channel_mean": s["channel_mean"].tolist(),
            "channel_std": s["channel_std"].tolist(),
            "channel_zero_frac": s["channel_zero_frac"].tolist(),
            "channel_max": s["channel_max"].tolist(),
        })

    with open(output_dir / "channel_activation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_dir / 'channel_activation_metrics.json'}")

    visualize(block_stats, args.threshold, output_dir)


if __name__ == "__main__":
    main()
