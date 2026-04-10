"""
Experiment 06 — CKA (Centered Kernel Alignment) Similarity Between Layers

Compute CKA between every pair of ResBlock outputs to identify redundant layers.
If two consecutive blocks produce nearly identical representations (CKA > 0.95),
one is likely redundant and can be removed for model compression.

CKA is invariant to orthogonal transforms and isotropic scaling, making it a
robust measure of representational similarity.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited" (ICML 2019)

Usage:
    python run_cka_similarity.py
    python run_cka_similarity.py --num-states 2000
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
    parser = argparse.ArgumentParser(description="CKA similarity between ResBlocks.")
    parser.add_argument(
        "--weights",
        default="../../weights/model_latest_323k_shaped.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument("--num-states", type=int, default=1000, help="Number of game states to collect.")
    parser.add_argument("--num-games", type=int, default=100, help="Parallel games for state collection.")
    parser.add_argument("--max-loops", type=int, default=3000, help="Max collection loops.")
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


def extract_layer_representations(model, X, batch_size=256):
    """
    Extract the GAP-pooled representation after each ResBlock (and the stem).

    Returns:
        representations: list of (N, C) numpy arrays — [stem, block0, block1, ..., block9]
        labels: list of string labels for each representation
    """
    num_blocks = len(model.res_blocks)
    layer_outputs = [[] for _ in range(num_blocks + 1)]  # +1 for stem

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]

            # Stem
            out = F.relu(model.bn_input(model.conv_input(batch)))
            # GAP to get (B, C) representation
            stem_repr = F.adaptive_avg_pool2d(out, 1).flatten(start_dim=1)
            layer_outputs[0].append(stem_repr.numpy())

            # Each ResBlock
            for i, block in enumerate(model.res_blocks):
                out = block(out)
                block_repr = F.adaptive_avg_pool2d(out, 1).flatten(start_dim=1)
                layer_outputs[i + 1].append(block_repr.numpy())

    representations = [np.concatenate(lo, axis=0) for lo in layer_outputs]
    labels = ["Stem"] + [f"Block {i}" for i in range(num_blocks)]
    return representations, labels


def linear_cka(X, Y):
    """
    Compute linear CKA between two representation matrices.

    X: (N, p) — N samples, p features
    Y: (N, q) — N samples, q features

    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    # Center the representations
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices
    XtX = X.T @ X  # (p, p)
    YtY = Y.T @ Y  # (q, q)
    YtX = Y.T @ X  # (q, p)

    hsic_xy = np.linalg.norm(YtX, "fro") ** 2
    hsic_xx = np.linalg.norm(XtX, "fro")
    hsic_yy = np.linalg.norm(YtY, "fro")

    if hsic_xx * hsic_yy == 0:
        return 0.0

    return float(hsic_xy / (hsic_xx * hsic_yy))


def compute_cka_matrix(representations, labels):
    """Compute pairwise CKA between all layer representations."""
    n = len(representations)
    cka_matrix = np.zeros((n, n))

    total = n * (n + 1) // 2
    done = 0
    for i in range(n):
        for j in range(i, n):
            cka_val = linear_cka(representations[i], representations[j])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
            done += 1
            if done % 10 == 0:
                print(f"  CKA progress: {done}/{total} pairs computed")

    return cka_matrix


def visualize(cka_matrix, labels, consecutive_cka, save_dir):
    """Generate CKA visualizations."""

    # --- Plot 1: Full CKA heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cka_matrix, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("CKA Similarity Between Layers")

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cka_matrix[i, j] > 0.7 else "black"
            ax.text(j, i, f"{cka_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="CKA Similarity")
    plt.tight_layout()
    plt.savefig(save_dir / "cka_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Consecutive block CKA ---
    fig, ax = plt.subplots(figsize=(10, 5))
    pairs = [f"{consecutive_cka[k]['pair'][0]}\nvs\n{consecutive_cka[k]['pair'][1]}"
             for k in range(len(consecutive_cka))]
    cka_vals = [c["cka"] for c in consecutive_cka]
    colors = ["#e74c3c" if v > 0.95 else "#f39c12" if v > 0.90 else "#2ecc71" for v in cka_vals]

    bars = ax.bar(range(len(pairs)), cka_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=8)
    ax.set_ylabel("CKA Similarity")
    ax.set_title("CKA Between Consecutive Layers")
    ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1.5, label="Redundancy threshold (0.95)")
    ax.axhline(y=0.90, color="orange", linestyle="--", linewidth=1, label="High similarity (0.90)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar, val in zip(bars, cka_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "cka_consecutive.png", dpi=150)
    plt.close(fig)

    print(f"Saved visualizations to {save_dir}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    weights_path = resolve_weights_path(args.weights)
    model = load_checkpoint_model(weights_path)

    X = collect_states(args.num_games, args.num_states, args.max_loops, args.seed)

    print(f"Extracting representations from {len(model.res_blocks)} ResBlocks + stem...")
    representations, labels = extract_layer_representations(model, X)

    print("Computing pairwise CKA matrix...")
    cka_matrix = compute_cka_matrix(representations, labels)

    # Consecutive layer CKA
    consecutive_cka = []
    for i in range(len(labels) - 1):
        consecutive_cka.append({
            "pair": [labels[i], labels[i + 1]],
            "cka": float(cka_matrix[i, i + 1]),
        })

    # Summary
    output_dir = Path(__file__).resolve().parent

    print(f"\n=== CKA SIMILARITY SUMMARY ===")
    print("Consecutive layer pairs:")
    for c in consecutive_cka:
        marker = " *** REDUNDANT" if c["cka"] > 0.95 else " ** HIGH" if c["cka"] > 0.90 else ""
        print(f"  {c['pair'][0]:8s} <-> {c['pair'][1]:8s}: CKA = {c['cka']:.4f}{marker}")

    # Find redundant pairs
    redundant = [c for c in consecutive_cka if c["cka"] > 0.95]
    high_sim = [c for c in consecutive_cka if 0.90 < c["cka"] <= 0.95]

    print(f"\nRedundant pairs (CKA > 0.95): {len(redundant)}")
    for r in redundant:
        print(f"  {r['pair'][0]} <-> {r['pair'][1]}: {r['cka']:.4f}")

    print(f"High similarity pairs (0.90 < CKA <= 0.95): {len(high_sim)}")
    for h in high_sim:
        print(f"  {h['pair'][0]} <-> {h['pair'][1]}: {h['cka']:.4f}")

    if redundant:
        print(f"\nCompression suggestion: {len(redundant)} block(s) may be removable")

    # Save metrics
    metrics = {
        "num_states": int(len(X)),
        "num_layers": len(labels),
        "labels": labels,
        "cka_matrix": cka_matrix.tolist(),
        "consecutive_cka": consecutive_cka,
        "redundant_pairs_095": [{"pair": r["pair"], "cka": r["cka"]} for r in redundant],
        "high_similarity_090": [{"pair": h["pair"], "cka": h["cka"]} for h in high_sim],
    }

    with open(output_dir / "cka_similarity_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_dir / 'cka_similarity_metrics.json'}")

    visualize(cka_matrix, labels, consecutive_cka, output_dir)


if __name__ == "__main__":
    main()
