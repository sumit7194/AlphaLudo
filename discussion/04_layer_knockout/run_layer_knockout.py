"""
Experiment 04 — Layer Knockout

Skip each ResBlock one at a time (replace with identity) and measure win rate
against a random opponent. This reveals which blocks are critical vs redundant
for model compression.

Usage:
    python run_layer_knockout.py
    python run_layer_knockout.py --num-games 200 --max-turns 400
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.common import advance_stuck_turn, load_checkpoint_model
import td_ludo_cpp as ludo_cpp


def parse_args():
    parser = argparse.ArgumentParser(description="Layer knockout experiment.")
    parser.add_argument(
        "--weights",
        default="../../weights/model_latest_323k_shaped.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument("--num-games", type=int, default=200, help="Games per evaluation.")
    parser.add_argument("--max-turns", type=int, default=500, help="Max turns per game.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def resolve_weights_path(raw_path):
    path = Path(raw_path)
    if path.exists():
        return str(path)
    return str((Path(__file__).resolve().parent / raw_path).resolve())


class IdentityBlock(nn.Module):
    """Replaces a ResBlock — just passes input through."""

    def forward(self, x):
        return x


def evaluate_win_rate(model, num_games, max_turns, seed=0):
    """
    Play `num_games` 2-player games: player 0 uses the model, player 1 plays
    randomly. Returns win rate for the model player.

    Also returns mean policy KL-divergence and value MAE vs a baseline if
    baseline outputs are provided (handled externally).
    """
    rng = np.random.default_rng(seed)
    env = ludo_cpp.VectorGameState(batch_size=num_games, two_player_mode=True)

    wins = 0
    completed = 0

    for turn in range(max_turns):
        # Check which games are still active
        active_games = []
        for i in range(num_games):
            game = env.get_game(i)
            if game.is_terminal:
                continue
            active_games.append(i)

        if not active_games:
            break

        # Roll dice for games that need it
        for i in active_games:
            game = env.get_game(i)
            if game.current_dice_roll == 0:
                game.current_dice_roll = int(rng.integers(1, 7))

        legal_moves_batch = env.get_legal_moves()
        states_tensor = env.get_state_tensor()

        # Batch model inference for all player-0 games
        model_games = []
        for i in active_games:
            game = env.get_game(i)
            if game.current_player == 0 and legal_moves_batch[i]:
                model_games.append(i)

        model_actions = {}
        if model_games:
            batch_tensors = torch.tensor(
                np.array([states_tensor[i] for i in model_games]),
                dtype=torch.float32,
            )
            batch_masks = torch.zeros(len(model_games), 4, dtype=torch.float32)
            for j, i in enumerate(model_games):
                for m in legal_moves_batch[i]:
                    batch_masks[j, m] = 1.0

            with torch.no_grad():
                policy, _ = model(batch_tensors, batch_masks)

            for j, i in enumerate(model_games):
                model_actions[i] = int(policy[j].argmax().item())

        # Build action list
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

            if i in model_actions:
                actions.append(model_actions[i])
            elif game.current_player == 0:
                # Model player but no legal moves processed (shouldn't happen)
                actions.append(int(rng.choice(moves)))
            else:
                # Random opponent
                actions.append(int(rng.choice(moves)))

        _, _, _, infos = env.step(actions)

        for i, info in enumerate(infos):
            if info["is_terminal"]:
                completed += 1
                # Check if player 0 (model) won
                game = env.get_game(i)
                scores = np.array(game.scores, dtype=np.int8)
                if scores[0] >= scores[1]:
                    wins += 1
                env.reset_game(i)

    # Count any remaining unfinished games
    for i in range(num_games):
        game = env.get_game(i)
        if not game.is_terminal:
            completed += 1
            scores = np.array(game.scores, dtype=np.int8)
            if scores[0] >= scores[1]:
                wins += 1

    wr = wins / max(completed, 1) * 100.0
    return wr, completed


def run_layer_knockout(model, num_games, max_turns, seed):
    """Run knockout for each ResBlock and return results."""
    num_blocks = len(model.res_blocks)

    # Baseline: full model
    print("Evaluating baseline (all blocks active)...")
    baseline_wr, baseline_games = evaluate_win_rate(model, num_games, max_turns, seed)
    print(f"  Baseline WR: {baseline_wr:.1f}% ({baseline_games} games)")

    results = {
        "baseline": {"win_rate": baseline_wr, "games_completed": baseline_games},
        "knockouts": [],
    }

    for i in range(num_blocks):
        # Save original block and replace with identity
        original_block = model.res_blocks[i]
        model.res_blocks[i] = IdentityBlock()

        print(f"Evaluating with block {i} knocked out...")
        wr, games = evaluate_win_rate(model, num_games, max_turns, seed)
        delta = wr - baseline_wr
        print(f"  Block {i} skipped: WR={wr:.1f}% (delta={delta:+.1f}%), {games} games")

        results["knockouts"].append({
            "block": i,
            "win_rate": wr,
            "delta": delta,
            "games_completed": games,
        })

        # Restore
        model.res_blocks[i] = original_block

    return results


def visualize(results, save_path):
    """Generate a bar chart of WR deltas per block."""
    knockouts = results["knockouts"]
    blocks = [k["block"] for k in knockouts]
    deltas = [k["delta"] for k in knockouts]
    win_rates = [k["win_rate"] for k in knockouts]
    baseline_wr = results["baseline"]["win_rate"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Win rate per knockout
    colors = ["#e74c3c" if d < -5 else "#f39c12" if d < -2 else "#2ecc71" for d in deltas]
    bars = ax1.bar(blocks, win_rates, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=baseline_wr, color="blue", linestyle="--", linewidth=2, label=f"Baseline: {baseline_wr:.1f}%")
    ax1.set_xlabel("ResBlock Index")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Layer Knockout — Win Rate with Each Block Removed")
    ax1.set_xticks(blocks)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 2: Delta from baseline
    colors2 = ["#e74c3c" if d < -5 else "#f39c12" if d < -2 else "#2ecc71" for d in deltas]
    ax2.bar(blocks, deltas, color=colors2, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("ResBlock Index")
    ax2.set_ylabel("WR Delta (%)")
    ax2.set_title("Layer Knockout — Win Rate Change from Baseline")
    ax2.set_xticks(blocks)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate blocks
    for i, (block, delta) in enumerate(zip(blocks, deltas)):
        label = "CRITICAL" if delta < -5 else "important" if delta < -2 else "redundant"
        ax2.annotate(
            label,
            (block, delta),
            textcoords="offset points",
            xytext=(0, -15 if delta >= 0 else 10),
            ha="center",
            fontsize=8,
            color="gray",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    weights_path = resolve_weights_path(args.weights)
    model = load_checkpoint_model(weights_path)

    results = run_layer_knockout(model, args.num_games, args.max_turns, args.seed)

    output_dir = Path(__file__).resolve().parent
    save_path = output_dir / "layer_knockout_metrics.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {save_path}")

    visualize(results, output_dir / "layer_knockout_results.png")

    # Print summary
    print("\n=== LAYER KNOCKOUT SUMMARY ===")
    print(f"Baseline WR: {results['baseline']['win_rate']:.1f}%")
    print()
    sorted_knockouts = sorted(results["knockouts"], key=lambda k: k["delta"])
    for k in sorted_knockouts:
        marker = "***" if k["delta"] < -5 else "**" if k["delta"] < -2 else ""
        print(f"  Block {k['block']:2d}: WR={k['win_rate']:.1f}% (delta={k['delta']:+.1f}%) {marker}")

    # Identify redundant blocks
    redundant = [k["block"] for k in results["knockouts"] if k["delta"] > -2]
    critical = [k["block"] for k in results["knockouts"] if k["delta"] < -5]
    print(f"\nRedundant blocks (delta > -2%): {redundant}")
    print(f"Critical blocks (delta < -5%): {critical}")
    print(f"Potential compression: {len(redundant)} of 10 blocks may be removable")


if __name__ == "__main__":
    main()
