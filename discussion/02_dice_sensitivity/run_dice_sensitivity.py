import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add the project root to the path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.common import advance_stuck_turn, load_checkpoint_model
import td_ludo_cpp as ludo_cpp

BASE_POS = -1
HOME_POS = 99
SAFE_INDICES = {0, 8, 13, 21, 26, 34, 39, 47}


def parse_args():
    parser = argparse.ArgumentParser(description="Dice sensitivity analysis for AlphaLudo.")
    parser.add_argument(
        "--weights",
        default="../../weights/model_latest_323k_shaped.pt",
        help="Path to the exported model checkpoint.",
    )
    parser.add_argument("--num-games", type=int, default=100, help="Parallel games to simulate.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max rollout steps per collection loop.",
    )
    parser.add_argument(
        "--global-target",
        type=int,
        default=300,
        help="Number of random decision states for the global average.",
    )
    parser.add_argument(
        "--bucket-target",
        type=int,
        default=200,
        help="Target count per curated bucket (rare buckets use a smaller target).",
    )
    parser.add_argument(
        "--bucket-sample-prob",
        type=float,
        default=0.3,
        help="Probability of evaluating bucket predicates on a decision state.",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=5000,
        help="Safety cap on collection loops.",
    )
    parser.add_argument("--grid-states", type=int, default=15, help="Per-state grid size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--skip-buckets",
        action="store_true",
        help="Skip curated bucket collection and run only the global average.",
    )
    parser.add_argument(
        "--skip-masked",
        action="store_true",
        help="Skip masked policy analysis.",
    )
    parser.add_argument(
        "--skip-unmasked",
        action="store_true",
        help="Skip unmasked (raw preference) analysis.",
    )
    return parser.parse_args()


def resolve_weights_path(raw_path):
    path = Path(raw_path)
    if path.exists():
        return str(path)
    return str((Path(__file__).resolve().parent / raw_path).resolve())


def load_model(weights_path):
    return load_checkpoint_model(weights_path)


def snapshot_state(game, state_tensor):
    return {
        "tensor": state_tensor.copy(),
        "player_positions": np.array(game.player_positions, dtype=np.int8).copy(),
        "scores": np.array(game.scores, dtype=np.int8).copy(),
        "active_players": np.array(game.active_players, dtype=bool).copy(),
        "current_player": int(game.current_player),
        "current_dice_roll": int(game.current_dice_roll),
    }


def build_game_state(sample, roll_override=None):
    state = ludo_cpp.GameState()
    state.player_positions = sample["player_positions"]
    state.scores = sample["scores"]
    state.active_players = sample["active_players"]
    state.current_player = sample["current_player"]
    state.current_dice_roll = (
        sample["current_dice_roll"] if roll_override is None else int(roll_override)
    )
    state.is_terminal = False
    return state


def get_absolute_pos(player, relative_pos):
    if relative_pos < 0 or relative_pos > 50:
        return None
    return (int(relative_pos) + 13 * int(player)) % 52


def token_progress(positions):
    return np.where(positions == HOME_POS, 56, positions)


def leading_token_index(sample):
    current_player = sample["current_player"]
    positions = sample["player_positions"][current_player].astype(int)
    return int(np.argmax(token_progress(positions)))


def leading_token_in_danger(sample):
    current_player = sample["current_player"]
    positions = sample["player_positions"].astype(int)
    lead_idx = leading_token_index(sample)
    lead_pos = positions[current_player, lead_idx]

    if lead_pos < 0 or lead_pos > 50:
        return False

    lead_abs = get_absolute_pos(current_player, lead_pos)
    if lead_abs in SAFE_INDICES:
        return False

    same_square_count = 0
    for token_idx in range(4):
        pos = positions[current_player, token_idx]
        if 0 <= pos <= 50 and get_absolute_pos(current_player, pos) == lead_abs:
            same_square_count += 1
    if same_square_count > 1:
        return False

    for player in range(4):
        if player == current_player or not sample["active_players"][player]:
            continue

        for token_pos in positions[player]:
            if token_pos < 0 or token_pos > 50:
                continue

            opp_abs = get_absolute_pos(player, token_pos)
            distance = (lead_abs - opp_abs) % 52
            if 1 <= distance <= 6 and token_pos + distance <= 50:
                return True

    return False


def home_stretch_count(sample):
    current_player = sample["current_player"]
    positions = sample["player_positions"][current_player].astype(int)
    return int(np.sum((positions >= 51) & (positions <= 55)))


def can_capture_this_turn(sample, roll_override=None):
    current_player = sample["current_player"]
    current_positions = sample["player_positions"]
    state = build_game_state(sample, roll_override=roll_override)

    for move in ludo_cpp.get_legal_moves(state):
        next_state = ludo_cpp.apply_move(state, int(move))
        next_positions = np.array(next_state.player_positions, dtype=np.int8)

        for player in range(4):
            if player == current_player or not sample["active_players"][player]:
                continue

            before = current_positions[player]
            after = next_positions[player]
            was_captured = (
                (before != BASE_POS)
                & (before != HOME_POS)
                & (after == BASE_POS)
            )
            if np.any(was_captured):
                return True

    return False


def capture_rolls(sample):
    rolls = set()
    for roll in range(1, 7):
        if can_capture_this_turn(sample, roll_override=roll):
            rolls.add(roll)
    return rolls


def default_bucket_defs():
    return {
        "roll_6": lambda s: s["current_dice_roll"] == 6,
        "roll_3": lambda s: s["current_dice_roll"] == 3,
        "capture_available": lambda s: can_capture_this_turn(s),
        "capture_roll_3_only": lambda s: capture_rolls(s) == {3},
        "leading_token_in_danger": lambda s: leading_token_in_danger(s),
        "home_stretch_2plus": lambda s: home_stretch_count(s) >= 2,
    }


def default_bucket_targets(base_target):
    return {
        "roll_6": base_target,
        "roll_3": base_target,
        "capture_available": base_target,
        "capture_roll_3_only": max(25, base_target // 4),
        "leading_token_in_danger": base_target,
        "home_stretch_2plus": base_target,
    }


def collect_states(
    num_games=100,
    max_steps=200,
    global_target=300,
    bucket_defs=None,
    bucket_targets=None,
    bucket_sample_prob=0.3,
    max_loops=5000,
    seed=0,
):
    print(f"Collecting {global_target} decision states...")
    rng = np.random.default_rng(seed)
    env = ludo_cpp.VectorGameState(batch_size=num_games, two_player_mode=True)

    collected_samples = []
    bucket_defs = bucket_defs or {}
    bucket_targets = bucket_targets or {}
    bucket_data = {
        name: {"samples": []} for name in bucket_defs.keys()
    }

    loop_count = 0
    while loop_count < max_loops:
        loop_count += 1
        if loop_count % 200 == 0:
            bucket_summary = ", ".join(
                f"{name}:{len(data['samples'])}/{bucket_targets.get(name, 0)}"
                for name, data in bucket_data.items()
            )
            print(
                f"  ...loop {loop_count}, global {len(collected_samples)}/{global_target}"
                + (f", buckets [{bucket_summary}]" if bucket_summary else "")
            )

        for i in range(num_games):
            game = env.get_game(i)
            if not game.is_terminal and game.current_dice_roll == 0:
                game.current_dice_roll = int(rng.integers(1, 7))

        legal_moves_batch = env.get_legal_moves()
        states_tensor = env.get_state_tensor()

        actions = []
        for i in range(num_games):
            moves = legal_moves_batch[i]
            game = env.get_game(i)

            if game.is_terminal:
                actions.append(-1)
                continue

            if not moves:
                advance_stuck_turn(game)
                actions.append(-1)
                continue

            actions.append(int(rng.choice(moves)))

            sample = snapshot_state(game, states_tensor[i])
            if len(collected_samples) < global_target:
                collected_samples.append(sample)

            if bucket_defs and rng.random() < bucket_sample_prob:
                if any(
                    len(bucket_data[name]["samples"]) < bucket_targets.get(name, 0)
                    for name in bucket_defs
                ):
                    for name, predicate in bucket_defs.items():
                        if len(bucket_data[name]["samples"]) >= bucket_targets.get(name, 0):
                            continue
                        if predicate(sample):
                            bucket_data[name]["samples"].append(sample)

        _, _, _, infos = env.step(actions)

        for i, info in enumerate(infos):
            if info["is_terminal"]:
                env.reset_game(i)

        global_done = len(collected_samples) >= global_target
        buckets_done = all(
            len(bucket_data[name]["samples"]) >= bucket_targets.get(name, 0)
            for name in bucket_defs
        )
        if global_done and buckets_done:
            break

    if loop_count >= max_loops:
        print("Reached max collection loops before filling all targets.")
        for name, data in bucket_data.items():
            target = bucket_targets.get(name, 0)
            if len(data["samples"]) < target:
                print(f"  Bucket '{name}' has {len(data['samples'])}/{target} samples.")

    return collected_samples, bucket_data


def legal_mask_for_roll(samples, roll):
    masks = np.zeros((len(samples), 4), dtype=np.float32)
    for idx, sample in enumerate(samples):
        state = build_game_state(sample, roll_override=roll)
        moves = ludo_cpp.get_legal_moves(state)
        for move in moves:
            masks[idx, int(move)] = 1.0
    return torch.tensor(masks, dtype=torch.float32)


def run_dice_sweep(model, samples, masked=False):
    num_states = len(samples)
    policy_distributions = np.zeros((num_states, 6, 4), dtype=np.float32)
    value_predictions = np.zeros((num_states, 6), dtype=np.float32)

    X_base = torch.tensor(np.stack([s["tensor"] for s in samples]), dtype=torch.float32)
    masks_by_roll = None
    if masked:
        masks_by_roll = [legal_mask_for_roll(samples, roll) for roll in range(1, 7)]

    dummy_mask = torch.ones((num_states, 4), dtype=torch.float32)

    for roll in range(1, 7):
        X_sweep = X_base.clone()
        X_sweep[:, 11:17, :, :] = 0.0
        channel_idx = 11 + (roll - 1)
        X_sweep[:, channel_idx, :, :] = 1.0

        with torch.no_grad():
            if masked:
                policy, value = model(X_sweep, masks_by_roll[roll - 1])
            else:
                policy, value = model(X_sweep, dummy_mask)

        policy_distributions[:, roll - 1, :] = policy.numpy()
        value_predictions[:, roll - 1] = value.squeeze(-1).numpy()

    return policy_distributions, value_predictions


def entropy(p):
    eps = 1e-8
    return -np.sum(p * np.log(p + eps), axis=-1)


def js_divergence(p, q):
    eps = 1e-8
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)), axis=-1)
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)), axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def compute_metrics(policy_distributions, value_predictions):
    num_states = policy_distributions.shape[0]
    roll_entropies = entropy(policy_distributions)
    roll_top_probs = policy_distributions.max(axis=2)

    js_pairs = []
    js_vs_roll1 = []
    for r1 in range(6):
        for r2 in range(r1 + 1, 6):
            js_pairs.append(js_divergence(policy_distributions[:, r1], policy_distributions[:, r2]))
        if r1 > 0:
            js_vs_roll1.append(js_divergence(policy_distributions[:, 0], policy_distributions[:, r1]))

    js_pairs = np.stack(js_pairs, axis=0)
    js_vs_roll1 = np.stack(js_vs_roll1, axis=0) if js_vs_roll1 else np.zeros((1, num_states))

    argmax = policy_distributions.argmax(axis=2)
    flip_any = np.any(argmax != argmax[:, :1], axis=1)
    flip_roll6 = argmax[:, 5] != argmax[:, 0]

    value_std = value_predictions.std(axis=1)
    value_range = value_predictions.max(axis=1) - value_predictions.min(axis=1)

    return {
        "num_states": int(num_states),
        "flip_any_roll": int(flip_any.sum()),
        "flip_roll6_vs_roll1": int(flip_roll6.sum()),
        "roll_entropy_mean": roll_entropies.mean(axis=0).tolist(),
        "roll_topprob_mean": roll_top_probs.mean(axis=0).tolist(),
        "js_pairwise_mean": float(js_pairs.mean()),
        "js_pairwise_std": float(js_pairs.std()),
        "js_vs_roll1_mean": float(js_vs_roll1.mean()),
        "value_std_mean": float(value_std.mean()),
        "value_range_mean": float(value_range.mean()),
    }


def visualize_grid(policy_distributions, value_predictions, save_path, title):
    num_states = policy_distributions.shape[0]
    fig, axes = plt.subplots(nrows=num_states, ncols=1, figsize=(10, 2 * num_states))
    fig.suptitle(title, fontsize=16, y=0.99)

    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
    labels = ["Token 0", "Token 1", "Token 2", "Token 3"]

    for s in range(num_states):
        ax = axes[s] if num_states > 1 else axes
        pol = policy_distributions[s]
        x = np.arange(1, 7)
        bottoms = np.zeros(6)

        for t in range(4):
            ax.bar(
                x,
                pol[:, t],
                bottom=bottoms,
                color=colors[t],
                edgecolor="white",
                label=labels[t] if s == 0 else "",
            )
            bottoms += pol[:, t]

        for r in range(6):
            v = value_predictions[s, r]
            ax.text(x[r], 1.05, f"C:{v:+.2f}", ha="center", va="bottom", fontsize=8, rotation=45)

        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel(f"State {s + 1}")
        ax.set_xticks(x)
        if s == num_states - 1:
            ax.set_xlabel("Simulated Dice Roll")
        else:
            ax.set_xticklabels([])

    if num_states > 1:
        fig.legend(labels, loc="upper right", bbox_to_anchor=(0.95, 0.98), ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


def visualize_avg(policy_distributions, value_predictions, save_path, title):
    mean_policy = policy_distributions.mean(axis=0)
    mean_value = value_predictions.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
    labels = ["Token 0", "Token 1", "Token 2", "Token 3"]

    x = np.arange(1, 7)
    bottoms = np.zeros(6)
    for t in range(4):
        ax.bar(x, mean_policy[:, t], bottom=bottoms, color=colors[t], edgecolor="white", label=labels[t])
        bottoms += mean_policy[:, t]

    for r in range(6):
        ax.text(x[r], 1.05, f"C:{mean_value[r]:+.2f}", ha="center", va="bottom", fontsize=9, rotation=45)

    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Mean Policy Probability")
    ax.set_xlabel("Simulated Dice Roll")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


def save_metrics(metrics, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"Saved metrics to {save_path}")


def run_analysis(model, samples, output_prefix, grid_states, masked, title_prefix):
    policy, value = run_dice_sweep(model, samples, masked=masked)
    metrics = compute_metrics(policy, value)

    grid_count = min(grid_states, len(samples))
    if grid_count > 0:
        grid_idx = np.random.choice(len(samples), grid_count, replace=False)
        visualize_grid(
            policy[grid_idx],
            value[grid_idx],
            f"{output_prefix}_grid.png",
            f"{title_prefix} (Grid {grid_count})",
        )

    visualize_avg(
        policy,
        value,
        f"{output_prefix}_avg.png",
        f"{title_prefix} (Mean)",
    )

    return metrics


def main():
    args = parse_args()
    np.random.seed(args.seed)

    weights_path = resolve_weights_path(args.weights)
    model = load_model(weights_path)

    bucket_defs = {} if args.skip_buckets else default_bucket_defs()
    bucket_targets = {} if args.skip_buckets else default_bucket_targets(args.bucket_target)

    samples, bucket_data = collect_states(
        num_games=args.num_games,
        max_steps=args.max_steps,
        global_target=args.global_target,
        bucket_defs=bucket_defs,
        bucket_targets=bucket_targets,
        bucket_sample_prob=args.bucket_sample_prob,
        max_loops=args.max_loops,
        seed=args.seed,
    )

    output_dir = Path(__file__).resolve().parent
    metrics = {"global": {}, "buckets": {}}

    if not args.skip_unmasked:
        metrics["global"]["unmasked"] = run_analysis(
            model,
            samples,
            str(output_dir / "dice_sensitivity_global_unmasked"),
            args.grid_states,
            masked=False,
            title_prefix="Global Dice Sensitivity (Unmasked)",
        )

    if not args.skip_masked:
        metrics["global"]["masked"] = run_analysis(
            model,
            samples,
            str(output_dir / "dice_sensitivity_global_masked"),
            args.grid_states,
            masked=True,
            title_prefix="Global Dice Sensitivity (Masked)",
        )

    if bucket_data:
        for name, data in bucket_data.items():
            bucket_samples = data["samples"]
            if not bucket_samples:
                print(f"Skipping bucket '{name}' - no samples collected.")
                continue

            metrics["buckets"][name] = {}
            if not args.skip_unmasked:
                metrics["buckets"][name]["unmasked"] = run_analysis(
                    model,
                    bucket_samples,
                    str(output_dir / f"dice_sensitivity_{name}_unmasked"),
                    min(args.grid_states, len(bucket_samples)),
                    masked=False,
                    title_prefix=f"{name} Dice Sensitivity (Unmasked)",
                )
            if not args.skip_masked:
                metrics["buckets"][name]["masked"] = run_analysis(
                    model,
                    bucket_samples,
                    str(output_dir / f"dice_sensitivity_{name}_masked"),
                    min(args.grid_states, len(bucket_samples)),
                    masked=True,
                    title_prefix=f"{name} Dice Sensitivity (Masked)",
                )

    save_metrics(metrics, output_dir / "dice_sensitivity_metrics.json")


if __name__ == "__main__":
    main()
