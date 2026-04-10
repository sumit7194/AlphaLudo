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
    parser = argparse.ArgumentParser(description="Run channel ablation on AlphaLudo inputs.")
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
        default=500,
        help="Number of random decision states for the global average plot.",
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--skip-buckets",
        action="store_true",
        help="Skip curated bucket collection and run only the global average.",
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
    max_steps_per_game=200,
    global_target=500,
    bucket_defs=None,
    bucket_targets=None,
    bucket_sample_prob=0.3,
    max_loops=5000,
    seed=0,
):
    print(f"Collecting {global_target} random decision states...")
    rng = np.random.default_rng(seed)
    env = ludo_cpp.VectorGameState(batch_size=num_games, two_player_mode=True)

    collected_states = []
    collected_masks = []

    bucket_defs = bucket_defs or {}
    bucket_targets = bucket_targets or {}
    bucket_data = {
        name: {"states": [], "masks": []} for name in bucket_defs.keys()
    }

    loop_count = 0
    while loop_count < max_loops:
        loop_count += 1
        if loop_count % 200 == 0:
            bucket_summary = ", ".join(
                f"{name}:{len(data['states'])}/{bucket_targets.get(name, 0)}"
                for name, data in bucket_data.items()
            )
            print(
                f"  ...loop {loop_count}, global {len(collected_states)}/{global_target}"
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

            action = int(rng.choice(moves))
            actions.append(action)

            # Create legal mask once for this state
            mask = np.zeros(4, dtype=np.float32)
            for m in moves:
                mask[m] = 1.0

            if len(collected_states) < global_target:
                collected_states.append(states_tensor[i].copy())
                collected_masks.append(mask)

            if bucket_defs and rng.random() < bucket_sample_prob:
                if any(
                    len(bucket_data[name]["states"]) < bucket_targets.get(name, 0)
                    for name in bucket_defs
                ):
                    sample = snapshot_state(game, states_tensor[i])
                    for name, predicate in bucket_defs.items():
                        if len(bucket_data[name]["states"]) >= bucket_targets.get(name, 0):
                            continue
                        if predicate(sample):
                            bucket_data[name]["states"].append(states_tensor[i].copy())
                            bucket_data[name]["masks"].append(mask)

        _, _, _, infos = env.step(actions)

        for i, info in enumerate(infos):
            if info["is_terminal"]:
                env.reset_game(i)

        global_done = len(collected_states) >= global_target
        buckets_done = all(
            len(bucket_data[name]["states"]) >= bucket_targets.get(name, 0)
            for name in bucket_defs
        )
        if global_done and buckets_done:
            break

    if loop_count >= max_loops:
        print("Reached max collection loops before filling all targets.")
        for name, data in bucket_data.items():
            target = bucket_targets.get(name, 0)
            if len(data["states"]) < target:
                print(f"  Bucket '{name}' has {len(data['states'])}/{target} samples.")

    max_states = min(global_target, len(collected_states))
    indices = rng.choice(len(collected_states), max_states, replace=False)

    X = torch.tensor(np.array([collected_states[i] for i in indices]), dtype=torch.float32)
    masks = torch.tensor(np.array([collected_masks[i] for i in indices]), dtype=torch.float32)

    return X, masks, bucket_data


def kl_divergence(p, q):
    # p and q are batches of probability distributions (B, 4)
    eps = 1e-8
    p_safe = p + eps
    q_safe = q + eps
    return torch.sum(p * (torch.log(p_safe) - torch.log(q_safe)), dim=1)


def run_ablation(model, X, masks):
    print("Running baseline predictions...")
    with torch.no_grad():
        baseline_policy, baseline_value = model(X, masks)

    num_channels = X.shape[1]
    policy_kl_impacts = []
    value_mae_impacts = []

    print(f"Ablating {num_channels} channels individually...")
    for c in range(num_channels):
        X_ablated = X.clone()
        X_ablated[:, c, :, :] = 0.0

        with torch.no_grad():
            ablated_policy, ablated_value = model(X_ablated, masks)

        kl = kl_divergence(baseline_policy, ablated_policy).mean().item()
        v_mae = torch.abs(baseline_value - ablated_value).mean().item()

        policy_kl_impacts.append(kl)
        value_mae_impacts.append(v_mae)
        print(f"  Channel {c:2d} -> Policy KL: {kl:.4f}, Value MAE: {v_mae:.4f}")

    return policy_kl_impacts, value_mae_impacts


def visualize(policy_impacts, value_impacts, save_path, title_suffix=""):
    print(f"Generating visualization at {save_path}...")

    channel_names = [
        "0: My Token 0",
        "1: My Token 1",
        "2: My Token 2",
        "3: My Token 3",
        "4: Opp Density",
        "5: Safe Zones",
        "6: My Home Path",
        "7: Opp Home Path",
        "8: Score Diff",
        "9: My Locked %",
        "10: Opp Locked %",
        "11: Dice = 1",
        "12: Dice = 2",
        "13: Dice = 3",
        "14: Dice = 4",
        "15: Dice = 5",
        "16: Dice = 6",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    y_pos = np.arange(len(channel_names))

    ax1.barh(y_pos, policy_impacts, align="center", color="skyblue")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(channel_names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Average KL Divergence")
    ax1.set_title(f"Impact on Policy Decision{title_suffix}")
    ax1.grid(axis="x", linestyle="--", alpha=0.7)

    ax2.barh(y_pos, value_impacts, align="center", color="salmon")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(channel_names)
    ax2.invert_yaxis()
    ax2.set_xlabel("Average Absolute Critic Shift")
    ax2.set_title(f"Impact on Critic Output{title_suffix}")
    ax2.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print("Done!")


def save_metrics(metrics, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"Saved metrics to {save_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    weights_path = resolve_weights_path(args.weights)
    model = load_model(weights_path)

    bucket_defs = {} if args.skip_buckets else default_bucket_defs()
    bucket_targets = {} if args.skip_buckets else default_bucket_targets(args.bucket_target)

    X, masks, bucket_data = collect_states(
        num_games=args.num_games,
        max_steps_per_game=args.max_steps,
        global_target=args.global_target,
        bucket_defs=bucket_defs,
        bucket_targets=bucket_targets,
        bucket_sample_prob=args.bucket_sample_prob,
        max_loops=args.max_loops,
        seed=args.seed,
    )

    metrics = {}
    p_impact, v_impact = run_ablation(model, X, masks)
    metrics["global"] = {
        "num_samples": int(len(X)),
        "policy_kl": p_impact,
        "critic_mae": v_impact,
    }

    output_dir = Path(__file__).resolve().parent
    visualize(
        p_impact,
        v_impact,
        output_dir / "channel_ablation_results.png",
        title_suffix=" (Global)",
    )

    if bucket_data:
        metrics["buckets"] = {}
        for name, data in bucket_data.items():
            if not data["states"]:
                print(f"Skipping bucket '{name}' - no samples collected.")
                continue
            X_bucket = torch.tensor(np.array(data["states"]), dtype=torch.float32)
            masks_bucket = torch.tensor(np.array(data["masks"]), dtype=torch.float32)
            print(f"Running ablation for bucket '{name}' with {len(X_bucket)} samples...")
            p_bucket, v_bucket = run_ablation(model, X_bucket, masks_bucket)
            metrics["buckets"][name] = {
                "num_samples": int(len(X_bucket)),
                "policy_kl": p_bucket,
                "critic_mae": v_bucket,
            }
            save_path = output_dir / f"channel_ablation_results_{name}.png"
            visualize(p_bucket, v_bucket, save_path, title_suffix=f" ({name})")

    save_metrics(metrics, output_dir / "channel_ablation_metrics.json")


if __name__ == "__main__":
    main()
