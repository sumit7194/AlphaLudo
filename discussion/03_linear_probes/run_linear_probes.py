import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Add the project root to the path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.common import advance_stuck_turn, load_checkpoint_model
import td_ludo_cpp as ludo_cpp

BASE_POS = -1
HOME_POS = 99
SAFE_INDICES = {0, 8, 13, 21, 26, 34, 39, 47}


def parse_args():
    parser = argparse.ArgumentParser(description="Run linear probes on AlphaLudo GAP features.")
    parser.add_argument(
        "--weights",
        default="../../weights/model_latest_323k_shaped.pt",
        help="Path to the exported model checkpoint.",
    )
    parser.add_argument("--num-games", type=int, default=100, help="Parallel games to simulate.")
    parser.add_argument("--target-states", type=int, default=2500, help="Final labeled states to keep.")
    parser.add_argument(
        "--sample-prob",
        type=float,
        default=0.5,
        help="Probability of keeping a decision state inside each finished game.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=512,
        help="Safety cap before resetting an unusually long rollout.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Feature extraction batch size.")
    parser.add_argument("--device", default="cpu", help="Torch device for feature extraction.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip saving the summary figure.")
    return parser.parse_args()


def resolve_weights_path(raw_path):
    path = Path(raw_path)
    if path.exists():
        return str(path)
    return str((Path(__file__).resolve().parent / raw_path).resolve())


def get_absolute_pos(player, relative_pos):
    if relative_pos < 0 or relative_pos > 50:
        return None
    return (int(relative_pos) + 13 * int(player)) % 52


def snapshot_state(game, state_tensor):
    return {
        "tensor": state_tensor.copy(),
        "player_positions": np.array(game.player_positions, dtype=np.int8).copy(),
        "scores": np.array(game.scores, dtype=np.int8).copy(),
        "active_players": np.array(game.active_players, dtype=bool).copy(),
        "current_player": int(game.current_player),
        "current_dice_roll": int(game.current_dice_roll),
    }


def build_game_state(sample):
    state = ludo_cpp.GameState()
    state.player_positions = sample["player_positions"]
    state.scores = sample["scores"]
    state.active_players = sample["active_players"]
    state.current_player = sample["current_player"]
    state.current_dice_roll = sample["current_dice_roll"]
    state.is_terminal = False
    return state


def token_progress(positions):
    return np.where(positions == HOME_POS, 56, positions)


def leading_token_index(sample):
    current_player = sample["current_player"]
    positions = sample["player_positions"][current_player].astype(int)
    return int(np.argmax(token_progress(positions)))


def can_capture_this_turn(sample):
    current_player = sample["current_player"]
    current_positions = sample["player_positions"]
    state = build_game_state(sample)

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


def closest_token_to_home(sample):
    current_player = sample["current_player"]
    positions = sample["player_positions"][current_player].astype(int)
    return int(np.argmax(token_progress(positions)))


def compute_labels(samples):
    print("Computing labels from raw game-state snapshots...")
    labels = {
        "can_capture_this_turn": np.array([can_capture_this_turn(sample) for sample in samples]),
        "leading_token_in_danger": np.array([leading_token_in_danger(sample) for sample in samples]),
        "home_stretch_count": np.array([home_stretch_count(sample) for sample in samples]),
        "eventual_win": np.array([sample["eventual_win"] for sample in samples]),
        "closest_token_to_home": np.array([closest_token_to_home(sample) for sample in samples]),
    }

    for concept, values in labels.items():
        unique, counts = np.unique(values, return_counts=True)
        summary = ", ".join(f"{u}:{c}" for u, c in zip(unique.tolist(), counts.tolist()))
        print(f"  {concept}: {summary}")

    return labels


def collect_labeled_dataset(
    num_games=100,
    target_states=2500,
    sample_prob=0.5,
    max_episode_steps=512,
    seed=0,
):
    print(f"Collecting {target_states} labeled states from finished games...")
    rng = np.random.default_rng(seed)
    env = ludo_cpp.VectorGameState(batch_size=num_games, two_player_mode=True)

    pending_samples = [[] for _ in range(num_games)]
    finalized_samples = []
    episode_steps = np.zeros(num_games, dtype=int)

    loop_count = 0
    while len(finalized_samples) < target_states:
        loop_count += 1
        if loop_count % 100 == 0:
            print(f"  ...loop {loop_count}, finalized {len(finalized_samples)}")

        for game_idx in range(num_games):
            game = env.get_game(game_idx)
            if not game.is_terminal and game.current_dice_roll == 0:
                game.current_dice_roll = int(rng.integers(1, 7))

        legal_moves_batch = env.get_legal_moves()
        states_tensor = env.get_state_tensor()

        actions = []
        for game_idx in range(num_games):
            game = env.get_game(game_idx)
            moves = legal_moves_batch[game_idx]

            if game.is_terminal:
                actions.append(-1)
                continue

            if not moves:
                advance_stuck_turn(game)
                actions.append(-1)
                continue

            actions.append(int(rng.choice(moves)))

            if rng.random() < sample_prob:
                pending_samples[game_idx].append(snapshot_state(game, states_tensor[game_idx]))

        _, _, _, infos = env.step(actions)
        episode_steps += 1

        for game_idx, info in enumerate(infos):
            if info["is_terminal"]:
                winner = int(info["winner"])
                for sample in pending_samples[game_idx]:
                    sample["eventual_win"] = winner == sample["current_player"]
                    finalized_samples.append(sample)

                pending_samples[game_idx].clear()
                env.reset_game(game_idx)
                episode_steps[game_idx] = 0

            elif episode_steps[game_idx] > max_episode_steps:
                pending_samples[game_idx].clear()
                env.reset_game(game_idx)
                episode_steps[game_idx] = 0

    samples = finalized_samples[:target_states]
    X = torch.tensor(np.stack([sample["tensor"] for sample in samples]), dtype=torch.float32)
    labels = compute_labels(samples)
    print(f"Collected {len(samples)} states with tensor shape {tuple(X.shape)}")
    return X, labels


def extract_features(model, X, batch_size=256, device="cpu"):
    print("Extracting GAP features from the backbone...")
    features = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size].to(device)
            features.append(model._backbone(batch).cpu().numpy())

    features = np.vstack(features)
    print(f"Feature matrix shape: {features.shape}")
    return features


def train_probes(features, labels, seed=0):
    print("Training linear probes...")
    results = {}

    for concept, y in labels.items():
        valid_mask = np.ones(len(y), dtype=bool)
        unique_classes, counts = np.unique(y, return_counts=True)
        rare_classes = unique_classes[counts < 2]
        if len(rare_classes) > 0:
            for rare_class in rare_classes:
                valid_mask &= y != rare_class
            print(
                f"  Dropping rare classes for '{concept}': "
                f"{', '.join(str(cls) for cls in rare_classes.tolist())}"
            )

        X_concept = features[valid_mask]
        y_concept = y[valid_mask]
        unique_classes, counts = np.unique(y_concept, return_counts=True)
        if len(unique_classes) < 2:
            print(f"  Skipping '{concept}' - only one class present.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_concept,
            y_concept,
            test_size=0.2,
            random_state=seed,
            stratify=y_concept,
        )

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        majority_baseline = counts.max() / counts.sum()

        results[concept] = {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "baseline": float(majority_baseline),
            "num_classes": int(len(unique_classes)),
            "class_counts": {str(cls): int(count) for cls, count in zip(unique_classes, counts)},
        }
        print(
            f"  {concept}: accuracy={accuracy:.3f}, "
            f"balanced_acc={balanced_accuracy:.3f}, baseline={majority_baseline:.3f}"
        )

    return results


def save_metrics(results, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Saved metrics to {save_path}")


def visualize_probes(results, save_path):
    if not results:
        print("No valid probe results to visualize.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concepts = list(results.keys())
    accuracies = [results[concept]["accuracy"] for concept in concepts]
    baselines = [results[concept]["baseline"] for concept in concepts]
    balanced = [results[concept]["balanced_accuracy"] for concept in concepts]

    x = np.arange(len(concepts))
    width = 0.28

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, accuracies, width, label="Probe Accuracy", color="darkorange")
    ax.bar(x, balanced, width, label="Balanced Accuracy", color="steelblue")
    ax.bar(x + width, baselines, width, label="Majority Baseline", color="lightgray")

    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Linear Probe Accuracy on 128-dim GAP Features")
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=15, ha="right")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    weights_path = resolve_weights_path(args.weights)
    model = load_checkpoint_model(weights_path, device=args.device)

    X, labels = collect_labeled_dataset(
        num_games=args.num_games,
        target_states=args.target_states,
        sample_prob=args.sample_prob,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )

    features = extract_features(
        model,
        X,
        batch_size=args.batch_size,
        device=args.device,
    )
    results = train_probes(features, labels, seed=args.seed)

    output_dir = Path(__file__).resolve().parent
    save_metrics(results, output_dir / "linear_probe_metrics.json")

    if not args.skip_plot:
        visualize_probes(results, output_dir / "linear_probe_results.png")


if __name__ == "__main__":
    main()
