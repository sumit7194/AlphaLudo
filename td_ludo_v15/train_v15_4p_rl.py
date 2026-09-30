"""AlphaLudo 4-Player PPO Reinforcement Learning Pipeline.

Features:
1. PPO clipped surrogate loss on 225-cell source-cell policy.
2. 4-way Categorical Value Head trained with CrossEntropy against 1-hot winner.
3. Strict storage safety: sliding window of max 3 ghost snapshots on disk (pruning old ones).
4. Dual live dashboard integration (dashboard_4p.html) on HTTP port.
5. Mac MPS acceleration support.
"""
from __future__ import annotations

import argparse
import collections
import http.server
import json
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V15_ROOT = _REPO_ROOT / "td_ludo_v15"
_TD_ROOT = _REPO_ROOT / "td_ludo"

for p in (str(_V15_ROOT), str(_TD_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import td_ludo_cpp as cpp
from td_ludo.game.heuristic_bot import ExpertBot, HeuristicLudoBot
from td_ludo.game.strong_bots import ExpectimaxBot
from td_ludo.game.strong_bots_v2 import AggressiveExpectimaxBot
from td_ludo.game.reward_shaping import compute_shaped_reward

from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS,
    cell_to_index,
    index_to_cell,
    position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder_4p import encode_frame_4p
from td_ludo_v15.models.v15_4p import V15_4P_GraphTransformer

_BASE_POS = -1
_HOME_POS = 99
GAMMA = 0.99
MAX_GHOSTS = 3


# ─── 1. Ghost Snapshot Manager (Capped to MAX_GHOSTS on disk) ────────────────

class GhostManager:
    """Manages frozen past self-play checkpoints on disk with a strict sliding window."""
    def __init__(self, ghost_dir: Path, max_ghosts: int = MAX_GHOSTS):
        self.ghost_dir = ghost_dir
        self.max_ghosts = max_ghosts
        self.ghost_files: List[Path] = []
        self.ghost_dir.mkdir(parents=True, exist_ok=True)
        # Scan existing ghosts
        existing = sorted(self.ghost_dir.glob("ghost_*.pt"), key=os.path.getmtime)
        for f in existing:
            self.ghost_files.append(f)
        self._prune()

    def _prune(self):
        while len(self.ghost_files) > self.max_ghosts:
            oldest = self.ghost_files.pop(0)
            if oldest.exists():
                try:
                    oldest.unlink()
                except OSError:
                    pass

    def add_snapshot(self, model: nn.Module, game_count: int) -> Path:
        ghost_path = self.ghost_dir / f"ghost_{game_count}.pt"
        torch.save(model.state_dict(), ghost_path)
        self.ghost_files.append(ghost_path)
        self._prune()
        return ghost_path

    def sample_ghost_weights(self) -> Optional[dict]:
        if not self.ghost_files:
            return None
        target = random.choice(self.ghost_files)
        try:
            return torch.load(target, map_location="cpu", weights_only=True)
        except Exception:
            return None


# ─── 2. 4P PPO Trainer ───────────────────────────────────────────────────────

class V15_4P_PPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 3e-5,
        ppo_clip: float = 0.2,
        ppo_epochs: int = 3,
        ppo_buffer_games: int = 40,
        minibatch_size: int = 256,
        entropy_coeff: float = 0.02,
        value_coeff: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.ppo_buffer_games = ppo_buffer_games
        self.minibatch_size = minibatch_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

        self.buffer: List[dict] = []
        self.games_buffered = 0

    def add_trajectory(self, trajectory: List[dict], winner: int, student_seat: int):
        if not trajectory:
            return

        won = (winner == student_seat)
        terminal_return = 1.0 if won else -0.33

        # Compute discounted returns
        R = terminal_return
        for step in reversed(trajectory):
            R = step.get("step_reward", 0.0) + GAMMA * R
            step["return"] = R
            step["target_winner"] = (winner - step["mover"]) % 4
            self.buffer.append(step)

        self.games_buffered += 1

    def update_if_ready(self) -> Optional[dict]:
        if self.games_buffered < self.ppo_buffer_games or len(self.buffer) < self.minibatch_size:
            return None

        # Prepare batch tensors
        N = len(self.buffer)
        frames = np.stack([s["frame"] for s in self.buffer])            # (N, 15, 15, 5)
        masks = np.stack([s["legal_mask"] for s in self.buffer])        # (N, 225)
        actions = np.array([s["action_cell"] for s in self.buffer], dtype=np.int64)
        old_log_probs = np.array([s["old_log_prob"] for s in self.buffer], dtype=np.float32)
        returns = np.array([s["return"] for s in self.buffer], dtype=np.float32)
        target_winners = np.array([s["target_winner"] for s in self.buffer], dtype=np.int64)

        # Baseline advantage computation
        adv = returns - returns.mean()
        std = adv.std() + 1e-8
        adv = adv / std

        frames_t = torch.from_numpy(frames).to(self.device)
        masks_t = torch.from_numpy(masks).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        old_lp_t = torch.from_numpy(old_log_probs).to(self.device)
        adv_t = torch.from_numpy(adv).to(self.device)
        win_t = torch.from_numpy(target_winners).to(self.device)

        policy_losses = []
        policy_confs = []
        value_losses = []
        val_accs = []
        entropies = []

        self.model.train()
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(N)
            for start in range(0, N, self.minibatch_size):
                end = min(start + self.minibatch_size, N)
                mb_idx = indices[start:end]

                mb_x = frames_t[mb_idx]
                mb_mask = masks_t[mb_idx]
                mb_act = actions_t[mb_idx]
                mb_old_lp = old_lp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_win = win_t[mb_idx]

                self.optimizer.zero_grad(set_to_none=True)
                policy, value, pol_logits, val_logits = self.model(
                    mb_x, legal_mask=mb_mask, return_logits=True
                )

                # Policy ratio & clipped loss
                dist = torch.distributions.Categorical(probs=policy + 1e-12)
                new_lp = dist.log_prob(mb_act)
                ratio = torch.exp(new_lp - mb_old_lp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_adv
                loss_policy = -torch.min(surr1, surr2).mean()

                # Value loss: CrossEntropy with 4-way winner target
                loss_value = F.cross_entropy(val_logits, mb_win)

                # Entropy bonus
                entropy = dist.entropy().mean()
                total_loss = loss_policy + self.value_coeff * loss_value - self.entropy_coeff * entropy

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                policy_losses.append(loss_policy.item())
                value_losses.append(loss_value.item())
                entropies.append(entropy.item())

                with torch.no_grad():
                    conf = policy.max(dim=-1).values.mean().item()
                    v_acc = (val_logits.argmax(dim=-1) == mb_win).float().mean().item()
                    policy_confs.append(conf)
                    val_accs.append(v_acc)

        self.buffer.clear()
        self.games_buffered = 0

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "policy_acc": float(np.mean(policy_confs)),
            "value_loss": float(np.mean(value_losses)),
            "val_acc": float(np.mean(val_accs)),
            "entropy": float(np.mean(entropies)),
            "transitions_trained": N,
        }


# ─── 3. Rollout Worker (Self-Play, Ghosts & Bots) ──────────────────────────────

def play_rl_episode(
    live_model: nn.Module,
    ghost_model: Optional[nn.Module],
    device: torch.device,
    use_shaped_reward: bool = True,
) -> Tuple[List[dict], int, int]:
    """Plays 1 game. Student is assigned a random seat (0..3).
    Other seats are assigned to Live Self-Play, Ghost Model, or ExpertBot.
    Returns (student_trajectory, winner_seat, student_seat).
    """
    student_seat = random.randint(0, 3)

    # Opponent setup
    opp_types = {}
    for s in range(4):
        if s == student_seat:
            opp_types[s] = "student"
        else:
            r = random.random()
            if r < 0.40:
                opp_types[s] = "self"
            elif r < 0.70 and ghost_model is not None:
                opp_types[s] = "ghost"
            else:
                opp_types[s] = "bot"

    bot_instances = {
        s: ExpertBot(player_id=s) if random.random() < 0.6 else HeuristicLudoBot(player_id=s)
        for s, t in opp_types.items() if t == "bot"
    }

    state = cpp.create_initial_state()
    consec_sixes = [0, 0, 0, 0]
    mc = 0
    trajectory: List[dict] = []
    prev_scores = [0, 0, 0, 0]

    while not state.is_terminal and mc < 800:
        cp = int(state.current_player)

        if state.current_dice_roll == 0:
            d = random.randint(1, 6)
            if d == 6:
                consec_sixes[cp] += 1
                if consec_sixes[cp] >= 3:
                    consec_sixes[cp] = 0
                    state.current_player = (cp + 1) % 4
                    state.current_dice_roll = 0
                    continue
            else:
                consec_sixes[cp] = 0
            state.current_dice_roll = d

        legal = cpp.get_legal_moves(state)
        if not legal:
            state.current_player = (cp + 1) % 4
            state.current_dice_roll = 0
            continue

        ptype = opp_types[cp]

        if ptype in ("student", "self", "ghost"):
            m = ghost_model if ptype == "ghost" else live_model
            m.eval()
            frame = encode_frame_4p(state, pov_player=cp)
            x_t = torch.from_numpy(frame).unsqueeze(0).to(device)

            legal_mask = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
            for t in legal:
                pos = int(state.player_positions[cp][t])
                c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                legal_mask[cell_to_index(*c)] = 1.0
            mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device)

            with torch.no_grad():
                policy, _ = m(x_t, legal_mask=mask_t)

            probs = policy[0].cpu().numpy()
            probs = probs / (probs.sum() + 1e-12)

            if ptype == "student":
                # Sample with temperature for exploration
                cell_idx = int(np.random.choice(NUM_BOARD_CELLS, p=probs))
                log_prob = float(np.log(probs[cell_idx] + 1e-12))
            else:
                cell_idx = int(probs.argmax())
                log_prob = 0.0

            chosen_cell = index_to_cell(cell_idx)

            chosen_token = legal[0]
            for t in legal:
                pos = int(state.player_positions[cp][t])
                c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                if c == chosen_cell:
                    chosen_token = t
                    break
            action = chosen_token
        else:
            action = bot_instances[cp].select_move(state, list(legal))

        state_before = state
        state = cpp.apply_move(state, int(action))
        mc += 1

        if ptype == "student":
            step_reward = 0.0
            if use_shaped_reward and compute_shaped_reward is not None:
                step_reward = float(compute_shaped_reward(state_before, state, cp))
            else:
                cur_score = int(state.scores[cp])
                if cur_score > prev_scores[cp]:
                    step_reward += 0.5 * (cur_score - prev_scores[cp])
                    prev_scores[cp] = cur_score

            trajectory.append({
                "frame": frame,
                "legal_mask": legal_mask,
                "action_cell": cell_idx,
                "old_log_prob": log_prob,
                "mover": cp,
                "step_reward": step_reward,
            })

    winner = int(cpp.get_winner(state)) if state.is_terminal else -1
    return trajectory, winner, student_seat


# ─── 4. Evaluation ────────────────────────────────────────────────────────────

def evaluate_rl_agent(model: nn.Module, device: torch.device, n_games: int = 40) -> float:
    from train_v15_4p_online_sl import evaluate_student_vs_bots
    return evaluate_student_vs_bots(model, device, n_games=n_games)


# ─── 5. Main RL Loop ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AlphaLudo 4-Player PPO RL Trainer")
    parser.add_argument("--init", default=None, help="Initial SL checkpoint to resume from")
    parser.add_argument("--total-games", type=int, default=0, help="Total games to train (0 = run indefinitely until stopped)")
    parser.add_argument("--lr", type=float, default=3e-5, help="PPO learning rate")
    parser.add_argument("--port", type=int, default=8845, help="Dashboard port")
    parser.add_argument("--device", default=None, help="Device (mps, cpu, cuda)")
    parser.add_argument("--ghost-interval", type=int, default=3000, help="Games between ghost snapshots")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Games between evaluations")
    parser.add_argument("--save-dir", default="checkpoints/v15_4p_rl", help="Directory for checkpoints")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mode_str = f"Continuous (Infinite until stopped)" if args.total_games == 0 else f"{args.total_games:,} games"
    print(f"🚀 AlphaLudo 4-Player PPO RL Training [{mode_str}]")
    print(f"💻 Device: {device} | Ghost Window: Max {MAX_GHOSTS} snapshots")
    print(f"🌐 Dashboard will be live at: http://localhost:{args.port}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ghost_mgr = GhostManager(save_dir / "ghosts", max_ghosts=MAX_GHOSTS)

    # Initialize model
    model = V15_4P_GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256)
    start_game = 1
    total_states_init = 0
    batches_init = 0
    best_wr = 25.0

    meta_file = save_dir / "metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            start_game = meta.get("games_played", 0) + 1
            total_states_init = meta.get("total_states", 0)
            batches_init = meta.get("batches", 0)
            best_wr = meta.get("best_eval_wr", 25.0)
            print(f"📖 Loaded session metadata: resuming from Game {start_game:,} (States: {total_states_init:,}, Batches: {batches_init:,}, Best WR: {best_wr:.1f}%)")
        except Exception as e:
            print(f"⚠️ Could not read metadata: {e}")

    if (save_dir / "model_latest.pt").exists() and (args.init is None or not Path(args.init).exists() or meta_file.exists()):
        print(f"🔄 Resuming weights from: {save_dir / 'model_latest.pt'}")
        st = torch.load(save_dir / "model_latest.pt", map_location=device, weights_only=True)
        model.load_state_dict(st)
    elif args.init and Path(args.init).exists():
        print(f"📥 Loading initial weights from: {args.init}")
        st = torch.load(args.init, map_location=device, weights_only=True)
        model.load_state_dict(st)
    else:
        print("🌱 Initializing with fresh random weights")

    model.to(device)

    # Ghost auxiliary model
    ghost_model = V15_4P_GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256)
    ghost_model.to(device)
    ghost_model.eval()

    trainer = V15_4P_PPOTrainer(model, device, lr=args.lr)

    stats = {
        "phase": "PPO RL",
        "device": str(device),
        "total_states": total_states_init,
        "batches": batches_init,
        "games_played": start_game - 1,
        "policy_loss": 0.0,
        "policy_acc": 0.0,
        "val_loss": 0.0,
        "val_acc": 0.0,
        "eval_wr": best_wr,
        "best_eval_wr": best_wr,
        "gpm": 0.0,
        "fps": 0.0,
        "lr": args.lr,
        "elapsed": "00:00:00",
    }
    metrics_history: List[dict] = []

    def save_checkpoint(is_best: bool = False):
        """Atomically saves checkpoint and metadata to prevent corruption on interrupt."""
        tmp_model = save_dir / "model_latest.pt.tmp"
        torch.save(model.state_dict(), tmp_model)
        tmp_model.replace(save_dir / "model_latest.pt")
        if is_best:
            torch.save(model.state_dict(), save_dir / "model_best.pt")

        meta = {
            "games_played": stats["games_played"],
            "total_states": stats["total_states"],
            "batches": stats["batches"],
            "best_eval_wr": round(best_wr, 1),
            "policy_loss": round(stats["policy_loss"], 4),
            "val_loss": round(stats["val_loss"], 4),
            "last_saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp_meta = save_dir / "metadata.json.tmp"
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=2)
        tmp_meta.replace(meta_file)

    # Signal and graceful stop handlers
    stop_requested = False

    def handle_signal(sig, frame):
        nonlocal stop_requested
        sig_name = signal.Signals(sig).name if hasattr(signal, "Signals") else str(sig)
        print(f"\n🛑 Signal {sig_name} received. Safely finishing current step and saving checkpoint...")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Dashboard server
    from train_v15_4p_online_sl import start_dashboard_server
    html_file = _V15_ROOT / "td_ludo_v15" / "rich" / "dashboard_4p.html"
    start_dashboard_server(lambda: stats, lambda: metrics_history, str(html_file), port=args.port)

    t_start = time.time()
    t_last = time.time()
    games_last = start_game - 1

    print("🏁 Starting RL rollouts and PPO optimization loop...")
    print("💡 To pause/stop training safely at any time: press Ctrl+C, send SIGTERM, or run: touch checkpoints/v15_4p_rl/stop")

    g_idx = start_game
    try:
        while not stop_requested:
            if args.total_games > 0 and g_idx > args.total_games:
                print(f"\n🎯 Reached target game count ({args.total_games:,}). Finishing run.")
                break

            # Check for file-based stop trigger
            stop_file = save_dir / "stop"
            if stop_file.exists():
                print("\n🛑 Found 'stop' trigger file in checkpoints directory. Stopping gracefully...")
                try:
                    stop_file.unlink()
                except OSError:
                    pass
                break

            # Load ghost weights if available
            g_weights = ghost_mgr.sample_ghost_weights()
            if g_weights is not None:
                ghost_model.load_state_dict(g_weights)

            traj, winner, s_seat = play_rl_episode(model, ghost_model if g_weights else None, device)
            trainer.add_trajectory(traj, winner, s_seat)

            # Update PPO
            update_res = trainer.update_if_ready()
            if update_res is not None:
                stats["batches"] += 1
                stats["policy_loss"] = update_res["policy_loss"]
                stats["val_loss"] = update_res["value_loss"]
                stats["entropy"] = update_res["entropy"]
                stats["total_states"] += update_res["transitions_trained"]

            # Save ghost snapshot periodically
            if g_idx % args.ghost_interval == 0:
                ghost_path = ghost_mgr.add_snapshot(model, g_idx)
                print(f"👻 Ghost snapshot saved: {ghost_path.name} (Pruning active, max {MAX_GHOSTS} kept)")

            # Periodic Eval
            if g_idx % args.eval_interval == 0:
                eval_wr = evaluate_rl_agent(model, device, n_games=40)
                stats["eval_wr"] = eval_wr
                is_best = False
                if eval_wr > best_wr:
                    best_wr = eval_wr
                    stats["best_eval_wr"] = best_wr
                    is_best = True
                    print(f"\n🌟 NEW BEST RL WR: {best_wr:.1f}% vs Bots (saved to model_best.pt)!")

                save_checkpoint(is_best=is_best)

                metrics_history.append({
                    "batches": stats["batches"],
                    "states": stats["total_states"],
                    "policy_loss": round(stats["policy_loss"], 4),
                    "policy_acc": round(stats["policy_acc"], 3),
                    "value_loss": round(stats["val_loss"], 4),
                    "val_acc": round(stats["val_acc"], 3),
                    "eval_wr": round(eval_wr, 1),
                    "best_eval_wr": round(best_wr, 1),
                })
                if len(metrics_history) > 300:
                    metrics_history.pop(0)

                print(
                    f"[{stats['elapsed']}] Game {g_idx:6d} | "
                    f"PPO Batches: {stats['batches']:4d} | "
                    f"Loss P: {stats['policy_loss']:.4f} | "
                    f"Loss V: {stats['val_loss']:.4f} | "
                    f"Eval WR: {eval_wr:4.1f}% (Best: {best_wr:4.1f}%)"
                )

            # Telemetry rates
            stats["games_played"] = g_idx
            now = time.time()
            stats["elapsed"] = time.strftime("%H:%M:%S", time.gmtime(int(now - t_start)))
            if now - t_last >= 2.0:
                dt = now - t_last
                stats["gpm"] = ((g_idx - games_last) / dt) * 60.0
                t_last = now
                games_last = g_idx

            g_idx += 1

    except KeyboardInterrupt:
        print("\n🛑 RL training interrupted by KeyboardInterrupt.")
    finally:
        save_checkpoint(is_best=False)
        print(f"💾 Checkpoints and session metadata saved safely to {save_dir}. Clean exit.")


if __name__ == "__main__":
    main()
