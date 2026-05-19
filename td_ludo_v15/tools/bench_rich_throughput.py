"""Speed-bench harness for V15 rich pipeline.

USES A COPY OF V15 SL — never reads/touches the live training checkpoint.

Runs the player+trainer for a fixed wall-clock duration and reports:
  - GPM (games-per-minute)
  - FPS (states-per-second)
  - GPU mem usage (peak)
  - Per-component time breakdown

Run multiple configs with `--config` arg; pick the best, then use those
flags for the live training restart.

Usage on VM:
    cp checkpoints/v15_sl_v2/model_sl.pt /tmp/v15_test_init.pt
    python3 tools/bench_rich_throughput.py \\
        --init /tmp/v15_test_init.pt \\
        --opp-v135-rl /home/sumit/td_ludo/checkpoints/v135_prod_rl_local/model_latest.pt \\
        --opp-v135-sl /home/sumit/td_ludo/checkpoints/v135_full/model_latest.pt \\
        --opp-v132    /home/sumit/td_ludo/checkpoints/v132/model_latest.pt \\
        --parallel-games 256 --opp-device cpu --duration-sec 90
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LEGACY = ROOT.parent / "td_ludo"
sys.path.insert(0, str(LEGACY))

from td_ludo_v15.models.v15 import V15GraphTransformer
from td_ludo_v15.rich.v15_trainer import V15RichTrainer
from td_ludo_v15.rich.v15_player import V15RichPlayer

# Reuse legacy opponent loaders + pickers (incl. bots and self)
from train_v15_rich import (
    load_v135_opponent, load_v132_opponent,
    make_v135_picker, make_v132_picker, make_self_picker, make_bot_picker,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="V15 SL CKPT TO COPY (test model)")
    # Neural opp checkpoints (off by default)
    p.add_argument("--opp-v135-rl", default=None)
    p.add_argument("--opp-v135-sl", default=None)
    p.add_argument("--opp-v132", default=None)
    # Self
    p.add_argument("--opp-weight-self", type=float, default=30.0)
    # Bots
    p.add_argument("--opp-weight-heuristic", type=float, default=15.0)
    p.add_argument("--opp-weight-expert", type=float, default=15.0)
    p.add_argument("--opp-weight-aggressive", type=float, default=10.0)
    p.add_argument("--opp-weight-defensive", type=float, default=10.0)
    p.add_argument("--opp-weight-racing", type=float, default=10.0)
    p.add_argument("--opp-weight-random", type=float, default=10.0)
    # Neural opps (default 0)
    p.add_argument("--opp-weight-v135-rl", type=float, default=0.0)
    p.add_argument("--opp-weight-v135-sl", type=float, default=0.0)
    p.add_argument("--opp-weight-v132", type=float, default=0.0)
    p.add_argument("--parallel-games", type=int, default=256)
    p.add_argument("--ppo-buffer-games", type=int, default=64)
    p.add_argument("--ppo-minibatch-size", type=int, default=256)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--opp-device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps"))
    p.add_argument("--duration-sec", type=int, default=60)
    p.add_argument("--label", default="bench")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device(args.device)
    opp_device = torch.device(args.opp_device)

    # Load student (test copy, not live ckpt)
    ck = torch.load(args.init, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    student = V15GraphTransformer().to(device)
    student.load_state_dict(sd, strict=False)
    student.train()
    n = sum(p.numel() for p in student.parameters())
    print(f"[{args.label}] student loaded: {n:,} params on {device}")

    opp_picks = {}
    opp_probs = {}
    if args.opp_weight_self > 0:
        opp_picks["SelfPlay"] = make_self_picker(student, device)
        opp_probs["SelfPlay"] = args.opp_weight_self
    bot_specs = [
        ("Heuristic", args.opp_weight_heuristic),
        ("Expert", args.opp_weight_expert),
        ("Aggressive", args.opp_weight_aggressive),
        ("Defensive", args.opp_weight_defensive),
        ("Racing", args.opp_weight_racing),
        ("Random", args.opp_weight_random),
    ]
    for bot_name, w in bot_specs:
        if w > 0:
            opp_picks[bot_name] = make_bot_picker(bot_name)
            opp_probs[bot_name] = w
    if args.opp_v135_rl and args.opp_weight_v135_rl > 0:
        m, _ = load_v135_opponent(args.opp_v135_rl, opp_device)
        opp_picks["Hist_V13_5_RL"] = make_v135_picker(m, opp_device)
        opp_probs["Hist_V13_5_RL"] = args.opp_weight_v135_rl
    if args.opp_v135_sl and args.opp_weight_v135_sl > 0:
        m, _ = load_v135_opponent(args.opp_v135_sl, opp_device)
        opp_picks["Hist_V13_5_SL"] = make_v135_picker(m, opp_device)
        opp_probs["Hist_V13_5_SL"] = args.opp_weight_v135_sl
    if args.opp_v132 and args.opp_weight_v132 > 0:
        m, _ = load_v132_opponent(args.opp_v132, opp_device)
        opp_picks["Hist_V13_2"] = make_v132_picker(m, opp_device)
        opp_probs["Hist_V13_2"] = args.opp_weight_v132

    print(f"[{args.label}] opponents: {list(opp_picks.keys())} on {opp_device}")
    print(f"[{args.label}] parallel_games={args.parallel_games} "
          f"ppo_buffer_games={args.ppo_buffer_games} "
          f"ppo_minibatch={args.ppo_minibatch_size}")
    print(f"[{args.label}] duration={args.duration_sec}s")

    trainer = V15RichTrainer(
        student, device, learning_rate=1e-5,
        ppo_buffer_games=args.ppo_buffer_games,
        ppo_minibatch_size=args.ppo_minibatch_size,
        ppo_epochs=args.ppo_epochs, entropy_coeff=0.03, kl_anchor_coeff=0.0,
    )
    player = V15RichPlayer(
        batch_size=args.parallel_games,
        opponents=opp_picks, opponent_probs=opp_probs,
        max_game_len=400, seed=42,
    )

    # Reset CUDA peak memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    states_processed = 0
    games_done = 0
    updates = 0
    spin_time = 0.0
    student_fwd_time = 0.0
    train_time = 0.0
    last_print = t0

    while time.time() - t0 < args.duration_sec:
        ts = time.time()
        decisions, finished = player.collect_student_decisions()
        spin_time += time.time() - ts

        if decisions:
            ts = time.time()
            v15_xs = np.stack([d["v15_x"] for d in decisions], axis=0)
            v15_ms = np.stack([d["v15_mask"] for d in decisions], axis=0)
            with torch.no_grad():
                x = torch.from_numpy(v15_xs).to(device, dtype=torch.float32)
                m = torch.from_numpy(v15_ms).to(device, dtype=torch.float32)
                policy, val = student(x, m)
                sampled = torch.multinomial(policy + 1e-9, num_samples=1).squeeze(1)
                lp = torch.log(policy.gather(1, sampled.unsqueeze(1)).squeeze(1) + 1e-8)
            student_fwd_time += time.time() - ts
            player.apply_student_actions(
                decisions, sampled.cpu().numpy(), lp.cpu().numpy(),
                np.ones(len(decisions), dtype=np.float32))

        for game in finished:
            ts = time.time()
            metrics = trainer.train_on_game(
                game["trajectory"], game["winner"], game["model_player"])
            train_time += time.time() - ts
            if metrics is not None:
                updates += 1
            states_processed += game["trajectory_length"]
            games_done += 1

        if time.time() - last_print > 10:
            elapsed = time.time() - t0
            gpm = (games_done / elapsed) * 60.0
            fps = states_processed / elapsed
            print(f"[{args.label}] t={elapsed:.0f}s  g={games_done}  "
                  f"st={states_processed}  fps={fps:.0f}  gpm={gpm:.1f}  "
                  f"upd={updates}")
            last_print = time.time()

    total_elapsed = time.time() - t0
    gpm = (games_done / total_elapsed) * 60.0
    fps = states_processed / total_elapsed
    print()
    print(f"=== [{args.label}] FINAL ===")
    print(f"  duration:        {total_elapsed:.1f}s")
    print(f"  games:           {games_done}")
    print(f"  states:          {states_processed:,}")
    print(f"  GPM:             {gpm:.1f}")
    print(f"  FPS:             {fps:.0f}")
    print(f"  PPO updates:     {updates}")
    print(f"  spin_time:       {spin_time:.1f}s ({100*spin_time/total_elapsed:.0f}%)")
    print(f"  student_fwd:     {student_fwd_time:.1f}s ({100*student_fwd_time/total_elapsed:.0f}%)")
    print(f"  trainer (PPO):   {train_time:.1f}s ({100*train_time/total_elapsed:.0f}%)")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  GPU peak alloc:  {peak:.2f} GB")
    print()


if __name__ == "__main__":
    main()
