"""Throughput benchmark for V12.2 RL pipeline.

Methodology:
  - Load V12.2 weights from repo (1.36M params, 3 ResBlocks × 128ch).
  - Drive `play_step` for a fixed wall-clock window after a warmup.
  - Report: marginal GPM (post-warmup), per-step latency, leaf count
    if search is on.
  - Two configs: search OFF, search ON (alpha=0.5, fraction=0.25).
  - PROD mode (BATCH_SIZE=512) to match L4 production.

Note: sandbox is 4-core CPU, no GPU. Absolute numbers underrepresent L4
(~6× slower in our prior measurement). Relative deltas across optimization
tiers should still hold for CPU-bound bottlenecks (Python loops, pybind
crossings). GPU-only opts (pinned memory, non_blocking) won't show up.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

# Force PROD config (BATCH_SIZE=512), unique run name to keep checkpoints
# isolated from any other test runs in the repo.
os.environ.setdefault("TD_LUDO_MODE", "PROD")
os.environ.setdefault("TD_LUDO_RUN_NAME", "ac_v12_2_perf_bench")

sys.path.insert(0, "/home/user/AlphaLudo/td_ludo")

import td_ludo_cpp  # noqa: E402
from td_ludo.models.v12 import AlphaLudoV12  # noqa: E402
from td_ludo.training.trainer_v10 import ActorCriticTrainerV10  # noqa: E402
from td_ludo.game.players.v11 import VectorACGamePlayer  # noqa: E402
from src.config import BATCH_SIZE  # noqa: E402


CKPT_PATH = "/home/user/AlphaLudo/td_ludo/play/model_weights/v12_2/model_latest.pt"
WARMUP_SEC = 30.0
MEASURE_SEC = 90.0


def build_player(search_on: bool, device: torch.device):
    model = AlphaLudoV12(
        num_res_blocks=3, num_channels=128,
        num_attn_layers=2, num_heads=4, ffn_ratio=4,
        dropout=0.0, in_channels=33,
    ).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd, strict=False)
    model.train()  # PPO update path expects train mode

    alpha = 0.5 if search_on else 0.0
    trainer = ActorCriticTrainerV10(
        model, device, learning_rate=1e-5, alpha_search=alpha,
    )
    player = VectorACGamePlayer(
        trainer, BATCH_SIZE, device, model_factory=None, elo_tracker=None,
        search_enabled=search_on,
        search_target_fraction=0.25 if search_on else 0.0,
        search_label_smoothing=0.1,
    )
    return player, trainer


def run(search_on: bool, device: torch.device, label: str):
    print(f"\n{'='*60}\n  {label}\n  search={search_on}, BATCH_SIZE={BATCH_SIZE}, device={device}\n{'='*60}")
    player, trainer = build_player(search_on, device)

    # Warmup
    t_warm0 = time.time()
    n_warm_steps = 0
    n_warm_games = 0
    while time.time() - t_warm0 < WARMUP_SEC:
        results = player.play_step(train=True)
        n_warm_games += len(results)
        n_warm_steps += 1
    games_at_warm_end = trainer.total_games
    t_warm_end = time.time()
    print(f"[warmup {time.time()-t_warm0:.1f}s] {n_warm_steps} steps, {n_warm_games} game completions")

    # Measurement window
    t_meas0 = time.time()
    n_meas_steps = 0
    while time.time() - t_meas0 < MEASURE_SEC:
        results = player.play_step(train=True)
        n_meas_steps += 1
    t_meas_end = time.time()
    games_after = trainer.total_games

    measured_games = games_after - games_at_warm_end
    measured_seconds = t_meas_end - t_meas0
    gpm = measured_games / (measured_seconds / 60.0) if measured_seconds > 0 else 0.0
    step_ms = 1000.0 * measured_seconds / max(1, n_meas_steps)
    games_per_step = measured_games / max(1, n_meas_steps)

    print(f"[measure {measured_seconds:.1f}s] "
          f"steps={n_meas_steps} games={measured_games} "
          f"GPM={gpm:.1f} step_ms={step_ms:.1f} games/step={games_per_step:.2f}")

    if search_on:
        diag = player.get_search_diagnostics()
        print(f"[search] searches={diag['searches_done']} "
              f"leaves={diag['leaf_count']} "
              f"opp_queries={diag['opp_query_count']} "
              f"top1_agree={diag['top1_agreement_pct']:.1f}%")

    return {
        "label": label,
        "search_on": search_on,
        "gpm": gpm,
        "step_ms": step_ms,
        "games_per_step": games_per_step,
        "measured_games": measured_games,
        "measured_seconds": measured_seconds,
    }


if __name__ == "__main__":
    device = torch.device("cpu")
    torch.set_num_threads(4)

    results = []
    results.append(run(search_on=False, device=device, label="baseline / search OFF"))
    results.append(run(search_on=True,  device=device, label="baseline / search ON"))

    print("\n\n=== SUMMARY ===")
    for r in results:
        print(f"{r['label']:35s} GPM={r['gpm']:6.1f} step_ms={r['step_ms']:5.1f} g/step={r['games_per_step']:.2f}")
