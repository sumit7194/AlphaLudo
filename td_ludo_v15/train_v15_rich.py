"""V15 RL — full Phase-L-equivalent pipeline.

Differences from train_v15_rl.py:
  - Imports legacy EloTracker, GameDB for proper ELO + per-game tracking
  - Bot-grid eval via v15_bot_eval (per-bot WR for the dashboard)
  - V15RichTrainer mirrors V13.5 ActorCriticTrainerV10 PPO formulae
    (Monte-Carlo discounted return + EMA return normalization +
    win-prob BCE + entropy bonus + ratio clamp)
  - Dashboard JSON written in the shape v13_dashboard.html expects
    (`/api/stats`, `/api/metrics`, `/api/elo`, `/api/games`, `/api/system`,
    `/api/chain`)
  - Checkpoint rotation (model_latest.pt / model_prev.pt / model_prev2.pt)
  - Opponent mix Phase-L style: V13_2 0.40 / V13_5_SL 0.30 / Self 0.20 / V12_2 0.10
    (V12.2 swappable for V13.5_RL; see flags)

Usage on VM:
    TD_LUDO_RUN_NAME=v15_rich_phase_l python3 train_v15_rich.py \\
        --init checkpoints/v15_sl_v2/model_sl.pt \\
        --opp-v135-rl /home/sumit/td_ludo/checkpoints/v135_prod_rl_local/model_latest.pt \\
        --opp-v135-sl /home/sumit/td_ludo/checkpoints/v135_full/model_latest.pt \\
        --opp-v132    /home/sumit/td_ludo/checkpoints/v132/model_latest.pt \\
        --target-states 20000000 --port 8790
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Bridge to legacy code
_LEGACY_ROOT = Path(__file__).resolve().parent.parent / "td_ludo"
if str(_LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGACY_ROOT))

from td_ludo.eval.elo_tracker import EloTracker  # type: ignore
from td_ludo.data.game_db import GameDB  # type: ignore
from td_ludo.game.encoder_v17 import encode_state_v17  # type: ignore
from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric  # type: ignore
from td_ludo.game.rank_mapping import (  # type: ignore
    state_to_rank_mapping,
    legal_mask_per_rank,
    rank_to_token_id,
)
from td_ludo.models.v13_5 import V135Symmetric, compute_rank_masks  # type: ignore
from experiments.distillation_14ch.model_14ch import MinimalCNN14  # type: ignore

import td_ludo_cpp as _legacy_cpp  # legacy engine
import td_ludo_v15_cpp as _v15_cpp

from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS,
    cell_to_index,
    position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.models.v15 import V15GraphTransformer
from td_ludo_v15.rich.v15_trainer import V15RichTrainer
from td_ludo_v15.rich import v15_player as _v15_player_mod
from td_ludo_v15.rich import v15_bot_eval as _v15_bot_eval_mod
from td_ludo_v15.rich.v15_player import V15RichPlayer
from td_ludo_v15.rich.v15_bot_eval import evaluate_v15_against_bots
from td_ludo_v15.rich.dashboard import start_dashboard
# NOTE: don't import HISTORY_LEN / TOTAL_FRAMES from v15_player at module
# scope — Python's `from X import Y` binds Y to whatever X.Y is at import
# time. After configure_history() mutates v15_player.TOTAL_FRAMES, a stale
# local copy would still be 8. Always read via _v15_player_mod.TOTAL_FRAMES.


_BASE_POS = _v15_cpp.BASE_POS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=None, required=False,
                   help="V15 SL checkpoint to initialize student from")
    p.add_argument("--resume", action="store_true",
                   help="Resume from <ckpt-dir>/model_latest.pt")
    p.add_argument("--kl-anchor", default=None,
                   help="KL anchor target (defaults to --init).")
    p.add_argument("--kl-anchor-coeff", type=float, default=0.0,
                   help="KL coefficient (V13.5 Phase-L used 0; we default to 0).")
    # Opponent checkpoints. Pass the file path AND keep a positive weight to
    # enable. Neural opponents are sequential per-game inference (slower than
    # scripted bots) but provide the only signal that actually matters at
    # this skill level — random/aggressive/defensive/racing bots are too weak
    # for a model already at 80%+ bot-WR and just inject noise into the
    # gradient. Defaults now favor neural opponents + one strong scripted
    # bot (Expert) for variety. Pass explicit weights to override.
    p.add_argument("--opp-v135-rl", default=None)
    p.add_argument("--opp-v135-sl", default=None)
    p.add_argument("--opp-v132", default=None)
    # Default mix (sums to 100):
    #   25 self  +  30 V13.5_RL  +  25 Expert  +  20 Heuristic
    # Heavy-neural mixes (e.g. V13.5_SL + V13.2 also turned on) crater
    # throughput because legacy opp inference is sequential per-game on CPU:
    # ~2.5× slowdown on L4. Keeping ONE strong neural opp (V13.5_RL) gives
    # the signal we want without the throughput hit; Expert + Heuristic are
    # the two strongest scripted bots and run for ~free.
    # Weak bots (Random / Aggressive / Defensive / Racing) default to 0 —
    # they're harmful past the SL ceiling, model just learns to abuse their
    # predictable mistakes instead of refining strong play.
    p.add_argument("--opp-weight-self", type=float, default=25.0)
    p.add_argument("--opp-weight-expert", type=float, default=25.0)
    p.add_argument("--opp-weight-heuristic", type=float, default=20.0)
    p.add_argument("--opp-weight-aggressive", type=float, default=0.0)
    p.add_argument("--opp-weight-defensive", type=float, default=0.0)
    p.add_argument("--opp-weight-racing", type=float, default=0.0)
    p.add_argument("--opp-weight-random", type=float, default=0.0)
    # Strong non-neural bots (added 2026-05-20). Off by default — they're
    # 2-3 orders of magnitude slower than the scripted bots, so enabling
    # them tanks GPM unless they're a small fraction of the mix. Use them
    # as "qualitatively different opponents" — see POPULATION_TRAINING_PLAN.md.
    p.add_argument("--opp-weight-expectimax", type=float, default=0.0,
                   help="1-step expectimax with dice expectation. ~50ms/move.")
    p.add_argument("--opp-weight-mcts-pure", type=float, default=0.0,
                   help="MCTS with random-rollout leaves (no neural net). "
                        "Slow (~100ms+ per move) but qualitatively different.")
    p.add_argument("--mcts-pure-sims", type=int, default=30,
                   help="MCTS simulations per move for the MCTSPure opponent.")
    p.add_argument("--mcts-pure-rollouts", type=int, default=4,
                   help="Random rollouts per MCTS leaf for the MCTSPure opponent.")
    # Neural opps — paths must also be passed; if a weight is positive but
    # no path supplied, the opp is simply skipped (no crash). V13.5_SL and
    # V13.2 default to 0 because they're slower than V13.5_RL and weaker —
    # the gradient signal they add is dominated by V13.5_RL's.
    p.add_argument("--opp-weight-v135-rl", type=float, default=30.0)
    p.add_argument("--opp-weight-v135-sl", type=float, default=0.0)
    p.add_argument("--opp-weight-v132", type=float, default=0.0)
    # Training / PPO
    p.add_argument("--target-states", type=int, default=20_000_000)
    p.add_argument("--max-game-len", type=int, default=400)
    p.add_argument("--parallel-games", type=int, default=64)
    p.add_argument("--ppo-buffer-games", type=int, default=64)
    p.add_argument("--ppo-minibatch-size", type=int, default=256)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--ppo-clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--entropy-coeff", type=float, default=0.03)  # Phase L
    p.add_argument("--win-bce-coeff", type=float, default=0.5)
    # Sampling
    p.add_argument("--temperature", type=float, default=1.0)
    # Eval cadence (Phase L used 10K interval × 2K games)
    p.add_argument("--eval-interval", type=int, default=10_000,
                   help="Bucket-based eval triggered every N games (not states)")
    p.add_argument("--eval-games", type=int, default=2_000)
    # Save cadence
    p.add_argument("--save-interval-sec", type=int, default=120)
    # Logging cadence
    p.add_argument("--log-every", type=int, default=5)
    # Model arch (must match --init)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=512)
    # History window — V15=8 frames (1 current + 7 past), V15.1=2.
    # MUST match the --init checkpoint's stack depth: the model's
    # input_mlp.in_features = history_len * 3.
    p.add_argument("--history-len", type=int, default=8,
                   help="Total frames T in the stack (1 current + (T-1) past). "
                        "V15=8, V15.1=2. Must match --init checkpoint.")
    # Misc
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--opp-device", default="cuda", choices=("cpu", "cuda", "mps"),
                   help="Device for neural opponent inference. cuda is fastest "
                        "(5ms/call vs 50ms on CPU) but uses ~2-3 GB extra VRAM. "
                        "On Mac, 'mps' may help small models — try it; fall back "
                        "to 'cpu' if MPS dispatch overhead exceeds savings.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def pick_device(name):
    if name in ("cuda", "cpu", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Model loaders ─────────────────────────────────────────────────────────
def load_v15_student(path, args, device):
    model = V15GraphTransformer(
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ffn_dim=args.ffn_dim,
        history_len=args.history_len,
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model, ckpt


def _probe_v135_arch(state_dict):
    cw = state_dict.get("conv_input.weight")
    num_channels = int(cw.shape[0])
    idxs = set()
    for k in state_dict.keys():
        if k.startswith("res_blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                idxs.add(int(parts[1]))
    return (max(idxs) + 1 if idxs else 0), num_channels


def load_v135_opponent(path, device=None):
    """Load V13.5-arch opponent. Default device=CPU — opponents do batch=1
    inference per game-turn so GPU buys nothing and costs significant VRAM.
    The student trains on GPU; opponents stay on CPU."""
    if device is None:
        device = torch.device("cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    if any(k.startswith("inner.") for k in sd):
        sd = {k[len("inner."):]: v for k, v in sd.items() if k.startswith("inner.")}
    nrb, nc = _probe_v135_arch(sd)
    model = V135Symmetric(num_res_blocks=nrb, num_channels=nc, in_channels=13)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, device


def load_v132_opponent(path, device=None):
    if device is None:
        device = torch.device("cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    cw = sd.get("conv_input.weight")
    nc = int(cw.shape[0])
    idxs = set()
    for k in sd:
        if k.startswith("res_blocks."):
            try:
                idxs.add(int(k.split(".")[1]))
            except Exception:
                pass
    nrb = max(idxs) + 1 if idxs else 10
    model = MinimalCNN14(num_res_blocks=nrb, num_channels=nc, in_channels=17)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, device


# ─── Opponent pickers ──────────────────────────────────────────────────────
def make_v135_picker(model, device):
    def pick(state, legal):
        if len(legal) == 1:
            return legal[0]
        pp = state.player_positions[int(state.current_player)]
        _, rank_tokens = state_to_rank_mapping(pp)
        rank_legal = legal_mask_per_rank(legal, rank_tokens).astype(np.float32)
        enc = encode_state_v18_symmetric(state).astype(np.float32)
        rm = compute_rank_masks(state).astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(enc).unsqueeze(0).to(device)
            rmt = torch.from_numpy(rm).unsqueeze(0).to(device)
            lmt = torch.from_numpy(rank_legal).unsqueeze(0).to(device)
            logits = model.forward_policy_only(x, rmt, lmt)
            rank = int(logits.argmax(dim=1).item())
        a = rank_to_token_id(rank, legal, rank_tokens)
        return a if a in legal else legal[0]
    return pick


def make_v132_picker(model, device):
    def pick(state, legal):
        if len(legal) == 1:
            return legal[0]
        enc = encode_state_v17(state).astype(np.float32)
        mask = np.zeros(4, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0
        with torch.no_grad():
            x = torch.from_numpy(enc).unsqueeze(0).to(device)
            m = torch.from_numpy(mask).unsqueeze(0).to(device)
            policy, _, _ = model(x, m)
            a = int(policy.argmax(dim=1).item())
        return a if a in legal else legal[0]
    return pick


def make_bot_picker(bot_type: str, **bot_kwargs):
    """Return a picker that calls the named bot.

    Resolves bot_type via two registries:
      1. Strong non-neural bots from td_ludo.game.strong_bots
         (Expectimax, MCTSPure) — these accept extra kwargs.
      2. Legacy scripted bots from td_ludo.game.heuristic_bot
         (Heuristic, Expert, Aggressive, Defensive, Racing, Random).

    Each call instantiates a fresh bot with the current player_id so it
    behaves correctly for either player slot. Bots are cached per
    player_id to avoid construction overhead.

    `bot_kwargs` only matters for the strong bots (e.g. n_sims=50 for
    MCTSPure). Ignored by legacy heuristic bots.
    """
    # First check the strong-bot registry (lazily imported to avoid the
    # MCTS engine import for runs that don't use it).
    strong_factory = None
    try:
        from td_ludo.game.strong_bots import STRONG_BOT_REGISTRY
        strong_factory = STRONG_BOT_REGISTRY.get(bot_type)
    except ImportError:
        pass

    if strong_factory is not None:
        _cache = {}
        def pick_strong(state, legal):
            cp = int(state.current_player)
            if cp not in _cache:
                _cache[cp] = strong_factory(player_id=cp, **bot_kwargs)
            try:
                a = _cache[cp].select_move(state, list(legal))
            except Exception:
                a = legal[0]
            return a if a in legal else legal[0]
        return pick_strong

    # Fall through to the legacy scripted-bot registry
    from td_ludo.game.heuristic_bot import get_bot  # legacy
    _cache = {}
    def pick(state, legal):
        cp = int(state.current_player)
        if cp not in _cache:
            _cache[cp] = get_bot(bot_type, player_id=cp)
        try:
            a = _cache[cp].select_move(state, list(legal))
        except Exception:
            a = legal[0]
        return a if a in legal else legal[0]
    return pick


def make_self_picker(student, device):
    """Greedy self-play picker using live student weights."""
    def pick(state, legal):
        if len(legal) == 1:
            return legal[0]
        cp = int(state.current_player)
        v15_x = np.zeros((_v15_player_mod.TOTAL_FRAMES, 15, 15, 3), dtype=np.float32)
        v15_x[-1] = encode_frame(state, pov_player=cp)
        v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
        legal_cells = []
        for t in legal:
            pos = int(state.player_positions[cp][t])
            c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
            v15_legal[cell_to_index(*c)] = 1.0
            legal_cells.append((t, c))
        student.eval()
        with torch.no_grad():
            xt = torch.from_numpy(v15_x).unsqueeze(0).to(device)
            mt = torch.from_numpy(v15_legal).unsqueeze(0).to(device)
            p, _ = student(xt, mt)
            idx = int(p.argmax(dim=-1).item())
        student.train()
        chosen_cell = divmod(idx, 15)
        for t, c in legal_cells:
            if c == chosen_cell:
                return t
        return legal[0]
    return pick


# ─── Checkpoint rotation ───────────────────────────────────────────────────
SAFE_BACKUP_NAMES = ["model_latest.pt", "model_prev.pt", "model_prev2.pt"]


def safe_save_with_rotation(student, optimizer, ckpt_dir: Path,
                             total_games: int, total_updates: int,
                             best_eval_wr: float, is_best: bool = False):
    """Rotate model_latest → model_prev → model_prev2 and write fresh latest.
    Optionally writes model_best.pt."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prev = ckpt_dir / "model_prev.pt"
    prev2 = ckpt_dir / "model_prev2.pt"
    latest = ckpt_dir / "model_latest.pt"
    if prev.exists():
        shutil.copy2(prev, prev2)
    if latest.exists():
        shutil.copy2(latest, prev)
    save_dict = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_games": int(total_games),
        "total_updates": int(total_updates),
        "best_eval_wr": float(best_eval_wr),
    }
    tmp = ckpt_dir / "model_latest.tmp"
    torch.save(save_dict, str(tmp))
    os.replace(str(tmp), str(latest))
    if is_best:
        best = ckpt_dir / "model_best.pt"
        torch.save({
            "model_state_dict": student.state_dict(),
            "total_games": int(total_games),
            "best_eval_wr": float(best_eval_wr),
        }, str(best))


# ─── Stats writers ─────────────────────────────────────────────────────────
def write_stats_json(
    stats_path: str, trainer: V15RichTrainer, player: V15RichPlayer,
    elo_tracker: EloTracker, game_db: GameDB,
    win_rate_100: float, gpm: float, best_eval_wr: float,
    eval_wr: Optional[float] = None,
    play_alarm: bool = False, is_stagnated: bool = False,
):
    """Write the rich /api/stats JSON v13_dashboard.html consumes."""
    diag = trainer.get_diagnostic_means()
    main_elo = elo_tracker.ratings.get("Model", 1500.0) if elo_tracker else 1500.0
    rankings = elo_tracker.get_rankings(top_n=15) if elo_tracker else []
    opp_stats = {}
    recent_opp_stats = {}
    try:
        opp_stats = game_db.get_opponent_stats("Model") if game_db else {}
    except Exception:
        pass
    # Recent opponent stats from last 500 rows
    try:
        if game_db:
            recent_opp_stats = _compute_recent_opp_stats(game_db, n_recent=500)
    except Exception:
        pass
    payload = {
        "total_games": int(trainer.total_games),
        "total_updates": int(trainer.total_updates),
        "win_rate_100": float(round(win_rate_100, 1)),
        "policy_entropy": float(round(diag["policy_entropy"], 4)),
        "avg_value_loss": float(round(diag["avg_value_loss"], 6)),
        "avg_policy_loss": float(round(diag["avg_policy_loss"], 6)),
        "avg_advantage": float(round(diag["avg_advantage"], 4)),
        "clip_fraction": float(round(diag["clip_fraction"], 4)),
        "approx_kl": float(round(diag["approx_kl"], 4)),
        "temperature": 1.0,
        "games_per_minute": float(round(gpm, 1)),
        "best_eval_win_rate": float(best_eval_wr),
        "ghost_count": 0,
        "is_stagnated": bool(is_stagnated),
        "play_alarm": bool(play_alarm),
        "timestamp": time.time(),
        "main_elo": float(main_elo),
        "elo_rankings": [{"name": n, "elo": float(round(e, 1))} for n, e in rankings],
        "opponent_stats": opp_stats,
        "db_total": (game_db.get_total_games() if game_db else 0),
        "recent_opponent_stats": recent_opp_stats,
    }
    if eval_wr is not None:
        # Dashboard expects FRACTION (0..1) and multiplies by 100 for display.
        payload["eval_win_rate"] = float(eval_wr)
    tmp = stats_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, stats_path)


def _compute_recent_opp_stats(game_db: GameDB, n_recent: int = 500) -> dict:
    """Aggregate the last `n_recent` games of GameDB into per-opponent WR.

    Note: legacy `GameDB.get_recent_games` returns dicts shaped as
    {"players": [p0,p1,p2,p3], "winner": int, "model_player_idx": int, ...}
    — NOT raw p0/p1/p2/p3 fields. Use the `players` list.
    """
    try:
        recent = game_db.get_recent_games(n_recent)
    except Exception:
        return {}
    counts: dict = {}
    for g in recent:
        winner = g.get("winner")
        identities = g.get("players") or []
        model_idx = g.get("model_player_idx")
        if model_idx is None or not identities:
            continue
        for i, name in enumerate(identities):
            if i == model_idx or not name:
                continue
            d = counts.setdefault(name, {"wins": 0, "games": 0})
            d["games"] += 1
            if winner == model_idx:
                d["wins"] += 1
    out: dict = {}
    for name, d in counts.items():
        if d["games"] == 0:
            continue
        out[name] = {
            "wins": d["wins"],
            "games": d["games"],
            "win_rate": round(100.0 * d["wins"] / d["games"], 1),
        }
    return out


def append_metrics_snapshot(metrics_path: str, snapshot: dict):
    """Append one snapshot to the /api/metrics list (eval-time)."""
    data = []
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []
    data.append(snapshot)
    tmp = metrics_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, metrics_path)


def write_chain(path: str, phase: str, run_name: str):
    with open(path, "w") as f:
        json.dump({"stage": "RL", "phase": phase, "arch": "v15",
                   "run_name": run_name, "ts": int(time.time())}, f)


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    # Wire history depth into the player + eval modules BEFORE instantiation.
    # The V15RichPlayer's per-game deques use HISTORY_LEN at __init__ time,
    # and the bot-eval harness reads TOTAL_FRAMES at every call — both need
    # the same T as the model (T*3 = input_mlp.in_features).
    _v15_player_mod.configure_history(args.history_len)
    _v15_bot_eval_mod.configure_history(args.history_len)

    if args.run_name:
        os.environ["TD_LUDO_RUN_NAME"] = args.run_name
    run_name = os.environ.get("TD_LUDO_RUN_NAME", "v15_rich_phase_l")
    CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / run_name
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    stats_path = str(CKPT_DIR / "stats.json")
    metrics_path = str(CKPT_DIR / "metrics.json")
    chain_path = str(CKPT_DIR / "chain_status.json")
    log_path = str(CKPT_DIR / "rl.log")
    game_db_path = str(CKPT_DIR / "games.sqlite")
    elo_path = str(CKPT_DIR / "elo.json")

    if args.resume:
        args.init = str(CKPT_DIR / "model_latest.pt")
        if not os.path.exists(args.init):
            print(f"ERROR: --resume but {args.init} not found")
            sys.exit(1)
    elif not args.init:
        print("ERROR: --init is required (or --resume)")
        sys.exit(1)

    print("=" * 70)
    print(f"V15 RICH RL — Phase-L pipeline  (run={run_name})")
    print("=" * 70)
    print(f"  device:           {device}")
    print(f"  init:             {args.init}")
    print(f"  checkpoint dir:   {CKPT_DIR}")
    print(f"  parallel_games:   {args.parallel_games}")
    print(f"  ppo_buffer_games: {args.ppo_buffer_games}")
    print(f"  ppo_minibatch:    {args.ppo_minibatch_size} × {args.ppo_epochs} epochs")
    print(f"  target_states:    {args.target_states:,}")
    print(f"  lr:               {args.lr}")
    print(f"  entropy_coeff:    {args.entropy_coeff}")
    print(f"  eval cadence:     every {args.eval_interval} games × {args.eval_games} games")
    print(f"  opp weights:      self={args.opp_weight_self} "
          f"v135_rl={args.opp_weight_v135_rl} "
          f"v135_sl={args.opp_weight_v135_sl} v132={args.opp_weight_v132}")
    print("=" * 70)

    # ── Student ────────────────────────────────────────────────────────────
    student, ckpt = load_v15_student(args.init, args, device)
    student.to(device).train()
    print(f"[student] V15 params: {sum(p.numel() for p in student.parameters()):,}")

    # ── KL anchor ──────────────────────────────────────────────────────────
    kl_anchor_model = None
    if args.kl_anchor_coeff > 0:
        anchor_path = args.kl_anchor or args.init
        kl_anchor_model, _ = load_v15_student(anchor_path, args, device)
        kl_anchor_model.to(device).eval()
        for p in kl_anchor_model.parameters():
            p.requires_grad = False
        print(f"[kl-anchor] loaded ({anchor_path}, coeff={args.kl_anchor_coeff})")

    # ── Opponents ──────────────────────────────────────────────────────────
    # Default mix is SelfPlay + scripted bots (super fast, mirrors V13.5
    # default composition). Legacy neural opps are off by default — they
    # would be called batch=1 per game-turn which dominates throughput.
    opp_picks = {}
    opp_probs = {}
    if args.opp_weight_self > 0:
        opp_picks["SelfPlay"] = make_self_picker(student, device)
        opp_probs["SelfPlay"] = args.opp_weight_self
    # Scripted bots (instant, no neural inference)
    bot_specs = [
        ("Heuristic", args.opp_weight_heuristic, {}),
        ("Expert", args.opp_weight_expert, {}),
        ("Aggressive", args.opp_weight_aggressive, {}),
        ("Defensive", args.opp_weight_defensive, {}),
        ("Racing", args.opp_weight_racing, {}),
        ("Random", args.opp_weight_random, {}),
        # Strong non-neural bots — qualitatively different from the
        # scripted family. Add per-bot kwargs as needed (MCTSPure tunes
        # n_sims + rollouts_per_leaf).
        ("Expectimax", args.opp_weight_expectimax, {}),
        ("MCTSPure",   args.opp_weight_mcts_pure, {
            "n_sims": args.mcts_pure_sims,
            "rollouts_per_leaf": args.mcts_pure_rollouts,
        }),
    ]
    for bot_name, w, kwargs in bot_specs:
        if w > 0:
            opp_picks[bot_name] = make_bot_picker(bot_name, **kwargs)
            opp_probs[bot_name] = w
            label = "scripted" if bot_name not in ("Expectimax", "MCTSPure") else "strong"
            print(f"[opp] {bot_name} ({label}, weight={w}"
                  f"{', ' + ', '.join(f'{k}={v}' for k, v in kwargs.items()) if kwargs else ''})")
    # Legacy neural opps — opt-in via positive weight + path
    opp_device = torch.device(args.opp_device)
    if args.opp_v135_rl and args.opp_weight_v135_rl > 0:
        m, _ = load_v135_opponent(args.opp_v135_rl, opp_device)
        opp_picks["Hist_V13_5_RL"] = make_v135_picker(m, opp_device)
        opp_probs["Hist_V13_5_RL"] = args.opp_weight_v135_rl
        print(f"[opp] Hist_V13_5_RL loaded on {opp_device}: {args.opp_v135_rl}")
    if args.opp_v135_sl and args.opp_weight_v135_sl > 0:
        m, _ = load_v135_opponent(args.opp_v135_sl, opp_device)
        opp_picks["Hist_V13_5_SL"] = make_v135_picker(m, opp_device)
        opp_probs["Hist_V13_5_SL"] = args.opp_weight_v135_sl
        print(f"[opp] Hist_V13_5_SL loaded on {opp_device}: {args.opp_v135_sl}")
    if args.opp_v132 and args.opp_weight_v132 > 0:
        m, _ = load_v132_opponent(args.opp_v132, opp_device)
        opp_picks["Hist_V13_2"] = make_v132_picker(m, opp_device)
        opp_probs["Hist_V13_2"] = args.opp_weight_v132
        print(f"[opp] Hist_V13_2 loaded on {opp_device}: {args.opp_v132}")
    if not opp_probs:
        print("ERROR: need at least one opponent")
        sys.exit(1)

    # ── Trainer + Player ───────────────────────────────────────────────────
    trainer = V15RichTrainer(
        model=student, device=device, learning_rate=args.lr,
        ppo_clip=args.ppo_clip, ppo_epochs=args.ppo_epochs,
        ppo_buffer_games=args.ppo_buffer_games,
        ppo_minibatch_size=args.ppo_minibatch_size,
        entropy_coeff=args.entropy_coeff,
        win_bce_coeff=args.win_bce_coeff,
        kl_anchor_coeff=args.kl_anchor_coeff,
        kl_anchor_model=kl_anchor_model,
    )
    if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt and args.resume:
        try:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            trainer.total_games = int(ckpt.get("total_games", 0))
            trainer.total_updates = int(ckpt.get("total_updates", 0))
            print(f"[resume] state restored: games={trainer.total_games} "
                  f"updates={trainer.total_updates}")
        except Exception as e:
            print(f"[resume] optimizer state mismatch ({e}); fresh optimizer")

    player = V15RichPlayer(
        batch_size=args.parallel_games,
        opponents=opp_picks, opponent_probs=opp_probs,
        max_game_len=args.max_game_len, seed=args.seed,
    )

    # ── ELO + GameDB ───────────────────────────────────────────────────────
    elo_tracker = EloTracker(k_factor=32, save_path=elo_path)
    elo_tracker.load()  # no-op if file doesn't exist
    game_db = GameDB(game_db_path)

    # ── Dashboard ──────────────────────────────────────────────────────────
    write_chain(chain_path, "training", run_name)
    if not args.no_dashboard:
        start_dashboard(args.port, str(_LEGACY_ROOT), stats_path,
                        metrics_path, chain_path,
                        elo_tracker=elo_tracker, game_db=game_db)

    # ── Logging ────────────────────────────────────────────────────────────
    log_f = open(log_path, "a")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()
    log(f"starting V15 rich RL: target {args.target_states:,} states, "
        f"init={args.init}, opponents={list(opp_picks.keys())}")

    # ── Training loop ──────────────────────────────────────────────────────
    t_start = time.time()
    session_games_start = trainer.total_games  # for accurate GPM on resume
    states_processed = 0
    last_save_t = time.time()
    last_eval_bucket = trainer.total_games // max(1, args.eval_interval)
    best_eval_wr = 0.0
    win_window: list = []  # rolling 500 game win/loss

    while states_processed < args.target_states:
        # 1. Collect a batch of decision points + finished games
        decisions, finished = player.collect_student_decisions()

        # 2. Forward + sample for student decisions
        if decisions:
            v15_xs = np.stack([d["v15_x"] for d in decisions], axis=0)
            v15_ms = np.stack([d["v15_mask"] for d in decisions], axis=0)
            with torch.no_grad():
                x = torch.from_numpy(v15_xs).to(device, dtype=torch.float32)
                m = torch.from_numpy(v15_ms).to(device, dtype=torch.float32)
                policy, win_prob = student(x, m)
                T = max(1e-3, args.temperature)
                if T != 1.0:
                    logits_T = torch.log(policy + 1e-8) / T
                    logits_T = logits_T.masked_fill(m < 0.5, -1e9)
                    sampling_policy = torch.softmax(logits_T, dim=1)
                else:
                    sampling_policy = policy
                sampled = torch.multinomial(sampling_policy + 1e-9, num_samples=1).squeeze(1)
                log_p_all = torch.log(sampling_policy + 1e-8)
                lp_chosen = log_p_all.gather(1, sampled.unsqueeze(1)).squeeze(1)
                chosen_cells = sampled.cpu().numpy()
                lp_old = lp_chosen.cpu().numpy()
                temperatures = np.full(len(decisions), float(T), dtype=np.float32)
            player.apply_student_actions(decisions, chosen_cells, lp_old, temperatures)

        # 3. Process finished games — feed to trainer, ELO, GameDB
        update_metrics: Optional[dict] = None
        for game in finished:
            game_num = trainer.total_games + 1
            metrics = trainer.train_on_game(
                game["trajectory"], game["winner"], game["model_player"],
            )
            if metrics is not None:
                update_metrics = metrics
            # ELO
            try:
                elo_tracker.update_from_game(
                    game["identities"], game["winner"], game_num=game_num)
            except Exception as e:
                log(f"[elo] update failed: {e}")
            # GameDB
            try:
                game_db.add_game(
                    game_num=game_num, identities=game["identities"],
                    winner=game["winner"], game_length=game["total_moves"],
                    model_player_idx=game["model_player"],
                )
            except Exception as e:
                log(f"[db] add_game failed: {e}")
            win_window.append(1 if game["model_won"] else 0)
            if len(win_window) > 500:
                win_window.pop(0)
            states_processed += game["trajectory_length"]

        # 4. Stats + log every N updates
        if update_metrics is not None and (trainer.total_updates % args.log_every == 0):
            elapsed = time.time() - t_start
            session_games = trainer.total_games - session_games_start
            gpm = (session_games / max(1, elapsed)) * 60.0
            fps = states_processed / max(1e-6, elapsed)
            wr100 = 100.0 * (sum(win_window) / max(1, len(win_window)))
            diag = trainer.get_diagnostic_means()
            opp_breakdown = ", ".join(
                f"{k}={v}" for k, v in player.opp_game_counts.most_common())
            log(f"upd {trainer.total_updates:>5} | st {states_processed:>9,}/"
                f"{args.target_states:,} | g {trainer.total_games} | "
                f"fps {fps:.0f} gpm {gpm:.1f} | wr500 {wr100:.1f}% | "
                f"pol {diag['avg_policy_loss']:+.4f} val {diag['avg_value_loss']:.3f} "
                f"ent {diag['policy_entropy']:.3f} clip {diag['clip_fraction']:.3f} "
                f"kl {diag['approx_kl']:.3f}")
            if trainer.total_updates % (args.log_every * 5) == 0:
                log(f"  [opp-mix] {opp_breakdown}")
            try:
                write_stats_json(
                    stats_path, trainer, player, elo_tracker, game_db,
                    win_rate_100=wr100, gpm=gpm, best_eval_wr=best_eval_wr,
                )
            except Exception as e:
                log(f"[stats] write failed: {e}")

        # 5. Auto-save by time
        now = time.time()
        if now - last_save_t >= args.save_interval_sec:
            try:
                safe_save_with_rotation(
                    student, trainer.optimizer, CKPT_DIR,
                    total_games=trainer.total_games,
                    total_updates=trainer.total_updates,
                    best_eval_wr=best_eval_wr,
                )
                elo_tracker.save()
                log(f"[checkpoint] rotated model_latest.pt @ games={trainer.total_games}")
            except Exception as e:
                log(f"[checkpoint] save failed: {e}")
            last_save_t = now

        # 6. Bucket-based eval
        cur_bucket = trainer.total_games // max(1, args.eval_interval)
        if cur_bucket > last_eval_bucket and trainer.total_games > 0:
            log(f"[eval] starting ({args.eval_games} games at game {trainer.total_games})...")
            result = evaluate_v15_against_bots(
                student, device, num_games=args.eval_games)
            eval_wr = result["win_rate"]
            log(f"[eval] WR = {eval_wr*100:.1f}% at game {trainer.total_games}  "
                f"(per-bot: " +
                ", ".join(f"{k}={v['win_rate']:.1f}%/{v['games']}g"
                          for k, v in result["per_bot"].items()) + ")")
            is_best = eval_wr > best_eval_wr
            if is_best:
                best_eval_wr = eval_wr
            # Save with best-flag
            try:
                safe_save_with_rotation(
                    student, trainer.optimizer, CKPT_DIR,
                    total_games=trainer.total_games,
                    total_updates=trainer.total_updates,
                    best_eval_wr=best_eval_wr, is_best=is_best,
                )
            except Exception as e:
                log(f"[checkpoint] post-eval save failed: {e}")
            # Append metrics snapshot
            snapshot = {
                "games": trainer.total_games,
                "updates": trainer.total_updates,
                "win_rate": float(sum(win_window) / max(1, len(win_window))),
                "policy_entropy": float(trainer.get_diagnostic_means()["policy_entropy"]),
                "avg_value_loss": float(trainer.get_diagnostic_means()["avg_value_loss"]),
                "avg_policy_loss": float(trainer.get_diagnostic_means()["avg_policy_loss"]),
                # FRACTION (0..1) — v13_dashboard multiplies by 100 for display.
                "eval_win_rate": float(eval_wr),
                "per_bot": result["per_bot"],
                "timestamp": time.time(),
            }
            append_metrics_snapshot(metrics_path, snapshot)
            try:
                write_stats_json(
                    stats_path, trainer, player, elo_tracker, game_db,
                    win_rate_100=100.0 * (sum(win_window) / max(1, len(win_window))),
                    gpm=((trainer.total_games - session_games_start)
                         / max(1, time.time() - t_start)) * 60.0,
                    best_eval_wr=best_eval_wr, eval_wr=eval_wr,
                )
            except Exception:
                pass
            last_eval_bucket = cur_bucket

    # Final save
    try:
        safe_save_with_rotation(
            student, trainer.optimizer, CKPT_DIR,
            total_games=trainer.total_games,
            total_updates=trainer.total_updates,
            best_eval_wr=best_eval_wr, is_best=False,
        )
        elo_tracker.save()
    except Exception as e:
        log(f"[final-save] failed: {e}")
    write_chain(chain_path, "completed", run_name)
    log(f"[done] processed {states_processed:,} states across {trainer.total_games} games")
    log_f.close()


if __name__ == "__main__":
    main()
