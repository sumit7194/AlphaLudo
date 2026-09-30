"""V13.5 RL — self-play REINFORCE-with-baseline on V135Symmetric.

Mirrors train_v133_rl.py's recipe (KL anchor + multi-legal filter + cosine
LR + entropy bonus) with three structural differences:

1. **No history.** V13.5 is stateless single-frame; we drop the K=8 deque.
2. **Rank-indexed action space.** Model outputs 4 logits per canonical
   rank (rank 0 = most-advanced own token). At rollout time we sample
   a rank and map it to a legal token-ID via `rank_to_token_id`.
   Trajectories store the chosen rank, not the token-ID — gradients
   flow through rank logits.
3. **KL anchor target = V13.5_SL, not V13.2.** Anchoring to V13.2 would
   pull the policy back through the rank→token aggregation, defeating
   the symmetry constraint. V13.5_SL is the natural anchor since both
   live in the same per-rank policy space.

Optional H2H gating: every --h2h-gate-every states, run a 200-game H2H
vs V13.2 (in-memory, mirrored seeds, greedy). The result is logged to
rl_stats.json under "h2h_history" and printed; this is the only signal
that's not teacher-bound.

Usage
-----
    TD_LUDO_RUN_NAME=v135_rl python train_v135_rl.py \\
        --init checkpoints/v135_full/model_latest.pt \\
        --kl-teacher checkpoints/v135_full/model_latest.pt \\
        --h2h-opponent checkpoints/v132/model_latest.pt \\
        --target-states 20000000 \\
        --port 8799
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import os
import random
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS
from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric, V18_CHANNELS
from td_ludo.game.rank_mapping import (
    state_to_rank_mapping,
    legal_mask_per_rank,
    rank_to_token_id,
)
from td_ludo.models.v13_5 import V135Symmetric, compute_rank_masks
from experiments.distillation_14ch.model_14ch import MinimalCNN14

# Strong-bot integration (Expectimax, MCTSPure, personality variants, etc.).
# Mirrors the recipe used by td_ludo_v15/train_v15_rich.py so V13.5 RL
# can train against the same mix and we can compare growth apples-to-apples.
from td_ludo.game.strong_bots import STRONG_BOT_REGISTRY  # noqa: E402
from src.heuristic_bot import get_bot as _legacy_get_bot  # noqa: E402
from td_ludo.eval.elo_tracker import EloTracker  # noqa: E402
from td_ludo.game.reward_shaping import compute_shaped_reward  # noqa: E402
from td_ludo.game.dense_rewards import compute_kill_penalty  # noqa: E402
# Bias penalties (champion recipe) — opt-in via env, matching the v11
# player's gating. Default OFF for backward compat.
try:
    from td_ludo.game.bias_penalties import compute_bias_penalties  # noqa: E402
except Exception:
    compute_bias_penalties = None
_BIAS_PENALTIES_ENABLED = (
    compute_bias_penalties is not None
    and os.environ.get('LUDO_BIAS_PENALTIES', '0').lower() in ('1', 'true', 'yes')
)

# Slim eval roster — matches V15 trainer's EVAL_BOT_NAMES_FAST.
# Excludes Depth2*, MCTSExpertPrior, MCTSExpectimaxPrior (slow per-move),
# VoteExpectimax, AdaptiveExpectimax (cost without informative signal),
# all rule-bots (saturated near 88% — no model differentiation).
EVAL_BOT_NAMES_FAST = [
    "Heuristic", "Aggressive", "Defensive", "Racing", "Random",
    "Expert", "Expectimax", "MCTSPure",
    "AggressiveExpectimax", "DefensiveExpectimax",
    "RacingExpectimax", "MinimaxExpectimax", "BlockadeExpectimax",
]


def get_unified_bot(name, player_id=None, *,
                    mcts_pure_sims=30, mcts_pure_rollouts=4):
    """Factory across strong-bot + legacy heuristic registries.

    Strong bots (Expectimax, MCTSPure, etc.) accept extra kwargs;
    legacy scripted bots ignore them.
    """
    if name == "MCTSPure":
        return STRONG_BOT_REGISTRY["MCTSPure"](
            player_id=player_id,
            n_sims=mcts_pure_sims,
            rollouts_per_leaf=mcts_pure_rollouts,
        )
    if name in STRONG_BOT_REGISTRY:
        return STRONG_BOT_REGISTRY[name](player_id=player_id)
    return _legacy_get_bot(name, player_id=player_id)


def build_opp_sampler(opp_probs):
    """Build a function () -> bot_name that samples from opp_probs.

    opp_probs is a dict {name: weight}. Weights need not sum to 1 — we
    normalize internally. A "SelfPlay" entry signals pure-self-play
    (no opp; trajectory records both players' moves).
    """
    names = list(opp_probs.keys())
    weights = np.asarray([opp_probs[n] for n in names], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        # Degenerate — fall back to pure self-play.
        return lambda: "SelfPlay"
    probs = weights / total

    def sampler():
        return random.choices(names, weights=probs, k=1)[0]
    return sampler


# ── Ghost pool (AlphaZero-style self-snapshot opponents) ────────────────────
import glob as _glob


class GhostBot:
    """Frozen past-student snapshot used as a self-play opponent.

    Mirrors the student's exact inference path (V18-symmetric encode →
    rank-indexed policy → rank_to_token_id) but with frozen weights and
    greedy argmax selection. Exposes `.select_move(game, legal)` so it
    drops into SelfPlayEnv's opponent slot like any scripted bot.

    Greedy (not sampled) is intentional — matches how the V15 self-picker
    plays: a stable, deterministic past-self gives a cleaner gradient
    target than a stochastic one.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def select_move(self, game, legal):
        cp = int(game.current_player)
        pp = game.player_positions[cp]
        _, rank_tokens = state_to_rank_mapping(pp)
        rank_legal = legal_mask_per_rank(legal, rank_tokens).astype(np.float32)
        enc = encode_state_v18_symmetric(game).astype(np.float32)
        rmasks = compute_rank_masks(game).astype(np.float32)
        x = torch.from_numpy(enc).unsqueeze(0).to(self.device)
        rm = torch.from_numpy(rmasks).unsqueeze(0).to(self.device)
        rl = torch.from_numpy(rank_legal).unsqueeze(0).to(self.device)
        out = self.model(x, rm, rl)
        policy = out[0] if isinstance(out, (tuple, list)) else out
        chosen_rank = int(policy.argmax(dim=-1).item())
        if rank_legal[chosen_rank] == 0:
            legal_ranks = np.where(rank_legal > 0)[0]
            chosen_rank = int(legal_ranks[0]) if len(legal_ranks) else 0
        return rank_to_token_id(chosen_rank, legal, rank_tokens)


class GhostPool:
    """Saves/prunes/loads frozen student snapshots for the Ghost opp slot."""

    def __init__(self, ghosts_dir, args, device, max_ghosts=8):
        self.dir = ghosts_dir
        self.args = args
        self.device = device
        self.max_ghosts = max_ghosts
        os.makedirs(self.dir, exist_ok=True)
        self._cache = {}  # path -> GhostBot (load once, reuse)

    def save(self, student, games_played):
        path = os.path.join(self.dir, f"ghost_{games_played:08d}.pt")
        torch.save({"model_state_dict": student.state_dict()}, path)
        self._prune()
        return path

    def _prune(self):
        ghosts = sorted(_glob.glob(os.path.join(self.dir, "ghost_*.pt")))
        while len(ghosts) > self.max_ghosts:
            victim = ghosts.pop(0)  # oldest first
            try:
                os.remove(victim)
                self._cache.pop(victim, None)
            except OSError:
                pass

    def list_ghosts(self):
        return sorted(_glob.glob(os.path.join(self.dir, "ghost_*.pt")))

    def count(self):
        return len(self.list_ghosts())

    def random_ghost(self):
        """A GhostBot for a random snapshot, or None if the pool is empty
        (caller then falls back to self-play for that game)."""
        ghosts = self.list_ghosts()
        if not ghosts:
            return None
        path = random.choice(ghosts)
        if path not in self._cache:
            model, _ = _load_v135(path, self.args, self.device)
            model.to(self.device).eval()   # _load_v135 doesn't move to device
            for p in model.parameters():
                p.requires_grad = False
            self._cache[path] = GhostBot(model, self.device)
        return self._cache[path]


# ── Args ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=None,
                   help="Initial student checkpoint (V13.5_SL recommended)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from <ckpt-dir>/model_latest.pt")
    p.add_argument("--kl-teacher", default=None,
                   help="V13.5_SL checkpoint for KL anchor (defaults to --init)")
    p.add_argument("--h2h-opponent", default=None,
                   help="V13.2 checkpoint for periodic H2H gating (optional)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--target-states", type=int, default=20_000_000)
    p.add_argument("--max-game-len", type=int, default=400)
    p.add_argument("--parallel-games", type=int, default=64)
    p.add_argument("--train-chunk", type=int, default=2048)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--train-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-end", type=float, default=5e-6)
    p.add_argument("--entropy-coeff", type=float, default=0.02)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--kl-anchor-coeff", type=float, default=0.1)
    p.add_argument("--save-every-games", type=int, default=20_000)
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--eval-every-games", type=int, default=20_000)
    p.add_argument("--eval-every", type=int, default=200_000)
    p.add_argument("--eval-games", type=int, default=3000)
    p.add_argument("--h2h-gate-every", type=int, default=2_000_000,
                   help="Every N states, run H2H vs --h2h-opponent (0=disabled)")
    p.add_argument("--h2h-games", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    # V13.5 architecture (must match the SL checkpoint)
    p.add_argument("--num-res-blocks", type=int, default=10)
    p.add_argument("--num-channels", type=int, default=128)
    p.add_argument("--head-hidden", type=int, default=64)
    # V13.2 H2H opponent architecture
    p.add_argument("--v132-num-res-blocks", type=int, default=10)
    p.add_argument("--v132-num-channels", type=int, default=128)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--no-dashboard", action="store_true")

    # ── Opponent mix (mirrors td_ludo_v15/train_v15_rich.py flags).
    # When ALL weights are 0, defaults to pure self-play (backward
    # compatible). Otherwise each game samples one opponent type and
    # the student trains against it; trajectory records only the
    # student's moves (with the bot acting as a fixed teacher).
    p.add_argument("--opp-weight-self", type=float, default=100.0,
                   help="Pure self-play weight. Default 100 = backward compat")
    # Legacy scripted bots
    p.add_argument("--opp-weight-aggressive", type=float, default=0.0)
    p.add_argument("--opp-weight-defensive", type=float, default=0.0)
    p.add_argument("--opp-weight-racing", type=float, default=0.0)
    p.add_argument("--opp-weight-heuristic", type=float, default=0.0)
    p.add_argument("--opp-weight-expert", type=float, default=0.0)
    p.add_argument("--opp-weight-random", type=float, default=0.0)
    # Strong bots (Expectimax + variants)
    p.add_argument("--opp-weight-expectimax", type=float, default=0.0)
    p.add_argument("--opp-weight-aggressive-expectimax", type=float, default=0.0)
    p.add_argument("--opp-weight-defensive-expectimax", type=float, default=0.0)
    p.add_argument("--opp-weight-racing-expectimax", type=float, default=0.0)
    p.add_argument("--opp-weight-minimax-expectimax", type=float, default=0.0)
    p.add_argument("--opp-weight-blockade-expectimax", type=float, default=0.0)
    p.add_argument("--opp-weight-vote-expectimax", type=float, default=0.0)
    p.add_argument("--opp-weight-depth2-expectimax", type=float, default=0.0)
    p.add_argument("--opp-weight-depth2-aggressive", type=float, default=0.0)
    p.add_argument("--opp-weight-depth2-defensive", type=float, default=0.0)
    # MCTS-family
    p.add_argument("--opp-weight-mcts-pure", type=float, default=0.0)
    p.add_argument("--mcts-pure-sims", type=int, default=30)
    p.add_argument("--mcts-pure-rollouts", type=int, default=4)
    p.add_argument("--opp-weight-mcts-expert-prior", type=float, default=0.0)
    p.add_argument("--opp-weight-mcts-expectimax-prior", type=float, default=0.0)
    # Rule bots
    p.add_argument("--opp-weight-max-capture", type=float, default=0.0)
    p.add_argument("--opp-weight-two-stack", type=float, default=0.0)
    p.add_argument("--opp-weight-home-rush", type=float, default=0.0)
    p.add_argument("--opp-weight-stack-home-rush", type=float, default=0.0)
    # Adaptive
    p.add_argument("--opp-weight-adaptive-expectimax", type=float, default=0.0)

    # ── Ghost pool (AlphaZero-style self-snapshot rotation). Ported from
    # train_v12.py (2026-05-29). When --opp-weight-ghost > 0, the trainer
    # snapshots the live student every --ghost-save-interval games into
    # <ckpt-dir>/ghosts/, and the "Ghost" opponent slot loads a RANDOM
    # past snapshot to play against. This is the ingredient V15.2/V13.6
    # lacked (pure SelfPlay plateaued at 79.6%); fighting past selves
    # prevents self-play equilibrium stagnation.
    p.add_argument("--opp-weight-ghost", type=float, default=0.0,
                   help="Weight for Ghost opponents (frozen past student "
                        "snapshots). 0 = disabled (backward compat).")
    p.add_argument("--ghost-save-interval", type=int, default=5000,
                   help="Snapshot the student into the ghost pool every N games.")
    p.add_argument("--max-ghosts", type=int, default=8,
                   help="Max ghost snapshots to retain (oldest pruned).")

    # ── Exploration knobs (carried over from V15 trainer)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Action sampling temperature for self-play rollouts.")

    # ── Reward shaping (per training journal: dense rewards are the
    # proven recipe; sparse-terminal-only learning collapses on
    # 150-move games due to credit-assignment failure).
    p.add_argument("--use-shaped-reward", type=int, default=1,
                   help="1 = dense reward shaping (v1 recipe). 0 = terminal-only.")

    return p.parse_args()


def pick_device(name):
    if name in ("cuda", "cpu", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Self-play / opp-vs-student env (no history; per-rank policy) ──────────
class SelfPlayEnv:
    """B parallel games. Each step computes V18 encoding + rank-masks +
    per-rank legal mask. Trajectories store (enc, rmasks, rlegal, chosen_rank,
    v_pred, cp).

    Each game is bound to an OPP role at reset:
      - opp_name == "SelfPlay": both players use student → trajectory
        records both players' moves (pure self-play, original behavior)
      - opp_name == <bot>: model_player ∈ {0, 2} is the student. The
        other player (opp_player) makes moves via bot.select_move(); those
        moves do NOT enter the trajectory. Reward at end is from the
        student's POV.
    """

    def __init__(self, batch_size, max_game_len=400,
                 opp_sampler=None, get_bot_fn=None, elo_tracker=None,
                 use_shaped_reward=True):
        self.batch_size = batch_size
        self.max_game_len = max_game_len
        # Either both opp_sampler+get_bot_fn must be set, or neither
        # (pure-self-play, backward-compatible).
        self.opp_sampler = opp_sampler
        self.get_bot_fn = get_bot_fn
        self._opp_enabled = (opp_sampler is not None and get_bot_fn is not None)
        # Optional EloTracker — updated on each opp-game finalization.
        # SelfPlay games are excluded (self-vs-self provides no Elo signal).
        self.elo_tracker = elo_tracker
        # ── Reward shaping (per journal: dense rewards needed for credit
        # assignment on long stochastic games). Each trajectory entry now
        # stores a 7th field: the per-step shaped reward computed at the
        # moment that entry's action was applied (state_before → state_after).
        # _finalize / _truncate aggregate forward to compute returns.
        self.use_shaped_reward = use_shaped_reward
        # Running mean shaped-reward magnitude for telemetry.
        self.recent_shaped_per_game = collections.deque(maxlen=200)
        # ── Got-killed tracking (2026-06-10 fix). compute_shaped_reward's
        # own −0.20 check can never fire: it looks for own tokens going to
        # base inside the (pre→post) window of the player's OWN move, but
        # captures happen during the opponent's intervening turn. Track each
        # player's at-base count at their previous own decision and charge
        # the delta — same mechanism as v11.py's compute_kill_penalty
        # (the champion v1_dense path). None = no baseline yet.
        self.prev_own_at_base = [[None] * 4 for _ in range(batch_size)]

        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        # trajectory[i] = list of (enc, rmasks, rlegal, chosen_rank, v_pred, cp)
        self.trajectory = [[] for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)
        self.games_played = 0
        self.game_lengths = []

        # Per-game opp binding. For SelfPlay-named games, opp_bot is None.
        self.model_players = [random.choice([0, 2]) for _ in range(batch_size)]
        self.opp_names = ["SelfPlay"] * batch_size
        self.opp_bots = [None] * batch_size
        self.opp_game_counts = collections.Counter()
        # Rolling per-opp results (model_won bool) over the last N finalized
        # opp games. Used to compute `recent_opponent_stats` for the dashboard.
        self.recent_opp_results = collections.deque(maxlen=500)
        # Rolling overall win rate over the last 100 finalized games (for
        # dashboard's win_rate_100 metric).
        self.recent_win_results = collections.deque(maxlen=100)
        for i in range(batch_size):
            self._init_opp(i)

    def _init_opp(self, i):
        if not self._opp_enabled:
            self.opp_names[i] = "SelfPlay"
            self.opp_bots[i] = None
            self.opp_game_counts["SelfPlay"] += 1
            return
        name = self.opp_sampler()
        self.opp_names[i] = name
        if name == "SelfPlay":
            self.opp_bots[i] = None
        else:
            opp_player = 2 if self.model_players[i] == 0 else 0
            self.opp_bots[i] = self.get_bot_fn(name, opp_player)
        self.opp_game_counts[name] += 1

    def _reset(self, i):
        self.games[i] = ludo_cpp.create_initial_state_2p()
        self.trajectory[i] = []
        self.consec_sixes[i] = 0
        self.step_count[i] = 0
        self.model_players[i] = random.choice([0, 2])
        self.prev_own_at_base[i] = [None] * 4
        self._init_opp(i)

    def _record_game_result(self, i, model_won):
        """Update rolling per-opp results + last-100 WR tracker + Elo."""
        opp = self.opp_names[i]
        # Track WR per opp (excludes SelfPlay since "model_won" isn't
        # meaningful when both players are the student).
        if opp != "SelfPlay":
            self.recent_opp_results.append((opp, bool(model_won)))
        self.recent_win_results.append(bool(model_won))
        # ELO update — only opp games (SelfPlay is model-vs-model, no signal).
        if self.elo_tracker is not None and opp != "SelfPlay":
            mp = self.model_players[i]
            op = 2 - mp
            identities = ["Inactive"] * 4
            identities[mp] = "Model"
            identities[op] = opp
            winner_idx = mp if model_won else op
            self.elo_tracker.update_from_game(
                identities, winner_idx, game_num=self.games_played,
            )

    def _finalize(self, i, winner):
        """Convert trajectory[i] entries into (..., G) tuples.

        With shaped rewards enabled, trajectory entries are 7-tuples
        (enc, rm, rl, chosen_rank, v_pred, cp, shaped_r). G per entry =
        sum of future shaped rewards by the same cp + terminal ±1 for
        that cp. Reverse pass, γ=1.

        With shaped rewards disabled (terminal-only), entries are
        6-tuples and G = ±1 from cp's terminal POV.
        """
        traj = self.trajectory[i]
        out = [None] * len(traj)
        # Reverse pass: γ=0.999 DISCOUNTED return-to-go per cp, matching the
        # champion recipe (trainer_v10.py: R = r_t + γ·R). Each player's
        # terminal ±1 enters at their LAST move and discounts forward
        # through their earlier moves. Previously this was an UNDISCOUNTED
        # sum (G = Σshaped + terminal), which inflates long-game returns and
        # diverges from the proven recipe. (2026-06-01 reward revert.)
        GAMMA = 0.999
        future_return = {}  # discounted return-to-go per cp (incl. terminal)
        total_shaped = 0.0
        for t in reversed(range(len(traj))):
            entry = traj[t]
            if len(entry) == 7:
                enc, rm, rl, cr, vp, cp_t, shaped_t = entry
            else:
                enc, rm, rl, cr, vp, cp_t = entry
                shaped_t = 0.0
            total_shaped += shaped_t
            terminal = 1.0 if cp_t == winner else -1.0
            if cp_t not in future_return:
                # cp's last decision: terminal enters here (discounted once).
                G = shaped_t + GAMMA * terminal
            else:
                G = shaped_t + GAMMA * future_return[cp_t]
            future_return[cp_t] = G
            out[t] = (enc, rm, rl, cr, vp, G)
        # Telemetry: avg shaped reward per move (only meaningful with
        # shaped rewards enabled).
        if traj:
            self.recent_shaped_per_game.append(total_shaped / max(1, len(traj)))
        # Record outcome from the student's POV.
        self._record_game_result(i, winner == self.model_players[i])
        self.games_played += 1
        self.game_lengths.append(len(traj))
        if len(self.game_lengths) > 200:
            self.game_lengths.pop(0)
        self.trajectory[i] = []
        return out

    def _truncate(self, i):
        """Treat in-progress trajectory as a draw.

        With shaped rewards: G per entry = sum of future shaped (no
        terminal, since no winner). Without shaped: G = 0.
        """
        traj = self.trajectory[i]
        out = [None] * len(traj)
        # γ=0.999 discounted return-to-go, no terminal (draw). Mirrors
        # _finalize's discounting for consistency.
        GAMMA = 0.999
        future_return = {}
        total_shaped = 0.0
        for t in reversed(range(len(traj))):
            entry = traj[t]
            if len(entry) == 7:
                enc, rm, rl, cr, vp, cp_t, shaped_t = entry
            else:
                enc, rm, rl, cr, vp, cp_t = entry
                shaped_t = 0.0
            total_shaped += shaped_t
            G = shaped_t + GAMMA * future_return.get(cp_t, 0.0)
            future_return[cp_t] = G
            out[t] = (enc, rm, rl, cr, vp, G)
        if traj:
            self.recent_shaped_per_game.append(total_shaped / max(1, len(traj)))
        # Treat truncation as a half-loss for WR tracking (neither side
        # actually finished — matches V15 trainer's "draw" treatment).
        self._record_game_result(i, False)
        self.games_played += 1
        self.game_lengths.append(len(traj))
        if len(self.game_lengths) > 200:
            self.game_lengths.pop(0)
        self.trajectory[i] = []
        return out

    def get_recent_opp_stats(self):
        """Aggregate the rolling deque into {opp_name: {wins, games, win_rate}}."""
        counts = {}
        for name, won in self.recent_opp_results:
            d = counts.setdefault(name, {"wins": 0, "games": 0})
            d["games"] += 1
            if won:
                d["wins"] += 1
        out = {}
        for name, d in counts.items():
            if d["games"] == 0:
                continue
            out[name] = {
                "wins": d["wins"], "games": d["games"],
                "win_rate": round(100.0 * d["wins"] / d["games"], 1),
            }
        return out

    def get_win_rate_100(self):
        """Recent-100-game WR (percentage). 0 if no games yet."""
        if not self.recent_win_results:
            return 0.0
        return 100.0 * sum(self.recent_win_results) / len(self.recent_win_results)

    def spin_to_decision(self):
        """Advance every game to a student-decision state.

        Inline-applies any opp moves (and any necessary game-resets that
        happen mid-spin if a game terminates during an opp move).
        Returns the usual decision tensors PLUS a list of finished
        trajectories collected during opp moves (which need to be
        merged into the training pool by the caller).
        """
        finished_from_opp = []
        decision_idxs = []
        cps = []
        encs = []
        rmasks_list = []
        rlegals = []
        legal_lists = []
        rank_token_ids_list = []

        for i in range(self.batch_size):
            while True:
                game = self.games[i]
                # ── Handle prior termination / truncation (left over
                # from a previous apply_actions call, OR from an
                # opp-move termination earlier in this spin) ──
                if game.is_terminal:
                    if self.trajectory[i]:
                        winner = int(ludo_cpp.get_winner(game))
                        finished_from_opp.extend(self._finalize(i, winner))
                    self._reset(i)
                    game = self.games[i]
                elif self.step_count[i] >= self.max_game_len:
                    if self.trajectory[i]:
                        finished_from_opp.extend(self._truncate(i))
                    self._reset(i)
                    game = self.games[i]

                cp = int(game.current_player)
                if not game.active_players[cp]:
                    nxt = (cp + 1) % 4
                    while not game.active_players[nxt]:
                        nxt = (nxt + 1) % 4
                    game.current_player = nxt
                    continue

                if game.current_dice_roll == 0:
                    roll = random.randint(1, 6)
                    game.current_dice_roll = roll
                    if roll == 6:
                        self.consec_sixes[i, cp] += 1
                    else:
                        self.consec_sixes[i, cp] = 0
                    if self.consec_sixes[i, cp] >= 3:
                        nxt = (cp + 1) % 4
                        while not game.active_players[nxt]:
                            nxt = (nxt + 1) % 4
                        game.current_player = nxt
                        game.current_dice_roll = 0
                        self.consec_sixes[i, cp] = 0
                        continue

                legal = ludo_cpp.get_legal_moves(game)
                if not legal:
                    nxt = (cp + 1) % 4
                    while not game.active_players[nxt]:
                        nxt = (nxt + 1) % 4
                    game.current_player = nxt
                    game.current_dice_roll = 0
                    continue

                # ── If this is an opp-bound game and it's the opp's
                # turn, apply the bot's move inline and continue
                # spinning (don't emit a decision state for training).
                if self.opp_bots[i] is not None and cp != self.model_players[i]:
                    action = self.opp_bots[i].select_move(game, list(legal))
                    self.games[i] = ludo_cpp.apply_move(game, int(action))
                    self.step_count[i] += 1
                    continue

                # ── Student's turn: emit decision state ──
                pp = game.player_positions[cp]
                _, rank_tokens = state_to_rank_mapping(pp)
                rank_legal = legal_mask_per_rank(legal, rank_tokens)
                enc = encode_state_v18_symmetric(game).astype(np.float32)
                rmasks = compute_rank_masks(game).astype(np.float32)

                decision_idxs.append(i)
                cps.append(cp)
                encs.append(enc)
                rmasks_list.append(rmasks)
                rlegals.append(rank_legal.astype(np.float32))
                legal_lists.append(legal)
                rank_token_ids_list.append(rank_tokens)
                break

        # Guard against the (impossible-in-practice) case that no game
        # produced a decision state this round. Caller checks length.
        encs_arr = (np.stack(encs, axis=0) if encs else
                    np.zeros((0, V18_CHANNELS, 15, 15), dtype=np.float32))
        rmasks_arr = (np.stack(rmasks_list, axis=0) if rmasks_list else
                      np.zeros((0, 4, 15, 15), dtype=np.float32))
        rlegals_arr = (np.stack(rlegals, axis=0) if rlegals else
                       np.zeros((0, 4), dtype=np.float32))

        return (
            decision_idxs,
            cps,
            encs_arr,
            rmasks_arr,
            rlegals_arr,
            legal_lists,
            rank_token_ids_list,
            finished_from_opp,
        )

    def apply_actions(self, decision_idxs, cps, encs, rmasks, rlegals,
                      legal_lists, rank_token_ids_list, ranks, v_preds):
        """ranks: chosen rank per state (B,). Map to token-id, apply, store.

        With --use-shaped-reward (default), each trajectory entry stores
        the per-step shaped reward computed via compute_shaped_reward()
        on the (state_before → state_after) transition for the moving
        player. Otherwise the entry is the legacy 6-tuple.
        """
        finished = []
        for k, i in enumerate(decision_idxs):
            chosen_rank = int(ranks[k])
            # Defensive: if rank_legal[k][chosen_rank] == 0, fall back to first legal rank
            if rlegals[k][chosen_rank] == 0:
                legal_ranks = np.where(rlegals[k] > 0)[0]
                chosen_rank = int(legal_ranks[0]) if len(legal_ranks) else 0
            token_id = rank_to_token_id(chosen_rank, legal_lists[k], rank_token_ids_list[k])
            if token_id < 0 or token_id not in legal_lists[k]:
                token_id = legal_lists[k][0]
                chosen_rank = 0  # mismatch fallback — shouldn't happen
            # Snapshot state before applying so we can compute shaped reward.
            state_before = self.games[i]
            self.games[i] = ludo_cpp.apply_move(state_before, int(token_id))
            self.step_count[i] += 1
            # Per-step shaped reward (v1 recipe: leave-base / forward /
            # home-stretch / score / capture / killed + strategic).
            if self.use_shaped_reward:
                shaped = float(
                    compute_shaped_reward(state_before, self.games[i], int(cps[k]))
                )
                # Got-killed −0.20 per token captured during the opponent's
                # turn(s) since this player's previous own decision. Compare
                # at-base count NOW (pre-move) vs the post-move snapshot from
                # their last decision. (2026-06-10 fix — see __init__ note.)
                _cp = int(cps[k])
                _cur_at_base = sum(
                    1 for p in state_before.player_positions[_cp] if int(p) == -1
                )
                shaped += float(compute_kill_penalty(
                    self.prev_own_at_base[i][_cp], _cur_at_base,
                ))
                self.prev_own_at_base[i][_cp] = sum(
                    1 for p in self.games[i].player_positions[_cp] if int(p) == -1
                )
                # Bias penalties (champion recipe, LUDO_BIAS_PENALTIES=1):
                # penalize leaving a laggard token at base while pushing the
                # leader, and moving the leader into opponent capture range.
                # ≤ 0; sums into the shaped reward. (2026-06-01 reward revert.)
                if _BIAS_PENALTIES_ENABLED:
                    try:
                        _bias_total, _ = compute_bias_penalties(
                            state_before, self.games[i], int(cps[k]),
                            {
                                'dice': int(getattr(state_before, 'current_dice_roll', 0)),
                                'legal_moves': list(legal_lists[k]),
                                'action': int(token_id),
                                'move_count': int(self.step_count[i]),
                            },
                        )
                        shaped += float(_bias_total)
                    except Exception:
                        pass  # bias penalty must never crash a rollout
                self.trajectory[i].append((
                    encs[k], rmasks[k], rlegals[k],
                    chosen_rank, float(v_preds[k]), int(cps[k]),
                    shaped,
                ))
            else:
                self.trajectory[i].append((
                    encs[k], rmasks[k], rlegals[k],
                    chosen_rank, float(v_preds[k]), int(cps[k]),
                ))

            game = self.games[i]
            if game.is_terminal:
                winner = int(ludo_cpp.get_winner(game))
                finished.extend(self._finalize(i, winner))
            elif self.step_count[i] >= self.max_game_len:
                finished.extend(self._truncate(i))

        return finished


# ── Eval (V13.5 vs slim strong-bot roster) ─────────────────────────────────
def quick_eval(student, device, n_games=200,
               mcts_pure_sims=30, mcts_pure_rollouts=4):
    """Mirrors v15_bot_eval.evaluate_v15_against_bots' contract: samples
    uniformly from EVAL_BOT_NAMES_FAST and reports overall + per-bot WR.

    Returns: (overall_wr_percent, per_bot_dict).
    """
    from src.config import MAX_MOVES_PER_GAME
    bot_types = list(EVAL_BOT_NAMES_FAST)
    student.eval()
    wins = 0
    per_bot_wins = collections.Counter()
    per_bot_games = collections.Counter()
    for _ in range(n_games):
        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        bot_type = random.choice(bot_types)
        bot = get_unified_bot(bot_type, player_id=opp_player,
                              mcts_pure_sims=mcts_pure_sims,
                              mcts_pure_rollouts=mcts_pure_rollouts)
        per_bot_games[bot_type] += 1
        state = ludo_cpp.create_initial_state_2p()
        csix = [0, 0, 0, 0]
        mc = 0
        while not state.is_terminal and mc < MAX_MOVES_PER_GAME:
            cp = int(state.current_player)
            if not state.active_players[cp]:
                n = (cp + 1) % 4
                while not state.active_players[n]:
                    n = (n + 1) % 4
                state.current_player = n
                continue
            if state.current_dice_roll == 0:
                state.current_dice_roll = random.randint(1, 6)
                if state.current_dice_roll == 6:
                    csix[cp] += 1
                else:
                    csix[cp] = 0
                if csix[cp] >= 3:
                    n = (cp + 1) % 4
                    while not state.active_players[n]:
                        n = (n + 1) % 4
                    state.current_player = n
                    state.current_dice_roll = 0
                    csix[cp] = 0
                    continue
            legal = ludo_cpp.get_legal_moves(state)
            if not legal:
                n = (cp + 1) % 4
                while not state.active_players[n]:
                    n = (n + 1) % 4
                state.current_player = n
                state.current_dice_roll = 0
                continue
            if cp == model_player:
                if len(legal) == 1:
                    action = legal[0]
                else:
                    pp = state.player_positions[cp]
                    _, rank_tokens = state_to_rank_mapping(pp)
                    rank_legal = legal_mask_per_rank(legal, rank_tokens)
                    enc = encode_state_v18_symmetric(state).astype(np.float32)
                    rm = compute_rank_masks(state).astype(np.float32)
                    with torch.no_grad():
                        x = torch.from_numpy(enc).unsqueeze(0).to(device)
                        rmt = torch.from_numpy(rm).unsqueeze(0).to(device)
                        lmt = torch.from_numpy(rank_legal.astype(np.float32)).unsqueeze(0).to(device)
                        logits = student.forward_policy_only(x, rmt, lmt)
                        rank = int(logits.argmax(dim=1).item())
                    action = rank_to_token_id(rank, legal, rank_tokens)
                    if action not in legal:
                        action = random.choice(legal)
            else:
                action = bot.select_move(state, list(legal))
            state = ludo_cpp.apply_move(state, int(action))
            mc += 1
        if state.is_terminal and ludo_cpp.get_winner(state) == model_player:
            wins += 1
            per_bot_wins[bot_type] += 1
    student.train()
    per_bot = {}
    for bt, g in per_bot_games.items():
        if g > 0:
            per_bot[bt] = {
                "win_rate": 100.0 * per_bot_wins[bt] / g,
                "wins": int(per_bot_wins[bt]),
                "games": int(g),
            }
    return 100 * wins / n_games, per_bot


# ── In-memory H2H vs V13.2 (mirrored seeds, greedy) ────────────────────────
def _v135_select(student, device, state, legal):
    if len(legal) == 1:
        return legal[0]
    pp = state.player_positions[int(state.current_player)]
    _, rank_tokens = state_to_rank_mapping(pp)
    rank_legal = legal_mask_per_rank(legal, rank_tokens)
    enc = encode_state_v18_symmetric(state).astype(np.float32)
    rm = compute_rank_masks(state).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(enc).unsqueeze(0).to(device)
        rmt = torch.from_numpy(rm).unsqueeze(0).to(device)
        lmt = torch.from_numpy(rank_legal.astype(np.float32)).unsqueeze(0).to(device)
        logits = student.forward_policy_only(x, rmt, lmt)
        rank = int(logits.argmax(dim=1).item())
    a = rank_to_token_id(rank, legal, rank_tokens)
    return a if a in legal else legal[0]


def _v132_select(v132_model, device, state, legal):
    if len(legal) == 1:
        return legal[0]
    enc = encode_state_v17(state).astype(np.float32)
    mask = np.zeros(4, dtype=np.float32)
    for a in legal:
        mask[a] = 1.0
    with torch.no_grad():
        x = torch.from_numpy(enc).unsqueeze(0).to(device)
        m = torch.from_numpy(mask).unsqueeze(0).to(device)
        policy, _, _ = v132_model(x, m)
        a = int(policy.argmax(dim=1).item())
    return a if a in legal else legal[0]


def _play_one(state, model_player, picks):
    """Run a game to terminal with the two selectors keyed by player id."""
    csix = [0, 0, 0, 0]
    mc = 0
    while not state.is_terminal and mc < 400:
        cp = int(state.current_player)
        if state.current_dice_roll == 0:
            state.current_dice_roll = random.randint(1, 6)
            if state.current_dice_roll == 6:
                csix[cp] += 1
            else:
                csix[cp] = 0
            if csix[cp] >= 3:
                n = (cp + 1) % 4
                while not state.active_players[n]:
                    n = (n + 1) % 4
                state.current_player = n
                state.current_dice_roll = 0
                csix[cp] = 0
                continue
        legal = ludo_cpp.get_legal_moves(state)
        if not legal:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            state.current_dice_roll = 0
            continue
        action = picks[cp](state, list(legal))
        state = ludo_cpp.apply_move(state, int(action))
        mc += 1
    if state.is_terminal:
        return int(ludo_cpp.get_winner(state))
    return -1  # truncated


def quick_h2h_vs_v132(student, v132_model, device, n_games=200, seed_base=12345):
    """In-memory V13.5 vs V13.2 H2H, mirrored seeds (V13.5 plays each seed
    once as P0 and once as P2). Returns (v135_wins, v132_wins, draws).
    """
    student.eval()
    v135_pick = lambda s, l: _v135_select(student, device, s, l)
    v132_pick = lambda s, l: _v132_select(v132_model, device, s, l)
    v135_w = v132_w = draws = 0
    for i in range(n_games):
        seed = seed_base + (i // 2)
        random.seed(seed)
        np.random.seed(seed)
        v135_player = 0 if (i % 2 == 0) else 2
        v132_player = 2 if v135_player == 0 else 0
        state = ludo_cpp.create_initial_state_2p()
        picks = {v135_player: v135_pick, v132_player: v132_pick}
        winner = _play_one(state, v135_player, picks)
        if winner == v135_player:
            v135_w += 1
        elif winner == v132_player:
            v132_w += 1
        else:
            draws += 1
    student.train()
    return v135_w, v132_w, draws


# ── Dashboard — RICH (mirrors V15 dashboard at td_ludo_v15/rich/dashboard.py)
# Endpoints:
#   /api/stats    — live trainer metrics (loss, entropy, GPM, ELO, opp stats)
#   /api/metrics  — list of per-eval snapshots (overall + per-bot WR)
#   /api/elo      — stub (V13.5 trainer doesn't run a full ELO tracker)
#   /api/games    — stub (no GameDB yet)
#   /api/system   — psutil CPU/RAM/PID
#   /api/chain    — pipeline stage status
class _RichDashboardHandler(SimpleHTTPRequestHandler):
    """Same /api/* shape as V15's v13_dashboard.html consumes."""

    def __init__(self, *args, directory=None,
                 stats_path=None, metrics_path=None, chain_path=None,
                 landing=None, **kw):
        self._stats_path = stats_path
        self._metrics_path = metrics_path
        self._chain_path = chain_path
        self._landing = landing
        super().__init__(*args, directory=directory, **kw)

    def log_message(self, *a, **kw):
        return

    def do_GET(self):  # noqa: N802
        try:
            if self.path in ("/", ""):
                if self._landing:
                    self.path = "/" + self._landing
                return super().do_GET()
            if self.path == "/api/stats":
                return self._serve_json_file(self._stats_path)
            if self.path == "/api/metrics":
                return self._serve_json_file(self._metrics_path)
            if self.path == "/api/chain":
                return self._serve_json_file(self._chain_path)
            if self.path == "/api/sl_stats":
                # Some old dashboards poll this; fall through to stats.
                return self._serve_json_file(self._stats_path)
            if self.path == "/api/elo":
                return self._send_json(b'{"rankings": [], "history": {}}')
            if self.path.startswith("/api/games"):
                return self._send_json(b'{"games": []}')
            if self.path == "/api/system":
                return self._serve_system()
            return super().do_GET()
        except (ConnectionResetError, BrokenPipeError):
            pass

    def _serve_json_file(self, path):
        if not path or not os.path.exists(path):
            self.send_response(404); self.end_headers(); return
        try:
            with open(path) as f:
                data = f.read()
        except OSError:
            self.send_response(500); self.end_headers(); return
        self._send_json(data.encode())

    def _serve_system(self):
        try:
            import psutil
            payload = {
                "cpu_percent": float(psutil.cpu_percent(interval=None)),
                "memory_percent": float(psutil.virtual_memory().percent),
                "pid": os.getpid(),
            }
        except Exception:
            payload = {"cpu_percent": 0.0, "memory_percent": 0.0,
                       "pid": os.getpid()}
        self._send_json(json.dumps(payload).encode())

    def _send_json(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def start_dashboard(port, stats_path, metrics_path, chain_path, dashboard_dir):
    """Start rich dashboard. Prefers v13_dashboard.html as the landing page
    (the V15-style rich UI). Falls back to rl_dashboard.html if the v13
    HTML isn't present in `dashboard_dir`."""
    landing = None
    for cand in ("v13_dashboard.html", "rl_dashboard.html",
                 "sl_dashboard.html", "index.html"):
        if os.path.exists(os.path.join(dashboard_dir, cand)):
            landing = cand
            break
    handler = functools.partial(
        _RichDashboardHandler, directory=dashboard_dir,
        stats_path=stats_path, metrics_path=metrics_path,
        chain_path=chain_path, landing=landing,
    )
    server = HTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[Dashboard] http://0.0.0.0:{port}/{landing or ''}", flush=True)


def write_stats_json(
    stats_path, *,
    total_games, total_updates, win_rate_100, policy_entropy,
    avg_policy_loss, avg_value_loss, approx_kl, gpm,
    best_eval_wr, temperature, recent_opp_stats=None,
    opp_game_counts=None, main_elo=1500.0, eval_wr=None,
    elo_rankings=None, run_info=None,
):
    """Write /api/stats JSON in V15-compatible shape.

    `run_info` (dict) carries dashboard text that varies per model/run
    (model_name, arch_summary, description, eval cadence, etc.). The
    dashboard HTML reads these and renders them into the layout so the
    same HTML file works for V13.5, V15, V15.1, etc. without edits.
    """
    if elo_rankings is None:
        elo_rankings = [{"name": "Model", "elo": float(main_elo)}]
    payload = {
        "total_games": int(total_games),
        "total_updates": int(total_updates),
        "win_rate_100": float(round(win_rate_100, 1)),
        "policy_entropy": float(round(policy_entropy, 4)),
        "avg_value_loss": float(round(avg_value_loss, 6)),
        "avg_policy_loss": float(round(avg_policy_loss, 6)),
        "avg_advantage": 0.0,
        "clip_fraction": 0.0,
        "approx_kl": float(round(approx_kl, 4)),
        "temperature": float(temperature),
        "games_per_minute": float(round(gpm, 1)),
        "best_eval_win_rate": float(best_eval_wr),
        "ghost_count": 0,
        "is_stagnated": False,
        "play_alarm": False,
        "timestamp": time.time(),
        "main_elo": float(main_elo),
        "elo_rankings": list(elo_rankings),
        "opponent_stats": dict(opp_game_counts or {}),
        "db_total": int(total_games),
        "recent_opponent_stats": recent_opp_stats or {},
        "run_info": run_info or {},
    }
    if eval_wr is not None:
        # Dashboard expects FRACTION (0..1).
        payload["eval_win_rate"] = float(eval_wr)
    tmp = stats_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, stats_path)


def append_metrics_snapshot(metrics_path, snapshot):
    """Append one eval snapshot to /api/metrics list."""
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


# ── Training step ──────────────────────────────────────────────────────────
def train_on_chunk(student, optimizer, chunk, device, args, kl_teacher=None):
    """chunk: list of (enc, rmasks, rlegal, chosen_rank, v_pred_old, G)."""
    if not chunk:
        return None
    encs = np.stack([c[0] for c in chunk], axis=0)
    rmasks = np.stack([c[1] for c in chunk], axis=0)
    rlegals = np.stack([c[2] for c in chunk], axis=0)
    ranks = np.array([c[3] for c in chunk], dtype=np.int64)
    v_old = np.array([c[4] for c in chunk], dtype=np.float32)
    Gs_raw = np.array([c[5] for c in chunk], dtype=np.float32)

    # ── Return + advantage normalization (ported from production
    # trainer.py:237-272, the recipe that produced the V13.5 champion).
    # WITHOUT this, dense shaped returns (≈+4, always positive) overwhelm
    # the [-1,1] win-prob value head → value_loss ~7, oversized
    # advantages (~+3), policy gradient explosion → entropy blowup +
    # regression (the exact failure this V13.6 run hit). Normalizing
    # returns to ~N(0,1) via a slow EMA keeps the value target tractable;
    # normalizing advantages keeps the policy gradient well-scaled.
    _b_mean = float(Gs_raw.mean())
    _b_std = float(Gs_raw.std())
    _st = train_on_chunk._ret_stats
    if not _st["init"]:
        _st["mean"], _st["std"], _st["init"] = _b_mean, max(_b_std, 1e-6), True
    else:
        _st["mean"] = 0.99 * _st["mean"] + 0.01 * _b_mean
        _st["std"] = 0.99 * _st["std"] + 0.01 * max(_b_std, 1e-6)
    Gs = (Gs_raw - _st["mean"]) / (_st["std"] + 1e-8)
    # Advantages over the whole chunk, normalized once.
    _adv_raw = Gs - v_old
    advs = (_adv_raw - _adv_raw.mean()) / (_adv_raw.std() + 1e-8)

    N = encs.shape[0]
    metrics = {"loss": 0.0, "loss_pol": 0.0, "loss_val": 0.0, "entropy": 0.0,
               "loss_kl": 0.0, "n_steps": 0}

    for epoch in range(args.train_epochs):
        order = np.random.permutation(N)
        for s in range(0, N, args.minibatch_size):
            idx = order[s:s + args.minibatch_size]
            x = torch.from_numpy(encs[idx]).to(device, dtype=torch.float32)
            rm = torch.from_numpy(rmasks[idx]).to(device, dtype=torch.float32)
            rl = torch.from_numpy(rlegals[idx]).to(device, dtype=torch.float32)
            r = torch.from_numpy(ranks[idx]).to(device)
            G = torch.from_numpy(Gs[idx]).to(device)            # normalized return (value target)
            advantage = torch.from_numpy(advs[idx]).to(device)  # normalized advantage

            _out = student(x, rm, rl)
            policy, win_prob = _out[0], _out[1]
            v = 2.0 * win_prob - 1.0

            multi = (rl.sum(dim=1) > 1).float()
            multi_n = multi.sum().clamp(min=1.0)

            log_p = torch.log(policy.gather(1, r.unsqueeze(1)).squeeze(1) + 1e-8)
            loss_pol = -(advantage * log_p * multi).sum() / multi_n

            loss_val = F.smooth_l1_loss(v, G)  # champion recipe (was MSE)

            entropy_per = -(policy * torch.log(policy + 1e-8)).sum(dim=1)
            entropy = (entropy_per * multi).sum() / multi_n
            loss_ent = -args.entropy_coeff * entropy

            loss = loss_pol + args.value_coeff * loss_val + loss_ent

            loss_kl_val = 0.0
            if kl_teacher is not None and args.kl_anchor_coeff > 0:
                with torch.no_grad():
                    _t_out = kl_teacher(x, rm, rl)
                    t_pol = _t_out[0]
                loss_kl = F.kl_div(torch.log(policy + 1e-8), t_pol,
                                   reduction="batchmean", log_target=False)
                loss = loss + args.kl_anchor_coeff * loss_kl
                loss_kl_val = float(loss_kl.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            n = idx.shape[0]
            metrics["loss"]     += float(loss.item()) * n
            metrics["loss_pol"] += float(loss_pol.item()) * n
            metrics["loss_val"] += float(loss_val.item()) * n
            metrics["entropy"]  += float(entropy.item()) * n
            metrics["loss_kl"]  += loss_kl_val * n
            metrics["n_steps"]  += n

    if metrics["n_steps"]:
        for k in ("loss", "loss_pol", "loss_val", "entropy", "loss_kl"):
            metrics[k] /= metrics["n_steps"]
    return metrics


# Persistent running return statistics for normalization (EMA across
# chunks). Lives on the function so it survives across train_on_chunk
# calls without a global. Re-initializes on process start (the EMA
# re-converges within a few chunks, fine for resume).
train_on_chunk._ret_stats = {"init": False, "mean": 0.0, "std": 1.0}


# ── Loaders ────────────────────────────────────────────────────────────────
def _load_v135(path, args, device):
    model = V135Symmetric(
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        head_hidden=args.head_hidden,
        in_channels=V18_CHANNELS,
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    # ── Production-adapter checkpoints (e.g. v135_prod_rl_local) wrap
    # V135Symmetric inside V135ProductionAdapter, so their state_dict
    # keys are prefixed with "inner.". Strip the prefix here to load
    # them directly into the bare V135Symmetric. (The same logic exists
    # in load_v135_opponent.) Without this, load_state_dict(strict=False)
    # silently loads NOTHING → the model is left at random init.
    if any(k.startswith("inner.") for k in sd):
        sd = {k[len("inner."):]: v for k, v in sd.items() if k.startswith("inner.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] _load_v135: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[WARN] _load_v135: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    return model, ckpt


def load_v135_kl_teacher(path, args, device):
    model, _ = _load_v135(path, args, device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_v132_h2h_opponent(path, args, device):
    model = MinimalCNN14(
        num_res_blocks=args.v132_num_res_blocks,
        num_channels=args.v132_num_channels,
        in_channels=17,
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)

    if args.run_name:
        os.environ["TD_LUDO_RUN_NAME"] = args.run_name
    from src.config import CHECKPOINT_DIR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    rl_stats_path = os.path.join(CHECKPOINT_DIR, "rl_stats.json")
    chain_path = os.path.join(CHECKPOINT_DIR, "chain_status.json")
    rl_log_path = os.path.join(CHECKPOINT_DIR, "rl.log")
    stats_path = os.path.join(CHECKPOINT_DIR, "stats.json")
    metrics_path = os.path.join(CHECKPOINT_DIR, "metrics.json")
    elo_path = os.path.join(CHECKPOINT_DIR, "elo.json")

    if args.resume:
        args.init = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
        if not os.path.exists(args.init):
            print(f"ERROR: --resume but {args.init} not found"); sys.exit(1)
    elif not args.init:
        print("ERROR: either --init or --resume is required"); sys.exit(1)

    if args.kl_teacher is None and args.kl_anchor_coeff > 0:
        # Default: anchor to the same checkpoint we initialized from (= V13.5_SL)
        args.kl_teacher = args.init

    print("=" * 70)
    print("V13.5 RL — self-play REINFORCE (rank-indexed)")
    print("=" * 70)
    print(f"  device:           {device}")
    print(f"  init:             {args.init}")
    print(f"  KL teacher:       {args.kl_teacher}")
    print(f"  H2H opponent:     {args.h2h_opponent or '(none)'}")
    print(f"  checkpoint dir:   {CHECKPOINT_DIR}")
    print(f"  parallel_games:   {args.parallel_games}")
    print(f"  train_chunk:      {args.train_chunk}")
    print(f"  minibatch:        {args.minibatch_size}  × {args.train_epochs} epochs")
    print(f"  target_states:    {args.target_states:,}")
    print(f"  lr:               {args.lr} → {args.lr_end} (cosine)")
    print(f"  entropy_coeff:    {args.entropy_coeff}")
    print(f"  kl_anchor_coeff:  {args.kl_anchor_coeff}")
    print(f"  arch:             V135Symmetric ({args.num_res_blocks}x{args.num_channels})")
    print("=" * 70)

    student, _ckpt = _load_v135(args.init, args, device)
    _resume_meta = None
    if isinstance(_ckpt, dict) and "model_state_dict" in _ckpt and args.resume:
        _resume_meta = _ckpt
    student.to(device).train()
    print(f"[Student] V135Symmetric params: {sum(p.numel() for p in student.parameters()):,}  "
          f"(loaded from {args.init})")

    kl_teacher = None
    if args.kl_teacher and args.kl_anchor_coeff > 0:
        kl_teacher = load_v135_kl_teacher(args.kl_teacher, args, device)
        print(f"[KL teacher] V13.5_SL loaded for KL anchor (coeff={args.kl_anchor_coeff})")

    v132_h2h = None
    if args.h2h_opponent and args.h2h_gate_every > 0:
        v132_h2h = load_v132_h2h_opponent(args.h2h_opponent, args, device)
        print(f"[H2H] V13.2 opponent loaded ({args.h2h_games} games every "
              f"{args.h2h_gate_every:,} states)")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    if _resume_meta and "optimizer_state_dict" in _resume_meta:
        try:
            optimizer.load_state_dict(_resume_meta["optimizer_state_dict"])
            print("[Resume] Optimizer state restored")
        except Exception as e:
            print(f"[Resume] optimizer state mismatch ({e}); starting fresh optimizer")

    def write_chain(phase):
        with open(chain_path, "w") as f:
            json.dump({"stage": "RL", "phase": phase, "arch": "v13_5",
                       "run_name": os.environ.get("TD_LUDO_RUN_NAME", "-"),
                       "ts": int(time.time())}, f)
    write_chain("training")

    if not args.no_dashboard:
        dash_dir = os.path.dirname(os.path.abspath(__file__))
        start_dashboard(args.port, stats_path, metrics_path,
                        chain_path, dash_dir)

    rl_log = open(rl_log_path, "a")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        rl_log.write(line + "\n")
        rl_log.flush()

    # ── Build opponent mix ────────────────────────────────────────────
    # Map CLI flag → bot registry name. SelfPlay is special-cased inside
    # SelfPlayEnv (no bot, both players use student).
    opp_specs = [
        ("SelfPlay", args.opp_weight_self),
        # Legacy scripted
        ("Aggressive", args.opp_weight_aggressive),
        ("Defensive", args.opp_weight_defensive),
        ("Racing", args.opp_weight_racing),
        ("Heuristic", args.opp_weight_heuristic),
        ("Expert", args.opp_weight_expert),
        ("Random", args.opp_weight_random),
        # Strong (Expectimax + variants)
        ("Expectimax", args.opp_weight_expectimax),
        ("AggressiveExpectimax", args.opp_weight_aggressive_expectimax),
        ("DefensiveExpectimax", args.opp_weight_defensive_expectimax),
        ("RacingExpectimax", args.opp_weight_racing_expectimax),
        ("MinimaxExpectimax", args.opp_weight_minimax_expectimax),
        ("BlockadeExpectimax", args.opp_weight_blockade_expectimax),
        ("VoteExpectimax", args.opp_weight_vote_expectimax),
        ("Depth2Expectimax", args.opp_weight_depth2_expectimax),
        ("Depth2AggressiveExpectimax", args.opp_weight_depth2_aggressive),
        ("Depth2DefensiveExpectimax", args.opp_weight_depth2_defensive),
        # MCTS family
        ("MCTSPure", args.opp_weight_mcts_pure),
        ("MCTSExpertPrior", args.opp_weight_mcts_expert_prior),
        ("MCTSExpectimaxPrior", args.opp_weight_mcts_expectimax_prior),
        # Rule bots
        ("MaxCapture", args.opp_weight_max_capture),
        ("TwoStack", args.opp_weight_two_stack),
        ("HomeRush", args.opp_weight_home_rush),
        ("StackHomeRush", args.opp_weight_stack_home_rush),
        # Adaptive
        ("AdaptiveExpectimax", args.opp_weight_adaptive_expectimax),
        # Ghost pool (frozen past students — handled specially in factory)
        ("Ghost", args.opp_weight_ghost),
    ]
    opp_probs = {n: w for n, w in opp_specs if w > 0}
    if not opp_probs:
        opp_probs = {"SelfPlay": 1.0}
    print("[Opp mix]")
    _total_weight = sum(opp_probs.values())
    for n, w in sorted(opp_probs.items(), key=lambda kv: -kv[1]):
        print(f"  {n:32s} weight={w:>6.2f}  ({100*w/_total_weight:.1f}%)")

    # Ghost pool — only active when --opp-weight-ghost > 0.
    ghost_pool = None
    if args.opp_weight_ghost > 0:
        ghost_pool = GhostPool(
            os.path.join(CHECKPOINT_DIR, "ghosts"),
            args, device, max_ghosts=args.max_ghosts,
        )
        print(f"[Ghost] pool enabled: dir={ghost_pool.dir}  "
              f"save_interval={args.ghost_save_interval}g  "
              f"max_ghosts={args.max_ghosts}  "
              f"existing={ghost_pool.count()}")

    def _bot_factory(name, opp_player):
        if name == "Ghost":
            # Random frozen past-self. None if pool still empty (early in
            # the run) → env treats as self-play for that game.
            return ghost_pool.random_ghost() if ghost_pool is not None else None
        return get_unified_bot(
            name, player_id=opp_player,
            mcts_pure_sims=args.mcts_pure_sims,
            mcts_pure_rollouts=args.mcts_pure_rollouts,
        )

    opp_sampler = build_opp_sampler(opp_probs)

    # ELO tracker — persisted across resumes via elo.json.
    elo_tracker = EloTracker(k_factor=32, initial_rating=1500,
                              save_path=elo_path)
    n_loaded = len(elo_tracker.ratings)
    if n_loaded:
        print(f"[Elo] Loaded {n_loaded} ratings from {elo_path}")
    else:
        print(f"[Elo] Fresh tracker (saving to {elo_path})")

    env = SelfPlayEnv(
        args.parallel_games, max_game_len=args.max_game_len,
        opp_sampler=opp_sampler, get_bot_fn=_bot_factory,
        elo_tracker=elo_tracker,
        use_shaped_reward=bool(args.use_shaped_reward),
    )
    print(f"[reward] {'DENSE (v1 shaped)' if args.use_shaped_reward else 'TERMINAL ONLY'}")

    # ── Build run_info for the dashboard. These strings replace the
    # hardcoded V15.1-specific labels in v13_dashboard.html — the same
    # HTML file is shared across V13.5 / V15 / V15.1 trainers.
    n_params = sum(p.numel() for p in student.parameters())
    run_name = os.environ.get("TD_LUDO_RUN_NAME", "v135_rl")
    reward_mode = "dense (v1 shaped)" if args.use_shaped_reward else "terminal ±1"
    anchor_blurb = (
        f"KL anchor → V13.5_SL (coeff={args.kl_anchor_coeff})"
        if args.kl_anchor_coeff > 0
        else "KL anchor DISABLED"
    )
    run_info = {
        "model_name": "V13.5 Symmetric (CNN)",
        "badge": f"V13.5 RL · {reward_mode}",
        "arch_summary": (
            f"V13.5 Sym: {args.num_res_blocks} res blocks · "
            f"{args.num_channels} channels · {n_params/1e6:.2f}M params · "
            f"V18-symmetric 13ch encoder · rank-indexed policy"
        ),
        "run_name": run_name,
        "description": (
            f"Run <b>{run_name}</b> — initialized from <b>{os.path.basename(os.path.dirname(args.init))}/"
            f"{os.path.basename(args.init)}</b>. "
            f"Reward: <b>{reward_mode}</b>. {anchor_blurb}. "
            f"Opponent mix: SelfPlay + 12 bots (Expectimax-family / scripted / MCTSPure / rule). "
            f"Goal: break past V13.5's 75.5% bot-WR ceiling by giving the model "
            f"dense per-step credit assignment + harder opponents."
        ),
        "first_eval_at": int(args.eval_every_games),
        "eval_every_games": int(args.eval_every_games),
        "eval_games": int(args.eval_games),
        "target_wr": 0.85,
        "baseline_wr": 0.755,  # V13.5 RL pre-experiment baseline
        "baseline_label": "V13.5 RL baseline (don't regress)",
        "target_label": "85% target (stretch goal)",
        "params_total": int(n_params),
    }
    log(f"Starting RL: target {args.target_states:,} states, init={args.init}")
    log(f"[temperature] {args.temperature}  entropy_coeff={args.entropy_coeff}")

    pool = []
    pool_size = 0
    total = 0
    step = 0
    last_save = 0
    last_eval = 0
    last_h2h = 0
    last_save_games = 0
    last_eval_games = 0
    last_ghost_games = 0
    t_start = time.time()
    eval_history = []
    h2h_history = []
    last_metrics = None

    if _resume_meta:
        total = _resume_meta.get("total", 0)
        step = _resume_meta.get("step", 0)
        last_save = total
        last_eval = total
        last_h2h = total
        eval_history = _resume_meta.get("eval_history", [])
        h2h_history = _resume_meta.get("h2h_history", [])
        env.games_played = _resume_meta.get("games_played", 0)
        last_save_games = env.games_played
        last_eval_games = env.games_played
        last_ghost_games = env.games_played
        log(f"[Resume] step={step}, states={total:,}, games={env.games_played}, "
            f"evals={len(eval_history)}, h2h={len(h2h_history)}")

    # Baselines for rate (fps/gpm) telemetry. On resume, `total` and
    # `games_played` are restored to large cumulative values, but t_start
    # resets each process — so dividing the cumulative count by
    # since-start elapsed inflates fps/gpm massively (e.g. gpm=70128).
    # Measure rates from the DELTA since this process started instead.
    _states_at_start = total
    _games_at_start = env.games_played

    while total < args.target_states:
        (decision_idxs, cps, encs, rmasks, rlegals, legal_lists,
         rank_token_ids_list, finished_from_opp) = env.spin_to_decision()

        # Trajectories finalized during opp moves go straight to pool —
        # they bypass the student inference path entirely.
        if finished_from_opp:
            pool.extend(finished_from_opp)
            pool_size += len(finished_from_opp)

        if len(decision_idxs) > 0:
            with torch.no_grad():
                x = torch.from_numpy(encs).to(device, dtype=torch.float32)
                rm = torch.from_numpy(rmasks).to(device, dtype=torch.float32)
                rl_ = torch.from_numpy(rlegals).to(device, dtype=torch.float32)
                _student_out = student(x, rm, rl_)
                policy, win_prob = _student_out[0], _student_out[1]
                # Apply temperature: raise log-probs to 1/T, renormalize.
                if abs(args.temperature - 1.0) > 1e-6:
                    logp = torch.log(policy.clamp_min(1e-12))
                    policy = torch.softmax(logp / args.temperature, dim=-1)
                    # Re-mask: temperature can leak mass onto illegal ranks
                    # in numerical edge cases. The rl_ mask is binary.
                    policy = policy * rl_
                    policy = policy / policy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                ranks = torch.multinomial(policy, num_samples=1).squeeze(1).cpu().numpy()
                v_preds = (2.0 * win_prob - 1.0).cpu().numpy()

            finished = env.apply_actions(
                decision_idxs, cps, encs, rmasks, rlegals,
                legal_lists, rank_token_ids_list, ranks, v_preds,
            )
            pool.extend(finished)
            pool_size += len(finished)

        if pool_size >= args.train_chunk:
            progress = total / max(1, args.target_states)
            cur_lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * progress))
            for g in optimizer.param_groups:
                g["lr"] = cur_lr
            metrics = train_on_chunk(student, optimizer, pool, device, args, kl_teacher=kl_teacher)
            last_metrics = metrics
            total += pool_size
            step += 1
            pool = []
            pool_size = 0

            if step % max(1, args.log_every) == 0 and metrics is not None:
                elapsed = time.time() - t_start
                fps = (total - _states_at_start) / max(1e-6, elapsed)
                avg_glen = float(np.mean(env.game_lengths)) if env.game_lengths else 0.0
                log(f"step {step:>5} | states {total:>9,}/{args.target_states:,} "
                    f"| fps {fps:>5.0f} | lr {cur_lr:.1e} | games {env.games_played} "
                    f"| avg_glen {avg_glen:.0f} "
                    f"| L {metrics['loss']:.4f} (pol {metrics['loss_pol']:+.4f} "
                    f"val {metrics['loss_val']:.4f} ent {metrics['entropy']:.3f}"
                    + (f" kl {metrics['loss_kl']:.3f}" if kl_teacher is not None else "")
                    + ")")
                # [opp-mix] cumulative game counts per bot. Useful sanity
                # check that the sampler is hitting the configured weights.
                if step % max(1, args.log_every * 5) == 0:
                    top = env.opp_game_counts.most_common(20)
                    log("  [opp-mix] " + ", ".join(f"{n}={c}" for n, c in top))
                # Write V15-compatible stats.json for the rich dashboard.
                gpm = ((env.games_played - _games_at_start) / max(1e-6, elapsed)) * 60.0
                best_eval_wr = (max(e[1] for e in eval_history) / 100.0
                                if eval_history else 0.0)
                # Pull ELO rankings (already updated live by SelfPlayEnv).
                main_elo = elo_tracker.ratings.get("Model", 1500.0)
                rankings = elo_tracker.get_rankings(top_n=15)
                elo_rankings = [
                    {"name": n, "elo": float(round(e, 1))}
                    for n, e in rankings
                ]
                try:
                    write_stats_json(
                        stats_path,
                        total_games=env.games_played,
                        total_updates=step,
                        win_rate_100=env.get_win_rate_100(),
                        policy_entropy=metrics["entropy"],
                        avg_policy_loss=metrics["loss_pol"],
                        avg_value_loss=metrics["loss_val"],
                        approx_kl=metrics["loss_kl"],
                        gpm=gpm,
                        best_eval_wr=best_eval_wr,
                        temperature=args.temperature,
                        recent_opp_stats=env.get_recent_opp_stats(),
                        opp_game_counts=env.opp_game_counts,
                        main_elo=main_elo,
                        elo_rankings=elo_rankings,
                        run_info=run_info,
                    )
                except Exception as e:
                    print(f"[stats] write failed: {e}")
                # Persist ELO ratings (covers crashes / resumes).
                try:
                    elo_tracker.save()
                except Exception as e:
                    print(f"[elo] save failed: {e}")
                # Legacy rl_stats.json — keep writing for any external reader
                # that might still depend on the old schema.
                try:
                    with open(rl_stats_path, "w") as f:
                        json.dump({
                            "stage": "RL", "arch": "v13_5",
                            "step": step, "states": total,
                            "target": args.target_states, "fps": fps,
                            "elapsed_sec": elapsed, "lr": cur_lr,
                            "games_played": env.games_played,
                            "avg_game_len": avg_glen,
                            "loss": metrics["loss"],
                            "loss_pol": metrics["loss_pol"],
                            "loss_val": metrics["loss_val"],
                            "entropy": metrics["entropy"],
                            "loss_kl": metrics["loss_kl"],
                            "eval_history": eval_history,
                            "h2h_history": h2h_history,
                            "ts": int(time.time()),
                        }, f)
                except Exception:
                    pass

            _save_g = args.save_every_games > 0 and env.games_played - last_save_games >= args.save_every_games
            _save_s = args.save_every_games == 0 and total - last_save >= args.save_every
            if _save_g or _save_s:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"rl_{total // 1000}K.pt")
                save_dict = {
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step, "total": total,
                    "games_played": env.games_played,
                    "eval_history": eval_history,
                    "h2h_history": h2h_history,
                }
                torch.save(save_dict, ckpt_path)
                latest_path = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
                torch.save(save_dict, latest_path)
                log(f"[checkpoint] {ckpt_path} + model_latest.pt (games={env.games_played})")
                last_save = total
                last_save_games = env.games_played

            # ── Ghost snapshot: save a frozen self into the pool ──
            if (ghost_pool is not None
                    and env.games_played - last_ghost_games >= args.ghost_save_interval):
                gpath = ghost_pool.save(student, env.games_played)
                last_ghost_games = env.games_played
                log(f"[ghost] saved {os.path.basename(gpath)} "
                    f"(pool size={ghost_pool.count()})")

            _eval_g = args.eval_every_games > 0 and env.games_played - last_eval_games >= args.eval_every_games
            _eval_s = args.eval_every_games == 0 and total - last_eval >= args.eval_every
            if (_eval_g or _eval_s) and total > 0:
                log(f"[eval] starting ({args.eval_games} games, at RL game {env.games_played})...")
                wr, per_bot = quick_eval(
                    student, device, n_games=args.eval_games,
                    mcts_pure_sims=args.mcts_pure_sims,
                    mcts_pure_rollouts=args.mcts_pure_rollouts,
                )
                eval_history.append([total, wr])
                per_bot_str = "  ".join(
                    f"{n}={d['win_rate']:.1f}%/{d['games']}g"
                    for n, d in sorted(per_bot.items())
                )
                log(f"[eval] WR = {wr:.1f}% at {total:,} states ({env.games_played} games)  "
                    f"per-bot: {per_bot_str}")
                # Append rich-dashboard metrics snapshot. Field name is
                # `eval_win_rate` (not `win_rate`) to match the V15-family
                # dashboard's filter at v13_dashboard.html:
                # `const evals = m.filter(e => e.eval_win_rate != null);`
                try:
                    append_metrics_snapshot(metrics_path, {
                        "timestamp": time.time(),
                        "games": env.games_played,
                        "states": total,
                        "eval_win_rate": wr / 100.0,   # FRACTION 0..1
                        "win_rate_percent": wr,        # PERCENT 0..100
                        "wins": int(round(wr / 100.0 * args.eval_games)),
                        "total": int(args.eval_games),
                        "per_bot": per_bot,
                    })
                except Exception as e:
                    print(f"[metrics] write failed: {e}")
                last_eval = total
                last_eval_games = env.games_played

            if v132_h2h is not None and total - last_h2h >= args.h2h_gate_every:
                log(f"[h2h-gate] starting ({args.h2h_games} games vs V13.2)...")
                v135_w, v132_w, draws = quick_h2h_vs_v132(
                    student, v132_h2h, device, n_games=args.h2h_games)
                n = max(1, v135_w + v132_w + draws)
                v135_wr = 100 * v135_w / n
                v132_wr = 100 * v132_w / n
                h2h_history.append([total, v135_wr, v132_wr, draws])
                log(f"[h2h-gate] V13.5 {v135_w}/{n}={v135_wr:.1f}%  vs  "
                    f"V13.2 {v132_w}/{n}={v132_wr:.1f}%  (draws {draws})")
                last_h2h = total

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "model_rl.pt")
    final_latest = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
    save_dict = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "total": total,
        "games_played": env.games_played,
        "eval_history": eval_history,
        "h2h_history": h2h_history,
    }
    torch.save(save_dict, final_path)
    torch.save(save_dict, final_latest)
    log(f"[done] processed {total:,} states across {env.games_played} games.")
    log(f"[done] saved → {final_path} + {final_latest}")
    write_chain("completed")
    rl_log.close()


if __name__ == "__main__":
    main()
