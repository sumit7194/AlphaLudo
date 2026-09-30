"""V15RichPlayer — rollout actor with opponent mix for V15 RL training.

Roughly mirrors `td_ludo/td_ludo/game/players/v11.py::VectorACGamePlayer` but:
  - V15-specific state encoding (8-frame chronological history per game)
  - 225-cell source-cell action space (vs V13.5's 4-way rank-indexed)
  - Stores trajectory entries in the shape `V15RichTrainer.train_on_game` expects
  - Returns per-completed-game result dicts the main loop can feed into
    EloTracker.update_from_game and GameDB.add_game

Opponent picking is parameterised by a `pick_fn(state, legal) → token_id`
dict, identical to the simpler trainer. Self-play uses a frozen-weight
snapshot picker; consider passing in a learner-self picker for the live model.
"""
from __future__ import annotations

import collections
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import td_ludo_cpp as _legacy_cpp
import td_ludo_v15_cpp as _v15_cpp

from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS,
    cell_to_index,
    position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder import encode_frame
# Full v1 dense reward shaping (compute_shaped_reward): leave-base /
# forward / home-stretch / score / capture / killed + strategic
# (safety / danger / leader / endgame). Imported lazily so this module
# stays import-cheap when use_full_shaping=False.
try:
    from td_ludo.game.reward_shaping import compute_shaped_reward
    _HAS_FULL_SHAPING = True
except ImportError:
    compute_shaped_reward = None  # type: ignore
    _HAS_FULL_SHAPING = False


# ─── Multiprocessing worker code ──────────────────────────────────────────
# The V15 player previously ran opp.select_move() sequentially on a single
# core. With Depth2Expectimax (~1.2s/move) or MCTSExpectimaxPrior (~1s/move)
# in the mix, this pinned the trainer to 1/8 CPU cores (GPM ~95). The
# workers below parallelize opp moves across an mp.Pool while the main
# process still handles SelfPlay (which needs the live student model on
# GPU) and student inference batches.

_WORKER_CPP = None
_WORKER_BOTS = None  # dict: bot_name -> bot_instance


def _serialize_state(state) -> dict:
    """Pickle-able snapshot of a cpp.GameState (only the fields a bot reads).

    All eight settable fields are captured so the reconstructed state in the
    worker has full fidelity (scores matter for Expectimax leaf evaluations,
    idle_counter / streak / last_moved for some shaping decisions).
    """
    return {
        'pp':         np.asarray(state.player_positions).copy(),
        'cp':         int(state.current_player),
        'dice':       int(state.current_dice_roll),
        'scores':     np.asarray(state.scores).copy(),
        'active':     np.asarray(state.active_players).copy(),
        'idle':       np.asarray(state.idle_counter).copy(),
        'streak':     np.asarray(state.streak).copy(),
        'last_moved': np.asarray(state.last_moved_token).copy(),
    }


def _worker_init(bot_specs: dict):
    """Per-worker init: import cpp + build bot instances ONCE.

    bot_specs: dict mapping bot_name -> kwargs for construction.
        e.g. {"Expectimax": {}, "MCTSExpectimaxPrior": {"n_sims": 30}}
    Bots are stateless across games so a single instance per worker is fine.
    """
    global _WORKER_CPP, _WORKER_BOTS
    import td_ludo_cpp as cpp_mod
    _WORKER_CPP = cpp_mod
    from td_ludo.game.strong_bots import STRONG_BOT_REGISTRY
    _WORKER_BOTS = {}
    for name, kwargs in bot_specs.items():
        if name not in STRONG_BOT_REGISTRY:
            continue  # only strong bots are dispatched to workers
        _WORKER_BOTS[name] = STRONG_BOT_REGISTRY[name](**kwargs)


def _worker_pick(args):
    """Compute one opp move in the worker process.

    args: (game_idx, opp_name, state_dict, legal_list)
    Returns: (game_idx, chosen_token_id)
    """
    game_idx, opp_name, state_dict, legal = args
    s = _WORKER_CPP.create_initial_state_2p()
    s.player_positions  = state_dict['pp'].tolist()
    s.current_player    = state_dict['cp']
    s.current_dice_roll = state_dict['dice']
    s.scores            = state_dict['scores'].tolist()
    s.active_players    = state_dict['active'].tolist()
    s.idle_counter      = state_dict['idle'].tolist()
    s.streak            = state_dict['streak'].tolist()
    s.last_moved_token  = state_dict['last_moved'].tolist()
    bot = _WORKER_BOTS[opp_name]
    token = int(bot.select_move(s, list(legal)))
    if token not in legal:
        token = legal[0]
    return (game_idx, token)


_BASE_POS = _v15_cpp.BASE_POS
NUM_PLAYERS = 4
# History/stack depth — V15=8 frames (1 current + 7 past), V15.1=2.
# Modules that read these as defaults capture them at import time; if you
# need to switch depths, call configure_history(T) BEFORE constructing
# V15RichPlayer instances (the deques inside use HISTORY_LEN at __init__).
HISTORY_LEN = 7
TOTAL_FRAMES = 8

def configure_history(total_frames: int):
    """Switch the module-level stack depth at runtime (V15=8, V15.1=2)."""
    global HISTORY_LEN, TOTAL_FRAMES
    if total_frames < 1:
        raise ValueError(f"history-len must be >= 1, got {total_frames}")
    TOTAL_FRAMES = total_frames
    HISTORY_LEN = total_frames - 1

OpponentPickFn = Callable[[object, List[int]], int]


class V15RichPlayer:
    """Vectorised V15 rollout with per-game opponent assignment.

    Args:
        batch_size: number of parallel games
        opponents: dict opp_name → pick_fn(state, legal)→token_id.
                   Special name "self" should resolve to a picker that uses
                   the live student model. Special name "SelfPlay_Ghost"
                   should use a frozen ghost snapshot (or fall back to self).
        opponent_probs: dict opp_name → weight (renormalized at construction)
        max_game_len: truncation cap (draw on truncate)
        score_reward: per-token-scored reward shaping (matches V13.5 ACV10)
    """

    def __init__(
        self,
        batch_size: int,
        opponents: Dict[str, OpponentPickFn],
        opponent_probs: Dict[str, float],
        max_game_len: int = 400,
        score_reward: float = 0.40,
        seed: Optional[int] = None,
        use_full_shaping: bool = False,
        worker_bot_specs: Optional[Dict[str, dict]] = None,
        n_workers: int = 0,
        record_aux: bool = False,
    ):
        if not opponents:
            raise ValueError("At least one opponent required")
        common = set(opponents.keys()) & set(opponent_probs.keys())
        if not common:
            raise ValueError("opponents and opponent_probs have no shared keys")
        self.batch_size = batch_size
        self.opponents = dict(opponents)
        self.max_game_len = max_game_len
        self.score_reward = score_reward
        # When True, replace score-only step_reward (+0.40 per token scored)
        # with full v1 compute_shaped_reward (leave-base/forward/home-stretch/
        # score/capture/killed + strategic safety/danger/leader/endgame).
        # Per training_journal.md: this is the proven recipe for dense
        # credit assignment on long stochastic Ludo games.
        self.use_full_shaping = bool(use_full_shaping)
        # v16 only: also record sparse tactical targets per student decision.
        # Default False keeps v15 trajectories byte-identical.
        self.record_aux = bool(record_aux)
        if self.use_full_shaping and not _HAS_FULL_SHAPING:
            raise ImportError(
                "use_full_shaping=True requires td_ludo.game.reward_shaping; "
                "ensure /home/sumit/td_ludo is on PYTHONPATH."
            )
        # Multiprocessing worker pool for slow opp bots (Depth2 / MCTSPrior /
        # Expectimax-family). SelfPlay and any non-listed opps stay in main
        # process (SelfPlay needs the live student model on GPU).
        # worker_bot_specs: dict bot_name -> kwargs. n_workers=0 disables.
        self.worker_bot_specs: Dict[str, dict] = dict(worker_bot_specs or {})
        self.n_workers: int = int(n_workers)
        self._pool = None   # lazy — built on first collect_student_decisions call
        names = [n for n in opponent_probs if n in opponents]
        weights = np.array([float(opponent_probs[n]) for n in names], dtype=np.float64)
        if weights.sum() <= 0:
            raise ValueError("opponent_probs must sum to > 0")
        self.opp_names = names
        self.opp_probs = weights / weights.sum()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Per-game state
        self.games = [_legacy_cpp.create_initial_state_2p() for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, NUM_PLAYERS), dtype=np.int32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)
        self.student_player = np.zeros(batch_size, dtype=np.int32)
        self.opp_per_game: List[str] = ["self"] * batch_size
        self.history: List[collections.deque] = [
            collections.deque(maxlen=HISTORY_LEN) for _ in range(batch_size)
        ]
        # Per-game trajectory of STUDENT decisions only.
        self.trajectory: List[List[dict]] = [[] for _ in range(batch_size)]
        # Per-game prev-scored snapshot for delta-score reward shaping.
        self.last_scores = np.zeros(batch_size, dtype=np.int32)

        # Counters & windows
        self.games_played = 0
        self.game_lengths: collections.deque = collections.deque(maxlen=500)
        self.opp_game_counts: collections.Counter = collections.Counter()

        for i in range(batch_size):
            self._reset(i)

    def _reset(self, i: int):
        self.games[i] = _legacy_cpp.create_initial_state_2p()
        self.consec_sixes[i] = 0
        self.step_count[i] = 0
        self.history[i].clear()
        self.trajectory[i] = []
        self.student_player[i] = random.choice([0, 2])
        self.opp_per_game[i] = self.opp_names[int(np.random.choice(
            len(self.opp_names), p=self.opp_probs))]
        self.last_scores[i] = 0

    def _finalize(self, i: int, winner: int) -> dict:
        """Build the per-game result dict and reset."""
        sp = int(self.student_player[i])
        opp = self.opp_per_game[i]
        self.opp_game_counts[opp] += 1
        self.games_played += 1
        glen = len(self.trajectory[i])
        self.game_lengths.append(glen)
        # Identities: 2-player → P0 and P2 are active, others None
        identities = [None] * NUM_PLAYERS
        identities[sp] = "Model"
        identities[2 if sp == 0 else 0] = opp
        return {
            "identities": identities,
            "winner": int(winner),
            "model_player": sp,
            "model_won": (winner == sp),
            "opponent": opp,
            "total_moves": int(self.step_count[i]),
            "trajectory": list(self.trajectory[i]),
            "trajectory_length": glen,
        }

    # ── Pool management ──────────────────────────────────────────────────────
    def _ensure_pool(self):
        """Lazily build the multiprocessing pool on first use."""
        if self._pool is None and self.worker_bot_specs and self.n_workers > 0:
            import multiprocessing as mp
            # 'spawn' avoids fork-after-cuda-init issues on Linux.
            ctx = mp.get_context('spawn')
            self._pool = ctx.Pool(
                processes=self.n_workers,
                initializer=_worker_init,
                initargs=(self.worker_bot_specs,),
            )

    def close_pool(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        try:
            self.close_pool()
        except Exception:
            pass

    # ── Per-game step kernel (no inner loop) ─────────────────────────────────
    def _step_one_game(self, i: int):
        """Take ONE atomic step on game i. Returns one of:
          ('finished', game_result_dict)        — game ended; caller resets
          ('decision', decision_dict)           — student's turn; collect
          ('opp_worker', (i, name, state, legal)) — dispatch to worker pool
          ('opp_inline', None)                  — opp handled in-process (SelfPlay)
          ('skip', None)                        — dice/no-legal/3-sixes → loop again
        """
        game = self.games[i]
        if game.is_terminal:
            winner = int(_legacy_cpp.get_winner(game))
            return ('finished', self._finalize(i, winner))
        if self.step_count[i] >= self.max_game_len:
            return ('finished', self._finalize(i, -1))

        cp = int(game.current_player)
        # Dice roll
        if game.current_dice_roll == 0:
            d = random.randint(1, 6)
            game.current_dice_roll = d
            if d == 6:
                self.consec_sixes[i, cp] += 1
            else:
                self.consec_sixes[i, cp] = 0
            if self.consec_sixes[i, cp] >= 3:
                nxt = (cp + 1) % NUM_PLAYERS
                while not game.active_players[nxt]:
                    nxt = (nxt + 1) % NUM_PLAYERS
                game.current_player = nxt
                game.current_dice_roll = 0
                self.consec_sixes[i, cp] = 0
                return ('skip', None)

        legal = _legacy_cpp.get_legal_moves(game)
        if not legal:
            nxt = (cp + 1) % NUM_PLAYERS
            while not game.active_players[nxt]:
                nxt = (nxt + 1) % NUM_PLAYERS
            game.current_player = nxt
            game.current_dice_roll = 0
            return ('skip', None)

        sp = int(self.student_player[i])
        if cp == sp:
            # Build student decision payload.
            past = list(self.history[i])
            pad = HISTORY_LEN - len(past)
            v15_x = np.zeros((TOTAL_FRAMES, 15, 15, 3), dtype=np.float32)
            real_frames = [None] * pad + past + [game]
            for t_idx, st in enumerate(real_frames):
                if st is None:
                    continue
                v15_x[t_idx] = encode_frame(st, pov_player=cp)
            v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
            for t in legal:
                pos = int(game.player_positions[cp][t])
                c = position_to_cell_in_pov(
                    _BASE_POS if pos == _BASE_POS else pos, cp, cp)
                v15_legal[cell_to_index(*c)] = 1.0
            return ('decision', {
                "game_idx": i,
                "v15_x": v15_x,
                "v15_mask": v15_legal,
                "legal": list(legal),
            })

        # Opp turn
        opp_name = self.opp_per_game[i]
        if opp_name in self.worker_bot_specs and self._pool is not None:
            # Dispatch to worker pool.
            return ('opp_worker',
                    (i, opp_name, _serialize_state(game), list(legal)))
        # In-process pick (SelfPlay, or worker pool disabled)
        pick_fn = self.opponents.get(opp_name)
        if pick_fn is None:
            token = random.choice(legal)
        else:
            token = pick_fn(game, list(legal))
        if token not in legal:
            token = legal[0]
        self.history[i].append(game)
        self.games[i] = _legacy_cpp.apply_move(game, int(token))
        self.step_count[i] += 1
        return ('opp_inline', None)

    # ── Rollout step ─────────────────────────────────────────────────────────
    def collect_student_decisions(self):
        """Advance every game until it's the STUDENT's turn (or a game ends).

        Multi-pass design:
          - Pass: step each "advancing" game until it either becomes
            decision-ready, finishes, or hits a worker-dispatch opp turn.
          - If any worker-dispatched opp moves accumulated, run them in
            parallel via the mp.Pool, then apply results and re-add those
            games to the advancing set.
          - Loop until no game is left advancing.

        Returns:
            decisions: list of {"game_idx", "v15_x", "v15_mask", "legal"} dicts
            finished_games: list of per-game result dicts (from _finalize)
        """
        self._ensure_pool()
        finished_games: List[dict] = []
        decisions: List[dict] = []
        # Track which games still need processing.
        advancing = set(range(self.batch_size))

        while advancing:
            pending_opp: List[tuple] = []
            done_this_pass: List[int] = []

            for i in list(advancing):
                # Run until this game either becomes decision-ready,
                # finishes (and resets — but stays in advancing for the
                # new game), or hits an opp-worker dispatch.
                while True:
                    status, payload = self._step_one_game(i)
                    if status == 'finished':
                        finished_games.append(payload)
                        self._reset(i)
                        # Restart from fresh game on next iteration.
                        continue
                    if status == 'decision':
                        decisions.append(payload)
                        done_this_pass.append(i)
                        break
                    if status == 'opp_worker':
                        pending_opp.append(payload)
                        # Wait for worker; don't remove from advancing
                        # but DON'T loop here either.
                        break
                    if status == 'opp_inline':
                        # Move applied; loop to advance further.
                        continue
                    if status == 'skip':
                        # Dice/turn skip; loop to try again.
                        continue
                    raise RuntimeError(f"unknown step status: {status}")

            for i in done_this_pass:
                advancing.discard(i)

            if pending_opp:
                # Parallel opp-move computation. With chunksize=1 every
                # worker grabs the next task as soon as it's free → no
                # straggler bottleneck.
                results = self._pool.map(_worker_pick, pending_opp)
                for game_idx, token in results:
                    game = self.games[game_idx]
                    if token not in _legacy_cpp.get_legal_moves(game):
                        token = _legacy_cpp.get_legal_moves(game)[0]
                    self.history[game_idx].append(game)
                    self.games[game_idx] = _legacy_cpp.apply_move(game, int(token))
                    self.step_count[game_idx] += 1
                    # Game can continue advancing in next pass.
                    # (It's already in `advancing` from before the dispatch.)

        return decisions, finished_games

    def apply_student_actions(
        self,
        decisions: List[dict],
        chosen_cells: np.ndarray,
        log_probs_old: np.ndarray,
        temperatures: np.ndarray,
        v_pred_old: Optional[np.ndarray] = None,
    ):
        """Apply student's chosen actions, record trajectory entry."""
        for k, dec in enumerate(decisions):
            i = dec["game_idx"]
            game = self.games[i]
            cp = int(game.current_player)
            chosen_idx = int(chosen_cells[k])
            chosen_cell = divmod(chosen_idx, 15)
            # Map chosen cell to a legal token-id (lowest token at that cell)
            token = None
            for t in sorted(dec["legal"]):
                pos = int(game.player_positions[cp][t])
                c = position_to_cell_in_pov(
                    _BASE_POS if pos == _BASE_POS else pos, cp, cp)
                if c == chosen_cell:
                    token = t
                    break
            if token is None:
                token = dec["legal"][0]
                pos = int(game.player_positions[cp][token])
                c = position_to_cell_in_pov(
                    _BASE_POS if pos == _BASE_POS else pos, cp, cp)
                chosen_idx = cell_to_index(*c)
            # Snapshot state BEFORE the move; we need both halves to
            # compute the shaped reward delta.
            state_before = game
            self.history[i].append(game)
            self.games[i] = _legacy_cpp.apply_move(game, int(token))
            self.step_count[i] += 1
            state_after = self.games[i]
            if self.use_full_shaping:
                # Full v1 dense shaping: leave-base / forward / home-stretch
                # / score / capture / killed + strategic. Loud per-step
                # signal for credit assignment on 150-move games.
                step_reward = float(
                    compute_shaped_reward(state_before, state_after, cp)
                )
            else:
                # Legacy score-only fallback (the V15 native recipe).
                prev_scored = int(state_before.scores[cp])
                new_scored = int(state_after.scores[cp])
                step_reward = (new_scored - prev_scored) * self.score_reward

            entry = {
                "v15_x": dec["v15_x"],
                "v15_mask": dec["v15_mask"],
                "action": int(chosen_idx),
                "old_log_prob": float(log_probs_old[k]),
                "temperature": float(temperatures[k]),
                "step_reward": float(step_reward),
            }
            if self.record_aux:
                # v16: sparse tactical targets, computed from the RAW state at
                # the decision point (the encoded frame is POV and cannot be
                # inverted back to token positions). Off by default, so v15
                # trajectories are byte-identical to before.
                from .v16_aux import aux_targets
                entry["aux_target"] = aux_targets(state_before, cp, dec["legal"])
            self.trajectory[i].append(entry)

    # ── Stats helpers ────────────────────────────────────────────────────────
    def avg_game_length(self) -> float:
        return float(np.mean(self.game_lengths)) if self.game_lengths else 0.0
