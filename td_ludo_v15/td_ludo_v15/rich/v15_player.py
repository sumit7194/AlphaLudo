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

    # ── Rollout step ─────────────────────────────────────────────────────────
    def collect_student_decisions(self):
        """Advance every game until it's the STUDENT's turn (or a game ends).

        Returns:
            decisions: list of dicts (one per game-at-decision):
                {"game_idx": int,
                 "v15_x": (8,15,15,3) float32,
                 "v15_mask": (225,) float32,
                 "legal": List[int]}
            finished_games: list of per-game result dicts (from _finalize)
        """
        finished_games: List[dict] = []
        decisions: List[dict] = []

        for i in range(self.batch_size):
            while True:
                game = self.games[i]
                if game.is_terminal:
                    winner = int(_legacy_cpp.get_winner(game))
                    finished_games.append(self._finalize(i, winner))
                    self._reset(i)
                    game = self.games[i]
                if self.step_count[i] >= self.max_game_len:
                    # Truncate as draw
                    finished_games.append(self._finalize(i, -1))
                    self._reset(i)
                    game = self.games[i]

                cp = int(game.current_player)
                # Roll dice if needed
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
                        continue
                legal = _legacy_cpp.get_legal_moves(game)
                if not legal:
                    nxt = (cp + 1) % NUM_PLAYERS
                    while not game.active_players[nxt]:
                        nxt = (nxt + 1) % NUM_PLAYERS
                    game.current_player = nxt
                    game.current_dice_roll = 0
                    continue

                sp = int(self.student_player[i])
                if cp == sp:
                    # Encode for student
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
                    decisions.append({
                        "game_idx": i,
                        "v15_x": v15_x,
                        "v15_mask": v15_legal,
                        "legal": list(legal),
                    })
                    break
                else:
                    # Opponent turn
                    opp_name = self.opp_per_game[i]
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
            # Compute step_reward = score delta * score_reward
            prev_scored = int(game.scores[cp])
            self.history[i].append(game)
            self.games[i] = _legacy_cpp.apply_move(game, int(token))
            self.step_count[i] += 1
            new_scored = int(self.games[i].scores[cp])
            step_reward = (new_scored - prev_scored) * self.score_reward

            self.trajectory[i].append({
                "v15_x": dec["v15_x"],
                "v15_mask": dec["v15_mask"],
                "action": int(chosen_idx),
                "old_log_prob": float(log_probs_old[k]),
                "temperature": float(temperatures[k]),
                "step_reward": float(step_reward),
            })

    # ── Stats helpers ────────────────────────────────────────────────────────
    def avg_game_length(self) -> float:
        return float(np.mean(self.game_lengths)) if self.game_lengths else 0.0
