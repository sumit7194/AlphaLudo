"""Stronger, non-neural Ludo bots for breaking the trained-model ceiling.

Why this file exists
--------------------
Every neural model in this project (V6 → V13.5 → V15.1) was trained
against the same 6 scripted bots: Heuristic, Aggressive, Defensive,
Racing, Random, Expert. They all converge to ~85% WR against that mix
because they're optimizing the same objective. Cross-model H2H shows
they're all within ~3pp of each other.

To break that ceiling we need opponents that are **qualitatively
different** from the scripted family — opponents that DO things the
scripted bots don't (lookahead, dice-expectation, search). This file
adds two such bots:

  ExpectimaxBot
    1-step lookahead with dice expectation. For each legal move, simulate
    applying it; then for each possible opponent dice (1..6), find the
    opponent's best greedy response; score the result; average over dice.
    Pick the move with highest expected score. ~50× slower than a
    scripted heuristic but a fundamentally different decision rule.

  MCTSPureBot
    Adapter over the existing PUCT engine in experiments/mcts_v1 with
    random-rollout leaf evaluation (no neural net). N rollouts per move
    (default 50). Slow (~500ms/move) but as different from the trained
    models as it gets — pure search + dice probabilities, no policy
    pattern-matching.

Both conform to the BOT_REGISTRY interface (`__init__(player_id=None)`,
`select_move(state, legal_moves) -> token_id`) so they plug into the
existing training/eval harness as drop-in opponents.

Usage from a training script
----------------------------
    from td_ludo.game.strong_bots import ExpectimaxBot, MCTSPureBot
    bot = ExpectimaxBot(player_id=2)
    action = bot.select_move(state, legal_moves)

These are deliberately written in Python (not C++) — correctness over
speed. Once we know they help, we can port to C++ for production use.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence

import numpy as np

import td_ludo_cpp as ludo_cpp


# Same safety constants the heuristic bots use, for consistency in the
# scoring function. Globes + stars from the standard Ludo board.
_SAFE_GLOBES = {0, 8, 13, 21, 26, 34, 39, 47}
_STAR_POSITIONS = {5, 18, 31, 44, 11, 24, 37, 50}
_SAFE_INDICES = _SAFE_GLOBES | _STAR_POSITIONS

_BASE_POS = -1
_HOME_POS = 99


# ─── Scoring helper ─────────────────────────────────────────────────────


def _absolute_pos(player: int, relative_pos: int) -> Optional[int]:
    """Convert a player's relative path position to absolute board index 0..51."""
    if relative_pos < 0 or relative_pos > 50:
        return None
    return (int(relative_pos) + 13 * int(player)) % 52


def _is_safe(player: int, pos: int) -> bool:
    """True if pos (player's relative) is a globe / star / home-stretch / home."""
    if pos == _BASE_POS or pos == _HOME_POS:
        return True
    if pos >= 51:  # in own home stretch
        return True
    abs_pos = _absolute_pos(player, pos)
    return abs_pos in _SAFE_INDICES


def _token_progress_score(player: int, positions: np.ndarray) -> float:
    """Sum of token progress (0..56 per token). Home = 56, base = -10 (penalty)."""
    s = 0.0
    for tok in range(4):
        pos = int(positions[player][tok])
        if pos == _HOME_POS:
            s += 56
        elif pos == _BASE_POS:
            s += -10  # at base = far behind, slight penalty
        else:
            s += pos
    return s


def _exposure_penalty(player: int, positions: np.ndarray,
                      active_players: np.ndarray) -> float:
    """Sum penalty for each of my unsafe tokens that an opponent could capture
    within 6 squares on next turn. ~estimate, not exact."""
    pen = 0.0
    for tok in range(4):
        pos = int(positions[player][tok])
        if pos == _BASE_POS or pos == _HOME_POS:
            continue
        if _is_safe(player, pos):
            continue
        my_abs = _absolute_pos(player, pos)
        if my_abs is None:
            continue
        # Check each opponent's tokens; if any is 1-6 squares behind me, danger.
        for opp in range(4):
            if opp == player or not active_players[opp]:
                continue
            for opp_tok in range(4):
                opp_pos = int(positions[opp][opp_tok])
                if opp_pos == _BASE_POS or opp_pos == _HOME_POS or opp_pos >= 51:
                    continue
                opp_abs = _absolute_pos(opp, opp_pos)
                if opp_abs is None:
                    continue
                dist = (my_abs - opp_abs) % 52
                if 1 <= dist <= 6:
                    pen += 15.0  # constant per-threat penalty
                    break  # one opp counts once for this token
    return pen


def _score_position(player: int, state) -> float:
    """Score a state from `player`'s POV. Higher = better.

    Components: own progress + own scored tokens × bonus - exposure - opp progress."""
    positions = np.asarray(state.player_positions, dtype=np.int8)
    active = np.asarray(state.active_players, dtype=bool)
    own_progress = _token_progress_score(player, positions)
    own_scored = int(state.scores[player]) * 60  # big bonus per scored token
    exposure = _exposure_penalty(player, positions, active)
    opp_progress = 0.0
    for opp in range(4):
        if opp == player or not active[opp]:
            continue
        opp_progress += _token_progress_score(opp, positions)
        opp_progress += int(state.scores[opp]) * 60
    # Weight: own counts double — we're optimizing FOR us, not equally
    return (own_progress + own_scored - exposure) * 2.0 - opp_progress


# ─── Game-state cloning helpers ──────────────────────────────────────────


def _clone_state(state):
    """Build a fresh GameState that mirrors `state`'s fields.

    `td_ludo_cpp.GameState` is C-bound and doesn't deepcopy cleanly, so we
    snapshot the array fields into a new GameState. This is the same trick
    used elsewhere in the project (e.g., the V15 history collector).
    """
    new = ludo_cpp.GameState()
    new.player_positions = np.array(state.player_positions, dtype=np.int8).copy()
    new.scores = np.array(state.scores, dtype=np.int8).copy()
    new.active_players = np.array(state.active_players, dtype=bool).copy()
    new.current_player = int(state.current_player)
    new.current_dice_roll = int(state.current_dice_roll)
    new.is_terminal = bool(getattr(state, "is_terminal", False))
    return new


def _next_active_player(state, current_player: int) -> int:
    """Next active player (cyclic) starting from current+1. Mirrors engine logic."""
    active = state.active_players
    n = (current_player + 1) % 4
    while not active[n]:
        n = (n + 1) % 4
    return n


# ─── ExpectimaxBot ──────────────────────────────────────────────────────


class ExpectimaxBot:
    """1-step lookahead with dice expectation.

    For each legal move `a`:
      apply(a) → s'
      For each possible NEXT dice value d ∈ {1..6} (the OPPONENT's roll):
        find opponent's best greedy response in s' under dice=d
        apply that → s''
        score s'' from MY perspective
      expected_score(a) = mean over d of score(s'')
    Pick a with highest expected_score.

    Cost ~ 4 own-moves × 6 dice × 4 opp-moves × score_eval ≈ 100 lightweight
    forward simulations per decision. ~50ms on Python.

    Why this is qualitatively different from the heuristic bots:
      The scripted bots score the IMMEDIATE post-move state. This bot
      considers the OPPONENT's response. That's a fundamental shift —
      from "what's best for me right now" to "what's best for me after
      they react". Closer to game-theoretic play.
    """

    def __init__(self, player_id: Optional[int] = None):
        self.player_id = player_id

    def select_move(self, state, legal_moves: Sequence[int]) -> int:
        if not legal_moves:
            return -1
        if len(legal_moves) == 1:
            return int(legal_moves[0])

        me = self.player_id if self.player_id is not None else int(state.current_player)
        # Pick the (in 2P mode) other active player. In our env this is (me+2) % 4.
        active = state.active_players
        opp = None
        for p in range(4):
            if p != me and active[p]:
                opp = p
                break
        if opp is None:
            # Degenerate: no opponent active. Just take first legal.
            return int(legal_moves[0])

        best_action = int(legal_moves[0])
        best_expected = -float("inf")

        for a in legal_moves:
            # Step 1: apply our move
            try:
                after_mine = ludo_cpp.apply_move(state, int(a))
            except Exception:
                continue

            # If our move ended the game, max score — pick this.
            if after_mine.is_terminal:
                winner = ludo_cpp.get_winner(after_mine) if hasattr(ludo_cpp, "get_winner") else (
                    me if state.scores[me] == 3 else -1
                )
                expected = 1e9 if winner == me else 0.0
            else:
                # Step 2: for each possible opp dice, find opp's best greedy move
                expected = 0.0
                for d in range(1, 7):
                    sim = _clone_state(after_mine)
                    # Engine sets current_player after apply_move; we trust it.
                    # But ensure dice is set for the opponent's roll.
                    sim.current_dice_roll = d
                    cp = int(sim.current_player)
                    if cp == me:
                        # Bonus-turn case: we get to move again with dice d.
                        # Score the position as-is (ignore the bonus-move expectation
                        # for cost reasons — it'd recurse).
                        expected += _score_position(me, sim) / 6.0
                        continue
                    # Find opp's legal moves under dice=d
                    opp_legal = ludo_cpp.get_legal_moves(sim)
                    if not opp_legal:
                        # Opp passes; advance turn back to me.
                        # Just score the current sim from my POV.
                        expected += _score_position(me, sim) / 6.0
                        continue
                    # Opp picks greedily (max score from opp POV — = min from mine)
                    best_opp_action = opp_legal[0]
                    best_opp_score = -float("inf")
                    for oa in opp_legal:
                        try:
                            after_opp = ludo_cpp.apply_move(sim, int(oa))
                        except Exception:
                            continue
                        s_opp = _score_position(cp, after_opp)
                        if s_opp > best_opp_score:
                            best_opp_score = s_opp
                            best_opp_action = oa
                    try:
                        after_opp = ludo_cpp.apply_move(sim, int(best_opp_action))
                    except Exception:
                        after_opp = sim
                    expected += _score_position(me, after_opp) / 6.0

            if expected > best_expected:
                best_expected = expected
                best_action = int(a)

        return best_action


# ─── MCTSPureBot ────────────────────────────────────────────────────────


class MCTSPureBot:
    """Pure-search MCTS — PUCT with random-rollout leaf eval. No neural net.

    Adapter over the existing mcts_engine. The engine expects a `net_fn`
    callable that returns (prior, value) for a leaf state. We pass a
    closure that:
      - Returns a uniform prior over legal actions
      - Runs N random rollouts from the leaf and returns mean outcome
        from the root player's POV as the value

    Sims-per-move and rollouts-per-leaf are configurable. Defaults err
    on the side of cheap (50 sims, 8 rollouts each) so it's usable in
    self-play. Strong-test mode (200 sims, 32 rollouts) at ~5s/move.
    """

    def __init__(
        self,
        player_id: Optional[int] = None,
        n_sims: int = 50,
        rollouts_per_leaf: int = 8,
        rollout_max_depth: int = 200,
        c_puct: float = 1.5,
    ):
        self.player_id = player_id
        self.n_sims = n_sims
        self.rollouts_per_leaf = rollouts_per_leaf
        self.rollout_max_depth = rollout_max_depth
        self.c_puct = c_puct
        # Lazy import — only pay the MCTS-engine import cost when this bot
        # is actually used. Keeps the strong_bots module light.
        self._engine = None

    def _ensure_engine(self):
        if self._engine is not None:
            return
        import sys
        from pathlib import Path
        engine_dir = (Path(__file__).resolve().parent.parent.parent
                      / "experiments" / "mcts_v1")
        if str(engine_dir.parent.parent) not in sys.path:
            sys.path.insert(0, str(engine_dir.parent.parent))
        from experiments.mcts_v1 import mcts_engine
        self._engine = mcts_engine

    def _random_rollout(self, state, root_player: int) -> float:
        """Play out the game with uniform-random moves. Return +1 if root
        wins, -1 if loses, 0 if truncated."""
        sim = _clone_state(state)
        depth = 0
        while depth < self.rollout_max_depth:
            if sim.is_terminal:
                winner = ludo_cpp.get_winner(sim) if hasattr(ludo_cpp, "get_winner") else -1
                if winner == root_player:
                    return 1.0
                if winner == -1:
                    return 0.0
                return -1.0
            cp = int(sim.current_player)
            if not sim.active_players[cp]:
                # Skip inactive player
                sim.current_player = _next_active_player(sim, cp)
                continue
            if sim.current_dice_roll == 0:
                sim.current_dice_roll = random.randint(1, 6)
            legal = ludo_cpp.get_legal_moves(sim)
            if not legal:
                # Pass turn
                sim.current_player = _next_active_player(sim, cp)
                sim.current_dice_roll = 0
                continue
            a = random.choice(legal)
            try:
                sim = ludo_cpp.apply_move(sim, int(a))
            except Exception:
                break
            depth += 1
        # Truncated — no decision
        return 0.0

    def select_move(self, state, legal_moves: Sequence[int]) -> int:
        if not legal_moves:
            return -1
        if len(legal_moves) == 1:
            return int(legal_moves[0])
        self._ensure_engine()
        me = self.player_id if self.player_id is not None else int(state.current_player)

        # Build a RolloutEvaluator that mimics NetworkEvaluator's interface
        # (`evaluate_batch(states) -> (priors, values)`) but uses uniform
        # priors + random-rollout values.
        rollouts_per_leaf = self.rollouts_per_leaf
        rollout_max_depth = self.rollout_max_depth
        self_random_rollout = self._random_rollout

        class _RolloutEvaluator:
            def __init__(self, root_player):
                self.root_player = root_player
            def evaluate_batch(self, states):
                if not states:
                    return (np.zeros((0, 4), dtype=np.float32),
                            np.zeros((0,), dtype=np.float32))
                priors = np.zeros((len(states), 4), dtype=np.float32)
                values = np.zeros(len(states), dtype=np.float32)
                for i, s in enumerate(states):
                    legal = ludo_cpp.get_legal_moves(s)
                    for a in legal:
                        priors[i, a] = 1.0
                    ps = priors[i].sum()
                    if ps > 0:
                        priors[i] /= ps
                    # Random rollouts for value. Returns mean outcome from
                    # ROOT player's POV (already sign-correct).
                    outs = [self_random_rollout(s, self.root_player)
                            for _ in range(rollouts_per_leaf)]
                    values[i] = float(np.mean(outs)) if outs else 0.0
                return priors, values

        evaluator = _RolloutEvaluator(root_player=me)
        mcts = self._engine.MCTS(
            evaluator,
            c_puct=self.c_puct,
            n_sims=self.n_sims,
            # No Dirichlet noise — this is eval/play mode, not training data
            # collection. Deterministic best-move selection.
            dirichlet_eps=0.0,
        )
        root = mcts.search(_clone_state(state), training=False)

        # root.N is a 4-element array of visit counts indexed by token-id.
        visits = np.asarray(root.N, dtype=np.float64)
        # Greedy: argmax over legal actions
        masked = np.full(4, -1.0)
        for a in legal_moves:
            masked[a] = visits[a]
        return int(np.argmax(masked))


# ─── Registry exports (for plug-in compatibility) ────────────────────────


STRONG_BOT_REGISTRY = {
    "Expectimax": ExpectimaxBot,
    "MCTSPure":   MCTSPureBot,
}


def get_strong_bot(name: str, player_id: Optional[int] = None, **kwargs):
    """Factory: `get_strong_bot('Expectimax', player_id=2)`."""
    cls = STRONG_BOT_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strong bot: {name}. "
                         f"Choices: {list(STRONG_BOT_REGISTRY)}")
    return cls(player_id=player_id, **kwargs)
