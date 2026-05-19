"""MCTS with informed priors — Phase 4 of STRONG_BOTS_PLAN.

The existing `MCTSPureBot` uses a uniform prior over legal actions in
its PUCT search. With pure-random rollouts at leaves, that means the
search has to do a lot of visits to find decent moves. Performance
suffers — `MCTSPureBot(30/4)` loses to Expert (32%) while base
`ExpectimaxBot` beats Expert (68%) at lower compute.

This module wraps `MCTSPureBot` with a **non-uniform prior** sourced
from another bot's policy. The pattern:

  prior_bot.select_move(s, legal) → preferred action a*
  prior over legal = one-hot at a* (smoothed by epsilon)

Two preset variants:

  MCTSExpertPriorBot      — uses ExpertBot's choice as the prior
  MCTSExpectimaxPriorBot  — uses ExpectimaxBot's choice as the prior

Both keep the existing random-rollout value evaluator. The expectation
is that informed-prior + random-rollout-value approximates the strength
of "depth-N expectimax with the prior bot's scoring function" at a
cheaper amortized cost (you can stop early once visits concentrate).

Cost model: each MCTS sim calls the prior bot once at the root expansion
+ random rollouts (8 default) for leaf eval. So adding the prior adds
one `prior_bot.select_move` call per expanded node — that's the dominant
cost when prior_bot is expensive (Expectimax: ~50ms). At n_sims=30,
Expert-prior should be cheap; Expectimax-prior should be ~1.5s/move.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np

import td_ludo_cpp as ludo_cpp

from td_ludo.game.strong_bots import (
    MCTSPureBot, _clone_state, _next_active_player,
)


class _MCTSWithPriorBase(MCTSPureBot):
    """Base for MCTS-with-informed-prior. Subclasses override
    `_make_prior_bot()` to return a bot instance whose argmax becomes
    the one-hot prior at each expanded node.

    The smoothing fraction `prior_smooth` puts that much mass on the
    other legal actions (uniform), keeping the search from fully
    collapsing on the prior bot's choice — important because the prior
    bot can be wrong.
    """

    NAME = "MCTSWithPriorBase"
    PRIOR_BOT_NAME = None  # subclass sets

    def __init__(
        self,
        player_id: Optional[int] = None,
        n_sims: int = 30,
        rollouts_per_leaf: int = 8,
        rollout_max_depth: int = 200,
        c_puct: float = 1.5,
        prior_smooth: float = 0.15,
    ):
        super().__init__(
            player_id=player_id,
            n_sims=n_sims,
            rollouts_per_leaf=rollouts_per_leaf,
            rollout_max_depth=rollout_max_depth,
            c_puct=c_puct,
        )
        self.prior_smooth = float(prior_smooth)
        self._prior_bot_cache = None  # one instance reused per select_move

    def _make_prior_bot(self):
        """Subclass override: return a bot instance with a working
        `select_move(state, legal) -> int` method. Called once per
        `select_move` call so internal state can be reset."""
        raise NotImplementedError

    def _prior_action(self, state, legal: Sequence[int]) -> int:
        """Run the prior bot, defensively coerce to a legal action."""
        if self._prior_bot_cache is None:
            self._prior_bot_cache = self._make_prior_bot()
        if hasattr(self._prior_bot_cache, "player_id"):
            self._prior_bot_cache.player_id = int(state.current_player)
        try:
            a = self._prior_bot_cache.select_move(state, list(legal))
        except Exception:
            return int(legal[0]) if legal else 0
        a = int(a)
        if a < 0 or a not in legal:
            a = int(legal[0]) if legal else 0
        return a

    def select_move(self, state, legal_moves: Sequence[int]) -> int:
        if not legal_moves:
            return -1
        if len(legal_moves) == 1:
            return int(legal_moves[0])
        # Reset prior bot per call (clears any internal accumulator).
        self._prior_bot_cache = None
        self._ensure_engine()
        me = self.player_id if self.player_id is not None else int(state.current_player)

        rollouts_per_leaf = self.rollouts_per_leaf
        self_random_rollout = self._random_rollout
        prior_action_fn = self._prior_action
        prior_smooth = self.prior_smooth

        class _InformedPriorEvaluator:
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
                    if not legal:
                        # No legal moves — uniform-zero prior, will be
                        # handled by the MCTS engine.
                        continue
                    # Get the prior bot's preferred action
                    preferred = prior_action_fn(s, legal)
                    # Smoothed one-hot: (1 - eps) at preferred,
                    # eps/(K-1) at the others, where K = len(legal).
                    n_legal = len(legal)
                    if n_legal == 1:
                        priors[i, preferred] = 1.0
                    else:
                        other_mass = prior_smooth / (n_legal - 1)
                        for a in legal:
                            priors[i, a] = other_mass
                        priors[i, preferred] = 1.0 - prior_smooth
                    # Random rollouts for value (same as MCTSPureBot)
                    outs = [self_random_rollout(s, self.root_player)
                            for _ in range(rollouts_per_leaf)]
                    values[i] = float(np.mean(outs)) if outs else 0.0
                return priors, values

        evaluator = _InformedPriorEvaluator(root_player=me)
        mcts = self._engine.MCTS(
            evaluator,
            c_puct=self.c_puct,
            n_sims=self.n_sims,
            dirichlet_eps=0.0,
        )
        root = mcts.search(_clone_state(state), training=False)

        visits = np.asarray(root.N, dtype=np.float64)
        masked = np.full(4, -1.0)
        for a in legal_moves:
            masked[a] = visits[a]
        return int(np.argmax(masked))


# ─── Preset priors ────────────────────────────────────────────────────────


class MCTSExpertPriorBot(_MCTSWithPriorBase):
    """MCTS with Expert as prior. Expert is fast (rule-based), so the
    overhead per node is minimal; should run at ~MCTSPure speed but
    converge to good moves with fewer sims."""

    NAME = "MCTSExpertPrior"
    PRIOR_BOT_NAME = "Expert"

    def _make_prior_bot(self):
        from td_ludo.game.heuristic_bot import ExpertBot
        return ExpertBot()


class MCTSExpectimaxPriorBot(_MCTSWithPriorBase):
    """MCTS with ExpectimaxBot as prior. Expectimax itself is ~50ms per
    decision, so this adds ~50ms per expanded node — significantly
    slower than the Expert-prior variant. Compensates with stronger
    initial priors. Best for low-sim regimes (n_sims=20-40)."""

    NAME = "MCTSExpectimaxPrior"
    PRIOR_BOT_NAME = "Expectimax"

    def _make_prior_bot(self):
        from td_ludo.game.strong_bots import ExpectimaxBot
        return ExpectimaxBot()


# ─── Registry ─────────────────────────────────────────────────────────────


MCTS_PRIOR_REGISTRY = {
    "MCTSExpertPrior":     MCTSExpertPriorBot,
    "MCTSExpectimaxPrior": MCTSExpectimaxPriorBot,
}
