"""AlphaZero-style PUCT MCTS for Ludo.

Designed to be clearer than fast; readable over optimal. Once results
justify production-grade speed, port to C++.

Tree structure
--------------
Ludo's flow per turn: dice roll → token pick → apply → (maybe bonus turn).
This makes the natural tree alternation:

    DecisionNode (state has dice rolled, cp picks a token)
        ├── action a₁ → ChanceNode (the dice-roll for the NEXT decision)
        │       ├── dice = 1 → DecisionNode (new state, new dice)
        │       ├── dice = 2 → DecisionNode
        │       ├── ...
        │       └── dice = 6 → DecisionNode
        ├── action a₂ → ChanceNode
        └── ...

PUCT selects at DecisionNodes. At ChanceNodes we sample dice uniformly
(equivalent to 6 visit-count slots, one per dice value, expanded as
visited — converges to the 1/6 expectation).

Terminal handling
-----------------
If after applying an action the game is terminal, the chance node is
replaced with a "terminal leaf" whose value is the actual outcome
(+1 / -1 from the root player's POV).

If after applying an action AND rolling the dice, the cp at the new
state has NO legal moves, the cp passes (turn skip). We follow the
cpp engine's `get_legal_moves` and `current_player` rotation directly
— the engine handles all the dice/skip/bonus mechanics for us.

Network leaf evaluation
-----------------------
At an unexpanded DecisionNode, we call the network ONCE to get:
  - prior π(a|s) for each legal action (rest get 0 mass)
  - V(s) value estimate from root-player POV (sigmoid·2−1 ∈ [-1, +1])

V is treated as the leaf return for backup.

Backup uses player-perspective: when backing up through a node where
cp ≠ root_player, the value is FLIPPED. (In a 2-player game, what's
good for opp is bad for me.)

Hyperparameters
---------------
- c_puct = 1.5
- Dirichlet noise at root: α=0.3, ε=0.25 (only when collecting training data)
- Move temperature: τ=1.0 for first 30 moves, τ=0.001 thereafter
  (greedy after early-game exploration)
"""
from __future__ import annotations

import math
import random
from typing import Optional, Dict, List, Callable, Tuple

import numpy as np

import td_ludo_cpp as ludo_cpp


# ─── Tree node types ──────────────────────────────────────────────────────
class DecisionNode:
    """A state where current_player must pick an action.

    State invariants:
      - state.current_dice_roll != 0 (dice already rolled for this decision)
      - get_legal_moves(state) is non-empty
      - state is NOT terminal
    """

    __slots__ = ("state", "cp", "legal", "P", "N", "W", "Q",
                 "children", "_expanded")

    def __init__(self, state, legal: List[int]):
        self.state = state
        self.cp = int(state.current_player)
        self.legal = legal
        # Prior P over LEGAL actions (4-element vector but only legal entries non-zero)
        self.P = np.zeros(4, dtype=np.float32)
        # Visit counts per action
        self.N = np.zeros(4, dtype=np.int32)
        # Cumulative value per action (from root_player POV)
        self.W = np.zeros(4, dtype=np.float64)
        # Mean value per action (W / N), maintained
        self.Q = np.zeros(4, dtype=np.float64)
        # Children: action_id → ChanceNode (lazy, expanded on first visit)
        self.children: Dict[int, "ChanceNode"] = {}
        self._expanded = False  # set True after first network eval

    def total_visits(self) -> int:
        return int(self.N.sum())

    def visit_distribution(self, temperature: float = 1.0) -> np.ndarray:
        """Get the post-search policy distribution π_search(a) ∝ N(a)^(1/τ).

        Returns a (4,) array, illegal actions = 0.
        """
        if temperature < 1e-3:
            # Greedy: argmax visits
            out = np.zeros(4, dtype=np.float32)
            best = -1
            best_n = -1
            for a in self.legal:
                if self.N[a] > best_n:
                    best_n = self.N[a]
                    best = a
            if best >= 0:
                out[best] = 1.0
            return out
        # Soft policy
        counts = self.N.astype(np.float64)
        # Zero out illegal
        legal_mask = np.zeros(4, dtype=np.float64)
        for a in self.legal:
            legal_mask[a] = 1.0
        counts = counts * legal_mask
        if temperature == 1.0:
            total = counts.sum()
        else:
            counts = np.power(np.maximum(counts, 0.0), 1.0 / temperature)
            total = counts.sum()
        if total <= 0:
            # No visits yet — fall back to uniform-over-legal
            out = legal_mask / max(1.0, legal_mask.sum())
            return out.astype(np.float32)
        return (counts / total).astype(np.float32)


class ChanceNode:
    """After an action is applied. Children are dice values 1..6 each leading
    to the next DecisionNode (or terminal leaf if the move ended the game).

    Note: a "chance node" in our setup represents the moment between
    applying an action and rolling the dice for the NEXT decision.
    """

    __slots__ = ("state_after_action", "terminal_value", "N", "children")

    def __init__(self, state_after_action, terminal_value: Optional[float] = None):
        # If state_after_action is terminal, terminal_value is the cached
        # ±1 outcome from the root_player POV. Otherwise None.
        self.state_after_action = state_after_action
        self.terminal_value = terminal_value
        self.N = 0  # total visits to this chance node
        # children[dice] = DecisionNode (or None if dice leads to skip/terminal)
        self.children: Dict[int, Optional[DecisionNode]] = {}

    def is_terminal(self) -> bool:
        return self.terminal_value is not None


# ─── Network evaluator interface ──────────────────────────────────────────
class NetworkEvaluator:
    """Wraps a model + encoder into a leaf-evaluation callable.

    Caller provides:
      - model: a V135ProductionAdapter-shaped module (or similar)
      - device: torch device
      - encoder_fn: state → np.ndarray (e.g., encode_state_v18_production)
      - root_player: which player's POV the V is computed from. Required
        for value sign-flip in 2-player Ludo.
    """

    def __init__(self, model, device, encoder_fn, root_player: int):
        import torch  # local import to keep numpy-only deps for unit tests
        self._torch = torch
        self.model = model
        self.device = device
        self.encoder_fn = encoder_fn
        self.root_player = root_player

    def evaluate_batch(self, states: List) -> Tuple[np.ndarray, np.ndarray]:
        """Run model on a batch of states.

        Returns:
          priors:  (B, 4) per-state policy (over token-ids 0..3, masked-softmax over legal)
          values:  (B,)   per-state value from root_player POV in [-1, +1]
        """
        import torch
        if not states:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        encoded = np.stack([self.encoder_fn(s) for s in states], axis=0)
        # Build per-state legal masks
        legal_masks = np.zeros((len(states), 4), dtype=np.float32)
        for i, s in enumerate(states):
            for a in ludo_cpp.get_legal_moves(s):
                legal_masks[i, a] = 1.0
        with torch.no_grad():
            x = torch.from_numpy(encoded).to(self.device, dtype=torch.float32)
            m = torch.from_numpy(legal_masks).to(self.device, dtype=torch.float32)
            out = self.model(x, m)
            policy = out[0]
            win_prob = out[1]
        priors = policy.cpu().numpy().astype(np.float32)  # already masked-softmaxed
        # Value in [-1, +1] from CURRENT PLAYER POV.
        v_from_cp = (2.0 * win_prob - 1.0).cpu().numpy().reshape(-1).astype(np.float32)
        # Flip sign if cp != root_player (2-player Ludo)
        values = np.zeros(len(states), dtype=np.float32)
        for i, s in enumerate(states):
            sign = +1.0 if int(s.current_player) == self.root_player else -1.0
            values[i] = sign * v_from_cp[i]
        return priors, values


# ─── MCTS algorithm ────────────────────────────────────────────────────────
class MCTS:
    """AlphaZero-style PUCT MCTS for Ludo.

    Usage:
        mcts = MCTS(network_evaluator, c_puct=1.5, n_sims=100,
                    dirichlet_alpha=0.3, dirichlet_eps=0.25)
        pi_search = mcts.search(root_state, training=True)
    """

    def __init__(
        self,
        evaluator: NetworkEvaluator,
        c_puct: float = 1.5,
        n_sims: int = 100,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        rng: Optional[random.Random] = None,
    ):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.n_sims = n_sims
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = rng or random.Random()

    # ── Public API ─────────────────────────────────────────────────────────
    def search(self, root_state, training: bool = True) -> DecisionNode:
        """Run n_sims MCTS simulations starting from root_state.

        Returns the root DecisionNode (visit-counts populated).
        Caller can extract `pi_search = root.visit_distribution(temperature)`.
        """
        # Set the evaluator's root_player to match this search's root
        self.evaluator.root_player = int(root_state.current_player)

        # Build root node
        legal = list(ludo_cpp.get_legal_moves(root_state))
        if not legal:
            raise ValueError("root state has no legal moves")
        root = DecisionNode(root_state, legal)

        # Initial network call to seed root priors + value
        priors, _values = self.evaluator.evaluate_batch([root_state])
        root.P = priors[0]
        root._expanded = True

        # Optional Dirichlet noise at root (training only)
        if training and self.dirichlet_eps > 0 and len(legal) > 1:
            self._apply_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.n_sims):
            self._simulate(root)

        return root

    # ── Simulation = select + expand + backup ──────────────────────────────
    def _simulate(self, root: DecisionNode):
        """Run one MCTS simulation from root."""
        path = []  # list of (DecisionNode, action_id) tuples for backup
        node = root

        while True:
            # PUCT select an action at this DecisionNode
            action = self._puct_select(node)
            path.append((node, action))

            # Move to or expand the corresponding ChanceNode
            chance = node.children.get(action)
            if chance is None:
                # First visit: apply the action to get the state-after-action
                state_after = ludo_cpp.apply_move(node.state, int(action))
                if state_after.is_terminal:
                    winner = int(ludo_cpp.get_winner(state_after))
                    rp = self.evaluator.root_player
                    if winner == rp:
                        leaf_v = +1.0
                    elif winner < 0:
                        leaf_v = 0.0
                    else:
                        leaf_v = -1.0  # 2-player Ludo
                    chance = ChanceNode(state_after, terminal_value=leaf_v)
                else:
                    chance = ChanceNode(state_after)
                node.children[action] = chance
                # Backup the leaf value (terminal or unevaluated chance)
                if chance.is_terminal():
                    self._backup(path, chance.terminal_value)
                    return
                # Need to sample dice + evaluate next DecisionNode (LEAF)
                next_node, leaf_v = self._roll_dice_and_expand(chance)
                self._backup(path, leaf_v)
                return

            chance.N += 1  # account for this visit to the chance node

            # Existing chance node
            if chance.is_terminal():
                self._backup(path, chance.terminal_value)
                return

            # Sample a dice value and follow (or expand if first-visit dice)
            dice = self.rng.randint(1, 6)
            child = chance.children.get(dice)
            if child is None:
                # First visit to this dice value — expand new DecisionNode
                child, leaf_v = self._expand_dice_child(chance, dice)
                chance.children[dice] = child
                if child is None:
                    # Skip / no legal moves after this dice — treat as
                    # passing turn. leaf_v from the V-eval of whoever's next.
                    self._backup(path, leaf_v)
                    return
                # New leaf — backup the V-eval result
                self._backup(path, leaf_v)
                return

            # Already-expanded dice child → recurse into it (path continues)
            node = child

    def _puct_select(self, node: DecisionNode) -> int:
        """Pick action argmax_a [Q(a) + c_puct * P(a) * sqrt(ΣN) / (1 + N(a))]."""
        sum_n = max(1, node.total_visits())
        sqrt_sum = math.sqrt(sum_n)
        best_a = -1
        best_score = -float("inf")
        for a in node.legal:
            u = self.c_puct * node.P[a] * sqrt_sum / (1.0 + node.N[a])
            score = node.Q[a] + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _expand_dice_child(self, chance: ChanceNode, dice: int) -> Tuple[Optional[DecisionNode], float]:
        """Given a chance node and a sampled dice value, create the
        next DecisionNode (or handle the no-legal-moves skip case).

        Returns:
          (child_node, leaf_value)  where leaf_value is the V-eval to backup.
        """
        # Make a copy of the state-after-action and set its dice
        s = _copy_state(chance.state_after_action)
        s.current_dice_roll = int(dice)

        # Skip turns where cp has no legal moves (turn passes)
        # Use a small advance loop to handle 3-six bonus and passes
        # NOTE: we DON'T re-roll the dice — the dice was just set above.
        skipped = False
        legal = ludo_cpp.get_legal_moves(s)
        while not legal and not s.is_terminal:
            # No legal moves → pass to next active player
            cp = int(s.current_player)
            nxt = (cp + 1) % 4
            while not s.active_players[nxt]:
                nxt = (nxt + 1) % 4
            s.current_player = nxt
            s.current_dice_roll = 0  # next player gets a fresh roll
            skipped = True
            # Roll a fresh dice for them too (random — chance branch)
            s.current_dice_roll = self.rng.randint(1, 6)
            legal = ludo_cpp.get_legal_moves(s)

        if s.is_terminal:
            winner = int(ludo_cpp.get_winner(s))
            rp = self.evaluator.root_player
            if winner == rp:
                return None, +1.0
            elif winner < 0:
                return None, 0.0
            else:
                return None, -1.0

        if not legal:
            # Stuck — shouldn't happen but be safe
            return None, 0.0

        # Evaluate this new state via the network
        priors, values = self.evaluator.evaluate_batch([s])
        child = DecisionNode(s, list(legal))
        child.P = priors[0]
        child._expanded = True
        return child, float(values[0])

    def _roll_dice_and_expand(self, chance: ChanceNode) -> Tuple[Optional[DecisionNode], float]:
        """Roll a dice + expand a new DecisionNode for the first time."""
        dice = self.rng.randint(1, 6)
        child, leaf_v = self._expand_dice_child(chance, dice)
        chance.children[dice] = child
        return child, leaf_v

    def _backup(self, path: List[Tuple[DecisionNode, int]], leaf_value: float):
        """Backup leaf_value (already in root_player POV) along the path."""
        for node, action in reversed(path):
            node.N[action] += 1
            node.W[action] += leaf_value
            node.Q[action] = node.W[action] / max(1, node.N[action])

    def _apply_dirichlet_noise(self, root: DecisionNode):
        """Add Dirichlet noise to root priors (Eric/AlphaZero standard)."""
        legal = root.legal
        if len(legal) < 2:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
        eps = self.dirichlet_eps
        new_P = np.zeros(4, dtype=np.float32)
        for k, a in enumerate(legal):
            new_P[a] = (1 - eps) * root.P[a] + eps * float(noise[k])
        # Re-normalize over legal
        s = sum(new_P[a] for a in legal)
        if s > 0:
            for a in legal:
                new_P[a] /= s
        root.P = new_P


# ─── Helpers ──────────────────────────────────────────────────────────────
def _copy_state(state):
    """Deep copy of a GameState. Mirrors the pattern from
    generate_search_data._copy_state — pybind GameState has no copy ctor,
    so we manually copy each field via numpy."""
    s = ludo_cpp.GameState()
    s.player_positions = np.array(state.player_positions, dtype=np.int8).copy()
    s.scores = np.array(state.scores, dtype=np.int8).copy()
    s.active_players = np.array(state.active_players, dtype=bool).copy()
    s.idle_counter = np.array(state.idle_counter, dtype=np.int8).copy()
    s.last_moved_token = np.array(state.last_moved_token, dtype=np.int8).copy()
    s.streak = np.array(state.streak, dtype=np.int8).copy()
    s.current_player = int(state.current_player)
    s.current_dice_roll = int(state.current_dice_roll)
    s.is_terminal = bool(state.is_terminal)
    return s
