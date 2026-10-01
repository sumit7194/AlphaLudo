"""AlphaZero Neural Search Engine for AlphaLudo V13.7.

Implements highly-optimized batched 2-ply Expectimax tree search:
1. Evaluates leaf states with V13.7's trained Value Network V_theta(s).
2. Uses V13.7's trained Policy Network P_theta(s, a) for move priors and opponent responses.
3. Batches all neural forward passes together for high-throughput multi-core inference.
4. Zero blind random rollouts. Full expectation over all 6 dice rolls.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17
from alphaludo.model_v137 import AlphaLudoV137


def _clone_state(state):
    """Builds an independent mutable copy of a C++ GameState."""
    new = ludo_cpp.GameState()
    new.player_positions = np.array(state.player_positions, dtype=np.int8).copy()
    new.scores = np.array(state.scores, dtype=np.int8).copy()
    new.active_players = np.array(state.active_players, dtype=bool).copy()
    new.current_player = int(state.current_player)
    new.current_dice_roll = int(state.current_dice_roll)
    new.is_terminal = bool(getattr(state, "is_terminal", False))
    return new


class AlphaZeroNeuralSearchBot:
    """True AlphaZero Neural Search Bot using V13.7 Dual-Head ResNet.

    Evaluates candidate moves via full-expectation lookahead over all 6 dice rolls.
    Replaces blind random rollouts with direct V_theta(s) leaf evaluation.
    """

    def __init__(
        self,
        model: AlphaLudoV137,
        device: torch.device | None = None,
        player_id: int | None = None,
        prior_weight: float = 0.20,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.player_id = player_id
        self.prior_weight = prior_weight
        self.model.eval().to(self.device)

    def select_move(self, state, legal_moves):
        legal = [int(a) for a in legal_moves]
        if not legal:
            return 0
        if len(legal) == 1:
            return legal[0]

        me = self.player_id if self.player_id is not None else int(state.current_player)

        # 1. Root policy prior
        enc_root = encode_state_v17(state)
        mask_root = np.zeros(4, dtype=np.float32)
        for m in legal:
            mask_root[m] = 1.0

        with torch.no_grad():
            x_root = torch.from_numpy(enc_root).unsqueeze(0).to(self.device)
            m_root = torch.from_numpy(mask_root).unsqueeze(0).to(self.device)
            logits_root, _ = self.model(x_root, m_root)
            probs_root = F.softmax(logits_root, dim=-1).squeeze(0).cpu().numpy()

        # 2. Branch over candidate moves
        move_scores = {m: 0.0 for m in legal}
        opp_policy_queries = []  # list of (m, sim_state, opp_legal, d_opp)
        leaf_states = []         # list of (state, m, weight)
        dice_prob = 1.0 / 6.0

        for m in legal:
            after_mine = ludo_cpp.apply_move(_clone_state(state), int(m))

            # Immediate winning move: always take it!
            if after_mine.is_terminal:
                if ludo_cpp.get_winner(after_mine) == me:
                    return m
                else:
                    move_scores[m] = -1.0
                    continue

            # Bonus turn (rolled 6, captured, or home)
            if int(after_mine.current_player) == me:
                leaf_states.append((after_mine, m, 1.0))
                continue

            # Opponent turn: test all 6 possible dice rolls
            for d_opp in range(1, 7):
                sim = _clone_state(after_mine)
                sim.current_dice_roll = d_opp
                opp_legal = ludo_cpp.get_legal_moves(sim)

                if not opp_legal:
                    # Opponent passes
                    leaf_states.append((sim, m, dice_prob))
                elif len(opp_legal) == 1:
                    after_opp = ludo_cpp.apply_move(sim, int(opp_legal[0]))
                    if after_opp.is_terminal:
                        win = ludo_cpp.get_winner(after_opp)
                        move_scores[m] += (1.0 if win == me else -1.0) * dice_prob
                    else:
                        leaf_states.append((after_opp, m, dice_prob))
                else:
                    opp_policy_queries.append((m, sim, opp_legal, dice_prob))

        # 3. Batched Opponent Policy Queries
        if opp_policy_queries:
            b_enc = np.stack([encode_state_v17(sim) for _, sim, _, _ in opp_policy_queries], axis=0)
            b_masks = np.zeros((len(opp_policy_queries), 4), dtype=np.float32)
            for i, (_, _, o_leg, _) in enumerate(opp_policy_queries):
                for om in o_leg:
                    b_masks[i, om] = 1.0

            with torch.no_grad():
                bx = torch.from_numpy(b_enc).to(self.device)
                bm = torch.from_numpy(b_masks).to(self.device)
                lo, _ = self.model(bx, bm)
                chosen_opp_acts = lo.argmax(dim=-1).cpu().numpy()

            for i, (m, sim, o_leg, weight) in enumerate(opp_policy_queries):
                act = int(chosen_opp_acts[i])
                if act not in o_leg:
                    act = int(o_leg[0])
                after_opp = ludo_cpp.apply_move(sim, act)
                if after_opp.is_terminal:
                    win = ludo_cpp.get_winner(after_opp)
                    move_scores[m] += (1.0 if win == me else -1.0) * weight
                else:
                    leaf_states.append((after_opp, m, weight))

        # 4. Batched Leaf Value Evaluations
        if leaf_states:
            batch_enc = np.stack([encode_state_v17(s) for s, _, _ in leaf_states], axis=0)
            batch_masks = np.ones((len(leaf_states), 4), dtype=np.float32)

            with torch.no_grad():
                bx = torch.from_numpy(batch_enc).to(self.device)
                bm = torch.from_numpy(batch_masks).to(self.device)
                _, b_vals = self.model(bx, bm)
                v_preds = b_vals.squeeze(-1).cpu().numpy()

            for idx, (s, m, weight) in enumerate(leaf_states):
                cur = int(s.current_player)
                val_for_me = float(v_preds[idx]) if cur == me else -float(v_preds[idx])
                move_scores[m] += val_for_me * weight

        # 5. Combine Q(s, a) with Policy Prior P(s, a)
        best_move = legal[0]
        best_score = -float("inf")

        for m in legal:
            score = move_scores[m] + self.prior_weight * probs_root[m]
            if score > best_score:
                best_score = score
                best_move = m

        return best_move
