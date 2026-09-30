"""Inference-time expectimax search wrapping the v15 policy/value net.

The decisive ceiling experiment (CEILING_RESEARCH.md lever 1): does adding
lookahead at PLAY time beat the raw policy (argmax) at identical weights? If
yes, search extracts skill the raw policy left on the table; if no, the
bottleneck is value-net accuracy.

Design (2-ply expectiminimax, the backgammon-relevant depth):
  For each of my legal moves m:
    apply m -> s'
    E over opponent dice d in 1..6 (prob 1/6, with 3-six handling skipped at
    the horizon for cost) of:
      opponent plays its POLICY argmax (the exact thing we test against)
      -> leaf state; value = net P(I win) from my POV (dice-neutral)
  pick m with max expected value.
Bonus-turn (my move earns another turn) and opp-pass cases score the position
as-is. Reuses the proven v13-engine mechanics from strong_bots.ExpectimaxBot.
Net calls are BATCHED (one pass for opp policies, one for leaf values).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V15_ROOT = HERE.parent / "td_ludo_v15"
sys.path.insert(0, str(V15_ROOT))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.strong_bots import _clone_state
from td_ludo_v15.game.encoder import encode_frame as _enc
from td_ludo_v15.game.cells import (
    position_to_cell_in_pov as _pos2cell, cell_to_index as _cell2idx,
    NUM_BOARD_CELLS as _NCELLS,
)
import td_ludo_v15_cpp as _v15cpp
from td_ludo_v15.models.v15 import V15GraphTransformer

_BASE = _v15cpp.BASE_POS


class V15Eval:
    """Batched value + policy access for the v15 GraphTransformer on v13 states."""

    def __init__(self, path, device=None):
        self.device = device or torch.device("cpu")
        m = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=1)
        ck = torch.load(path, map_location=self.device, weights_only=False)
        sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        m.load_state_dict(sd, strict=False)
        self.model = m.eval().to(self.device)

    def _encode(self, state, pov):
        # (B=1, T=1, 15, 15, 3) — history_len=1
        x = np.zeros((1, 1, 15, 15, 3), dtype=np.float32)
        x[0, 0] = _enc(state, pov_player=pov)
        return x

    def _legal_mask(self, state, legal, pov):
        mask = np.zeros(_NCELLS, dtype=np.float32)
        idx = {}
        for t in legal:
            pos = int(state.player_positions[pov][int(t)])
            c = _pos2cell(_BASE if pos == _BASE else pos, pov, pov)
            i = _cell2idx(*c)
            mask[i] = 1.0
            idx[int(t)] = i
        return mask, idx

    @torch.no_grad()
    def values(self, states, povs):
        """Batched P(pov wins) for a list of (state, pov)."""
        if not states:
            return np.zeros(0, dtype=np.float32)
        xs = np.concatenate([self._encode(s, p) for s, p in zip(states, povs)], axis=0)
        masks = np.ones((len(states), _NCELLS), dtype=np.float32)  # value head is mask-indep
        xt = torch.from_numpy(xs).to(self.device)
        mt = torch.from_numpy(masks).to(self.device)
        _, wp = self.model(xt, mt)
        return wp.detach().cpu().numpy().reshape(-1)

    @torch.no_grad()
    def policy_argmax_batch(self, items):
        """items: list of (state, legal_list, pov). Returns list of chosen token."""
        if not items:
            return []
        xs, masks, idxs = [], [], []
        for state, legal, pov in items:
            xs.append(self._encode(state, pov))
            mask, idx = self._legal_mask(state, legal, pov)
            masks.append(mask[None, :]); idxs.append(idx)
        xt = torch.from_numpy(np.concatenate(xs, 0)).to(self.device)
        mt = torch.from_numpy(np.concatenate(masks, 0)).to(self.device)
        pol, _ = self.model(xt, mt)
        pol = pol.detach().cpu().numpy()
        out = []
        for row, idx in zip(pol, idxs):
            out.append(max(idx.items(), key=lambda kv: row[kv[1]])[0])
        return out

    def policy_argmax(self, state, legal, pov):
        if len(legal) <= 1:
            return int(legal[0]) if legal else -1
        return int(self.policy_argmax_batch([(state, list(legal), pov)])[0])


def _other_active(state, me):
    for p in range(4):
        if p != me and state.active_players[p]:
            return p
    return None


class NeuralExpectimaxBot:
    """2-ply expectimax with the v15 value net at leaves, v15 policy as the
    opponent model. select_move(state, legal) -> token (matches bot interface)."""

    def __init__(self, ev: V15Eval, player_id=None, depth=2, mode="anchored", margin=0.03):
        self.ev = ev
        self.player_id = player_id
        self.depth = depth  # only 2 supported here
        # mode "value": pick argmax of the 2-ply value-lookahead (naive, ignores
        #   the policy — degenerates toward random when the value head is flat).
        # mode "anchored": default to the raw policy's move, override ONLY if a
        #   different move's lookahead Q exceeds it by `margin` (win-prob units).
        #   Guaranteed ≥ raw policy by construction; tests "does value-lookahead
        #   catch the policy's blunders?" (the backgammon blunder-elimination win).
        self.mode = mode
        self.margin = margin

    def select_move(self, state, legal_moves, history=None):
        legal = [int(a) for a in legal_moves]
        if not legal:
            return -1
        if len(legal) == 1:
            return legal[0]
        me = self.player_id if self.player_id is not None else int(state.current_player)
        opp = _other_active(state, me)

        # Pass A: for each move, build the (opp-dice) sub-states; gather opp-policy queries.
        per_move = {a: [] for a in legal}     # a -> list of ("val", state) leaf requests OR ("opp", marker)
        opp_queries = []                       # (a, d, sim, opp_legal)
        direct_leaves = []                     # (a, d, state)  -> value(me) directly
        terminal_val = {}                      # a -> forced value if my move ends game
        for a in legal:
            after = ludo_cpp.apply_move(state, a)
            if after.is_terminal:
                terminal_val[a] = 1.0 if ludo_cpp.get_winner(after) == me else 0.0
                continue
            for d in range(1, 7):
                sim = _clone_state(after)
                sim.current_dice_roll = d
                cp = int(sim.current_player)
                if cp == me or opp is None:           # bonus turn / no opp -> score as-is
                    direct_leaves.append((a, d, sim))
                else:
                    opp_legal = [int(x) for x in ludo_cpp.get_legal_moves(sim)]
                    if not opp_legal:                  # opp passes
                        direct_leaves.append((a, d, sim))
                    else:
                        opp_queries.append((a, d, sim, opp_legal))

        # Pass B: batch opp-policy argmax, apply, those become leaves too.
        if opp_queries:
            chosen = self.ev.policy_argmax_batch([(s, l, int(s.current_player)) for _, _, s, l in opp_queries])
            for (a, d, sim, _legal), mv in zip(opp_queries, chosen):
                after_opp = ludo_cpp.apply_move(sim, int(mv))
                direct_leaves.append((a, d, after_opp))

        # Pass C: batch leaf values P(me wins).
        states = [s for _, _, s in direct_leaves]
        vals = self.ev.values(states, [me] * len(states))
        acc = {a: 0.0 for a in legal}
        cnt = {a: 0 for a in legal}
        for (a, d, _s), v in zip(direct_leaves, vals):
            acc[a] += float(v); cnt[a] += 1

        q = {}
        for a in legal:
            q[a] = (terminal_val[a] + 1.0) if a in terminal_val else acc[a] / max(1, cnt[a])

        if self.mode == "value":
            return int(max(legal, key=lambda a: q[a]))

        # anchored: default to raw policy move; override only on a clear Q margin.
        pol_pick = self.ev.policy_argmax(state, legal, me)
        best_a, best_q = pol_pick, q[pol_pick]
        for a in legal:
            if a != pol_pick and q[a] > best_q + self.margin:
                best_a, best_q = a, q[a]
        return int(best_a)
