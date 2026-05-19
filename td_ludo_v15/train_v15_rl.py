"""V15 RL — PPO with opponent mix on V15GraphTransformer.

Initialized from V15 SL (model_sl.pt @ 83% bot-eval, ≥ V13.5 teacher).

Differences vs V13.5 RL:
  - Source-cell policy (225 logits, masked softmax) instead of rank-indexed (4).
  - 8-frame history per game (V15 encoder is temporal).
  - PPO clipped (Phase-L style), not REINFORCE-with-baseline.
  - Opponent mix during rollout, NOT pure self-play:
        35% self-play (V15 vs V15)
        30% V13.5_RL  (strongest legacy)
        20% V13.5_SL
        15% V13.2
    Trajectories are collected ONLY from V15's own turns. Opponent turns
    just advance the game (no gradient).
  - KL anchor target = V15 SL (V13.5 anchor would pull through cross-arch
    projection — pointless).

Same RL dashboard as V13.5 (serves /api/rl_stats, /api/chain — same fields).

Usage on VM:
    TD_LUDO_RUN_NAME=v15_rl python3 train_v15_rl.py \\
        --init   checkpoints/v15_sl_v2/model_sl.pt \\
        --kl-anchor checkpoints/v15_sl_v2/model_sl.pt \\
        --opp-v135-rl /home/sumit/td_ludo/checkpoints/v135_prod_rl_local/model_latest.pt \\
        --opp-v135-sl /home/sumit/td_ludo/checkpoints/v135_full/model_latest.pt \\
        --opp-v132    /home/sumit/td_ludo/checkpoints/v132/model_latest.pt \\
        --target-states 20000000 --port 8799
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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bridge to legacy code for V13.5 / V13.2 opponents
_LEGACY_ROOT = Path(__file__).resolve().parent.parent / "td_ludo"
if str(_LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGACY_ROOT))

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
import td_ludo_v15_cpp as _v15_cpp  # for constants

from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS,
    cell_to_index,
    position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.models.v15 import V15GraphTransformer


_BASE_POS = _v15_cpp.BASE_POS
_HOME_POS = _v15_cpp.HOME_POS
NUM_PLAYERS = 4
# History/stack depth — set by configure_history() once args are parsed.
# Module-level so the env, eval harness, and model all see the same T.
# Defaults match V15 (8-frame stack); V15.1 sets T=2 via --history-len 2.
HISTORY_LEN = 7          # past frames (TOTAL_FRAMES - 1)
TOTAL_FRAMES = 8         # current + past

def configure_history(total_frames: int):
    """Switch the module-level stack depth (default V15=8, V15.1=2)."""
    global HISTORY_LEN, TOTAL_FRAMES
    if total_frames < 1:
        raise ValueError(f"history-len must be >= 1, got {total_frames}")
    TOTAL_FRAMES = total_frames
    HISTORY_LEN = total_frames - 1


# ─── Args ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=None,
                   help="V15 SL checkpoint to initialize the student from "
                        "(typically checkpoints/v15_sl_v2/model_sl.pt)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from <ckpt-dir>/model_latest.pt")
    p.add_argument("--kl-anchor", default=None,
                   help="V15 SL checkpoint for KL anchor (defaults to --init).")
    # Opponent model paths
    p.add_argument("--opp-v135-rl", default=None,
                   help="V13.5 RL checkpoint (strongest legacy opponent)")
    p.add_argument("--opp-v135-sl", default=None,
                   help="V13.5 SL checkpoint")
    p.add_argument("--opp-v132", default=None,
                   help="V13.2 checkpoint (asymmetric per-token policy)")
    # Opponent mix weights (auto-renormalized).
    p.add_argument("--opp-weight-self", type=float, default=35.0)
    p.add_argument("--opp-weight-v135-rl", type=float, default=30.0)
    p.add_argument("--opp-weight-v135-sl", type=float, default=20.0)
    p.add_argument("--opp-weight-v132", type=float, default=15.0)
    # PPO knobs
    p.add_argument("--target-states", type=int, default=20_000_000)
    p.add_argument("--max-game-len", type=int, default=400)
    p.add_argument("--parallel-games", type=int, default=64)
    p.add_argument("--train-chunk", type=int, default=2048)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--train-epochs", type=int, default=2)
    p.add_argument("--ppo-clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-end", type=float, default=5e-6)
    p.add_argument("--entropy-coeff", type=float, default=0.03)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--kl-anchor-coeff", type=float, default=0.1)
    # Save / eval cadence
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--eval-every", type=int, default=200_000)
    p.add_argument("--eval-games", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    # Model arch (must match --init)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=512)
    # History window — V15=8 frames (1 current + 7 past), V15.1=2.
    # MUST match whatever the --init checkpoint was trained with: the
    # input_mlp's in_features depends on it (T*3). A mismatch yields a
    # shape error at load_state_dict time.
    p.add_argument("--history-len", type=int, default=8,
                   help="Total frames T in the stack (1 current + (T-1) past). "
                        "V15=8, V15.1=2. Must match the --init checkpoint.")
    # Misc
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8799)
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


# ─── V15 student loader ────────────────────────────────────────────────────
def load_v15_student(path, args, device):
    model = V15GraphTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        history_len=args.history_len,
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model, ckpt


def load_v15_kl_anchor(path, args, device):
    model, _ = load_v15_student(path, args, device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ─── Opponent loaders + action pickers ─────────────────────────────────────
def _probe_v135_arch(state_dict):
    cw = state_dict.get("conv_input.weight")
    if cw is None:
        raise RuntimeError("V13.5 checkpoint missing conv_input.weight")
    num_channels = int(cw.shape[0])
    idxs = set()
    for k in state_dict.keys():
        if k.startswith("res_blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                idxs.add(int(parts[1]))
    num_res_blocks = max(idxs) + 1 if idxs else 0
    return num_res_blocks, num_channels


def load_v135_opponent(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
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
    return model


def load_v132_opponent(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    # Probe arch
    cw = sd.get("conv_input.weight")
    num_channels = int(cw.shape[0])
    idxs = set()
    for k in sd:
        if k.startswith("res_blocks."):
            try:
                idxs.add(int(k.split(".")[1]))
            except Exception:
                pass
    nrb = max(idxs) + 1 if idxs else 10
    model = MinimalCNN14(num_res_blocks=nrb, num_channels=num_channels, in_channels=17)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def v135_pick(model, device, state, legal):
    """Greedy action selection for a V13.5-arch opponent (rank-indexed)."""
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


def v132_pick(model, device, state, legal):
    """Greedy action selection for a V13.2-arch opponent (per-token policy)."""
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


# ─── Self-play env with opponent mix ───────────────────────────────────────
class V15RLEnv:
    """B parallel games. Each game has a randomly-assigned (a) student player
    slot and (b) opponent type. Only V15 student's decisions feed the
    trajectory; opponent turns just advance the game using the opponent's
    greedy action.
    """

    def __init__(self, batch_size, opponents, opp_probs, max_game_len=400):
        """
        opponents: dict name → pick function `f(state, legal) → token_id`.
                   Must include "self" — at rollout time, "self" decisions
                   are deferred to the main loop (which runs the student).
        opp_probs: dict name → float (renormalized). Probability of each
                   opponent type when a new game starts.
        """
        self.batch_size = batch_size
        self.max_game_len = max_game_len
        self.opponents = opponents
        names = list(opp_probs.keys())
        weights = np.array([float(opp_probs[n]) for n in names], dtype=np.float64)
        weights = weights / max(1e-9, weights.sum())
        self.opp_names = names
        self.opp_probs = weights
        self.games = [_legacy_cpp.create_initial_state_2p() for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, NUM_PLAYERS), dtype=np.int32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)
        self.student_player = np.zeros(batch_size, dtype=np.int32)
        self.opp_per_game = ["self"] * batch_size
        # Per-game history of pre-decision GameStates (POV-agnostic; encoded
        # at extraction time in the current student-decision's POV).
        self.history = [collections.deque(maxlen=HISTORY_LEN) for _ in range(batch_size)]
        # Per-game trajectory (only the student's decision points).
        self.trajectory = [[] for _ in range(batch_size)]
        # Stats
        self.games_played = 0
        self.game_lengths = []
        self.opp_game_counts = collections.Counter()
        for i in range(batch_size):
            self._reset(i)

    def _reset(self, i):
        self.games[i] = _legacy_cpp.create_initial_state_2p()
        self.consec_sixes[i] = 0
        self.step_count[i] = 0
        self.history[i].clear()
        self.trajectory[i] = []
        # Assign random student slot and opponent
        self.student_player[i] = random.choice([0, 2])
        self.opp_per_game[i] = self.opp_names[int(np.random.choice(len(self.opp_names),
                                                                   p=self.opp_probs))]

    def _finalize(self, i, winner):
        """Convert trajectory to terminal-return tuples and reset."""
        out = []
        sp = int(self.student_player[i])
        for entry in self.trajectory[i]:
            G = 1.0 if winner == sp else (0.0 if winner == -1 else -1.0)
            out.append({**entry, "G": G})
        opp = self.opp_per_game[i]
        self.opp_game_counts[opp] += 1
        self.games_played += 1
        self.game_lengths.append(len(self.trajectory[i]))
        if len(self.game_lengths) > 200:
            self.game_lengths.pop(0)
        return out

    def spin_to_decision(self):
        """Advance every game until it's the STUDENT's turn (or game ends
        / truncates, in which case finalize trajectory and start a new game).
        Returns the batched student decision states ready for one forward pass,
        plus any trajectories that finished during this spin.
        """
        finished = []
        decision_idxs = []
        v15_xs = []
        v15_masks = []
        legal_lists = []  # raw token-IDs per state

        for i in range(self.batch_size):
            while True:
                game = self.games[i]
                if game.is_terminal:
                    winner = int(_legacy_cpp.get_winner(game))
                    finished.extend(self._finalize(i, winner))
                    self._reset(i)
                    game = self.games[i]
                if self.step_count[i] >= self.max_game_len:
                    # Truncate as draw
                    finished.extend(self._finalize(i, -1))
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
                opp_name = self.opp_per_game[i]
                if cp == sp:
                    # ── Student decision: prepare encoding ─────────────────
                    past_states = list(self.history[i])
                    pad = HISTORY_LEN - len(past_states)
                    v15_x = np.zeros((TOTAL_FRAMES, 15, 15, 3), dtype=np.float32)
                    real_frames = [None] * pad + past_states + [game]
                    for t_idx, st in enumerate(real_frames):
                        if st is None:
                            continue
                        v15_x[t_idx] = encode_frame(st, pov_player=cp)
                    v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
                    for t in legal:
                        pos = int(game.player_positions[cp][t])
                        cell = position_to_cell_in_pov(
                            _BASE_POS if pos == _BASE_POS else pos, cp, cp)
                        v15_legal[cell_to_index(*cell)] = 1.0
                    decision_idxs.append(i)
                    v15_xs.append(v15_x)
                    v15_masks.append(v15_legal)
                    legal_lists.append(legal)
                    break  # this game is at a student decision
                else:
                    # ── Opponent decision: pick + apply, no trajectory ─────
                    # For "self" mix slot, use the student's CURRENT weights
                    # by also routing through the model — but we don't have
                    # access to the student here. Solution: when opp == "self",
                    # we also call decision path, BUT no trajectory step is
                    # collected for self-as-opponent moves. To keep code
                    # simple, we just treat opp == "self" the same as the
                    # student case AND skip trajectory collection: that
                    # means we double-bias on student moves. So instead we
                    # use a frozen snapshot of student via the kl_anchor
                    # opponent slot. Practical fix: "self" opponent uses the
                    # student model passed in via opponents dict.
                    pick_fn = self.opponents.get(opp_name)
                    if pick_fn is None:
                        # Unknown opponent: fallback random
                        token = random.choice(legal)
                    else:
                        token = pick_fn(game, list(legal))
                    if token not in legal:
                        token = legal[0]
                    # Push the pre-move state into history (so the student's
                    # next observation sees opp's prior move as an old frame).
                    self.history[i].append(self.games[i])
                    self.games[i] = _legacy_cpp.apply_move(self.games[i], int(token))
                    self.step_count[i] += 1
                    # continue spinning

        # Pack arrays only if we collected something
        if decision_idxs:
            v15_x_arr = np.stack(v15_xs, axis=0)
            v15_mask_arr = np.stack(v15_masks, axis=0)
        else:
            v15_x_arr = np.zeros((0, TOTAL_FRAMES, 15, 15, 3), dtype=np.float32)
            v15_mask_arr = np.zeros((0, NUM_BOARD_CELLS), dtype=np.float32)
        return {
            "decision_idxs": decision_idxs,
            "v15_x": v15_x_arr,
            "v15_mask": v15_mask_arr,
            "legal_lists": legal_lists,
            "finished": finished,
        }

    def apply_student_actions(self, decision_idxs, legal_lists,
                              chosen_cells, log_probs, v_preds, v15_xs, v15_masks):
        """Apply the student's chosen actions, recording trajectory entries.

        chosen_cells: per-decision cell index (0..224).
        Each cell maps back to a token-id by finding the legal token whose
        source cell matches; if multiple, lowest token-id wins (state-equiv).
        """
        for k, i in enumerate(decision_idxs):
            game = self.games[i]
            cp = int(game.current_player)
            chosen_r, chosen_c = divmod(int(chosen_cells[k]), 15)
            chosen_cell = (chosen_r, chosen_c)
            # Find a legal token whose source cell == chosen_cell
            token = None
            for t in sorted(legal_lists[k]):
                pos = int(game.player_positions[cp][t])
                cell = position_to_cell_in_pov(
                    _BASE_POS if pos == _BASE_POS else pos, cp, cp)
                if cell == chosen_cell:
                    token = t
                    break
            if token is None:
                token = legal_lists[k][0]
                # Fall back: pick THIS token's source cell so the trajectory
                # still references a legal cell. Cell index will be remapped
                # below for consistency.
                pos = int(game.player_positions[cp][token])
                cell = position_to_cell_in_pov(
                    _BASE_POS if pos == _BASE_POS else pos, cp, cp)
                chosen_cells[k] = cell_to_index(*cell)
            # Record trajectory step
            self.trajectory[i].append({
                "v15_x": v15_xs[k],
                "v15_mask": v15_masks[k],
                "chosen_cell": int(chosen_cells[k]),
                "log_prob_old": float(log_probs[k]),
                "v_pred_old": float(v_preds[k]),
                "cp": cp,
            })
            # Push pre-decision state into history; advance
            self.history[i].append(self.games[i])
            self.games[i] = _legacy_cpp.apply_move(self.games[i], int(token))
            self.step_count[i] += 1


# ─── Quick eval (vs heuristic-bot mix, mirrors V15 SL eval) ────────────────
def quick_eval(student, device, n_games: int = 200) -> float:
    from src.heuristic_bot import get_bot, BOT_REGISTRY  # legacy
    from src.config import MAX_MOVES_PER_GAME  # legacy
    bot_types = list(BOT_REGISTRY.keys())
    student.eval()
    wins = 0
    for _ in range(n_games):
        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        bot = get_bot(random.choice(bot_types), player_id=opp_player)
        state = _legacy_cpp.create_initial_state_2p()
        history: collections.deque = collections.deque(maxlen=HISTORY_LEN)
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
                d = random.randint(1, 6)
                if d == 6:
                    csix[cp] += 1
                    if csix[cp] >= 3:
                        csix[cp] = 0
                        n = (cp + 1) % 4
                        while not state.active_players[n]:
                            n = (n + 1) % 4
                        state.current_player = n
                        state.current_dice_roll = 0
                        continue
                else:
                    csix[cp] = 0
                state.current_dice_roll = d
            legal = _legacy_cpp.get_legal_moves(state)
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
                    past = list(history)
                    pad = HISTORY_LEN - len(past)
                    v15_x = np.zeros((TOTAL_FRAMES, 15, 15, 3), dtype=np.float32)
                    real_frames = [None] * pad + past + [state]
                    for t_idx, st in enumerate(real_frames):
                        if st is None:
                            continue
                        v15_x[t_idx] = encode_frame(st, pov_player=cp)
                    v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
                    legal_cells = []
                    for t in legal:
                        pos = int(state.player_positions[cp][t])
                        cell = position_to_cell_in_pov(
                            _BASE_POS if pos == _BASE_POS else pos, cp, cp)
                        v15_legal[cell_to_index(*cell)] = 1.0
                        legal_cells.append((t, cell))
                    with torch.no_grad():
                        xt = torch.from_numpy(v15_x).unsqueeze(0).to(device)
                        mt = torch.from_numpy(v15_legal).unsqueeze(0).to(device)
                        policy, _ = student(xt, mt)
                        chosen_idx = int(policy.argmax(dim=-1).item())
                    chosen_cell = divmod(chosen_idx, 15)
                    action = None
                    for t, c in legal_cells:
                        if c == chosen_cell:
                            action = t
                            break
                    if action is None:
                        action = legal[0]
            else:
                action = bot.select_move(state, list(legal))
            history.append(state)
            state = _legacy_cpp.apply_move(state, int(action))
            mc += 1
        if state.is_terminal and _legacy_cpp.get_winner(state) == model_player:
            wins += 1
    student.train()
    return 100.0 * wins / n_games


# ─── PPO training step ─────────────────────────────────────────────────────
def train_on_chunk(student, optimizer, chunk, device, args, kl_anchor=None):
    """chunk: list of trajectory-step dicts with `G` filled in."""
    if not chunk:
        return None
    v15_xs = np.stack([c["v15_x"] for c in chunk], axis=0)         # (N,8,15,15,3)
    v15_masks = np.stack([c["v15_mask"] for c in chunk], axis=0)   # (N,225)
    chosen_cells = np.array([c["chosen_cell"] for c in chunk], dtype=np.int64)
    log_probs_old = np.array([c["log_prob_old"] for c in chunk], dtype=np.float32)
    v_preds_old = np.array([c["v_pred_old"] for c in chunk], dtype=np.float32)
    Gs = np.array([c["G"] for c in chunk], dtype=np.float32)

    N = v15_xs.shape[0]
    metrics = {"loss": 0.0, "loss_pol": 0.0, "loss_val": 0.0,
               "entropy": 0.0, "loss_kl": 0.0, "n_steps": 0,
               "approx_kl": 0.0, "clip_frac": 0.0}

    for epoch in range(args.train_epochs):
        order = np.random.permutation(N)
        for s in range(0, N, args.minibatch_size):
            idx = order[s:s + args.minibatch_size]
            x = torch.from_numpy(v15_xs[idx]).to(device, dtype=torch.float32)
            m = torch.from_numpy(v15_masks[idx]).to(device, dtype=torch.float32)
            cc = torch.from_numpy(chosen_cells[idx]).to(device)
            lp_old = torch.from_numpy(log_probs_old[idx]).to(device)
            vb_old = torch.from_numpy(v_preds_old[idx]).to(device)
            G = torch.from_numpy(Gs[idx]).to(device)

            policy, win_prob = student(x, m)
            v = 2.0 * win_prob - 1.0  # [0,1] sigmoid → [-1,+1]

            log_p_all = torch.log(policy + 1e-9)
            log_p_chosen = log_p_all.gather(1, cc.unsqueeze(1)).squeeze(1)

            # PPO clipped surrogate
            advantage = (G - vb_old).detach()
            ratio = torch.exp(log_p_chosen - lp_old)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - args.ppo_clip,
                                  1.0 + args.ppo_clip) * advantage
            loss_pol = -torch.min(unclipped, clipped).mean()

            # Value loss
            loss_val = F.mse_loss(v, G)

            # Entropy bonus (over the masked distribution)
            entropy = -(policy * log_p_all).sum(dim=1).mean()
            loss_ent = -args.entropy_coeff * entropy

            loss = loss_pol + args.value_coeff * loss_val + loss_ent

            # KL anchor to V15_SL
            loss_kl_val = 0.0
            if kl_anchor is not None and args.kl_anchor_coeff > 0:
                with torch.no_grad():
                    t_pol, _ = kl_anchor(x, m)
                # KL(teacher || student) — pull student toward teacher on
                # multi-legal states (matches V13.5 RL convention).
                loss_kl = F.kl_div(log_p_all, t_pol, reduction="batchmean",
                                   log_target=False)
                loss = loss + args.kl_anchor_coeff * loss_kl
                loss_kl_val = float(loss_kl.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            # Diagnostics
            with torch.no_grad():
                approx_kl = (lp_old - log_p_chosen).mean().item()
                clip_frac = ((ratio - 1.0).abs() > args.ppo_clip).float().mean().item()

            n = idx.shape[0]
            metrics["loss"]      += float(loss.item()) * n
            metrics["loss_pol"]  += float(loss_pol.item()) * n
            metrics["loss_val"]  += float(loss_val.item()) * n
            metrics["entropy"]   += float(entropy.item()) * n
            metrics["loss_kl"]   += loss_kl_val * n
            metrics["approx_kl"] += approx_kl * n
            metrics["clip_frac"] += clip_frac * n
            metrics["n_steps"]   += n

    if metrics["n_steps"]:
        for k in ("loss", "loss_pol", "loss_val", "entropy", "loss_kl",
                  "approx_kl", "clip_frac"):
            metrics[k] /= metrics["n_steps"]
    return metrics


# ─── Dashboard ──────────────────────────────────────────────────────────────
class _RLHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, rl_path=None, chain_path=None, landing=None, **kw):
        self._rl_path = rl_path
        self._chain_path = chain_path
        self._landing = landing
        super().__init__(*args, directory=directory, **kw)

    def log_message(self, *a, **kw):
        return

    def do_GET(self):
        if self.path in ("/", ""):
            if self._landing:
                self.path = "/" + self._landing
            return super().do_GET()
        if self.path == "/api/rl_stats":
            return self._serve(self._rl_path)
        if self.path == "/api/sl_stats":
            return self._serve(self._rl_path)
        if self.path == "/api/chain":
            return self._serve(self._chain_path)
        super().do_GET()

    def _serve(self, p):
        try:
            with open(p) as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode())
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()


def start_dashboard(port, rl_path, chain_path, dashboard_dir):
    landing = None
    for cand in ("rl_dashboard.html", "v13_dashboard.html", "sl_dashboard.html", "index.html"):
        if os.path.exists(os.path.join(dashboard_dir, cand)):
            landing = cand
            break
    handler = functools.partial(_RLHandler, directory=dashboard_dir,
                                rl_path=rl_path, chain_path=chain_path,
                                landing=landing)
    server = HTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[dashboard] http://localhost:{port}/{landing or ''}", flush=True)


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    # Wire history depth BEFORE constructing env/model so both agree on T.
    configure_history(args.history_len)

    if args.run_name:
        os.environ["TD_LUDO_RUN_NAME"] = args.run_name
    run_name = os.environ.get("TD_LUDO_RUN_NAME", "v15_rl")
    CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / run_name
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    rl_stats_path = str(CHECKPOINT_DIR / "rl_stats.json")
    chain_path = str(CHECKPOINT_DIR / "chain_status.json")
    rl_log_path = str(CHECKPOINT_DIR / "rl.log")

    if args.resume:
        args.init = str(CHECKPOINT_DIR / "model_latest.pt")
        if not os.path.exists(args.init):
            print(f"ERROR: --resume but {args.init} not found"); sys.exit(1)
    elif not args.init:
        print("ERROR: either --init or --resume is required"); sys.exit(1)

    if args.kl_anchor is None and args.kl_anchor_coeff > 0:
        args.kl_anchor = args.init

    print("=" * 70)
    print(f"V15 RL — PPO with opponent mix  (run={run_name})")
    print("=" * 70)
    print(f"  device:           {device}")
    print(f"  init:             {args.init}")
    print(f"  KL anchor:        {args.kl_anchor}")
    print(f"  checkpoint dir:   {CHECKPOINT_DIR}")
    print(f"  parallel_games:   {args.parallel_games}")
    print(f"  train_chunk:      {args.train_chunk}")
    print(f"  minibatch:        {args.minibatch_size} × {args.train_epochs} epochs")
    print(f"  target_states:    {args.target_states:,}")
    print(f"  lr:               {args.lr} → {args.lr_end} (cosine)")
    print(f"  ppo_clip:         {args.ppo_clip}")
    print(f"  entropy_coeff:    {args.entropy_coeff}")
    print(f"  kl_anchor_coeff:  {args.kl_anchor_coeff}")
    print(f"  opp weights:      self={args.opp_weight_self} "
          f"v135_rl={args.opp_weight_v135_rl} "
          f"v135_sl={args.opp_weight_v135_sl} "
          f"v132={args.opp_weight_v132}")
    print("=" * 70)

    student, _ck = load_v15_student(args.init, args, device)
    student.to(device).train()
    print(f"[student] V15 params: {sum(p.numel() for p in student.parameters()):,}")

    kl_anchor = None
    if args.kl_anchor and args.kl_anchor_coeff > 0:
        kl_anchor = load_v15_kl_anchor(args.kl_anchor, args, device)
        print(f"[kl-anchor] V15 SL loaded (coeff={args.kl_anchor_coeff})")

    # Build opponent picker dict
    opp_picks = {}
    opp_probs = {}
    if args.opp_weight_self > 0:
        # Self-play: use the student in eval mode (no grad) for the opp side
        student_eval_ref = [student]  # reference indirection so we always pull latest weights
        def self_pick(state, legal):
            if len(legal) == 1:
                return legal[0]
            cp = int(state.current_player)
            v15_x = np.zeros((TOTAL_FRAMES, 15, 15, 3), dtype=np.float32)
            v15_x[-1] = encode_frame(state, pov_player=cp)
            v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
            legal_cells = []
            for t in legal:
                pos = int(state.player_positions[cp][t])
                c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                v15_legal[cell_to_index(*c)] = 1.0
                legal_cells.append((t, c))
            m = student_eval_ref[0]
            m.eval()
            with torch.no_grad():
                xt = torch.from_numpy(v15_x).unsqueeze(0).to(device)
                mt = torch.from_numpy(v15_legal).unsqueeze(0).to(device)
                p, _ = m(xt, mt)
                idx = int(p.argmax(dim=-1).item())
            m.train()
            chosen_cell = divmod(idx, 15)
            for t, c in legal_cells:
                if c == chosen_cell:
                    return t
            return legal[0]
        opp_picks["self"] = self_pick
        opp_probs["self"] = args.opp_weight_self
    if args.opp_v135_rl and args.opp_weight_v135_rl > 0:
        m = load_v135_opponent(args.opp_v135_rl, device)
        opp_picks["v135_rl"] = lambda s, l, _m=m: v135_pick(_m, device, s, l)
        opp_probs["v135_rl"] = args.opp_weight_v135_rl
        print(f"[opp] V13.5 RL loaded: {args.opp_v135_rl}")
    if args.opp_v135_sl and args.opp_weight_v135_sl > 0:
        m = load_v135_opponent(args.opp_v135_sl, device)
        opp_picks["v135_sl"] = lambda s, l, _m=m: v135_pick(_m, device, s, l)
        opp_probs["v135_sl"] = args.opp_weight_v135_sl
        print(f"[opp] V13.5 SL loaded: {args.opp_v135_sl}")
    if args.opp_v132 and args.opp_weight_v132 > 0:
        m = load_v132_opponent(args.opp_v132, device)
        opp_picks["v132"] = lambda s, l, _m=m: v132_pick(_m, device, s, l)
        opp_probs["v132"] = args.opp_weight_v132
        print(f"[opp] V13.2 loaded: {args.opp_v132}")
    if not opp_probs:
        print("ERROR: at least one opponent must be enabled"); sys.exit(1)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    def write_chain(phase):
        with open(chain_path, "w") as f:
            json.dump({"stage": "RL", "phase": phase, "arch": "v15",
                       "run_name": run_name, "ts": int(time.time())}, f)
    write_chain("training")

    if not args.no_dashboard:
        # Serve the legacy rl_dashboard.html from the legacy td_ludo dir
        dash_dir = str(_LEGACY_ROOT)
        start_dashboard(args.port, rl_stats_path, chain_path, dash_dir)

    rl_log = open(rl_log_path, "a")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        rl_log.write(line + "\n")
        rl_log.flush()

    env = V15RLEnv(args.parallel_games, opp_picks, opp_probs, max_game_len=args.max_game_len)
    log(f"starting RL: target {args.target_states:,} states, init={args.init}")

    pool = []
    total = 0
    step = 0
    last_save = 0
    last_eval = 0
    t_start = time.time()
    eval_history = []
    last_metrics = None

    while total < args.target_states:
        spin = env.spin_to_decision()
        # finished trajectories appended to the pool
        pool.extend(spin["finished"])

        if len(spin["decision_idxs"]) > 0:
            # Student forward (no grad) to sample + record log_prob/v_pred
            with torch.no_grad():
                x = torch.from_numpy(spin["v15_x"]).to(device, dtype=torch.float32)
                m = torch.from_numpy(spin["v15_mask"]).to(device, dtype=torch.float32)
                policy, win_prob = student(x, m)
                sampled = torch.multinomial(policy + 1e-9, num_samples=1).squeeze(1)
                log_p_all = torch.log(policy + 1e-9)
                lp_chosen = log_p_all.gather(1, sampled.unsqueeze(1)).squeeze(1)
                v_pred = (2.0 * win_prob - 1.0).cpu().numpy()
                chosen_cells = sampled.cpu().numpy()
                lp_old = lp_chosen.cpu().numpy()
            env.apply_student_actions(
                spin["decision_idxs"], spin["legal_lists"],
                chosen_cells, lp_old, v_pred, spin["v15_x"], spin["v15_mask"],
            )

        if len(pool) >= args.train_chunk:
            progress = total / max(1, args.target_states)
            cur_lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * progress))
            for g in optimizer.param_groups:
                g["lr"] = cur_lr
            metrics = train_on_chunk(student, optimizer, pool, device, args,
                                      kl_anchor=kl_anchor)
            last_metrics = metrics
            total += len(pool)
            step += 1
            pool = []

            if step % max(1, args.log_every) == 0 and metrics is not None:
                elapsed = time.time() - t_start
                fps = total / max(1e-6, elapsed)
                avg_glen = float(np.mean(env.game_lengths)) if env.game_lengths else 0.0
                opp_summary = ", ".join(f"{k}={v}" for k, v in env.opp_game_counts.most_common())
                log(f"step {step:>5} | states {total:>9,}/{args.target_states:,} "
                    f"| fps {fps:>5.0f} | lr {cur_lr:.1e} | games {env.games_played} "
                    f"| avg_glen {avg_glen:.0f} "
                    f"| L {metrics['loss']:.4f} (pol {metrics['loss_pol']:+.4f} "
                    f"val {metrics['loss_val']:.4f} ent {metrics['entropy']:.3f} "
                    f"kl {metrics['loss_kl']:.3f} clip {metrics['clip_frac']:.3f})")
                if step % (max(1, args.log_every) * 10) == 0:
                    log(f"  [opp-mix games so far] {opp_summary}")
                try:
                    with open(rl_stats_path, "w") as f:
                        json.dump({
                            "stage": "RL", "arch": "v15", "run_name": run_name,
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
                            "approx_kl": metrics["approx_kl"],
                            "clip_frac": metrics["clip_frac"],
                            "eval_history": eval_history,
                            "opp_mix": dict(env.opp_game_counts),
                            "ts": int(time.time()),
                        }, f)
                except Exception as e:
                    print(f"[stats] write failed: {e}")

            if total - last_save >= args.save_every:
                ck_path = CHECKPOINT_DIR / f"rl_{total // 1000}K.pt"
                save_dict = {
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step, "total": total,
                    "games_played": env.games_played,
                    "eval_history": eval_history,
                }
                torch.save(save_dict, str(ck_path))
                torch.save(save_dict, str(CHECKPOINT_DIR / "model_latest.pt"))
                log(f"[checkpoint] {ck_path}")
                last_save = total

            if total - last_eval >= args.eval_every and total > 0:
                log(f"[eval] starting ({args.eval_games} games)...")
                wr = quick_eval(student, device, n_games=args.eval_games)
                eval_history.append([total, wr])
                log(f"[eval] WR = {wr:.1f}% at {total:,} states ({env.games_played} games)")
                last_eval = total

    # Final save
    final_path = CHECKPOINT_DIR / "model_rl.pt"
    save_dict = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "total": total,
        "games_played": env.games_played,
        "eval_history": eval_history,
    }
    torch.save(save_dict, str(final_path))
    torch.save(save_dict, str(CHECKPOINT_DIR / "model_latest.pt"))
    log(f"[done] saved {final_path}")
    write_chain("completed")
    rl_log.close()


if __name__ == "__main__":
    main()
