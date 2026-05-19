"""V15-aware bot-grid eval.

`evaluate_v11.evaluate_model` is V13-shaped (single-frame state + 4-way greedy
argmax + rank-to-token gather). V15 has:
  - 8-frame chronological history per game
  - 225-cell source-cell policy (we map cell → legal token-id at apply time)

This module mirrors the legacy eval's contract — returns the same dict shape
that the trainer's dashboard / training journal consumes — but plays V15
correctly. Per-bot win rates are tracked.

Return dict (matches `evaluate_v11.evaluate_model`):
    {
        "win_rate":         float (0..1),
        "wins":             int,
        "total":            int,
        "win_rate_percent": float (1 decimal),
        "elapsed_seconds":  float,
        "games_per_minute": float,
        "avg_game_length":  float,
        "per_bot": {
            bot_type: {
                "win_rate":   float (percent, 1 decimal),
                "wins":       int,
                "games":      int,
                "avg_length": float,
            },
            ...
        },
    }
"""
from __future__ import annotations

import collections
import random
import time
from typing import Optional, List

import numpy as np
import torch

import td_ludo_cpp as _legacy_cpp
from src.heuristic_bot import get_bot as _get_scripted_bot, BOT_REGISTRY  # legacy
from src.config import MAX_MOVES_PER_GAME  # legacy

# Strong non-neural bots (Expectimax, MCTSPure) — qualitatively different
# from the scripted family. Optional import: only fails if the engine
# layout shifts. See td_ludo/game/strong_bots.py.
try:
    from td_ludo.game.strong_bots import STRONG_BOT_REGISTRY
except ImportError:
    STRONG_BOT_REGISTRY = {}


# Unified bot registry. Eval treats scripted + strong bots uniformly —
# the eval loop just calls `bot.select_move(state, legal)`. The strong
# bots conform to the same interface.
ALL_BOT_NAMES = list(BOT_REGISTRY.keys()) + list(STRONG_BOT_REGISTRY.keys())


def get_bot(bot_type, player_id=None, **kwargs):
    """Factory across both legacy + strong registries.

    Strong bots (Expectimax, MCTSPure) accept extra kwargs (e.g.
    `n_sims=50` for MCTSPure). Legacy scripted bots ignore them.
    """
    if bot_type in STRONG_BOT_REGISTRY:
        return STRONG_BOT_REGISTRY[bot_type](player_id=player_id, **kwargs)
    return _get_scripted_bot(bot_type, player_id=player_id)

from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS,
    cell_to_index,
    position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder import encode_frame

import td_ludo_v15_cpp as _v15_cpp


_BASE_POS = _v15_cpp.BASE_POS
# History/stack depth — V15=8, V15.1=2. Call configure_history(T) before
# evaluate_v15_against_bots to switch.
HISTORY_LEN = 7
TOTAL_FRAMES = 8

def configure_history(total_frames: int):
    """Switch the module-level stack depth at runtime (V15=8, V15.1=2)."""
    global HISTORY_LEN, TOTAL_FRAMES
    if total_frames < 1:
        raise ValueError(f"history-len must be >= 1, got {total_frames}")
    TOTAL_FRAMES = total_frames
    HISTORY_LEN = total_frames - 1


def _v15_select(model, device, state, legal, history):
    """Greedy V15 action selection. Returns a legal token-id."""
    if len(legal) == 1:
        return legal[0]
    cp = int(state.current_player)
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
        c = position_to_cell_in_pov(
            _BASE_POS if pos == _BASE_POS else pos, cp, cp)
        v15_legal[cell_to_index(*c)] = 1.0
        legal_cells.append((t, c))
    with torch.no_grad():
        xt = torch.from_numpy(v15_x).unsqueeze(0).to(device)
        mt = torch.from_numpy(v15_legal).unsqueeze(0).to(device)
        policy, _ = model(xt, mt)
        chosen_idx = int(policy.argmax(dim=-1).item())
    chosen_cell = divmod(chosen_idx, 15)
    for t, c in legal_cells:
        if c == chosen_cell:
            return t
    return legal[0]


def evaluate_v15_against_bots(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int = 200,
    bot_types: Optional[List[str]] = None,
    seed_base: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Play V15 vs random heuristic-bot mix. Returns evaluation summary."""
    model.eval()
    # Default opponent pool now includes the strong non-neural bots
    # (Expectimax, MCTSPure) — they're qualitatively different from the
    # scripted family and our most informative eval signal. Callers can
    # still pass explicit `bot_types` to override.
    available_types = list(bot_types or ALL_BOT_NAMES)
    per_bot_wins = collections.Counter()
    per_bot_games = collections.Counter()
    per_bot_total_len = collections.Counter()
    wins = 0
    total_len = 0
    t_start = time.time()

    for g in range(num_games):
        if seed_base is not None:
            random.seed(seed_base + g)
        bot_type = random.choice(available_types)
        per_bot_games[bot_type] += 1

        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        bot = get_bot(bot_type, player_id=opp_player)

        state = _legacy_cpp.create_initial_state_2p()
        history: collections.deque = collections.deque(maxlen=HISTORY_LEN)
        csix = [0, 0, 0, 0]
        mc = 0
        won = False
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
                action = _v15_select(model, device, state, list(legal), history)
            else:
                action = bot.select_move(state, list(legal))
            history.append(state)
            state = _legacy_cpp.apply_move(state, int(action))
            mc += 1
        if state.is_terminal and _legacy_cpp.get_winner(state) == model_player:
            wins += 1
            per_bot_wins[bot_type] += 1
            won = True
        per_bot_total_len[bot_type] += mc
        total_len += mc

    model.train()
    elapsed = time.time() - t_start
    gpm = (num_games / elapsed * 60.0) if elapsed > 0 else 0.0

    per_bot = {}
    for bt, g in per_bot_games.items():
        if g == 0:
            continue
        per_bot[bt] = {
            "win_rate": 100.0 * per_bot_wins[bt] / g,
            "wins": int(per_bot_wins[bt]),
            "games": int(g),
            "avg_length": per_bot_total_len[bt] / g,
        }

    return {
        "win_rate": wins / max(1, num_games),
        "wins": int(wins),
        "total": int(num_games),
        "win_rate_percent": round(100.0 * wins / max(1, num_games), 1),
        "elapsed_seconds": elapsed,
        "games_per_minute": gpm,
        "avg_game_length": total_len / max(1, num_games),
        "per_bot": per_bot,
    }
