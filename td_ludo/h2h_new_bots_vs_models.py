"""How do the new strong bots stack up against our trained models?

The other Claude session ran extensive bot-vs-bot arenas
(STRONG_BOTS_RESULTS.md). What we don't have: bot-vs-model numbers.
This is the more decisive question — if a new bot beats V15.1 RL at
60%+, that bot is a training-worthy opponent. If it loses 30-40%,
it's just a sparring partner.

Tests each new bot vs V15.1 RL latest + V13.5. 100 games per pair
(alternating who plays player-0 vs player-2), CPU-only, ~30 min total.

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 python3 h2h_new_bots_vs_models.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V15_ROOT = HERE.parent / "td_ludo_v15"
sys.path.insert(0, str(V15_ROOT))

import td_ludo_cpp as cpp
from td_ludo.game.encoder_v18_production import encode_state_v18_production  # noqa
from td_ludo.models.v13_5_production import V135ProductionAdapter
from td_ludo.game.strong_bots import STRONG_BOT_REGISTRY

from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.models.v15 import V15GraphTransformer
import td_ludo_v15_cpp as v15_cpp

_BASE_POS = v15_cpp.BASE_POS
MAX_MOVES = 400


def _strip_prefixes(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


def load_v135(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip_prefixes(sd)
    m = V135ProductionAdapter(num_res_blocks=10, num_channels=128)
    m.load_state_dict(sd, strict=False)
    m.eval().to(device)
    return m


def load_v151(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip_prefixes(sd)
    m = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4,
                             ffn_dim=256, history_len=2)
    m.load_state_dict(sd, strict=False)
    m.eval().to(device)
    return m


def pick_v135(model, device, state, legal, _hist):
    """Greedy V13.5 action selection (V18 21-channel encoder, 4-token policy)."""
    if len(legal) == 1:
        return legal[0]
    enc = encode_state_v18_production(state).astype(np.float32)
    token_legal = np.zeros(4, dtype=np.float32)
    for a in legal:
        token_legal[a] = 1.0
    with torch.no_grad():
        x = torch.from_numpy(enc).unsqueeze(0).to(device)
        lmt = torch.from_numpy(token_legal).unsqueeze(0).to(device)
        out = model(x, lmt)
        policy = out[0] if isinstance(out, tuple) else out
        action = int(policy.argmax(dim=1).item())
    return action if action in legal else legal[0]


def make_pick_v151(model, device):
    HISTORY_LEN = 1

    def pick(state, legal, history):
        if len(legal) == 1:
            return legal[0]
        cp = int(state.current_player)
        past = list(history)
        v15_x = np.zeros((2, 15, 15, 3), dtype=np.float32)
        if past:
            v15_x[0] = encode_frame(past[-1], pov_player=cp)
        v15_x[1] = encode_frame(state, pov_player=cp)
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
    return pick


def make_pick_bot(bot):
    def pick(state, legal, _history):
        return int(bot.select_move(state, list(legal)))
    return pick


def run_game(picker_a, picker_b, seed):
    """One game, picker_a plays player 0, picker_b plays player 2.
    Returns 'a', 'b', or 'draw'."""
    random.seed(seed)
    state = cpp.create_initial_state_2p()
    history = []
    csix = [0, 0, 0, 0]
    mc = 0
    while not state.is_terminal and mc < MAX_MOVES:
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
        legal = cpp.get_legal_moves(state)
        if not legal:
            n = (cp + 1) % 4
            while not state.active_players[n]:
                n = (n + 1) % 4
            state.current_player = n
            state.current_dice_roll = 0
            continue
        if cp == 0:
            action = picker_a(state, list(legal), history)
        else:
            action = picker_b(state, list(legal), history)
        history.append(state)
        if len(history) > 8:
            history.pop(0)
        state = cpp.apply_move(state, int(action))
        mc += 1
    if state.is_terminal:
        w = cpp.get_winner(state)
        if w == 0:
            return "a", mc
        elif w == 2:
            return "b", mc
    return "draw", mc


def run_pair(name_a, pick_a, name_b, pick_b, games_per_orientation=50):
    """Plays both orientations to neutralize first-player advantage."""
    aw = bw = dr = 0
    total_len = 0
    t0 = time.time()
    for i in range(games_per_orientation):
        r, mc = run_game(pick_a, pick_b, seed=i)
        total_len += mc
        if r == "a":
            aw += 1
        elif r == "b":
            bw += 1
        else:
            dr += 1
    for i in range(games_per_orientation):
        r, mc = run_game(pick_b, pick_a, seed=10_000 + i)
        total_len += mc
        if r == "a":
            bw += 1
        elif r == "b":
            aw += 1
        else:
            dr += 1
    total = aw + bw + dr
    elapsed = time.time() - t0
    return {
        "a_wins": aw, "b_wins": bw, "draws": dr,
        "a_wr": 100.0 * aw / total if total else 0.0,
        "b_wr": 100.0 * bw / total if total else 0.0,
        "elapsed_sec": elapsed,
        "avg_len": total_len / total if total else 0.0,
    }


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Load both models
    v135 = load_v135(HERE / "play" / "model_weights" / "v13_5" / "model_latest.pt", device)
    v151 = load_v151(V15_ROOT / "checkpoints" / "v151_rl_strong" / "model_latest.pt", device)
    pick_v135_b = lambda s, l, h: pick_v135(v135, device, s, l, h)
    pick_v151_b = make_pick_v151(v151, device)

    # New strong bots to test. Order: fast→slow. Each gets matched
    # against both models.
    bots_to_test = [
        # Sanity-check baselines (already known)
        "Expectimax",
        "MCTSPure",
        # Personality variants (Phase 1 — proven beats base Expectimax)
        "AggressiveExpectimax",
        "DefensiveExpectimax",
        "MinimaxExpectimax",
        "RacingExpectimax",
        "BlockadeExpectimax",
        "VoteExpectimax",
        # Adaptive (proven worse than base — sanity test)
        "AdaptiveExpectimax",
        # Rule-based (low-difficulty diverse opponents)
        "MaxCapture",
        "TwoStack",
        "HomeRush",
        "StackHomeRush",
        # Depth-2 — the strongest practical drop-in. SLOW (~1s/move).
        "Depth2Expectimax",
        # MCTS+ExpectimaxPrior — strongest measured, but SLOWEST. Include
        # at smaller games-per-pair below.
        "MCTSExpectimaxPrior",
    ]

    results = {}
    fast_games = 50  # = 100 total (both orientations)
    slow_games = 15  # = 30 total — MCTSExpectimaxPrior is ~1s/move

    t_start = time.time()
    for bot_name in bots_to_test:
        bot_cls = STRONG_BOT_REGISTRY[bot_name]
        n_games = slow_games if bot_name == "MCTSExpectimaxPrior" else fast_games

        for model_name, model_pick in [("V15.1_RL_strong", pick_v151_b),
                                        ("V13.5", pick_v135_b)]:
            bot = bot_cls()
            pick_bot = make_pick_bot(bot)
            label = f"{model_name}_vs_{bot_name}"
            print(f"\n━━ {label} ({2*n_games} games) ━━", flush=True)
            r = run_pair(model_name, model_pick, bot_name, pick_bot,
                         games_per_orientation=n_games)
            results[label] = r
            print(f"   {model_name}: {r['a_wr']:.1f}%  | "
                  f"{bot_name}: {r['b_wr']:.1f}%  | "
                  f"draws: {r['draws']}  | avg_len: {r['avg_len']:.0f}  | "
                  f"{r['elapsed_sec']:.1f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}\nTotal: {elapsed/60:.1f} min")

    # Summary table sorted by V15.1_RL model WR ascending (hardest bots first)
    print("\n--- summary (sorted by V15.1_RL_strong WR ASC = hardest bots first) ---")
    print(f"{'Bot':<28} {'V15.1_RL WR':>12} {'V13.5 WR':>12}")
    rows = []
    for bot_name in bots_to_test:
        v15_r = results.get(f"V15.1_RL_strong_vs_{bot_name}", {})
        v13_r = results.get(f"V13.5_vs_{bot_name}", {})
        rows.append((bot_name, v15_r.get("a_wr", 0.0), v13_r.get("a_wr", 0.0)))
    rows.sort(key=lambda r: r[1])
    for name, v15, v13 in rows:
        print(f"{name:<28} {v15:>11.1f}% {v13:>11.1f}%")

    out_path = HERE / "h2h_new_bots_vs_models_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "elapsed_min": elapsed / 60.0}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
