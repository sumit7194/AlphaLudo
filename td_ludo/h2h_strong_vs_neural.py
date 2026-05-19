"""H2H: Expectimax + MCTSPure vs the trained neural models.

The single most useful number we don't have yet: how does our new strong
bot stack up against V13.5 and V15.1? If Expectimax beats them
substantially, the "trained-bot ceiling" hypothesis is confirmed.

Light run: 200 games per matchup. ~10 min total on CPU.

Usage:
    python h2h_strong_vs_neural.py
"""
from __future__ import annotations

import collections
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
from td_ludo.game.strong_bots import ExpectimaxBot, MCTSPureBot

from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.models.v15 import V15GraphTransformer
import td_ludo_v15_cpp as v15_cpp

_BASE_POS = v15_cpp.BASE_POS
MAX_MOVES = 400
GLOBAL_HISTORY_MAX = 8


# ─── Model loaders (mirror h2h_v151_tournament.py) ──────────────────────


def _strip_prefixes(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


def load_v135_prod(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip_prefixes(sd)
    m = V135ProductionAdapter(num_res_blocks=10, num_channels=128)
    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_v15_any(path, device, *, d_model, n_heads, n_layers, ffn_dim, history_len):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip_prefixes(sd)
    m = V15GraphTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        ffn_dim=ffn_dim, history_len=history_len,
    )
    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


# ─── Pickers ────────────────────────────────────────────────────────────


def pick_v135(model, device, state, legal, history=None):
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


def make_pick_v15(model, device, *, history_len):
    total_frames = history_len
    past_needed = history_len - 1

    def pick(state, legal, history):
        if len(legal) == 1:
            return legal[0]
        cp = int(state.current_player)
        past = list(history) if history else []
        if past_needed == 0:
            real_past = []
        else:
            past = past[-past_needed:]
            real_past = [None] * (past_needed - len(past)) + past
        real_frames = real_past + [state]
        v15_x = np.zeros((total_frames, 15, 15, 3), dtype=np.float32)
        for t_idx, st in enumerate(real_frames):
            if st is None:
                continue
            v15_x[t_idx] = encode_frame(st, pov_player=cp)
        v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
        legal_cells = []
        for t in legal:
            pos = int(state.player_positions[cp][t])
            c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
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
    """Wrap a strong-bot instance into the (state, legal, history) signature."""
    def pick(state, legal, history):
        # Set the bot's player_id dynamically each call — supports playing
        # either side of the table without rebuilding the bot.
        bot.player_id = int(state.current_player)
        return bot.select_move(state, list(legal))
    return pick


# ─── Game loop ──────────────────────────────────────────────────────────


def play_one(picks, seed):
    random.seed(seed)
    np.random.seed(seed)
    state = cpp.create_initial_state_2p()
    csix = [0, 0, 0, 0]
    mc = 0
    history = {p: collections.deque(maxlen=GLOBAL_HISTORY_MAX) for p in range(4)}
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
        pick_fn = picks.get(cp)
        action = pick_fn(state, list(legal), history.get(cp)) if pick_fn else legal[0]
        for p in range(4):
            history[p].append(state)
        state = cpp.apply_move(state, int(action))
        mc += 1
    return int(cpp.get_winner(state)) if state.is_terminal else -1


def run_pair(name_a, picker_a, name_b, picker_b, games_per_orientation=100,
             log_every=50):
    a_wins = b_wins = draws = 0
    total = games_per_orientation * 2
    t0 = time.time()
    for g in range(total):
        if g % 2 == 0:
            picks = {0: picker_a, 2: picker_b}
            a_pid, b_pid = 0, 2
        else:
            picks = {0: picker_b, 2: picker_a}
            a_pid, b_pid = 2, 0
        winner = play_one(picks, seed=42 + g // 2)
        if winner == a_pid: a_wins += 1
        elif winner == b_pid: b_wins += 1
        else: draws += 1
        if (g + 1) % log_every == 0:
            t = a_wins + b_wins + draws
            print(f"    [{g+1:>3}/{total}] {name_a} {100*a_wins/t:5.1f}% "
                  f"({a_wins}-{b_wins}, draws {draws})", flush=True)
    elapsed = time.time() - t0
    print(f"  FINAL: {name_a} {100*a_wins/total:.1f}%  vs  {name_b} {100*b_wins/total:.1f}%  "
          f"(draws {draws})  [{elapsed:.0f}s]")
    return a_wins, b_wins, draws


# ─── Main ───────────────────────────────────────────────────────────────


def main():
    device = torch.device("cpu")  # MPS adds tiny overhead for batch=1; CPU is fine
    print(f"Device: {device}")
    repo_root = HERE
    v15_root = HERE.parent / "td_ludo_v15"

    print("\nLoading models...")
    # V13.5 (pre-experiment)
    v135 = load_v135_prod(
        repo_root / "checkpoints" / "v135_prod_rl_local" / "model_latest.pt", device)
    # V15.1 RL (latest)
    v151_rl = load_v15_any(
        v15_root / "checkpoints" / "v151_rl" / "model_latest.pt",
        device, d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=2)
    # V15.1 SL (parent)
    v151_sl = load_v15_any(
        v15_root / "checkpoints" / "v151_sl" / "model_sl.pt",
        device, d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=2)

    # Bot pickers
    expectimax = ExpectimaxBot()
    pick_expectimax = make_pick_bot(expectimax)

    # Model pickers
    pick_v135_bound = lambda s, l, h: pick_v135(v135, device, s, l, h)
    pick_v151_rl_bound = make_pick_v15(v151_rl, device, history_len=2)
    pick_v151_sl_bound = make_pick_v15(v151_sl, device, history_len=2)

    matchups = [
        ("Expectimax", pick_expectimax, "V13.5",    pick_v135_bound),
        ("Expectimax", pick_expectimax, "V15.1_RL", pick_v151_rl_bound),
        ("Expectimax", pick_expectimax, "V15.1_SL", pick_v151_sl_bound),
    ]

    results = {}
    for a_name, a_pick, b_name, b_pick in matchups:
        print(f"\n━━ {a_name} vs {b_name} (200 games) ━━")
        aw, bw, dr = run_pair(a_name, a_pick, b_name, b_pick,
                              games_per_orientation=100)
        results[f"{a_name}_vs_{b_name}"] = {
            "a_wins": aw, "b_wins": bw, "draws": dr,
            "a_wr": 100 * aw / (aw + bw + dr),
            "b_wr": 100 * bw / (aw + bw + dr),
        }

    print("\n" + "=" * 60)
    print("Summary:")
    for k, v in results.items():
        print(f"  {k}: {v['a_wr']:.1f}% / {v['b_wr']:.1f}%")
    with open("h2h_strong_vs_neural_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → h2h_strong_vs_neural_results.json")


if __name__ == "__main__":
    main()
