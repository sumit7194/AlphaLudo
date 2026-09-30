"""Local side-by-side eval: V13.5 / V15 / V15.1 vs the same bot roster
that the VM training uses.

Matches the VM eval recipe:
  - EVAL_BOT_NAMES_FAST (13 bots: Heuristic, Aggressive, Defensive, Racing,
    Random, Expert, Expectimax, MCTSPure + 5 personality-Expectimax variants)
  - 2000 games per model, randomly sampled from the roster (~150g/bot)
  - Model plays player-0 or player-2 (random per game)
  - Same dice/3-six-rule loop as evaluate_v15_against_bots

Per-bot WR is what we want to compare. The roster is identical across
all three models so any WR delta is the model, not the eval.

Models:
  - V13.5  → CNN, V18 21ch encoder, 4-token policy
            (td_ludo/play/model_weights/v13_5/model_latest.pt)
  - V15    → GraphTransformer d=256 layers=8 heads=8 ffn=512 H=8
            (td_ludo_v15/checkpoints/v15_rich_phase_l/model_latest.pt)
  - V15.1  → GraphTransformer d=128 layers=4 heads=4 ffn=256 H=2
            (td_ludo_v15/checkpoints/v151_rl/model_latest.pt — pre-strong)

Usage:
    cd /Users/sumit/Github/AlphaLudo/td_ludo
    PYTHONPATH=.:../td_ludo_v15 \
      ./td_env/bin/python -u eval_v135_v15_v151_local.py
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
from td_ludo_v15.rich.v15_bot_eval import (
    EVAL_BOT_NAMES_FAST, get_bot, configure_history,
)
from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.models.v15 import V15GraphTransformer
import td_ludo_v15_cpp as v15_cpp

_BASE_POS = v15_cpp.BASE_POS
MAX_MOVES = 400


# ─── Model loaders ─────────────────────────────────────────────────────

def _strip(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


def load_v135(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip(sd)
    m = V135ProductionAdapter(num_res_blocks=10, num_channels=128)
    m.load_state_dict(sd, strict=False)
    m.eval().to(device)
    return m


def load_v15_any(path, device, *, d_model, n_heads, n_layers, ffn_dim, history_len):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = _strip(sd)
    m = V15GraphTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        ffn_dim=ffn_dim, history_len=history_len,
    )
    m.load_state_dict(sd, strict=False)
    m.eval().to(device)
    return m


# ─── Pickers ───────────────────────────────────────────────────────────

def make_picker_v135(model, device):
    def pick(state, legal, _history):
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
    return pick


def make_picker_v15(model, device, *, history_len):
    """history_len = number of total frames in input (V15=8, V15.1=2)."""
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
        v15_x = np.zeros((total_frames, 15, 15, 3), dtype=np.float32)
        real_frames = real_past + [state]
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
    return pick


# ─── Eval loop (mirrors evaluate_v15_against_bots exactly) ─────────────

def run_eval(model_picker, *, num_games, bot_types, seed_base, label,
             history_maxlen=8):
    per_bot_wins = collections.Counter()
    per_bot_games = collections.Counter()
    per_bot_total_len = collections.Counter()
    wins = 0
    total_len = 0
    t_start = time.time()

    for g in range(num_games):
        if seed_base is not None:
            random.seed(seed_base + g)
        bot_type = random.choice(bot_types)
        per_bot_games[bot_type] += 1

        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        bot = get_bot(bot_type, player_id=opp_player)

        state = cpp.create_initial_state_2p()
        history = collections.deque(maxlen=history_maxlen)
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
            if cp == model_player:
                action = model_picker(state, list(legal), history)
            else:
                action = bot.select_move(state, list(legal))
            history.append(state)
            state = cpp.apply_move(state, int(action))
            mc += 1
        if state.is_terminal and cpp.get_winner(state) == model_player:
            wins += 1
            per_bot_wins[bot_type] += 1
        per_bot_total_len[bot_type] += mc
        total_len += mc

        if (g + 1) % 200 == 0:
            elapsed = time.time() - t_start
            print(f"  [{label}] {g+1}/{num_games}  |  {(g+1)/elapsed:.1f} g/s  |  "
                  f"running WR {100*wins/(g+1):.1f}%", flush=True)

    elapsed = time.time() - t_start
    per_bot = {}
    for bt, gn in per_bot_games.items():
        per_bot[bt] = {
            "win_rate": 100.0 * per_bot_wins[bt] / gn,
            "wins": int(per_bot_wins[bt]),
            "games": int(gn),
            "avg_length": per_bot_total_len[bt] / gn,
        }
    return {
        "win_rate_percent": 100.0 * wins / max(1, num_games),
        "wins": int(wins),
        "total": int(num_games),
        "elapsed_seconds": elapsed,
        "games_per_sec": num_games / elapsed,
        "avg_game_length": total_len / max(1, num_games),
        "per_bot": per_bot,
    }


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    device = torch.device("cpu")
    n_games = 2000
    seed_base = 42

    print(f"Device: {device}")
    print(f"Roster ({len(EVAL_BOT_NAMES_FAST)} bots): {EVAL_BOT_NAMES_FAST}")
    print(f"Games per model: {n_games}, seed_base={seed_base}\n")

    print("Loading models...")
    v135 = load_v135(HERE / "play" / "model_weights" / "v13_5" / "model_latest.pt", device)
    v15 = load_v15_any(
        V15_ROOT / "checkpoints" / "v15_rich_phase_l" / "model_latest.pt",
        device, d_model=256, n_heads=8, n_layers=8, ffn_dim=512, history_len=8)
    v151 = load_v15_any(
        V15_ROOT / "checkpoints" / "v151_rl" / "model_latest.pt",
        device, d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=2)
    print(f"  V13.5:  {sum(p.numel() for p in v135.parameters()):,} params")
    print(f"  V15:    {sum(p.numel() for p in v15.parameters()):,} params")
    print(f"  V15.1:  {sum(p.numel() for p in v151.parameters()):,} params\n")

    pickers = [
        ("V13.5",  make_picker_v135(v135, device), 8),
        ("V15",    make_picker_v15(v15, device, history_len=8), 8),
        ("V15.1",  make_picker_v15(v151, device, history_len=2), 1),
    ]

    results = {}
    t_all = time.time()
    for name, picker, hmax in pickers:
        # V15 module uses module-level HISTORY_LEN — configure_history(T)
        # sets it. V13.5 doesn't use the history but we still set hmax=8
        # so the maxlen deque doesn't blow up. (history just buffered, not read.)
        if name == "V15":
            configure_history(8)
        elif name == "V15.1":
            configure_history(2)
        # V13.5: leave history config at whatever the previous model used.
        print(f"\n━━━━━━ {name} eval ({n_games} games) ━━━━━━")
        r = run_eval(picker, num_games=n_games,
                     bot_types=list(EVAL_BOT_NAMES_FAST),
                     seed_base=seed_base, label=name, history_maxlen=hmax)
        results[name] = r
        print(f"   {name} overall WR: {r['win_rate_percent']:.1f}%  "
              f"({r['games_per_sec']:.1f} g/s, avg_len {r['avg_game_length']:.0f})")

    total_min = (time.time() - t_all) / 60.0
    print(f"\n{'='*72}")
    print(f"Total wall: {total_min:.1f} min\n")

    # Side-by-side table
    print(f"{'Bot':<26} {'V13.5':>10} {'V15':>10} {'V15.1':>10}  Δ(V15.1-V13.5)")
    print("-" * 72)
    overall_row = ("OVERALL",
                   results["V13.5"]["win_rate_percent"],
                   results["V15"]["win_rate_percent"],
                   results["V15.1"]["win_rate_percent"])
    rows = []
    for bot in EVAL_BOT_NAMES_FAST:
        v135_wr = results["V13.5"]["per_bot"].get(bot, {}).get("win_rate", float("nan"))
        v15_wr = results["V15"]["per_bot"].get(bot, {}).get("win_rate", float("nan"))
        v151_wr = results["V15.1"]["per_bot"].get(bot, {}).get("win_rate", float("nan"))
        rows.append((bot, v135_wr, v15_wr, v151_wr))
    rows.sort(key=lambda r: r[3])  # hardest-for-V15.1 first
    print(f"{overall_row[0]:<26} {overall_row[1]:>9.1f}% {overall_row[2]:>9.1f}% "
          f"{overall_row[3]:>9.1f}%  {overall_row[3]-overall_row[1]:>+5.1f}pp")
    print("-" * 72)
    for name, v135_wr, v15_wr, v151_wr in rows:
        print(f"{name:<26} {v135_wr:>9.1f}% {v15_wr:>9.1f}% {v151_wr:>9.1f}%  "
              f"{v151_wr - v135_wr:>+5.1f}pp")

    out_path = HERE / "eval_v135_v15_v151_local_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "wall_min": total_min,
                   "n_games": n_games, "seed_base": seed_base}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
