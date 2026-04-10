#!/usr/bin/env python3
"""
Inference-time MCTS evaluation sweep for V6.1 (Step 1 of POST_V61_EXPERIMENT_PLAN).

Modes:
    --single               Run a single matchup (configurable via flags)
    --sweep                Run the full hard-coded sweep plan (default for the
                           experiment)

Each matchup plays `num_games` 2-player Ludo games, alternating the model's
seat (P0 / P2) to remove first-player bias, and logs progress every 50 games.

Model loading:
    Assumes V6.1 architecture: AlphaLudoV5(num_res_blocks=10, num_channels=128,
    in_channels=24). Both the primary and opponent models (if model vs model)
    must share this architecture.

MCTS specifics:
    - Uses the C++ MCTSEngine from td_ludo_cpp (now fixed to 24ch encoding)
    - Eval mode: dirichlet_eps=0.0 (no exploration noise)
    - Batched sims via parallel_sims inside a single game to keep the GPU fed
    - Values clamped to [-1, 1] before backprop (V5 value head has no tanh;
      returns can be outside that range with return normalization)
    - Temperature = 0 at the root (greedy pick of most-visited child)

Outputs:
    - Log lines printed to stdout and appended to `--log` path
    - Structured results (one dict per matchup) appended to `--out` (JSON list)
"""
import argparse
import functools
import json
import os
import random
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp  # noqa: E402
from src.heuristic_bot import get_bot  # noqa: E402
from src.model import AlphaLudoV5  # noqa: E402

MAX_MOVES_PER_GAME = 2000

# ---------------------------------------------------------------------------
# Graceful shutdown: set by SIGTERM/SIGINT, checked after each game
# ---------------------------------------------------------------------------

STOP_REQUESTED = threading.Event()


def _signal_handler(signum, frame):
    if STOP_REQUESTED.is_set():
        print("\n[Sweep] Force exit.", flush=True)
        sys.exit(1)
    print(f"\n[Sweep] Signal {signum} received — pausing after current game...",
          flush=True)
    STOP_REQUESTED.set()


# ---------------------------------------------------------------------------
# Paths for live state + partial checkpoint (set in main, used by run_matchup)
# ---------------------------------------------------------------------------

STATE_PATHS = {
    "live_stats": None,     # JSON updated every few games + on matchup boundaries
    "partial": None,        # Per-matchup mid-game checkpoint
    "results": None,        # Full results file (completed matchups only)
    "log": None,            # Detailed text log
    "sweep_started_at": None,
}


# ---------------------------------------------------------------------------
# Persistent state helpers: partial-matchup checkpoint + live stats JSON
# ---------------------------------------------------------------------------

def _atomic_write_json(path, data):
    if path is None:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def save_partial(state):
    """Write mid-matchup checkpoint. Called periodically during a matchup."""
    _atomic_write_json(STATE_PATHS["partial"], state)


def clear_partial():
    """Remove partial checkpoint after matchup finishes cleanly."""
    p = STATE_PATHS["partial"]
    if p and os.path.exists(p):
        try:
            os.remove(p)
        except Exception:
            pass


def load_partial(label):
    """If a partial exists for this label, return it; else None."""
    p = STATE_PATHS["partial"]
    if not p or not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        if data.get("label") == label:
            return data
    except Exception:
        return None
    return None


def write_live_stats(current_matchup, plan, all_results, last_line=""):
    """Update live_stats.json with current progress for dashboard."""
    start = STATE_PATHS["sweep_started_at"] or time.time()
    completed_labels = {r.get("label") for r in all_results}
    plan_summary = []
    for m in plan:
        state = "pending"
        if m["label"] in completed_labels:
            state = "done"
        elif current_matchup and current_matchup.get("label") == m["label"]:
            state = "running"
        plan_summary.append({
            "label": m["label"],
            "num_sims": m["num_sims"],
            "num_games": m["num_games"],
            "state": state,
        })
    stats = {
        "timestamp": time.time(),
        "uptime_sec": round(time.time() - start, 1),
        "current_matchup": current_matchup,
        "plan": plan_summary,
        "completed_count": sum(1 for m in plan_summary if m["state"] == "done"),
        "total_matchups": len(plan_summary),
        "recent_results": all_results[-10:],
        "last_log_line": last_line,
        "stop_requested": STOP_REQUESTED.is_set(),
    }
    _atomic_write_json(STATE_PATHS["live_stats"], stats)


# ---------------------------------------------------------------------------
# Dashboard HTTP server (very small, inspired by train_v6_1.py)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!doctype html>
<html><head><title>MCTS Sweep Dashboard</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 20px; background: #0b0f14; color: #d8e0e6; }
h1, h2 { color: #9cdcfe; }
.card { background: #111821; padding: 14px 18px; margin: 10px 0; border-radius: 8px; border: 1px solid #1e2a36; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 5px 10px; text-align: left; border-bottom: 1px solid #1e2a36; }
.done { color: #4ec9b0; }
.running { color: #dcdcaa; font-weight: bold; }
.pending { color: #6a7a8a; }
.big { font-size: 22px; font-weight: bold; color: #dcdcaa; }
pre { white-space: pre-wrap; background: #050810; padding: 8px; border-radius: 4px; font-size: 12px; }
.stop { color: #f48771; }
</style></head>
<body>
<h1>MCTS Sweep — Step 1</h1>
<div class="card" id="current"></div>
<div class="card"><h2>Plan</h2><div id="plan"></div></div>
<div class="card"><h2>Completed Matchups (recent)</h2><div id="results"></div></div>
<div class="card"><h2>Last log line</h2><pre id="last_line"></pre></div>
<script>
async function refresh() {
  try {
    const r = await fetch('/api/stats?t=' + Date.now());
    const s = await r.json();
    const cm = s.current_matchup;
    const pct = cm ? (100*cm.games_played/cm.num_games).toFixed(1) : 0;
    document.getElementById('current').innerHTML = cm
      ? `<h2>Current: ${cm.label}</h2>
         <div class="big">${(cm.win_rate*100).toFixed(1)}% WR — ${cm.games_played}/${cm.num_games} (${pct}%)</div>
         <div>sims=${cm.num_sims} gpm=${cm.gpm.toFixed(1)} W=${cm.wins} L=${cm.losses} D=${cm.draws}</div>
         <div>ETA: ${cm.eta_min.toFixed(1)} min | Uptime: ${(s.uptime_sec/60).toFixed(1)} min</div>
         ${s.stop_requested ? '<div class="stop">STOP requested — finishing current game</div>' : ''}`
      : `<h2>Idle</h2><div>Sweep: ${s.completed_count}/${s.total_matchups} matchups done</div>`;
    document.getElementById('plan').innerHTML = '<table><tr><th>#</th><th>Label</th><th>Sims</th><th>Games</th><th>State</th></tr>' +
      s.plan.map((m,i) => `<tr><td>${i+1}</td><td>${m.label}</td><td>${m.num_sims}</td><td>${m.num_games}</td><td class="${m.state}">${m.state}</td></tr>`).join('') + '</table>';
    document.getElementById('results').innerHTML = s.recent_results.length ? '<table><tr><th>Label</th><th>Sims</th><th>WR</th><th>W/L/D</th><th>GPM</th><th>Elapsed</th></tr>' +
      s.recent_results.map(r => `<tr><td>${r.label}</td><td>${r.num_sims}</td><td>${(r.win_rate*100).toFixed(1)}%</td><td>${r.wins}/${r.losses}/${r.draws}</td><td>${r.gpm.toFixed(1)}</td><td>${(r.elapsed_sec/60).toFixed(1)}m</td></tr>`).join('') + '</table>' : '<i>none</i>';
    document.getElementById('last_line').textContent = s.last_log_line || '';
  } catch (e) {}
}
refresh(); setInterval(refresh, 3000);
</script>
</body></html>
"""


class _DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, "text/html", _DASHBOARD_HTML.encode())
        elif self.path.startswith("/api/stats"):
            self._serve_file(STATE_PATHS["live_stats"], "application/json")
        elif self.path.startswith("/api/results"):
            self._serve_file(STATE_PATHS["results"], "application/json")
        else:
            self._send(404, "text/plain", b"not found")

    def _serve_file(self, path, content_type):
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                self._send(200, content_type, data)
            except Exception:
                self._send(500, "text/plain", b"err")
        else:
            self._send(200, content_type, b"{}")

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


def start_dashboard(port=8788):
    try:
        srv = HTTPServer(("0.0.0.0", port), _DashboardHandler)
    except OSError as e:
        print(f"[Dashboard] Could not bind to port {port}: {e}", flush=True)
        return None
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"[Dashboard] Started at http://localhost:{port}", flush=True)
    return srv


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_v61_model(ckpt_path, device):
    model = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Move selection
# ---------------------------------------------------------------------------

def _pick_legal(legal_moves):
    return legal_moves[0] if legal_moves else -1


def model_move_raw(model, state, legal_moves, device):
    """Single-state raw policy argmax with legal-move masking."""
    if len(legal_moves) == 1:
        return legal_moves[0]
    tensor = ludo_cpp.encode_state_v6(state)
    legal_mask = np.zeros(4, dtype=np.float32)
    for m in legal_moves:
        legal_mask[m] = 1.0
    with torch.no_grad():
        s_t = torch.from_numpy(tensor).unsqueeze(0).to(device, dtype=torch.float32)
        m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(device, dtype=torch.float32)
        logits = model.forward_policy_only(s_t, m_t)
        action = int(logits.argmax(dim=1).item())
    if action not in legal_moves:
        action = _pick_legal(legal_moves)
    return action


def model_move_mcts(model, state, legal_moves, device, num_sims, parallel_sims=16):
    """Run MCTS with the model as policy-prior + leaf value estimator, return greedy action."""
    if len(legal_moves) == 1:
        return legal_moves[0]
    if num_sims <= 0:
        return model_move_raw(model, state, legal_moves, device)

    # batch_size=1, no dirichlet noise at eval time
    mcts = ludo_cpp.MCTSEngine(
        1, c_puct=3.0, dirichlet_alpha=0.3, dirichlet_eps=0.0,
    )
    mcts.set_roots([state])

    sims_done = 0
    safety_no_progress = 0
    while sims_done < num_sims:
        bsz = min(parallel_sims, num_sims - sims_done)
        leaves = mcts.select_leaves(bsz)
        if not leaves:
            break
        tensors = mcts.get_leaf_tensors()  # np array (N, 24, 15, 15)
        if tensors.size == 0 or tensors.shape[0] == 0:
            safety_no_progress += 1
            if safety_no_progress > 3:
                break
            continue
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(tensors)).to(device, dtype=torch.float32)
            policy, value = model(t)
            # policy is already softmax (V5.forward returns softmax)
            # value is unbounded — clamp for UCB stability
            value = torch.clamp(value, -1.0, 1.0)
            policies_list = policy.cpu().numpy().astype(np.float32).tolist()
            values_list = value.squeeze(-1).cpu().numpy().astype(np.float32).tolist()
        mcts.expand_and_backprop(policies_list, values_list)
        sims_done += tensors.shape[0]

    action_probs = mcts.get_action_probs(0.0)[0]  # (4,), greedy: one-hot on argmax
    # Pick the legal action with highest prob; fall back to argmax then any legal.
    best_action = -1
    best_p = -1.0
    for m in legal_moves:
        if action_probs[m] > best_p:
            best_p = action_probs[m]
            best_action = m
    if best_action < 0:
        best_action = _pick_legal(legal_moves)
    return best_action


# ---------------------------------------------------------------------------
# Opponent abstraction
# ---------------------------------------------------------------------------

class BotOpp:
    def __init__(self, bot_name, player_id):
        self.bot = get_bot(bot_name, player_id=player_id)

    def select(self, model, state, legal_moves, device):
        return self.bot.select_move(state, legal_moves)


class ModelOpp:
    def __init__(self, opp_model, num_sims):
        self.model = opp_model
        self.num_sims = num_sims

    def select(self, _primary_model, state, legal_moves, device):
        if self.num_sims == 0:
            return model_move_raw(self.model, state, legal_moves, device)
        return model_move_mcts(self.model, state, legal_moves, device, self.num_sims)


# ---------------------------------------------------------------------------
# Single game loop (based on evaluate_v6_1.py structure)
# ---------------------------------------------------------------------------

def play_game(model, opponent, model_player, device, num_sims):
    state = ludo_cpp.create_initial_state_2p()
    consecutive_sixes = [0, 0, 0, 0]
    move_count = 0

    while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
        current_player = state.current_player

        if not state.active_players[current_player]:
            nxt = (current_player + 1) % 4
            while not state.active_players[nxt]:
                nxt = (nxt + 1) % 4
            state.current_player = nxt
            continue

        if state.current_dice_roll == 0:
            state.current_dice_roll = random.randint(1, 6)
            cp = state.current_player
            if state.current_dice_roll == 6:
                consecutive_sixes[cp] += 1
            else:
                consecutive_sixes[cp] = 0
            if consecutive_sixes[cp] >= 3:
                nxt = (cp + 1) % 4
                while not state.active_players[nxt]:
                    nxt = (nxt + 1) % 4
                state.current_player = nxt
                state.current_dice_roll = 0
                consecutive_sixes[cp] = 0
                continue

        legal_moves = ludo_cpp.get_legal_moves(state)
        if not legal_moves:
            nxt = (state.current_player + 1) % 4
            while not state.active_players[nxt]:
                nxt = (nxt + 1) % 4
            state.current_player = nxt
            state.current_dice_roll = 0
            continue

        if current_player == model_player:
            if num_sims == 0:
                action = model_move_raw(model, state, legal_moves, device)
            else:
                action = model_move_mcts(model, state, legal_moves, device, num_sims)
        else:
            action = opponent.select(model, state, legal_moves, device)

        if action not in legal_moves:
            action = _pick_legal(legal_moves)

        state = ludo_cpp.apply_move(state, action)
        move_count += 1

    winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
    return winner, move_count


# ---------------------------------------------------------------------------
# Matchup runner
# ---------------------------------------------------------------------------

def run_matchup(label, model_path, num_sims, opponent_spec, num_games, device,
                seed, log_path, plan=None, all_results_so_far=None):
    """Run one matchup. Supports mid-matchup pause/resume via STATE_PATHS['partial'].

    Returns (result_dict, paused_bool). If paused_bool is True, the caller
    should break the sweep loop — the partial state is saved to disk and will
    be resumed on next launch.
    """
    # Try to resume from partial
    partial = load_partial(label)
    if partial:
        start_game = partial.get("games_played", 0)
        wins = partial.get("wins", 0)
        losses = partial.get("losses", 0)
        draws = partial.get("draws", 0)
        lengths = partial.get("lengths", [])
        accumulated_sec = partial.get("elapsed_sec", 0.0)
        print(f"[Resume] Continuing {label} from game {start_game}/{num_games} "
              f"(wins={wins} losses={losses} draws={draws})", flush=True)
    else:
        start_game = 0
        wins = losses = draws = 0
        lengths = []
        accumulated_sec = 0.0

    # Seed is deterministic + offset by game index so resume gives same sequence
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Advance RNG state to match resume point (approximate — dice seed per game
    # is acceptable drift; we only need determinism across a fresh run)
    if start_game > 0:
        for _ in range(start_game * 400):  # rough per-game RNG draws
            random.random()

    model = load_v61_model(model_path, device)

    if opponent_spec["type"] == "bot":
        def make_opp(pid):
            return BotOpp(opponent_spec["bot"], pid)
    elif opponent_spec["type"] == "model":
        opp_model = load_v61_model(opponent_spec["path"], device)
        opp_sims = opponent_spec.get("num_sims", 0)
        def make_opp(pid):
            return ModelOpp(opp_model, opp_sims)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_spec}")

    session_start = time.time()
    paused = False
    last_line = ""

    header = (
        f"\n=== [{label}] model={os.path.basename(model_path)} sims={num_sims} "
        f"opp={opponent_spec} games={num_games} seed={seed} start_game={start_game} ==="
    )
    print(header, flush=True)
    with open(log_path, "a") as logf:
        logf.write(header + "\n")
        logf.flush()

        for g in range(start_game, num_games):
            model_player = 0 if (g % 2 == 0) else 2
            opp_player = 2 if model_player == 0 else 0
            opp = make_opp(opp_player)

            winner, mlen = play_game(model, opp, model_player, device, num_sims)
            lengths.append(mlen)

            if winner == model_player:
                wins += 1
            elif winner == -1:
                draws += 1
            else:
                losses += 1

            games_played = g + 1
            session_elapsed = time.time() - session_start
            total_elapsed = accumulated_sec + session_elapsed
            gpm = games_played / (total_elapsed / 60) if total_elapsed > 0 else 0
            wr = wins / games_played
            remaining = max(num_games - games_played, 0)
            eta_min = (remaining / gpm) if gpm > 0 else 0

            # Per-game live stats update (fast, cheap)
            current = {
                "label": label,
                "num_sims": num_sims,
                "num_games": num_games,
                "games_played": games_played,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": wr,
                "gpm": gpm,
                "eta_min": eta_min,
                "opponent_spec": opponent_spec,
            }
            if games_played % 5 == 0 or games_played == num_games:
                write_live_stats(current, plan or [], all_results_so_far or [], last_line)

            # Periodic partial checkpoint (every 20 games)
            if games_played % 20 == 0 or games_played == num_games:
                save_partial({
                    "label": label,
                    "model_path": model_path,
                    "num_sims": num_sims,
                    "opponent_spec": opponent_spec,
                    "num_games": num_games,
                    "seed": seed,
                    "games_played": games_played,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "lengths": lengths,
                    "elapsed_sec": total_elapsed,
                    "timestamp": time.time(),
                })

            if games_played % 50 == 0 or games_played == num_games:
                line = (
                    f"  [{games_played:>5}/{num_games}] WR={wr*100:5.1f}% "
                    f"({wins}W/{losses}L/{draws}D) "
                    f"len={np.mean(lengths[-50:]):.0f} GPM={gpm:5.1f} "
                    f"ETA={eta_min:.1f}m"
                )
                last_line = line
                print(line, flush=True)
                logf.write(line + "\n")
                logf.flush()

            if STOP_REQUESTED.is_set():
                paused = True
                # Save final partial for clean resume
                save_partial({
                    "label": label,
                    "model_path": model_path,
                    "num_sims": num_sims,
                    "opponent_spec": opponent_spec,
                    "num_games": num_games,
                    "seed": seed,
                    "games_played": games_played,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "lengths": lengths,
                    "elapsed_sec": total_elapsed,
                    "timestamp": time.time(),
                })
                print(f"[Sweep] Paused cleanly at {games_played}/{num_games} in {label}",
                      flush=True)
                logf.write(f"=== PAUSED [{label}] at {games_played}/{num_games} ===\n")
                break

    total_elapsed = accumulated_sec + (time.time() - session_start)
    wr = wins / num_games if num_games > 0 else 0.0
    result = {
        "label": label,
        "model_path": model_path,
        "num_sims": num_sims,
        "opponent_spec": opponent_spec,
        "num_games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wr, 4),
        "elapsed_sec": round(total_elapsed, 1),
        "gpm": round(num_games / (total_elapsed / 60), 2) if total_elapsed > 0 else 0,
        "avg_game_length": round(float(np.mean(lengths)), 1) if lengths else 0,
        "seed": seed,
        "timestamp": time.time(),
        "paused": paused,
    }
    if not paused:
        clear_partial()
        with open(log_path, "a") as logf:
            logf.write(f"=== RESULT [{label}] win_rate={wr*100:.1f}% "
                       f"wins={wins}/{num_games} elapsed={total_elapsed/60:.1f}min ===\n")
    return result, paused


def save_results(results, out_path):
    existing = []
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(results[-1] if results else {})
    # Actually rewrite whole file with full list for atomicity
    with open(out_path, "w") as f:
        json.dump(existing if len(existing) >= len(results) else results, f, indent=2)


def write_results_atomic(all_results, out_path):
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)


# ---------------------------------------------------------------------------
# Sweep definition (Step 1 of POST_V61_EXPERIMENT_PLAN)
# ---------------------------------------------------------------------------

def build_sweep_plan(primary_ckpt, best_ckpt=None):
    """
    Returns the ordered list of matchups for Step 1.
    Primary model: V6.1 model_latest.pt @ 239K (most-trained).
    """
    plan = []

    # Phase A — raw baseline vs Expert
    plan.append(dict(
        label="A_raw_vs_expert",
        model_path=primary_ckpt,
        num_sims=0,
        opponent_spec={"type": "bot", "bot": "Expert"},
        num_games=1000,
        seed=10001,
    ))

    # Phase B — MCTS sweep vs Expert
    plan.append(dict(
        label="B_mcts25_vs_expert",
        model_path=primary_ckpt,
        num_sims=25,
        opponent_spec={"type": "bot", "bot": "Expert"},
        num_games=2000,
        seed=10025,
    ))
    plan.append(dict(
        label="B_mcts50_vs_expert",
        model_path=primary_ckpt,
        num_sims=50,
        opponent_spec={"type": "bot", "bot": "Expert"},
        num_games=2500,
        seed=10050,
    ))
    plan.append(dict(
        label="B_mcts100_vs_expert",
        model_path=primary_ckpt,
        num_sims=100,
        opponent_spec={"type": "bot", "bot": "Expert"},
        num_games=2500,
        seed=10100,
    ))
    plan.append(dict(
        label="B_mcts200_vs_expert",
        model_path=primary_ckpt,
        num_sims=200,
        opponent_spec={"type": "bot", "bot": "Expert"},
        num_games=1500,
        seed=10200,
    ))

    # Phase C — head-to-head: MCTS(N) vs raw V6.1 (same weights, different search)
    plan.append(dict(
        label="C_mcts50_vs_raw",
        model_path=primary_ckpt,
        num_sims=50,
        opponent_spec={"type": "model", "path": primary_ckpt, "num_sims": 0},
        num_games=1500,
        seed=20050,
    ))
    plan.append(dict(
        label="C_mcts100_vs_raw",
        model_path=primary_ckpt,
        num_sims=100,
        opponent_spec={"type": "model", "path": primary_ckpt, "num_sims": 0},
        num_games=1500,
        seed=20100,
    ))
    plan.append(dict(
        label="C_mcts200_vs_raw",
        model_path=primary_ckpt,
        num_sims=200,
        opponent_spec={"type": "model", "path": primary_ckpt, "num_sims": 0},
        num_games=1000,
        seed=20200,
    ))

    # Phase D — sanity cross-model (optional; only if best_ckpt provided)
    if best_ckpt:
        plan.append(dict(
            label="D_mcts100_vs_best157k",
            model_path=primary_ckpt,
            num_sims=100,
            opponent_spec={"type": "model", "path": best_ckpt, "num_sims": 0},
            num_games=500,
            seed=30100,
        ))

    return plan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "sweep"], default="sweep")
    parser.add_argument("--model", type=str,
                        default="checkpoints/ac_v6_1_strategic/model_latest.pt",
                        help="Primary V6.1 checkpoint (for sweep mode) or the model under test (for single)")
    parser.add_argument("--best_ckpt", type=str,
                        default="checkpoints/ac_v6_1_strategic/model_best.pt",
                        help="Optional best-eval checkpoint for Phase D cross-model sanity")
    parser.add_argument("--sims", type=int, default=0, help="(single mode) MCTS sims per move")
    parser.add_argument("--games", type=int, default=100, help="(single mode) number of games")
    parser.add_argument("--opponent", type=str, default="Expert",
                        help="(single mode) bot name, or 'model:path:sims'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", type=str, default="mcts_sweep.log")
    parser.add_argument("--out", type=str, default="mcts_sweep_results.json")
    parser.add_argument("--skip_completed", action="store_true",
                        help="Skip matchups whose label already appears in --out")
    parser.add_argument("--live_stats", type=str, default="mcts_sweep_live_stats.json",
                        help="Path to live stats JSON (dashboard reads this)")
    parser.add_argument("--partial", type=str, default="mcts_sweep_partial.json",
                        help="Path to mid-matchup partial checkpoint")
    parser.add_argument("--dashboard", action="store_true", default=True,
                        help="Start dashboard HTTP server (default: true)")
    parser.add_argument("--no_dashboard", dest="dashboard", action="store_false",
                        help="Disable dashboard server")
    parser.add_argument("--dashboard_port", type=int, default=8788)
    args = parser.parse_args()

    device = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARN: cuda not available, falling back to cpu")
        device = "cpu"

    # Install STATE_PATHS globally
    STATE_PATHS["live_stats"] = args.live_stats
    STATE_PATHS["partial"] = args.partial
    STATE_PATHS["results"] = args.out
    STATE_PATHS["log"] = args.log
    STATE_PATHS["sweep_started_at"] = time.time()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Dashboard server (non-blocking, daemon thread)
    if args.dashboard:
        start_dashboard(port=args.dashboard_port)

    # Resolve model paths (if relative, anchor to project root = script dir)
    project_root = os.path.dirname(os.path.abspath(__file__))
    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    all_results = []
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = []
        except Exception:
            all_results = []

    if args.mode == "single":
        if args.opponent.startswith("model:"):
            _, path, sims = args.opponent.split(":", 2)
            opp_spec = {"type": "model", "path": resolve(path), "num_sims": int(sims)}
        else:
            opp_spec = {"type": "bot", "bot": args.opponent}
        single_plan = [dict(
            label=f"single_sims{args.sims}_vs_{args.opponent.replace('/','_')}",
            num_sims=args.sims, num_games=args.games,
        )]
        result, paused = run_matchup(
            label=single_plan[0]["label"],
            model_path=resolve(args.model),
            num_sims=args.sims,
            opponent_spec=opp_spec,
            num_games=args.games,
            device=device,
            seed=args.seed,
            log_path=args.log,
            plan=single_plan,
            all_results_so_far=all_results,
        )
        if not paused:
            result.pop("paused", None)
            all_results.append(result)
            write_results_atomic(all_results, args.out)
        print(f"\nSaved result to {args.out}")
        return

    # Sweep mode
    plan = build_sweep_plan(
        resolve(args.model),
        resolve(args.best_ckpt) if args.best_ckpt else None,
    )
    completed_labels = {r.get("label") for r in all_results}

    print(f"[Sweep] Plan has {len(plan)} matchups. Device={device}")
    print(f"[Sweep] Dashboard: http://localhost:{args.dashboard_port}/"
          if args.dashboard else "[Sweep] Dashboard disabled")
    print(f"[Sweep] Live stats: {args.live_stats}")
    print(f"[Sweep] Partial ckpt: {args.partial}")
    for i, m in enumerate(plan, 1):
        print(f"  {i:2d}. {m['label']:28s} sims={m['num_sims']:4d} games={m['num_games']:5d}")

    # Initial live stats write so dashboard has something to show immediately
    write_live_stats(None, plan, all_results)

    for idx, m in enumerate(plan, 1):
        if STOP_REQUESTED.is_set():
            print(f"\n[{idx}/{len(plan)}] STOP requested before {m['label']} — exiting")
            break
        if args.skip_completed and m["label"] in completed_labels:
            print(f"\n[{idx}/{len(plan)}] SKIP (already completed): {m['label']}")
            continue
        print(f"\n[{idx}/{len(plan)}] START: {m['label']}")
        result, paused = run_matchup(
            label=m["label"],
            model_path=m["model_path"],
            num_sims=m["num_sims"],
            opponent_spec=m["opponent_spec"],
            num_games=m["num_games"],
            device=device,
            seed=m["seed"],
            log_path=args.log,
            plan=plan,
            all_results_so_far=all_results,
        )
        if paused:
            print(f"[{idx}/{len(plan)}] PAUSED: {m['label']} — state saved, will resume on next launch")
            write_live_stats(None, plan, all_results, "paused mid-matchup")
            break
        result.pop("paused", None)
        all_results.append(result)
        write_results_atomic(all_results, args.out)
        write_live_stats(None, plan, all_results, f"completed {m['label']}")
        print(f"[{idx}/{len(plan)}] DONE: {m['label']} "
              f"WR={result['win_rate']*100:.1f}% "
              f"({result['wins']}/{result['num_games']}) "
              f"elapsed={result['elapsed_sec']/60:.1f}min")

    write_live_stats(None, plan, all_results, "sweep finished")
    print("\n=== SWEEP COMPLETE ===")
    for r in all_results:
        print(f"  {r.get('label','?'):28s} sims={r.get('num_sims','?'):>4} "
              f"WR={r.get('win_rate',0)*100:5.1f}% "
              f"({r.get('wins','?')}/{r.get('num_games','?')}) "
              f"gpm={r.get('gpm',0):.1f}")


if __name__ == "__main__":
    main()
