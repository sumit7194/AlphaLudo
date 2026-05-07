"""AI-vs-AI Watch & Comment

Runs the loaded model against itself (both players use the same model).
Client drives the pacing — calls /api/play_step on a JS interval. Pause
is just "stop calling." All decisions are kept in-memory so the UI can
scrub backward and let you comment on any past move without rewinding
the actual game state.

Comments append to play/decision_logs/ai_disagreements.jsonl in the same
shape the play UI's flag_ai_disagreement uses (with `source: auto_watch`
plus an optional null preferred_token for comment-only annotations).

Run:
    LUDO_MODEL=v12_2 python -m experiments.ai_vs_ai_watch.server
or
    cd td_ludo && python experiments/ai_vs_ai_watch/server.py
"""
import datetime
import json
import os
import random
import secrets
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
TD_LUDO_DIR = HERE.parent.parent
PLAY_DIR = TD_LUDO_DIR / 'play'

# Make td_ludo_cpp + play.server helpers importable.
sys.path.insert(0, str(TD_LUDO_DIR))
sys.path.insert(0, str(PLAY_DIR))

import td_ludo_cpp as ludo_cpp  # noqa: E402

# Importing play.server triggers its model load (a few seconds). We use
# its helpers for board geometry + model architecture so this file stays
# small. We intentionally do NOT use play.server's GameManager — it bakes
# in human/AI player roles that don't apply here.
import server as play_server  # noqa: E402
from server import (  # noqa: E402
    get_board_coord, generate_board_layout,
    MODEL_VERSION, DECISION_LOGS_DIR, DECISION_SCHEMA_VERSION,
)

# Search visualization: compute both reward-only and policy-blended search
# per decision so the UI can show both decision processes side-by-side.
# These are NOT used to drive gameplay — model.argmax still picks the move.
# Both call into the bias-penalty + shaped-reward modules we use during
# training, so what we visualize is exactly what the training would see.
os.environ.pop('LUDO_BIAS_PENALTIES', None)  # we call penalties separately
from td_ludo.game.reward_shaping import compute_shaped_reward  # noqa: E402
from td_ludo.game.bias_penalties import compute_bias_penalties  # noqa: E402

# Policy-blend alpha. α=2.0 was the best-performing value in
# experiments/search_validation/penalty_override_eval.py (+0.6% over vanilla
# vs bots). Score = policy_prob + alpha * (shaped_reward + penalty).
SEARCH_BLEND_ALPHA = 2.0

# Use the model + device that play.server already loaded — saves a 2nd
# load and ~5MB of duplicated weights.
MODEL = play_server.model
DEVICE = play_server.device

AI_DISAGREEMENTS_PATH = Path(DECISION_LOGS_DIR) / 'ai_disagreements.jsonl'
os.makedirs(DECISION_LOGS_DIR, exist_ok=True)

STATIC_DIR = HERE / 'static'
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')


# ── Encoding helpers (per loaded model version) ───────────────────────
def encode_state(state):
    if MODEL_VERSION == 'v12_2':
        return ludo_cpp.encode_state_v11(state)
    if MODEL_VERSION in ('v11', 'v12'):
        return ludo_cpp.encode_state_v10(state)
    if MODEL_VERSION == 'v6_3':
        # Auto-watch supports the new models cleanly; v6.x logging fields
        # (win_prob, etc.) get filled with zeros to keep a uniform schema.
        return ludo_cpp.encode_state_v6_3(state, 0)
    return ludo_cpp.encode_state_v6(state)


def model_forward(state, legal_moves):
    """One forward pass for the current player. Returns (policy, value, action)."""
    state_tensor = encode_state(state)
    legal_mask = np.zeros(4, dtype=np.float32)
    for m in legal_moves:
        legal_mask[int(m)] = 1.0
    with torch.no_grad():
        s_t = torch.from_numpy(np.asarray(state_tensor)).unsqueeze(0).to(
            DEVICE, dtype=torch.float32
        )
        m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(
            DEVICE, dtype=torch.float32
        )
        out = MODEL(s_t, m_t)
        policy = out[0].squeeze(0).cpu().numpy()
        try:
            value = float(out[1].squeeze().item())
        except (IndexError, AttributeError):
            value = 0.0
    action = int(policy.argmax())
    if action not in legal_moves:
        action = int(legal_moves[0])
    return policy, value, action


def positions_dict(state):
    out = {}
    for p in [0, 2]:
        out[str(p)] = [int(x) for x in state.player_positions[p]]
    return out


def _reconstruct_state(positions, current_player, dice):
    """Build a fresh state with given positions/dice (for search simulation).
    apply_move mutates so we need clones per candidate."""
    g = ludo_cpp.create_initial_state_2p()
    pp = list(g.player_positions)
    for pstr, plist in positions.items():
        pp[int(pstr)] = list(int(x) for x in plist)
    g.player_positions = pp
    sc = list(g.scores)
    for pstr, plist in positions.items():
        sc[int(pstr)] = sum(1 for x in plist if int(x) == 99)
    g.scores = sc
    g.current_player = int(current_player)
    g.current_dice_roll = int(dice)
    return g


def _detect_capture(pre_state, post_state, mover_player):
    """Did mover_player's move send any opp token to base?"""
    for opp in range(4):
        if opp == mover_player or not pre_state.active_players[opp]:
            continue
        for t in range(4):
            old_op = pre_state.player_positions[opp][t]
            new_op = post_state.player_positions[opp][t]
            if old_op >= 0 and old_op != 99 and new_op == -1:
                return True
    return False


def compute_search_analysis(positions, current_player, dice, legal,
                            policy, move_count):
    """Compute per-action search scores for visualization.

    For each legal action, simulates apply_move and computes:
      - shaped_reward (existing reward shaping)
      - bias_penalty (the 5 bias penalties)
      - score_reward = shaped_reward + bias_penalty (pure reward search)
      - score_blend  = policy_prob + alpha * (shaped_reward + bias_penalty)

    Returns dict with per-action data + which action each search mode picks.
    Does NOT modify any state — purely analytical.
    """
    actions_data = {}
    for a in legal:
        sim_pre = _reconstruct_state(positions, current_player, dice)
        sim_next = ludo_cpp.apply_move(sim_pre, int(a))
        # Re-fetch pre because apply_move mutated sim_pre
        sim_pre = _reconstruct_state(positions, current_player, dice)
        base_r = compute_shaped_reward(sim_pre, sim_next, current_player)
        ctx = {
            'dice': dice, 'legal_moves': list(legal),
            'action': int(a), 'move_count': move_count,
        }
        pen_total, pen_breakdown = compute_bias_penalties(
            sim_pre, sim_next, current_player, ctx,
        )
        captured = _detect_capture(sim_pre, sim_next, current_player)
        post_pos = int(sim_next.player_positions[current_player][a])
        pre_pos = int(sim_pre.player_positions[current_player][a])
        score_reward = base_r + pen_total
        score_blend = float(policy[a]) + SEARCH_BLEND_ALPHA * score_reward
        # Filter zero penalties from breakdown for cleaner display
        bd_nonzero = {k: round(float(v), 4) for k, v in pen_breakdown.items()
                      if abs(v) > 1e-9}
        actions_data[int(a)] = {
            'pre_pos': pre_pos,
            'post_pos': post_pos,
            'shaped_reward': round(float(base_r), 4),
            'penalty_total': round(float(pen_total), 4),
            'penalty_breakdown': bd_nonzero,
            'policy_prob': round(float(policy[a]), 4),
            'score_reward': round(float(score_reward), 4),
            'score_blend': round(float(score_blend), 4),
            'captured': captured,
        }

    # Picks per mode
    model_pick = int(np.argmax(policy))
    if model_pick not in legal:
        model_pick = int(legal[0])
    reward_pick = max(legal, key=lambda a: actions_data[a]['score_reward'])
    blend_pick = max(legal, key=lambda a: actions_data[a]['score_blend'])

    return {
        'alpha': SEARCH_BLEND_ALPHA,
        'actions': actions_data,
        'model_pick': model_pick,
        'reward_pick': int(reward_pick),
        'blend_pick': int(blend_pick),
    }


def scores_dict(state):
    return {str(p): int(state.scores[p]) for p in [0, 2]}


def token_coords(state):
    coords = {}
    for player in [0, 2]:
        positions = list(state.player_positions[player])
        coords[str(player)] = [
            {
                'row': r, 'col': c, 'pos': int(positions[t]),
                'in_base': int(positions[t]) == -1,
                'scored': int(positions[t]) == 99,
                'on_home_run': 50 < int(positions[t]) < 99,
            }
            for t in range(4)
            for r, c in [get_board_coord(player, int(positions[t]), t)]
        ]
    return coords


# ── Auto-play game manager ────────────────────────────────────────────
class AutoPlayGame:
    """Self-play game with full per-ply history kept in memory.

    Each call to step() advances exactly one ply (one model decision +
    apply_move). Forced moves (single legal) and skipped turns (no legal /
    triple-six) advance internally so the UI sees one "interesting" frame
    per step. History records cover every ply where the model actually
    chose between options, so they all become candidates for review.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = ludo_cpp.create_initial_state_2p()
        self.consecutive_sixes = [0, 0, 0, 0]
        self.move_count = 0
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.game_id = f"auto_{ts}_{secrets.token_hex(3)}"
        self.history = []   # per-ply records (see step())
        self.terminal = False
        self.winner = -1
        self.last_message = 'New game ready.'
        return self.snapshot()

    def _skip_inactive(self):
        cp = int(self.state.current_player)
        if not self.state.active_players[cp]:
            np_ = (cp + 1) % 4
            while not self.state.active_players[np_]:
                np_ = (np_ + 1) % 4
            self.state.current_player = np_

    def _pass_turn(self, cp):
        np_ = (cp + 1) % 4
        while not self.state.active_players[np_]:
            np_ = (np_ + 1) % 4
        self.state.current_player = np_
        self.state.current_dice_roll = 0

    def step(self):
        """Roll for the current player and play one decision. Returns a
        dict describing what happened (for the UI to log + render).

        Forced moves still get logged but with `forced: True` so the UI
        can de-emphasize them. Triple-6 / no-legal turns return a record
        with action=null and `passed: True`.
        """
        if self.terminal:
            return {'terminal': True, 'winner': self.winner}

        self._skip_inactive()
        cp = int(self.state.current_player)

        # Roll
        roll = random.randint(1, 6)
        self.state.current_dice_roll = roll
        if roll == 6:
            self.consecutive_sixes[cp] += 1
        else:
            self.consecutive_sixes[cp] = 0

        # Triple six → skip turn
        if self.consecutive_sixes[cp] >= 3:
            self.consecutive_sixes[cp] = 0
            self._pass_turn(cp)
            rec = self._record_passed(cp, roll, reason='triple_six')
            self.history.append(rec)
            return rec

        legal = list(ludo_cpp.get_legal_moves(self.state))
        if not legal:
            self._pass_turn(cp)
            rec = self._record_passed(cp, roll, reason='no_legal')
            self.history.append(rec)
            return rec

        # Pre-move snapshot (this is what gets stored as the decision state)
        pre_positions = positions_dict(self.state)
        pre_scores = scores_dict(self.state)

        # Model inference (even for forced moves — gives a uniform schema)
        policy, value, action = model_forward(self.state, legal)

        # If only one legal move, the decision is trivial. Mark forced.
        forced = len(legal) == 1
        if forced:
            action = int(legal[0])

        # Search visualization (only for multi-legal — forced moves have
        # nothing to compare). Computed BEFORE apply_move so simulations
        # see the same pre-state. Cheap (~5ms per decision).
        search_analysis = None
        if not forced:
            try:
                search_analysis = compute_search_analysis(
                    pre_positions, cp, roll, legal, policy, self.move_count,
                )
            except Exception as e:
                print(f'[search_analysis] failed: {e}')

        # Capture detection: snapshot opp positions before/after.
        opp = (cp + 2) % 4 if (cp + 2) % 4 in (0, 2) else None
        old_opp_positions = (
            [int(p) for p in self.state.player_positions[opp]] if opp is not None else None
        )

        old_pos = int(self.state.player_positions[cp][action])
        self.state = ludo_cpp.apply_move(self.state, action)
        self.move_count += 1
        new_pos = int(self.state.player_positions[cp][action])

        captured = False
        if opp is not None:
            new_opp_positions = [int(p) for p in self.state.player_positions[opp]]
            for t in range(4):
                if (old_opp_positions[t] >= 0 and old_opp_positions[t] != 99
                        and new_opp_positions[t] == -1):
                    captured = True
                    break

        # Bonus turn detector: if engine left current_player == cp and not
        # terminal, same player rolls again next step.
        next_cp = int(self.state.current_player)
        bonus = (next_cp == cp and not self.state.is_terminal)

        if self.state.is_terminal:
            self.terminal = True
            self.winner = int(ludo_cpp.get_winner(self.state))

        rec = {
            'move_idx': len(self.history),
            'game_id': self.game_id,
            'ply': self.move_count,
            'player': cp,
            'dice': roll,
            'pre_positions': pre_positions,
            'pre_scores': pre_scores,
            'post_positions': positions_dict(self.state),
            'post_scores': scores_dict(self.state),
            'legal_tokens': [int(m) for m in legal],
            'ai_policy': [round(float(p), 6) for p in policy],
            'ai_chosen': int(action),
            'ai_value': round(float(value), 6),
            'forced': forced,
            'from_pos': old_pos,
            'to_pos': new_pos,
            'captured': bool(captured),
            'bonus': bool(bonus),
            'terminal': bool(self.state.is_terminal),
            'winner': self.winner if self.state.is_terminal else -1,
            'token_coords_after': self._token_coords_for(post=True),
            'passed': False,
            'search_analysis': search_analysis,
        }
        self.history.append(rec)
        return rec

    def _record_passed(self, cp, roll, reason):
        """Build a 'turn passed' record (no model decision).

        Still includes the post-state for rendering.
        """
        return {
            'move_idx': len(self.history),
            'game_id': self.game_id,
            'ply': self.move_count,
            'player': cp,
            'dice': roll,
            'pre_positions': positions_dict(self.state),
            'pre_scores': scores_dict(self.state),
            'post_positions': positions_dict(self.state),
            'post_scores': scores_dict(self.state),
            'legal_tokens': [],
            'ai_policy': None,
            'ai_chosen': None,
            'ai_value': None,
            'forced': False,
            'captured': False,
            'bonus': False,
            'terminal': False,
            'winner': -1,
            'token_coords_after': self._token_coords_for(post=True),
            'passed': True,
            'pass_reason': reason,
        }

    def _token_coords_for(self, post=True):
        """Compute current board coords (post-step). For history scrubbing
        we keep these inline so the FE doesn't need a per-state lookup."""
        return token_coords(self.state)

    def snapshot(self):
        return {
            'game_id': self.game_id,
            'move_count': self.move_count,
            'terminal': self.terminal,
            'winner': self.winner,
            'history_len': len(self.history),
            'current_player': int(self.state.current_player),
            'token_coords': token_coords(self.state),
            'positions': positions_dict(self.state),
            'scores': scores_dict(self.state),
            'model_version': MODEL_VERSION,
        }


GAME = AutoPlayGame()


# ── Comments → ai_disagreements.jsonl ─────────────────────────────────
def append_disagreement(record):
    record['ts'] = datetime.datetime.now().isoformat(timespec='milliseconds')
    with open(AI_DISAGREEMENTS_PATH, 'a') as f:
        f.write(json.dumps(record, separators=(',', ':')) + '\n')


# ── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(str(STATIC_DIR), 'index.html')


@app.route('/api/layout')
def api_layout():
    return jsonify(generate_board_layout())


@app.route('/api/info')
def api_info():
    return jsonify({
        'model_version': MODEL_VERSION,
        'game_id': GAME.game_id,
        'history_len': len(GAME.history),
    })


@app.route('/api/new_game', methods=['POST'])
def api_new_game():
    GAME.reset()
    return jsonify(GAME.snapshot())


@app.route('/api/state')
def api_state():
    return jsonify(GAME.snapshot())


@app.route('/api/history')
def api_history():
    """Full per-ply history. Cheap (≤200 records per game).
    Optional ?since=N returns only records with move_idx >= N."""
    try:
        since = int(request.args.get('since', 0))
    except ValueError:
        since = 0
    history = [r for r in GAME.history if r.get('move_idx', 0) >= since]
    return jsonify({
        'game_id': GAME.game_id,
        'history_len': len(GAME.history),
        'records': history,
        'terminal': GAME.terminal,
        'winner': GAME.winner,
    })


@app.route('/api/play_step', methods=['POST'])
def api_play_step():
    """Advance exactly one ply. Returns the new history record."""
    if GAME.terminal:
        return jsonify({'terminal': True, 'winner': GAME.winner, 'record': None})
    rec = GAME.step()
    return jsonify({
        'record': rec,
        'history_len': len(GAME.history),
        'terminal': GAME.terminal,
        'winner': GAME.winner,
    })


@app.route('/api/comment', methods=['POST'])
def api_comment():
    """Attach a comment (and optional preferred_token) to a past decision.
    Writes to ai_disagreements.jsonl so the disagreement-review pipeline
    can pick it up."""
    data = request.get_json(force=True) or {}
    try:
        move_idx = int(data.get('move_idx'))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'bad move_idx'}), 400
    comment = data.get('comment', '')
    if not isinstance(comment, str):
        comment = str(comment)
    preferred = data.get('preferred_token')
    if preferred is not None:
        try:
            preferred = int(preferred)
        except (TypeError, ValueError):
            return jsonify({'ok': False, 'error': 'bad preferred_token'}), 400
        if preferred not in (0, 1, 2, 3):
            return jsonify({'ok': False, 'error': 'preferred_token must be 0..3'}), 400

    if not (0 <= move_idx < len(GAME.history)):
        return jsonify({'ok': False, 'error': 'move_idx out of range'}), 404
    rec = GAME.history[move_idx]
    if rec.get('passed') or rec.get('ai_chosen') is None:
        return jsonify({
            'ok': False,
            'error': 'cannot comment on a passed turn (no decision was made)',
        }), 400

    if preferred is not None:
        if preferred not in (rec.get('legal_tokens') or []):
            return jsonify({
                'ok': False,
                'error': 'preferred_token was not legal at decision time',
            }), 400
        if preferred == rec.get('ai_chosen'):
            return jsonify({
                'ok': False,
                'error': 'preferred_token equals ai_chosen — pick a different token or omit',
            }), 400

    if not comment.strip() and preferred is None:
        return jsonify({
            'ok': False,
            'error': 'comment or preferred_token required',
        }), 400

    out = {
        'schema_version': DECISION_SCHEMA_VERSION,
        'source': 'auto_watch',
        'game_id': rec.get('game_id'),
        'ai_decision_id': move_idx,           # uses history idx as id
        'model_version': MODEL_VERSION,
        'move_count': rec.get('ply'),
        'current_player': rec.get('player'),
        'dice': rec.get('dice'),
        'positions': rec.get('pre_positions'),
        'scores': rec.get('pre_scores'),
        'legal_tokens': rec.get('legal_tokens'),
        'ai_policy': rec.get('ai_policy'),
        'ai_chosen': rec.get('ai_chosen'),
        'ai_value': rec.get('ai_value'),
        'preferred_token': preferred,
        'comment': comment,
    }
    try:
        append_disagreement(out)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
    return jsonify({'ok': True, 'saved': out})


if __name__ == '__main__':
    print("\n" + '=' * 50)
    print(f"  AlphaLudo Auto-Watch ({MODEL_VERSION.upper()})")
    print(f"  Open: http://localhost:5053")
    print(f"  Disagreements append to: {AI_DISAGREEMENTS_PATH}")
    print('=' * 50 + '\n')
    app.run(host='0.0.0.0', port=5053, debug=False)
