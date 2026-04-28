"""
AlphaLudo Play — Web Server for Human vs AI Ludo

Flask backend that manages game state via the C++ engine and runs AI
inference. Supports V6.1, V6.3, V11.1, V12 (all 28ch input, different attn).

Select via env var:
    LUDO_MODEL=v6_1                  (loads model_weights/model.pt)
    LUDO_MODEL=v6_3                  (loads model_weights/model_v6_3.pt)
    LUDO_MODEL=v11                   (loads model_weights/model_v11.pt)
    LUDO_MODEL=v12   (default)       (loads model_weights/model_v12.pt)
"""

import math
import os
import sys
import json
import secrets
import random
import numpy as np
import torch

# Add parent td_ludo directory to path for C++ module access
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TD_LUDO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TD_LUDO_DIR)

import td_ludo_cpp as ludo_cpp
from flask import Flask, jsonify, request, send_from_directory
from model import AlphaLudoV5, AlphaLudoV63, AlphaLudoV11, AlphaLudoV12

# ── Configuration ──────────────────────────────────────────────
MODEL_VERSION = os.environ.get('LUDO_MODEL', 'v12').lower()
if MODEL_VERSION not in ('v6_1', 'v6_3', 'v11', 'v12'):
    raise ValueError(f"Unknown LUDO_MODEL='{MODEL_VERSION}'. Use 'v6_1', 'v6_3', 'v11', or 'v12'.")

MODEL_FILES = {
    'v6_1': os.path.join(SCRIPT_DIR, 'model_weights', 'model.pt'),
    'v6_3': os.path.join(SCRIPT_DIR, 'model_weights', 'model_v6_3.pt'),
    'v11':  os.path.join(SCRIPT_DIR, 'model_weights', 'model_v11.pt'),
    'v12':  os.path.join(SCRIPT_DIR, 'model_weights', 'model_v12.pt'),
}
MODEL_PATH = MODEL_FILES[MODEL_VERSION]

HUMAN_PLAYER = 0   # P0 = Human (top-left on standard board)
AI_PLAYER = 2       # P2 = AI (bottom-right on standard board)
MAX_MOVES = 10000

# ── Board coordinate lookup ────────────────────────────────────
# P0's 52-square path on the 15x15 grid (from game.cpp)
PATH_COORDS_P0 = [
    (6,1),(6,2),(6,3),(6,4),(6,5),           # 0-4
    (5,6),(4,6),(3,6),(2,6),(1,6),(0,6),      # 5-10
    (0,7),(0,8),                              # 11-12
    (1,8),(2,8),(3,8),(4,8),(5,8),            # 13-17
    (6,9),(6,10),(6,11),(6,12),(6,13),(6,14), # 18-23
    (7,14),(8,14),                            # 24-25
    (8,13),(8,12),(8,11),(8,10),(8,9),        # 26-30
    (9,8),(10,8),(11,8),(12,8),(13,8),(14,8), # 31-36
    (14,7),(14,6),                            # 37-38
    (13,6),(12,6),(11,6),(10,6),(9,6),        # 39-43
    (8,5),(8,4),(8,3),(8,2),(8,1),(8,0),      # 44-49
    (7,0),                                    # 50
]

HOME_RUN_P0 = [(7,1),(7,2),(7,3),(7,4),(7,5)]
HOME_COORD = (7,7)  # Center

BASE_COORDS = {
    0: [(2,2),(2,3),(3,2),(3,3)],
    1: [(2,11),(2,12),(3,11),(3,12)],
    2: [(11,11),(11,12),(12,11),(12,12)],
    3: [(11,2),(11,3),(12,2),(12,3)],
}

SAFE_INDICES = {0, 8, 13, 21, 26, 34, 39, 47}

def rotate_90cw(r, c):
    """Rotate a point 90° clockwise around center (7,7) on a 15x15 grid."""
    return (c, 14 - r)

def get_board_coord(player, pos, token_index=0):
    """Get (row, col) board coordinate for a token position."""
    if pos == -1:  # BASE
        return BASE_COORDS[player][token_index]
    if pos == 99:  # HOME (scored)
        local = HOME_COORD
    elif pos > 50:  # Home run (51-55)
        idx = pos - 51
        if idx < 5:
            local = HOME_RUN_P0[idx]
        else:
            local = HOME_COORD
    else:  # Main track (0-50)
        local = PATH_COORDS_P0[pos]
    
    # Rotate for player
    r, c = local
    for _ in range(player):
        r, c = rotate_90cw(r, c)
    return (r, c)


# ── Generate the full board layout data ────────────────────────
def generate_board_layout():
    """Pre-compute all path squares, safe squares, home runs, bases for frontend."""
    layout = {
        'path_squares': [],
        'safe_squares': [],
        'home_runs': {},
        'bases': BASE_COORDS,
        'home_center': list(HOME_COORD),
    }
    
    # Generate all 52 path squares for all active players (P0, P2)
    all_path = set()
    for player in [0, 2]:
        for pos in range(51):
            r, c = get_board_coord(player, pos, 0)
            all_path.add((r, c))
    layout['path_squares'] = [list(p) for p in sorted(all_path)]
    
    # Safe squares
    for player in [0, 2]:
        for si in SAFE_INDICES:
            abs_pos = si  # Safe indices are absolute (P0 view)
            # We need to find which relative position maps to this absolute for each player
            # abs = (rel + 13*player) % 52, so rel = (abs - 13*player) % 52
            rel = (abs_pos - 13 * player) % 52
            if rel <= 50:
                r, c = get_board_coord(player, rel)
                layout['safe_squares'].append([r, c])
    layout['safe_squares'] = [list(s) for s in set(tuple(s) for s in layout['safe_squares'])]
    
    # Home runs for each active player
    for player in [0, 2]:
        hrs = []
        for pos in range(51, 56):
            r, c = get_board_coord(player, pos)
            hrs.append([r, c])
            all_path.add((r, c))
        layout['home_runs'][str(player)] = hrs
    
    # Add home run path squares too
    layout['path_squares'] = [list(p) for p in sorted(all_path)]
    
    # Spawn positions (pos 0 for each player)
    layout['spawn_squares'] = {}
    for player in [0, 2]:
        r, c = get_board_coord(player, 0)
        layout['spawn_squares'][str(player)] = [r, c]
    
    return layout


# ── AI Model Loading ───────────────────────────────────────────
def load_model():
    device = torch.device('cpu')  # CPU for single-game inference is fine

    if MODEL_VERSION == 'v12':
        # V12: same CNN backbone as V11.1, but attention over 8 game-piece
        # tokens (not 225 cells). 4 ResBlocks × 96ch + 2 attn × 4 heads.
        model = AlphaLudoV12(
            num_res_blocks=4, num_channels=96,
            num_attn_layers=2, num_heads=4,
            ffn_ratio=4, dropout=0.0, in_channels=28,
        )
    elif MODEL_VERSION == 'v11':
        # V11.1: 4 ResBlocks + 1 attn × 2 heads × dim 64, 28ch input.
        model = AlphaLudoV11(
            num_res_blocks=4, num_channels=96,
            num_attn_layers=1, num_heads=2,
            ffn_ratio=4, attn_dim=64,
            dropout=0.0, in_channels=28,
        )
    elif MODEL_VERSION == 'v6_3':
        model = AlphaLudoV63(num_res_blocks=10, num_channels=128, in_channels=27)
    else:
        model = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    # Handle compiled model state dicts
    if isinstance(checkpoint, dict) and any(
        isinstance(k, str) and k.startswith('_orig_mod.') for k in checkpoint.keys()
    ):
        checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
    # Handle full checkpoint dict vs raw state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Play] Loaded {MODEL_VERSION.upper()} model from {MODEL_PATH} "
          f"({param_count:,} params)")
    return model, device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Game Event Logger ─────────────────────────────────────────
import datetime
LOGS_DIR = os.path.join(SCRIPT_DIR, 'game_logs')
os.makedirs(LOGS_DIR, exist_ok=True)


class GameLogger:
    """Writes every game event to a per-game file for post-game review."""

    def __init__(self, logs_dir=LOGS_DIR, model_version=MODEL_VERSION):
        self.logs_dir = logs_dir
        self.model_version = model_version
        self.current_file = None
        self.game_num = 0
        self._open_new_game()

    def _open_new_game(self):
        if self.current_file:
            self.current_file.close()
        self.game_num += 1
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.logs_dir, f'game_{ts}_{self.model_version}.log')
        self.current_file = open(path, 'w')
        self.log_path = path
        self.log(f"=== NEW GAME ({self.model_version.upper()}) — {ts} ===")

    def log(self, msg):
        if not self.current_file or self.current_file.closed:
            return
        t = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        line = f"[{t}] {msg}"
        self.current_file.write(line + '\n')
        self.current_file.flush()
        print(line)

    def new_game(self):
        self.log(f"--- game ended ---")
        self._open_new_game()

    def close(self):
        if self.current_file:
            self.current_file.close()


game_logger = GameLogger()


# ── Decision Logger (Eval Lens — Level 1) ──────────────────────
DECISION_LOGS_DIR = os.path.join(SCRIPT_DIR, 'decision_logs')

DECISION_SCHEMA_VERSION = 1
RATING_LABELS = {"v12_right", "human_right", "either", "both_bad"}


class DecisionLogger:
    """Append-only JSONL logger for human decisions (vs model recommendation).

    Three files in `decision_logs/`:
      - decisions_<game_id>.jsonl  : one record per human decision (per game)
      - outcomes.jsonl             : one record per finished game (global)
      - ratings.jsonl              : one record per Level-2 user label (global)

    Open-write-close per record so a crash can never corrupt history. The
    write volume is tiny (50 decisions/game).
    """

    def __init__(self, logs_dir=DECISION_LOGS_DIR, model_version=MODEL_VERSION):
        self.logs_dir = logs_dir
        self.model_version = model_version
        os.makedirs(self.logs_dir, exist_ok=True)
        self.outcomes_path = os.path.join(self.logs_dir, 'outcomes.jsonl')
        self.ratings_path = os.path.join(self.logs_dir, 'ratings.jsonl')

    @staticmethod
    def _now_iso():
        return datetime.datetime.now().isoformat(timespec='milliseconds')

    def start_game(self):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"g_{ts}_{secrets.token_hex(3)}"

    def decisions_path(self, game_id):
        return os.path.join(self.logs_dir, f'decisions_{game_id}.jsonl')

    def log_decision(self, record):
        try:
            path = self.decisions_path(record['game_id'])
            with open(path, 'a') as f:
                f.write(json.dumps(record, separators=(',', ':')) + '\n')
        except Exception as e:
            # Never let logging break a move.
            print(f"[DecisionLogger] log_decision failed: {e}")

    def log_outcome(self, game_id, winner, num_decisions, total_moves,
                    abandoned=False):
        try:
            record = {
                'schema_version': DECISION_SCHEMA_VERSION,
                'game_id': game_id,
                'ended_ts': self._now_iso(),
                'winner': int(winner),
                'human_won': bool(winner == HUMAN_PLAYER),
                'total_moves': int(total_moves),
                'num_human_decisions': int(num_decisions),
                'model_version': self.model_version,
                'abandoned': bool(abandoned),
            }
            with open(self.outcomes_path, 'a') as f:
                f.write(json.dumps(record, separators=(',', ':')) + '\n')
        except Exception as e:
            print(f"[DecisionLogger] log_outcome failed: {e}")

    def log_rating(self, game_id, decision_id, label):
        record = {
            'schema_version': DECISION_SCHEMA_VERSION,
            'game_id': game_id,
            'decision_id': int(decision_id),
            'label': label,
            'ts': self._now_iso(),
        }
        with open(self.ratings_path, 'a') as f:
            f.write(json.dumps(record, separators=(',', ':')) + '\n')

    def read_decisions(self, game_id):
        path = self.decisions_path(game_id)
        if not os.path.exists(path):
            return []
        out = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def latest_game_id(self):
        """Most recently modified decisions_*.jsonl in logs_dir, or None."""
        try:
            entries = []
            for name in os.listdir(self.logs_dir):
                if name.startswith('decisions_') and name.endswith('.jsonl'):
                    full = os.path.join(self.logs_dir, name)
                    entries.append((os.path.getmtime(full), name))
            if not entries:
                return None
            entries.sort(reverse=True)
            name = entries[0][1]
            # decisions_<game_id>.jsonl
            return name[len('decisions_'):-len('.jsonl')]
        except Exception:
            return None


decision_logger = DecisionLogger()


# ── Game State Manager ─────────────────────────────────────────
class GameManager:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.reset()

    def reset(self):
        # If we have an in-progress, non-terminal game, mark it abandoned
        # before starting fresh so eval data isn't silently dropped.
        if (
            getattr(self, '_initialized', False)
            and getattr(self, 'state', None) is not None
            and not self.state.is_terminal
            and getattr(self, '_human_decision_count', 0) > 0
        ):
            try:
                decision_logger.log_outcome(
                    self.game_id,
                    winner=-1,
                    num_decisions=self._human_decision_count,
                    total_moves=self.move_count,
                    abandoned=True,
                )
            except Exception as e:
                print(f"[DecisionLogger] abandoned-flush failed: {e}")

        self.state = ludo_cpp.create_initial_state_2p()
        self.consecutive_sixes = [0, 0, 0, 0]
        self.move_count = 0
        self.game_log = []
        self.pending_dice = None
        self.last_move_info = None
        self.game_id = decision_logger.start_game()
        self._human_decision_count = 0
        self._outcome_logged = False
        self._cached_prediction = None
        if hasattr(self, '_initialized'):
            game_logger.new_game()
        self._initialized = True
        return self._get_state_json()
    
    def _get_token_coords(self):
        """Get board coordinates for all tokens."""
        coords = {}
        for player in [0, 2]:
            token_coords = []
            positions = list(self.state.player_positions[player])
            for t in range(4):
                pos = int(positions[t])
                r, c = get_board_coord(player, pos, t)
                token_coords.append({
                    'row': r, 'col': c,
                    'pos': pos,
                    'in_base': pos == -1,
                    'scored': pos == 99,
                    'on_home_run': 50 < pos < 99,
                })
            coords[str(player)] = token_coords
        return coords
    
    def _compute_win_chance(self):
        """Run the value head and return AI's win probability in [0, 1].

        The value head is "expected outcome for the current player to move".
        To get a consistent metric across turn boundaries, we evaluate from
        AI's perspective ALWAYS — temporarily set current_player=AI and
        clear the dice (between-turns evaluation). This avoids the wild
        swings caused by the model's current-player-advantage bias.

        Returns None if the game is over.
        """
        if self.state.is_terminal:
            winner = int(ludo_cpp.get_winner(self.state))
            return 1.0 if winner == AI_PLAYER else 0.0 if winner != -1 else None

        if not self.state.active_players[AI_PLAYER]:
            return None

        # Save fields we'll temporarily override
        saved_cp = int(self.state.current_player)
        saved_dice = int(self.state.current_dice_roll)
        try:
            # Always evaluate from AI's POV with no dice rolled (neutral
            # between-turns snapshot).
            self.state.current_player = AI_PLAYER
            self.state.current_dice_roll = 0

            if MODEL_VERSION in ('v11', 'v12'):
                # V11 and V12 both use the V10 encoder (28 channels).
                state_tensor = ludo_cpp.encode_state_v10(self.state)
            elif MODEL_VERSION == 'v6_3':
                state_tensor = ludo_cpp.encode_state_v6_3(
                    self.state, int(self.consecutive_sixes[AI_PLAYER])
                )
            else:
                state_tensor = ludo_cpp.encode_state_v6(self.state)

            # Permissive mask — value head doesn't actually depend on it,
            # but the model API expects one.
            mask = np.ones(4, dtype=np.float32)

            with torch.no_grad():
                s_t = torch.from_numpy(np.asarray(state_tensor)).unsqueeze(0).to(
                    self.device, dtype=torch.float32
                )
                m_t = torch.from_numpy(mask).unsqueeze(0).to(
                    self.device, dtype=torch.float32
                )
                out = self.model(s_t, m_t)
                # V6.1:  (policy, value)
                # V6.3:  (policy, value, aux)
                # V11:   (policy, win_prob, moves_remaining)
                # V12:   (policy, win_prob, moves_remaining)
                v = float(out[1].squeeze().item())
        except Exception as e:
            print(f"[Play] win_chance failed: {e}")
            return None
        finally:
            # Restore actual game state
            self.state.current_player = saved_cp
            self.state.current_dice_roll = saved_dice

        if MODEL_VERSION in ('v11', 'v12'):
            # V11/V12 win_prob head is sigmoid-output, BCE-trained on actual
            # outcomes — already a calibrated probability in [0, 1].
            ai_win_prob = v
        else:
            # V6.x value head is unbounded; squash for display.
            import math
            ai_win_prob = 1.0 / (1.0 + math.exp(-v))
        return max(0.001, min(0.999, ai_win_prob))

    def _get_state_json(self):
        positions = {}
        scores = {}
        for player in [0, 2]:
            positions[str(player)] = [int(p) for p in self.state.player_positions[player]]
            scores[str(player)] = int(self.state.scores[player])

        win_chance = self._compute_win_chance()

        return {
            'positions': positions,
            'scores': scores,
            'current_player': int(self.state.current_player),
            'dice_roll': int(self.state.current_dice_roll),
            'is_terminal': bool(self.state.is_terminal),
            'winner': int(ludo_cpp.get_winner(self.state)) if self.state.is_terminal else -1,
            'move_count': self.move_count,
            'token_coords': self._get_token_coords(),
            'last_move': self.last_move_info,
            'consecutive_sixes': self.consecutive_sixes.copy(),
            'ai_win_chance': win_chance,  # 0.0..1.0 or None
            'game_id': getattr(self, 'game_id', None),
            'model_version': MODEL_VERSION,
        }
    
    def _skip_inactive_players(self):
        """Skip to next active player if current is inactive."""
        cp = int(self.state.current_player)
        if not self.state.active_players[cp]:
            next_p = (cp + 1) % 4
            while not self.state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            self.state.current_player = next_p
    
    def roll_dice(self):
        """Roll dice for the current player."""
        self._skip_inactive_players()
        
        if self.state.is_terminal:
            return {**self._get_state_json(), 'legal_moves': [], 'message': 'Game Over'}
        
        if self.state.current_dice_roll != 0:
            # Already rolled
            legal = ludo_cpp.get_legal_moves(self.state)
            return {**self._get_state_json(), 'legal_moves': [int(m) for m in legal]}
        
        # Roll
        roll = random.randint(1, 6)
        self.state.current_dice_roll = roll
        cp = int(self.state.current_player)
        who = 'AI' if cp == AI_PLAYER else 'Human'

        # Track consecutive sixes
        if roll == 6:
            self.consecutive_sixes[cp] += 1
        else:
            self.consecutive_sixes[cp] = 0

        game_logger.log(f"{who} (P{cp}) rolled {roll}"
                        + (f" [consec-sixes={self.consecutive_sixes[cp]}]"
                           if self.consecutive_sixes[cp] > 0 else ''))

        # Triple six penalty
        if self.consecutive_sixes[cp] >= 3:
            self.consecutive_sixes[cp] = 0
            self.state.current_dice_roll = 0
            next_p = (cp + 1) % 4
            while not self.state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            self.state.current_player = next_p

            self.game_log.append(f"P{cp} rolled triple 6! Turn lost.")
            game_logger.log(f"  -> TRIPLE 6 PENALTY, turn passes to P{next_p}")
            return {
                **self._get_state_json(),
                'legal_moves': [],
                'message': 'Triple 6! Turn lost.',
                'triple_six': True,
                'rolled': roll,  # preserve rolled value even after dice reset
            }

        legal = ludo_cpp.get_legal_moves(self.state)

        # No legal moves — pass turn
        if len(legal) == 0:
            self.state.current_dice_roll = 0
            next_p = (cp + 1) % 4
            while not self.state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            self.state.current_player = next_p
            game_logger.log(f"  -> no legal moves, turn passes to P{next_p}")

            return {
                **self._get_state_json(),
                'legal_moves': [],
                'message': f'P{cp} has no legal moves. Turn passed.',
                'no_moves': True,
                'rolled': roll,  # preserve rolled value even after dice reset
            }

        # Eval Lens — when it's the human about to choose, run V12 on the
        # *pre-move* state and ship the prediction with the roll response so
        # the frontend can highlight V12's recommended token in real time.
        # Cache the result so make_move's _log_human_decision doesn't have
        # to run a second forward pass on the same state.
        prediction = None
        if cp == HUMAN_PLAYER and len(legal) > 1:
            prediction = self._predict_human_policy(list(legal))
        elif cp == HUMAN_PLAYER and len(legal) == 1:
            # Single legal move — auto-played by the client. No real
            # decision to log, but the FE may still want to render the
            # forced pick consistently. Cheap to compute.
            prediction = self._predict_human_policy(list(legal))
        if prediction is not None:
            # Pin to (move_count, dice) so a stale cache can't be reused
            # across turns. _log_human_decision verifies the key matches.
            self._cached_prediction = {
                'move_count': self.move_count,
                'dice': roll,
                'data': prediction,
            }

        response = {
            **self._get_state_json(),
            'legal_moves': [int(m) for m in legal],
            'rolled': roll,  # actual rolled value, also in dice_roll on success
        }
        if prediction is not None:
            response['model_pick'] = prediction['argmax']
            response['model_policy'] = prediction['policy']
            response['model_win_prob'] = prediction['win_prob']
        return response

    def _predict_human_policy(self, legal):
        """Run the loaded V11/V12 model on the current state and return a
        dict with policy / argmax / win_prob / moves_remaining. Returns
        None for older models (different encoder, not supported)."""
        if MODEL_VERSION not in ('v11', 'v12'):
            return None

        legal_mask = np.zeros(4, dtype=np.float32)
        for m in legal:
            legal_mask[int(m)] = 1.0

        try:
            state_tensor = ludo_cpp.encode_state_v10(self.state)
            with torch.no_grad():
                s_t = torch.from_numpy(np.asarray(state_tensor)).unsqueeze(0).to(
                    self.device, dtype=torch.float32
                )
                m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(
                    self.device, dtype=torch.float32
                )
                out = self.model(s_t, m_t)
                policy = out[0].squeeze(0).cpu().numpy()
                win_prob = float(out[1].squeeze().item())
                try:
                    moves_remaining = float(out[2].squeeze().item())
                except (IndexError, AttributeError):
                    moves_remaining = None
        except Exception as e:
            print(f"[Play] _predict_human_policy failed: {e}")
            return None

        return {
            'policy': [round(float(p), 6) for p in policy],
            'argmax': int(np.argmax(policy)),
            'win_prob': round(win_prob, 6),
            'moves_remaining': (
                round(moves_remaining, 4) if moves_remaining is not None else None
            ),
        }

    def make_move(self, token_index):
        """Apply a move (for human player)."""
        if self.state.is_terminal:
            return {**self._get_state_json(), 'message': 'Game Over'}
        
        legal = ludo_cpp.get_legal_moves(self.state)
        if token_index not in legal:
            return {**self._get_state_json(), 'message': 'Illegal move', 'error': True}
        
        cp = int(self.state.current_player)
        old_pos = int(self.state.player_positions[cp][token_index])

        # Eval Lens — Level 1: log V12's would-have-chosen vs the human's
        # actual choice. Only when it's the human's turn (AI moves are
        # already logged via game_logger and aren't useful as eval signal).
        if cp == HUMAN_PLAYER:
            try:
                self._log_human_decision(token_index, list(legal))
            except Exception as e:
                # Never let logging break a move.
                print(f"[DecisionLogger] _log_human_decision failed: {e}")

        # Check for captures (compare pre/post state)
        old_opp_positions = {}
        for opp in [0, 2]:
            if opp != cp:
                old_opp_positions[opp] = [int(p) for p in self.state.player_positions[opp]]
        
        self.state = ludo_cpp.apply_move(self.state, token_index)
        self.move_count += 1
        
        new_pos = int(self.state.player_positions[cp][token_index])
        
        # Detect captures
        captured = False
        for opp, old_opp_pos in old_opp_positions.items():
            new_opp_pos = [int(p) for p in self.state.player_positions[opp]]
            for t in range(4):
                if old_opp_pos[t] >= 0 and old_opp_pos[t] != 99 and new_opp_pos[t] == -1:
                    captured = True
                    break
        
        self.last_move_info = {
            'player': cp,
            'token': token_index,
            'from_pos': old_pos,
            'to_pos': new_pos if new_pos != 99 else 'HOME',
            'captured': captured,
            'dice': int(self.state.current_dice_roll) if self.state.current_dice_roll else 0,
        }

        who = 'AI' if cp == AI_PLAYER else 'Human'
        to_str = 'HOME' if new_pos == 99 else str(new_pos)
        msg = f"  {who} (P{cp}) moved token {token_index}: pos {old_pos} -> {to_str}"
        if captured:
            msg += " [CAPTURED opponent!]"
        game_logger.log(msg)
        scores = f"Score: Human={self.state.scores[0]}/4  AI={self.state.scores[AI_PLAYER]}/4"
        game_logger.log(f"  [{scores}  move #{self.move_count}]")

        result = self._get_state_json()

        # Check if same player goes again (bonus turn)
        next_cp = int(self.state.current_player)
        result['bonus_turn'] = (next_cp == cp and not self.state.is_terminal)
        if result['bonus_turn']:
            game_logger.log(f"  -> {who} gets a BONUS TURN (rolled 6 or scored)")

        if self.state.is_terminal:
            winner = ludo_cpp.get_winner(self.state)
            win_who = 'AI' if winner == AI_PLAYER else 'Human' if winner == 0 else 'Unknown'
            game_logger.log(f"*** GAME OVER — {win_who} (P{winner}) wins in {self.move_count} moves ***")
            if not getattr(self, '_outcome_logged', False):
                decision_logger.log_outcome(
                    self.game_id,
                    winner=int(winner),
                    num_decisions=self._human_decision_count,
                    total_moves=self.move_count,
                )
                self._outcome_logged = True

        return result

    def _log_human_decision(self, token_index, legal):
        """Run the loaded model on the *pre-move* state and append a
        decision record to decisions_<game_id>.jsonl.

        Captures: dice, positions, scores, V12's full policy + win_prob,
        and the human's actual chosen token. Disagreement metrics
        (interest_score, kl, agree) are precomputed at log time so the
        review endpoint can sort without re-deriving them.
        """
        # Only V11/V12 share the 28-channel encoder we rely on. Older
        # models use a different encoder + value-head semantics; just
        # skip Level 1 for them.
        if MODEL_VERSION not in ('v11', 'v12'):
            return

        cp = int(self.state.current_player)
        dice = int(self.state.current_dice_roll)

        # Reuse the prediction computed in roll_dice() if it matches the
        # current (move_count, dice). Saves ~15ms by avoiding a redundant
        # forward pass on the exact same state.
        cache = getattr(self, '_cached_prediction', None)
        if (cache is not None
                and cache.get('move_count') == self.move_count
                and cache.get('dice') == dice):
            data = cache['data']
            policy = np.asarray(data['policy'], dtype=np.float32)
            win_prob = float(data['win_prob'])
            moves_remaining = data.get('moves_remaining')
        else:
            pred = self._predict_human_policy(legal)
            if pred is None:
                return
            policy = np.asarray(pred['policy'], dtype=np.float32)
            win_prob = float(pred['win_prob'])
            moves_remaining = pred.get('moves_remaining')

        v12_argmax = int(np.argmax(policy))
        v12_prob_of_human = float(policy[int(token_index)])
        # KL(human_one_hot || policy) = -log(policy[human_token])
        kl = -math.log(max(v12_prob_of_human, 1e-9))
        interest = float(np.max(policy)) * kl

        positions = {}
        scores = {}
        for player in [0, 2]:
            positions[str(player)] = [int(p) for p in self.state.player_positions[player]]
            scores[str(player)] = int(self.state.scores[player])

        record = {
            'schema_version': DECISION_SCHEMA_VERSION,
            'game_id': self.game_id,
            'decision_id': self._human_decision_count,
            'ts': DecisionLogger._now_iso(),
            'model_version': MODEL_VERSION,

            'current_player': cp,
            'dice': dice,
            'consecutive_sixes': int(self.consecutive_sixes[cp]),
            'positions': positions,
            'scores': scores,
            'move_count': self.move_count,
            'legal_tokens': [int(m) for m in legal],

            'v12_policy': [round(float(p), 6) for p in policy],
            'v12_argmax': v12_argmax,
            'v12_win_prob': round(win_prob, 6),
            'v12_moves_remaining': (
                round(moves_remaining, 4) if moves_remaining is not None else None
            ),

            'human_token': int(token_index),
            'agree': bool(v12_argmax == int(token_index)),
            'v12_prob_of_human': round(v12_prob_of_human, 6),
            'kl_v12_to_human': round(kl, 6),
            'interest_score': round(interest, 6),
        }
        decision_logger.log_decision(record)
        self._human_decision_count += 1

    def ai_move(self):
        """AI evaluates the board and makes a move."""
        if self.state.is_terminal:
            return {**self._get_state_json(), 'message': 'Game Over'}
        
        self._skip_inactive_players()
        cp = int(self.state.current_player)
        
        if cp != AI_PLAYER:
            return {**self._get_state_json(), 'message': 'Not AI turn', 'error': True}
        
        # Roll dice first
        roll_result = self.roll_dice()

        # Always carry forward the rolled value as ai_roll for the client
        rolled_for_ai = roll_result.get('rolled') or roll_result.get('dice_roll', 0)

        if roll_result.get('triple_six') or roll_result.get('no_moves'):
            roll_result['ai_roll'] = rolled_for_ai
            return roll_result

        legal = roll_result.get('legal_moves', [])
        if not legal:
            roll_result['ai_roll'] = rolled_for_ai
            return roll_result

        # If only one legal move, take it
        if len(legal) == 1:
            game_logger.log(f"  AI has only 1 legal move (token {legal[0]}) — forced")
            return {**self.make_move(legal[0]), 'ai_roll': rolled_for_ai}

        # Model inference — pick encoder matching the loaded model
        if MODEL_VERSION in ('v11', 'v12'):
            state_tensor = ludo_cpp.encode_state_v10(self.state)
        elif MODEL_VERSION == 'v6_3':
            state_tensor = ludo_cpp.encode_state_v6_3(
                self.state, int(self.consecutive_sixes[cp])
            )
        else:
            state_tensor = ludo_cpp.encode_state_v6(self.state)
        legal_mask = np.zeros(4, dtype=np.float32)
        for m in legal:
            legal_mask[m] = 1.0

        with torch.no_grad():
            s_t = torch.from_numpy(np.asarray(state_tensor)).unsqueeze(0).to(self.device, dtype=torch.float32)
            m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device, dtype=torch.float32)
            # Run full model for richer logging.
            # V6.1: (policy, value); V6.3: (policy, value, aux); V11: (policy, win_prob, moves_remaining)
            full_out = self.model(s_t, m_t)
            policy = full_out[0].squeeze(0).cpu().numpy()
            value = float(full_out[1].squeeze().item())  # win_prob for v11, value for v6.x
            probs = policy
            action = int(policy.argmax())

        if action not in legal:
            action = random.choice(legal)

        # Log the AI's full decision — policy dist, value, chosen action
        prob_str = ', '.join(
            f"T{i}={probs[i]:.3f}" + ('*' if i == action else '') + ('' if legal_mask[i] > 0 else '(X)')
            for i in range(4)
        )
        game_logger.log(
            f"  AI decision: legal={legal}, chose token {action} "
            f"(policy: {prob_str}, value={value:+.3f})"
        )

        dice_val = roll_result['dice_roll']
        result = self.make_move(action)
        result['ai_roll'] = dice_val
        result['ai_probs'] = [round(float(p), 3) for p in probs]
        result['ai_chosen'] = action

        return result


# ── Flask App ──────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')
model, device = load_model()
game = GameManager(model, device)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

MODEL_INFO = {
    'v6_1': {
        'label': 'V6.1 Strategic',
        'subtitle': 'V6.1 Strategic · 78.8% eval · 3M params',
    },
    'v6_3': {
        'label': 'V6.3 (SL+RL)',
        'subtitle': 'V6.3 SL+RL · 77.8% eval · 3M params',
    },
    'v11': {
        'label': 'V11.1 ResTNet',
        'subtitle': 'V11.1 CNN+Attention · 79.05% eval (best) · 780K params',
    },
    'v12': {
        'label': 'V12 Token-Entity Attn',
        'subtitle': 'V12 CNN + token-entity attn · 81.00% eval (best) · 951K params',
    },
}


@app.route('/api/info')
def model_info():
    """Return info about the loaded model for the UI."""
    info = dict(MODEL_INFO[MODEL_VERSION])
    info['version'] = MODEL_VERSION
    info['param_count'] = count_parameters(model)
    return jsonify(info)


@app.route('/api/layout')
def layout():
    """Return the pre-computed board layout for rendering."""
    return jsonify(generate_board_layout())

@app.route('/api/new_game', methods=['POST'])
def new_game():
    state = game.reset()
    return jsonify(state)

@app.route('/api/state')
def get_state():
    return jsonify(game._get_state_json())

@app.route('/api/roll_dice', methods=['POST'])
def roll_dice():
    result = game.roll_dice()
    return jsonify(result)

@app.route('/api/move', methods=['POST'])
def make_move():
    data = request.get_json()
    token = int(data.get('token', -1))
    result = game.make_move(token)
    return jsonify(result)

@app.route('/api/ai_turn', methods=['POST'])
def ai_turn():
    result = game.ai_move()
    return jsonify(result)


# ── Eval Lens — Level 2: Review endpoints ─────────────────────
def _decorate_review_decision(d):
    """Add precomputed token_coords so the frontend can render the
    decision's mini-board without rerunning the C++ engine."""
    out = {
        'decision_id': d.get('decision_id'),
        'dice': d.get('dice'),
        'positions': d.get('positions', {}),
        'scores': d.get('scores', {}),
        'legal_tokens': d.get('legal_tokens', []),
        'v12_policy': d.get('v12_policy'),
        'v12_argmax': d.get('v12_argmax'),
        'v12_win_prob': d.get('v12_win_prob'),
        'v12_prob_of_human': d.get('v12_prob_of_human'),
        'human_token': d.get('human_token'),
        'agree': d.get('agree'),
        'interest_score': d.get('interest_score'),
        'move_count': d.get('move_count'),
    }

    # Pre-compute token_coords like _get_token_coords() but from the
    # serialized positions dict so review cards can render directly.
    coords = {}
    for player_str, pos_list in (d.get('positions') or {}).items():
        try:
            player = int(player_str)
        except (TypeError, ValueError):
            continue
        token_coords = []
        for t, p in enumerate(pos_list):
            r, c = get_board_coord(player, int(p), t)
            token_coords.append({
                'row': r, 'col': c,
                'pos': int(p),
                'in_base': int(p) == -1,
                'scored': int(p) == 99,
                'on_home_run': 50 < int(p) < 99,
            })
        coords[player_str] = token_coords
    out['token_coords'] = coords
    return out


@app.route('/api/review_decisions/<game_id>')
def review_decisions(game_id):
    """Return top-N most-interesting decisions from a finished game.

    Query: n (default 5). Sorted by `interest_score` desc — V12
    confidence × KL(human || V12). Highest = V12 was confident and
    disagreed with the human.
    """
    try:
        n = int(request.args.get('n', 5))
    except ValueError:
        n = 5
    n = max(1, min(50, n))

    decisions = decision_logger.read_decisions(game_id)
    decisions.sort(key=lambda d: d.get('interest_score', 0.0), reverse=True)
    top = decisions[:n]
    return jsonify({
        'game_id': game_id,
        'total_decisions': len(decisions),
        'returned': len(top),
        'decisions': [_decorate_review_decision(d) for d in top],
    })


@app.route('/api/review_decisions/latest')
def review_decisions_latest():
    """Convenience: returns top-N for the most recent game."""
    game_id = decision_logger.latest_game_id()
    if game_id is None:
        return jsonify({'game_id': None, 'decisions': [], 'total_decisions': 0,
                        'returned': 0})
    return review_decisions(game_id)


@app.route('/api/submit_rating', methods=['POST'])
def submit_rating():
    data = request.get_json() or {}
    game_id = data.get('game_id')
    decision_id = data.get('decision_id')
    label = data.get('label')

    if not isinstance(game_id, str) or not game_id:
        return jsonify({'ok': False, 'error': 'missing game_id'}), 400
    if not isinstance(decision_id, int):
        try:
            decision_id = int(decision_id)
        except (TypeError, ValueError):
            return jsonify({'ok': False, 'error': 'bad decision_id'}), 400
    if label not in RATING_LABELS:
        return jsonify({
            'ok': False,
            'error': f'label must be one of {sorted(RATING_LABELS)}',
        }), 400

    try:
        decision_logger.log_rating(game_id, decision_id, label)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
    return jsonify({'ok': True})


if __name__ == '__main__':
    print("\n" + "="*50)
    print(f"  AlphaLudo Play — Human vs AI ({MODEL_VERSION.upper()})")
    print("  Open: http://localhost:5050")
    print(f"  Switch model: LUDO_MODEL=v6_1|v6_3 python3 server.py")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5050, debug=False)
