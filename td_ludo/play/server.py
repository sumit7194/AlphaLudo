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

import os
import sys
import json
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


# ── Game State Manager ─────────────────────────────────────────
class GameManager:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.reset()

    def reset(self):
        self.state = ludo_cpp.create_initial_state_2p()
        self.consecutive_sixes = [0, 0, 0, 0]
        self.move_count = 0
        self.game_log = []
        self.pending_dice = None
        self.last_move_info = None
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
            }
        
        return {
            **self._get_state_json(),
            'legal_moves': [int(m) for m in legal],
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

        return result
    
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
        
        if roll_result.get('triple_six') or roll_result.get('no_moves'):
            return roll_result
        
        legal = roll_result.get('legal_moves', [])
        if not legal:
            return roll_result
        
        # If only one legal move, take it
        if len(legal) == 1:
            game_logger.log(f"  AI has only 1 legal move (token {legal[0]}) — forced")
            return {**self.make_move(legal[0]), 'ai_roll': int(self.state.current_dice_roll) or roll_result['dice_roll']}

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


if __name__ == '__main__':
    print("\n" + "="*50)
    print(f"  AlphaLudo Play — Human vs AI ({MODEL_VERSION.upper()})")
    print("  Open: http://localhost:5050")
    print(f"  Switch model: LUDO_MODEL=v6_1|v6_3 python3 server.py")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5050, debug=False)
