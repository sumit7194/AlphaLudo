"""
AlphaLudo Play — Web Server for Human vs AI Ludo

Flask backend that manages game state via the C++ engine
and runs AI inference via the AlphaLudoV5 model (V6.1, 24-channel strategic encoding).
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
from model import AlphaLudoV5

# ── Configuration ──────────────────────────────────────────────
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_weights', 'model.pt')
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
    model = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    # Handle compiled model state dicts
    if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
        checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
    # Handle full checkpoint dict vs raw state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Play] Model loaded from {MODEL_PATH} ({param_count:,} params)")
    return model, device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    
    def _get_state_json(self):
        positions = {}
        scores = {}
        for player in [0, 2]:
            positions[str(player)] = [int(p) for p in self.state.player_positions[player]]
            scores[str(player)] = int(self.state.scores[player])
        
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
        
        # Track consecutive sixes
        if roll == 6:
            self.consecutive_sixes[cp] += 1
        else:
            self.consecutive_sixes[cp] = 0
        
        # Triple six penalty
        if self.consecutive_sixes[cp] >= 3:
            self.consecutive_sixes[cp] = 0
            self.state.current_dice_roll = 0
            next_p = (cp + 1) % 4
            while not self.state.active_players[next_p]:
                next_p = (next_p + 1) % 4
            self.state.current_player = next_p
            
            self.game_log.append(f"P{cp} rolled triple 6! Turn lost.")
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
        
        result = self._get_state_json()
        
        # Check if same player goes again (bonus turn)
        next_cp = int(self.state.current_player)
        result['bonus_turn'] = (next_cp == cp and not self.state.is_terminal)
        
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
            return {**self.make_move(legal[0]), 'ai_roll': int(self.state.current_dice_roll) or roll_result['dice_roll']}
        
        # Model inference
        state_tensor = ludo_cpp.encode_state_v6(self.state)
        legal_mask = np.zeros(4, dtype=np.float32)
        for m in legal:
            legal_mask[m] = 1.0
        
        with torch.no_grad():
            s_t = torch.from_numpy(state_tensor).unsqueeze(0).to(self.device, dtype=torch.float32)
            m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device, dtype=torch.float32)
            policy_logits = self.model.forward_policy_only(s_t, m_t)
            
            # Get probabilities for display
            probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            action = int(policy_logits.argmax(dim=1).item())
        
        if action not in legal:
            action = random.choice(legal)
        
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
    print("  AlphaLudo Play — Human vs AI")
    print("  Open: http://localhost:5050")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5050, debug=False)
