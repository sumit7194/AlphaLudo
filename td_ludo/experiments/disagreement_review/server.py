"""Disagreement review UI backend.

Loads disagreements_real.jsonl produced by scripts/analyze_disagreements.py,
serves them to a static HTML/JS frontend that renders the board state and
lets the user cycle through, comment, and mark each as
important/normal/dismiss.

Annotations are appended to annotations.jsonl in this folder. Re-marking
the same decision overwrites the prior entry on disk-load (latest entry
wins).

Run from td_ludo/ root:
  python -m experiments.disagreement_review.server
"""
import glob
import json
import os
import sys
import time
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / 'static'
LOGS_DIR = HERE.parent.parent / 'play' / 'decision_logs'
DISAGREE_PATH = LOGS_DIR / 'disagreements_real.jsonl'  # legacy fallback only
ANNOT_PATH = HERE / 'annotations.jsonl'

# Filter logic (mirrors scripts/analyze_disagreements.py — kept in sync).
HUMAN_PLAYER = 0
BASE_POS = -1
HOME_POS = 99


def _is_interchangeable(d, prob_tolerance=1e-3):
    cp = d.get('current_player', HUMAN_PLAYER)
    own_positions = d.get('positions', {}).get(str(cp), [])
    if len(own_positions) < 4:
        return False
    human_pick = d.get('human_token')
    ai_pick = d.get('v12_argmax')
    if human_pick is None or ai_pick is None or human_pick == ai_pick:
        return False
    h_pos = own_positions[human_pick]
    if h_pos == BASE_POS or h_pos == HOME_POS:
        return False
    stack_indices = [t for t in range(4) if own_positions[t] == h_pos]
    if len(stack_indices) < 2:
        return False
    if ai_pick not in stack_indices:
        return False
    policy = d.get('v12_policy', [0, 0, 0, 0])
    stack_probs = [policy[i] for i in stack_indices]
    if max(stack_probs) - min(stack_probs) > prob_tolerance:
        return False
    return True


def aggregate_disagreements_live():
    """Re-scan ALL decisions_*.jsonl files NOW. Auto-pulls in any games
    played since startup. Returns sorted by interest_score desc."""
    files = sorted(glob.glob(str(LOGS_DIR / 'decisions_*.jsonl')))
    real = []
    for path in files:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get('current_player') != HUMAN_PLAYER:
                        continue
                    if d.get('agree', True):
                        continue
                    if _is_interchangeable(d):
                        continue
                    real.append(d)
        except IOError:
            continue
    real.sort(key=lambda r: r.get('interest_score', 0.0), reverse=True)
    return real

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')

# ── Board geometry (pure Python, mirrors src/game.cpp constants) ─────────
PATH_COORDS_P0 = [
    (6,1),(6,2),(6,3),(6,4),(6,5),
    (5,6),(4,6),(3,6),(2,6),(1,6),(0,6),
    (0,7),(0,8),
    (1,8),(2,8),(3,8),(4,8),(5,8),
    (6,9),(6,10),(6,11),(6,12),(6,13),(6,14),
    (7,14),(8,14),
    (8,13),(8,12),(8,11),(8,10),(8,9),
    (9,8),(10,8),(11,8),(12,8),(13,8),(14,8),
    (14,7),(14,6),
    (13,6),(12,6),(11,6),(10,6),(9,6),
    (8,5),(8,4),(8,3),(8,2),(8,1),(8,0),
    (7,0),
]
HOME_RUN_P0 = [(7,1),(7,2),(7,3),(7,4),(7,5)]
HOME_COORD = (7,7)

# Fixed BASE_COORDS — matches the post-encoder-fix layout in src/game.cpp.
# After per-player rotation, slot t for ANY player lands at canonical
# cell t (T0→(2,2), T1→(2,3), T2→(3,2), T3→(3,3)).
BASE_COORDS = {
    0: [(2, 2),  (2, 3),  (3, 2),  (3, 3)],
    1: [(2, 12), (3, 12), (2, 11), (3, 11)],
    2: [(12, 12),(12, 11),(11, 12),(11, 11)],
    3: [(12, 2), (11, 2), (12, 3), (11, 3)],
}
SAFE_INDICES = {0, 8, 13, 21, 26, 34, 39, 47}


def rotate_90cw(r, c):
    return (c, 14 - r)


def get_board_coord(player, pos, token_index=0):
    if pos == -1:
        return BASE_COORDS[player][token_index]
    if pos == 99:
        local = HOME_COORD
    elif pos > 50:
        idx = pos - 51
        local = HOME_RUN_P0[idx] if idx < 5 else HOME_COORD
    else:
        local = PATH_COORDS_P0[pos]
    r, c = local
    for _ in range(player):
        r, c = rotate_90cw(r, c)
    return (r, c)


def build_board_layout():
    layout = {
        'board_size': 15,
        'path_squares': [],
        'safe_squares': [],
        'home_runs': {},
        'bases': {str(p): [list(c) for c in v] for p, v in BASE_COORDS.items()},
        'spawn_squares': {},
        'home_center': list(HOME_COORD),
    }
    all_path = set()
    safe_seen = set()
    for player in [0, 1, 2, 3]:
        for pos in range(0, 51):
            all_path.add(get_board_coord(player, pos))
        hrs = []
        for pos in range(51, 56):
            r, c = get_board_coord(player, pos)
            hrs.append([r, c])
            all_path.add((r, c))
        layout['home_runs'][str(player)] = hrs
        layout['spawn_squares'][str(player)] = list(get_board_coord(player, 0))
        for s in SAFE_INDICES:
            rel = (s - 13 * player) % 52
            if rel <= 50:
                safe_seen.add(get_board_coord(player, rel))
    layout['path_squares'] = [list(p) for p in sorted(all_path)]
    layout['safe_squares'] = [list(s) for s in safe_seen]
    return layout


def position_to_coord(player, pos, token_index=0):
    """Helper: render-ready coord for a token position."""
    return list(get_board_coord(player, pos, token_index))


# ── Disagreement data ────────────────────────────────────────────────────
def load_disagreements():
    if not DISAGREE_PATH.exists():
        return []
    out = []
    with open(DISAGREE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except json.JSONDecodeError: pass
    return out


def load_annotations():
    """Load latest annotation per (game_id, decision_id). Later entries win."""
    annots = {}
    if not ANNOT_PATH.exists():
        return annots
    with open(ANNOT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                a = json.loads(line)
                key = f"{a['game_id']}/{a['decision_id']}"
                annots[key] = a
            except (json.JSONDecodeError, KeyError):
                pass
    return annots


def append_annotation(annot):
    annot['ts'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(ANNOT_PATH, 'a') as f:
        f.write(json.dumps(annot) + '\n')


# ── Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(str(STATIC_DIR), 'index.html')


@app.route('/api/layout')
def api_layout():
    return jsonify(build_board_layout())


@app.route('/api/disagreements')
def api_disagreements():
    # Live-scan decisions_*.jsonl every request — picks up new gameplay
    # automatically with no manual re-run of analyze_disagreements.py.
    rows = aggregate_disagreements_live()
    annots = load_annotations()
    for r in rows:
        key = f"{r.get('game_id','')}/{r.get('decision_id','')}"
        if key in annots:
            r['_annotation'] = {
                'mark': annots[key].get('mark'),
                'comment': annots[key].get('comment', ''),
                'ts': annots[key].get('ts'),
            }
        # Pre-compute board coords for every token so the FE can place pieces.
        positions = r.get('positions', {})
        coord_map = {}
        for pstr, pos_list in positions.items():
            p = int(pstr)
            coord_map[pstr] = [
                {'pos': int(pos), 'coord': position_to_coord(p, int(pos), t)}
                for t, pos in enumerate(pos_list)
            ]
        r['_coords'] = coord_map
    return jsonify({'count': len(rows), 'rows': rows})


@app.route('/api/annotate', methods=['POST'])
def api_annotate():
    data = request.get_json(force=True)
    if 'game_id' not in data or 'decision_id' not in data:
        return jsonify({'error': 'game_id + decision_id required'}), 400
    if data.get('mark') not in (None, 'important', 'normal', 'dismiss'):
        return jsonify({'error': 'mark must be important|normal|dismiss|null'}), 400
    annot = {
        'game_id': data['game_id'],
        'decision_id': data['decision_id'],
        'mark': data.get('mark'),
        'comment': data.get('comment', ''),
    }
    append_annotation(annot)
    return jsonify({'ok': True, 'saved': annot})


if __name__ == '__main__':
    print(f'[Disagreement Review] http://localhost:5051')
    print(f'[Disagreement Review] data: {DISAGREE_PATH}')
    print(f'[Disagreement Review] annotations: {ANNOT_PATH}')
    app.run(host='0.0.0.0', port=5051, debug=False)
