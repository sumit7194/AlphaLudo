"""Search-candidate review UI backend.

Loads candidates.jsonl produced by select_candidates.py, serves them to a
static HTML/JS frontend that renders each board state and lets the user
mark each decision as:
  - search-critical (must search this state)
  - search-helpful  (search would be nice but not essential)
  - search-unnecessary (model handles this fine without search)
  - dismiss
plus optional depth (1/2/3) for "if you want search, how deep?" and a
free-text comment.

Annotations append to annotations.jsonl in this folder; latest entry per
candidate id wins on reload.
"""
import json
import os
import sys
import time
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / 'static'
CANDIDATES_PATH = HERE / 'candidates.jsonl'
ANNOT_PATH = HERE / 'annotations.jsonl'

# Reuse layout helper from disagreement_review/server.py (same constants).
sys.path.insert(0, str(HERE.parent / 'disagreement_review'))
from server import build_board_layout, position_to_coord  # noqa: E402

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')


def load_candidates():
    out = []
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: out.append(json.loads(line))
                    except json.JSONDecodeError: pass
    return out


def load_annotations():
    annots = {}
    if not ANNOT_PATH.exists():
        return annots
    with open(ANNOT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                a = json.loads(line)
                key = str(a['id'])
                annots[key] = a
            except (json.JSONDecodeError, KeyError):
                pass
    return annots


def append_annotation(annot):
    annot['ts'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(ANNOT_PATH, 'a') as f:
        f.write(json.dumps(annot) + '\n')


@app.route('/')
def index():
    return send_from_directory(str(STATIC_DIR), 'index.html')


@app.route('/api/layout')
def api_layout():
    return jsonify(build_board_layout())


@app.route('/api/candidates')
def api_candidates():
    rows = load_candidates()
    annots = load_annotations()
    for r in rows:
        key = str(r.get('id'))
        if key in annots:
            a = annots[key]
            r['_annotation'] = {
                'mark': a.get('mark'),
                'depth': a.get('depth'),
                'comment': a.get('comment', ''),
                'ts': a.get('ts'),
            }
        # Re-build _coords if missing (in case position_to_coord wasn't available
        # at selection time — fallback for older candidates.jsonl files).
        if '_coords' not in r or any(c['coord'] == [0, 0] for c in r['_coords'].get(str(r['current_player']), [])):
            cp = r['current_player']
            opp = (cp + 2) % 4
            own = [int(x) for x in r['own_pos'].split(',')]
            opp_pos = [int(x) for x in r['opp_pos'].split(',')]
            r['_coords'] = {
                str(cp): [{'pos': p, 'coord': position_to_coord(cp, p, t)} for t, p in enumerate(own)],
                str(opp): [{'pos': p, 'coord': position_to_coord(opp, p, t)} for t, p in enumerate(opp_pos)],
            }
    return jsonify({'count': len(rows), 'rows': rows})


@app.route('/api/annotate', methods=['POST'])
def api_annotate():
    data = request.get_json(force=True)
    if 'id' not in data:
        return jsonify({'error': 'id required'}), 400
    valid_marks = (None, 'critical', 'helpful', 'unnecessary', 'dismiss')
    if data.get('mark') not in valid_marks:
        return jsonify({'error': f'mark must be one of {valid_marks}'}), 400
    if data.get('depth') not in (None, 1, 2, 3):
        return jsonify({'error': 'depth must be 1, 2, 3, or null'}), 400
    annot = {
        'id': data['id'],
        'mark': data.get('mark'),
        'depth': data.get('depth'),
        'comment': data.get('comment', ''),
    }
    append_annotation(annot)
    return jsonify({'ok': True, 'saved': annot})


if __name__ == '__main__':
    print('[Search Candidate Review] http://localhost:5052')
    print(f'[Search Candidate Review] candidates: {CANDIDATES_PATH}')
    print(f'[Search Candidate Review] annotations: {ANNOT_PATH}')
    app.run(host='0.0.0.0', port=5052, debug=False)
