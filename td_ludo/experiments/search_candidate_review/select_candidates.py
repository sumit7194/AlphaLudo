"""Select ~100 diverse search-candidate decisions from the 100K self-play DB.

Diversity dimensions (stratified):
  - Game phase: early, early_mid, mid, endgame
  - Uncertainty: high (max_prob ≤ 0.4), med (0.4-0.7), low (0.7-0.9)
  - Tactical flags (computed): capture_available, danger_present
  - Position quality: V12.2 win_prob bucket (losing/neutral/winning)

Final pool ≈ 100 candidates that span (phase × uncertainty × tactical-flag)
combinations, plus a small "edge case" group of genuinely ambiguous
decisions, plus a random-serendipity subset.

Each candidate is enriched with:
  - capture_available (bool)  — Ch 22 of v11 encoding has any non-zero cell
  - leading_in_danger (bool)  — Ch 21 of v11 encoding has any non-zero cell
  - phase / uncertainty bucket labels
  - bucket origin tag (so you can see where in the stratification it came from)

Output: experiments/search_candidate_review/candidates.jsonl

Usage:
  python -m experiments.search_candidate_review.select_candidates \\
      --db experiments/synthetic_rlhf/v122_selfplay_100k.db \\
      --target-count 100
"""
import argparse
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import td_ludo_cpp as ludo_cpp

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / 'candidates.jsonl'

BASE_POS = -1
HOME_POS = 99


def parse_pos(s):
    return [int(x) for x in s.split(',')]


def reconstruct_state(own_pos, opp_pos, current_player, dice):
    g = ludo_cpp.create_initial_state_2p()
    pp = list(g.player_positions)
    pp[current_player] = list(own_pos)
    pp[(current_player + 2) % 4] = list(opp_pos)
    g.player_positions = pp
    g.current_player = current_player
    g.current_dice_roll = dice
    return g


def compute_tactical_flags(own_pos, opp_pos, current_player, dice):
    """Reconstruct state, encode v11, check Ch 21 (danger) and Ch 22 (capture)."""
    g = reconstruct_state(own_pos, opp_pos, current_player, dice)
    enc = np.asarray(ludo_cpp.encode_state_v11(g), dtype=np.float32)
    capture_available = bool(enc[22].sum() > 0)
    danger_present    = bool(enc[21].sum() > 0)
    return capture_available, danger_present


def categorize_phase(n_base, n_home):
    if n_base >= 3:  return 'early'
    if n_base >= 1:  return 'early_mid'
    if n_home == 0:  return 'mid'
    return 'endgame'


def is_artificial_spawn_uncertainty(own_pos, dice, legal_mask, policy):
    """True when the model's uncertainty is just spawn-choice noise.

    When dice=6 with ≥2 base tokens legal, choosing which base-token to
    spawn is functionally equivalent (they all land on the same spawn cell).
    Same logic for same-cell stacks: multiple legal own tokens on the
    same cell are interchangeable.

    We drop the candidate if MOST of the model's policy mass (>0.85) sits
    on these interchangeable-token sets. Real strategic uncertainty (e.g.,
    "spawn vs advance an existing token") preserves at least 15% mass on
    NON-equivalent options, so survives the filter.
    """
    # Build legal-token list with their positions
    legal_indices = [t for t in range(4) if legal_mask[t] == '1']
    if len(legal_indices) < 2:
        return False  # forced move — not in pool anyway

    # Group legal tokens by position (interchangeable groups)
    pos_groups = {}
    for t in legal_indices:
        p = own_pos[t]
        # Treat all base tokens as one group ONLY when dice=6 (spawn equivalent).
        # When dice != 6, base tokens aren't legal so this branch never fires.
        key = ('base_spawn', dice) if (p == BASE_POS and dice == 6) else ('cell', p)
        pos_groups.setdefault(key, []).append(t)

    # Find groups with ≥2 interchangeable tokens
    interchange_mass = 0.0
    has_any_interchange = False
    for key, group in pos_groups.items():
        if len(group) >= 2:
            has_any_interchange = True
            interchange_mass += sum(policy[t] for t in group)

    if not has_any_interchange:
        return False

    # If the model's mass is overwhelmingly on the interchangeable group(s),
    # the apparent uncertainty is spatial-bias noise, not strategic doubt.
    return interchange_mass > 0.85


def categorize_uncertainty(max_prob):
    if max_prob <= 0.4: return 'high_uncertainty'
    if max_prob <= 0.7: return 'med_uncertainty'
    return 'low_uncertainty'


def categorize_winprob(wp):
    if wp < 0.4: return 'losing'
    if wp > 0.6: return 'winning'
    return 'neutral'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='experiments/synthetic_rlhf/v122_selfplay_100k.db')
    p.add_argument('--target-count', type=int, default=100,
                   help='Approximate target candidate count')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--per-cell-sample', type=int, default=200,
                   help='Per-cell SQL sample size before tactical filtering. '
                        'Higher = better odds of finding capture/danger examples.')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    if not Path(args.db).exists():
        raise SystemExit(f'DB not found: {args.db}')

    conn = sqlite3.connect(args.db)
    print(f'[select] DB: {args.db}')

    # Build candidate ID pool per (phase, uncertainty) cell via SQL.
    # Stratification cells: 4 phases × 3 uncertainty levels = 12 cells.
    PHASE_SQL = {
        'early':       'n_own_at_base >= 3',
        'early_mid':   'n_own_at_base IN (1, 2)',
        'mid':         'n_own_at_base = 0 AND n_own_at_home = 0',
        'endgame':     'n_own_at_home >= 1',
    }
    UNCERTAINTY_SQL = {
        # SQLite has no built-in MAX-of-columns; use chained MAX nested.
        'high_uncertainty': 'MAX(MAX(policy_t0, policy_t1), MAX(policy_t2, policy_t3)) <= 0.4',
        'med_uncertainty':  'MAX(MAX(policy_t0, policy_t1), MAX(policy_t2, policy_t3)) > 0.4 AND MAX(MAX(policy_t0, policy_t1), MAX(policy_t2, policy_t3)) <= 0.7',
        'low_uncertainty':  'MAX(MAX(policy_t0, policy_t1), MAX(policy_t2, policy_t3)) > 0.7 AND MAX(MAX(policy_t0, policy_t1), MAX(policy_t2, policy_t3)) <= 0.9',
    }
    MULTI_LEGAL = "LENGTH(REPLACE(legal_mask, '0', '')) > 1"

    # Per-cell raw pool (oversample so we can apply tactical-flag filtering after).
    print('[select] pulling oversampled per-cell candidates from SQL...')
    raw = {}
    for phase, p_sql in PHASE_SQL.items():
        for unc, u_sql in UNCERTAINTY_SQL.items():
            cur = conn.execute(f'''
                SELECT id, game_id, move_idx, current_player, dice,
                       own_pos, opp_pos, n_own_at_base, n_own_on_track, n_own_at_home,
                       n_opp_at_base, n_opp_on_track, n_opp_at_home, legal_mask,
                       policy_t0, policy_t1, policy_t2, policy_t3, win_prob,
                       moves_remaining, action_chosen, winner
                FROM decisions
                WHERE {MULTI_LEGAL} AND {p_sql} AND {u_sql}
                ORDER BY RANDOM() LIMIT ?
            ''', (args.per_cell_sample,))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            raw[(phase, unc)] = rows
            print(f'  {phase:10}/{unc:18}: {len(rows)} raw candidates')

    # Enrich + filter out artificial uncertainty (spawn/stack interchangeable).
    print('[select] enriching with capture/danger flags + filtering noise...')
    enriched = []
    seen_ids = set()
    n_filtered = 0
    for cell_key, rows in raw.items():
        for r in rows:
            if r['id'] in seen_ids:
                continue
            seen_ids.add(r['id'])
            own = parse_pos(r['own_pos'])
            opp = parse_pos(r['opp_pos'])
            policy = [r['policy_t0'], r['policy_t1'], r['policy_t2'], r['policy_t3']]
            # Drop spawn/stack interchangeable cases — search useless there.
            if is_artificial_spawn_uncertainty(own, r['dice'], r['legal_mask'], policy):
                n_filtered += 1
                continue
            cap, dng = compute_tactical_flags(own, opp, r['current_player'], r['dice'])
            r['capture_available'] = cap
            r['danger_present']    = dng
            r['phase']             = cell_key[0]
            r['uncertainty']       = cell_key[1]
            r['win_bucket']        = categorize_winprob(r['win_prob'])
            r['_max_prob'] = max(policy)
            enriched.append(r)
    print(f'  enriched: {len(enriched)} unique decisions ({n_filtered} dropped as artificial)')

    # Stratified pick: walk through (phase × uncertainty × tactical-pattern) cells,
    # pick 1-2 each for diversity. Tactical patterns: none / capture / danger / both.
    TACTICAL_PATTERNS = ['none', 'capture', 'danger', 'both']

    def matches_tac(r, pat):
        if pat == 'none':    return not r['capture_available'] and not r['danger_present']
        if pat == 'capture': return r['capture_available'] and not r['danger_present']
        if pat == 'danger':  return not r['capture_available'] and r['danger_present']
        if pat == 'both':    return r['capture_available'] and r['danger_present']
        return False

    # Group enriched by (phase, uncertainty, tactical_pattern)
    cells = defaultdict(list)
    for r in enriched:
        for pat in TACTICAL_PATTERNS:
            if matches_tac(r, pat):
                cells[(r['phase'], r['uncertainty'], pat)].append(r)
                break  # each row in exactly one tactical bucket

    print()
    print('[select] cell counts (phase × uncertainty × tactical):')
    for k in sorted(cells.keys()):
        print(f'  {str(k):60s} {len(cells[k]):>4}')

    # Pick 1 candidate per cell (some cells may be empty). 4×3×4 = 48 max.
    selected = []
    selected_ids = set()
    for cell_key, candidates in cells.items():
        if not candidates: continue
        pick = random.choice(candidates)
        if pick['id'] in selected_ids: continue
        pick['_bucket'] = ' / '.join(cell_key)
        selected.append(pick)
        selected_ids.add(pick['id'])
    print(f'[select] one-per-cell pick: {len(selected)} candidates')

    # Add second pick from each cell that has ≥2 (more density per archetype)
    for cell_key, candidates in cells.items():
        if len(candidates) < 2: continue
        for c in candidates:
            if c['id'] not in selected_ids:
                c['_bucket'] = ' / '.join(cell_key) + ' (#2)'
                selected.append(c)
                selected_ids.add(c['id'])
                break
    print(f'[select] after second-pick: {len(selected)} candidates')

    # Edge cases: very high uncertainty (max_prob ≤ 0.35) — true coin-flips
    cur = conn.execute(f'''
        SELECT id, game_id, move_idx, current_player, dice,
               own_pos, opp_pos, n_own_at_base, n_own_on_track, n_own_at_home,
               n_opp_at_base, n_opp_on_track, n_opp_at_home, legal_mask,
               policy_t0, policy_t1, policy_t2, policy_t3, win_prob,
               moves_remaining, action_chosen, winner
        FROM decisions
        WHERE {MULTI_LEGAL}
        AND MAX(MAX(policy_t0, policy_t1), MAX(policy_t2, policy_t3)) <= 0.35
        ORDER BY RANDOM() LIMIT 30
    ''')
    cols = [d[0] for d in cur.description]
    edge_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    n_edge_added = 0
    for r in edge_rows:
        if r['id'] in selected_ids: continue
        own = parse_pos(r['own_pos']); opp = parse_pos(r['opp_pos'])
        policy = [r['policy_t0'], r['policy_t1'], r['policy_t2'], r['policy_t3']]
        if is_artificial_spawn_uncertainty(own, r['dice'], r['legal_mask'], policy):
            continue
        cap, dng = compute_tactical_flags(own, opp, r['current_player'], r['dice'])
        r['capture_available'] = cap; r['danger_present'] = dng
        r['phase'] = categorize_phase(r['n_own_at_base'], r['n_own_at_home'])
        r['uncertainty'] = categorize_uncertainty(max(r['policy_t0'], r['policy_t1'], r['policy_t2'], r['policy_t3']))
        r['win_bucket'] = categorize_winprob(r['win_prob'])
        r['_max_prob'] = max(r['policy_t0'], r['policy_t1'], r['policy_t2'], r['policy_t3'])
        r['_bucket'] = 'edge_case (true coin-flip)'
        selected.append(r)
        selected_ids.add(r['id'])
        n_edge_added += 1
        if len(selected) >= args.target_count:
            break
    print(f'[select] added {n_edge_added} edge-case candidates → {len(selected)} total')

    # Late-game random sample — n_own_at_home >= 2 means player is closing out.
    # Search may matter MORE here (small mistakes lose the lead), and the
    # archetypes here look different from generic mid-game.
    cur = conn.execute(f'''
        SELECT id, game_id, move_idx, current_player, dice,
               own_pos, opp_pos, n_own_at_base, n_own_on_track, n_own_at_home,
               n_opp_at_base, n_opp_on_track, n_opp_at_home, legal_mask,
               policy_t0, policy_t1, policy_t2, policy_t3, win_prob,
               moves_remaining, action_chosen, winner
        FROM decisions
        WHERE {MULTI_LEGAL} AND n_own_at_home >= 2
        AND MAX(MAX(policy_t0, policy_t1), MAX(policy_t2, policy_t3)) <= 0.85
        ORDER BY RANDOM() LIMIT 80
    ''')
    cols = [d[0] for d in cur.description]
    late_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    n_late_added = 0
    for r in late_rows:
        if r['id'] in selected_ids: continue
        own = parse_pos(r['own_pos']); opp = parse_pos(r['opp_pos'])
        policy = [r['policy_t0'], r['policy_t1'], r['policy_t2'], r['policy_t3']]
        if is_artificial_spawn_uncertainty(own, r['dice'], r['legal_mask'], policy):
            continue
        cap, dng = compute_tactical_flags(own, opp, r['current_player'], r['dice'])
        r['capture_available'] = cap; r['danger_present'] = dng
        r['phase'] = categorize_phase(r['n_own_at_base'], r['n_own_at_home'])
        r['uncertainty'] = categorize_uncertainty(max(policy))
        r['win_bucket'] = categorize_winprob(r['win_prob'])
        r['_max_prob'] = max(policy)
        r['_bucket'] = f'late-game random (n_home={r["n_own_at_home"]})'
        selected.append(r); selected_ids.add(r['id'])
        n_late_added += 1
        if n_late_added >= 15:
            break
    print(f'[select] added {n_late_added} late-game random candidates → {len(selected)} total')

    # Trim or pad to target_count
    if len(selected) > args.target_count:
        random.shuffle(selected)
        selected = selected[:args.target_count]
    print(f'[select] final candidate count: {len(selected)}')

    # Pre-compute board coords for the FE (mirror disagreement_review server code)
    sys.path.insert(0, str(HERE.parent / 'disagreement_review'))
    try:
        from server import position_to_coord
    except ImportError:
        # If standalone, replicate inline
        def position_to_coord(player, pos, t):
            return [0, 0]

    for r in selected:
        own = parse_pos(r['own_pos']); opp = parse_pos(r['opp_pos'])
        cp = r['current_player']
        coord_map = {
            str(cp): [{'pos': p, 'coord': position_to_coord(cp, p, t)} for t, p in enumerate(own)],
            str((cp + 2) % 4): [{'pos': p, 'coord': position_to_coord((cp + 2) % 4, p, t)} for t, p in enumerate(opp)],
        }
        r['_coords'] = coord_map

    # Write
    with open(OUT_PATH, 'w') as f:
        for r in selected:
            f.write(json.dumps(r) + '\n')
    print(f'[select] wrote {OUT_PATH}')

    # Quick summary
    print()
    print('[select] SUMMARY by phase:')
    for phase in ['early', 'early_mid', 'mid', 'endgame']:
        n = sum(1 for r in selected if r['phase'] == phase)
        print(f'  {phase:>10}: {n}')
    print('[select] SUMMARY by uncertainty:')
    for u in ['high_uncertainty', 'med_uncertainty', 'low_uncertainty']:
        n = sum(1 for r in selected if r['uncertainty'] == u)
        print(f'  {u:>18}: {n}')
    print('[select] SUMMARY by tactical:')
    n_cap = sum(1 for r in selected if r['capture_available'])
    n_dng = sum(1 for r in selected if r['danger_present'])
    n_both = sum(1 for r in selected if r['capture_available'] and r['danger_present'])
    print(f'  capture_available: {n_cap}')
    print(f'  danger_present:    {n_dng}')
    print(f'  both:              {n_both}')

    conn.close()


if __name__ == '__main__':
    main()
