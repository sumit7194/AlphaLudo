"""Trigger-specificity check for bias_penalties.

Two tests against bias_penalties.compute_bias_penalties:

1. Manual-flag fire rate
   For each record in play/decision_logs/ai_disagreements.jsonl, reconstruct
   the state and run penalties on the model's actual choice. Expect HIGH
   trigger rate (we hand-flagged these as "model wrong" — at least one
   penalty should fire on most of them).

2. Baseline false-positive rate
   Sample N multi-legal decisions from v122_selfplay_100k.db that we have
   no reason to believe are bad. Run penalties on the model's actual choice.
   Expect LOW trigger rate (penalties should be specific, not blanket).

If (1) ≥ 70% AND (2) ≤ 25%, the penalty signal is well-targeted.
If (1) is low, penalties are missing real failures.
If (2) is high, penalties will over-correct during training.

Run:
    cd td_ludo && td_env/bin/python experiments/search_validation/penalty_specificity.py
"""
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TD_LUDO_DIR = HERE.parent.parent

sys.path.insert(0, str(TD_LUDO_DIR))

import td_ludo_cpp as ludo_cpp  # noqa: E402
from td_ludo.game.bias_penalties import compute_bias_penalties  # noqa: E402

DISAGREE_PATH = TD_LUDO_DIR / 'play' / 'decision_logs' / 'ai_disagreements.jsonl'
DB_PATH = TD_LUDO_DIR / 'experiments' / 'synthetic_rlhf' / 'v122_selfplay_100k.db'

PENALTY_KEYS = [
    'unlock_with_better',
    'missed_capture',
    'missed_finish',
    'left_safe',
    'advanced_into_danger',
]


def reconstruct(positions_dict, current_player, dice):
    """Build a state with recorded positions; scores from positions==99 count."""
    g = ludo_cpp.create_initial_state_2p()
    pp = list(g.player_positions)
    for pstr, plist in positions_dict.items():
        pp[int(pstr)] = list(int(x) for x in plist)
    g.player_positions = pp
    sc = list(g.scores)
    for pstr, plist in positions_dict.items():
        sc[int(pstr)] = sum(1 for x in plist if int(x) == 99)
    g.scores = sc
    g.current_player = int(current_player)
    g.current_dice_roll = int(dice)
    return g


def fired_breakdown(breakdown):
    """Returns list of penalty keys that fired (negative) on this breakdown."""
    return [k for k in PENALTY_KEYS if breakdown.get(k, 0.0) < -1e-9]


# =============================================================================
# Test 1: manual-flag fire rate
# =============================================================================
def test_manual_flags():
    if not DISAGREE_PATH.exists():
        print(f'[err] {DISAGREE_PATH} not found.')
        return None
    records = []
    with open(DISAGREE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Dedupe by (game_id, ai_decision_id)
    seen = {}
    for r in records:
        key = (r.get('game_id'), r.get('ai_decision_id'))
        # Keep with longest comment (most informative)
        if key not in seen or len((r.get('comment') or '').strip()) > len(
            (seen[key].get('comment') or '').strip()
        ):
            seen[key] = r
    flagged = [r for r in seen.values() if r.get('preferred_token') is not None]

    print(f'\n=== Test 1: Manual-flag fire rate ===')
    print(f'Loaded {len(flagged)} unique flagged states\n')

    fired_count = 0
    per_penalty = Counter()
    move_count_dist = []
    print(f'{"#":>2} {"ply":>4} {"dice":>4} {"chose":>5} {"pref":>5} {"total":>8}  fired_penalties')
    print('-' * 90)
    for i, r in enumerate(flagged, 1):
        cp = int(r['current_player'])
        dice = int(r['dice'])
        positions = r['positions']
        action = int(r['ai_chosen'])
        pref = int(r['preferred_token'])
        legal = r.get('legal_tokens') or []
        move_count = int(r.get('move_count', 0))
        move_count_dist.append(move_count)

        ctx = {
            'dice': dice,
            'legal_moves': legal,
            'action': action,
            'move_count': move_count,
        }

        # State reconstruction (used as 'state' — pre-move).
        # For penalty calc we also need 'next_state' (post-move). Apply the
        # action to get post-state. For comment-only without preferred_token
        # we still want to evaluate 'did this action trigger a penalty'.
        state = reconstruct(positions, cp, dice)
        try:
            next_state = ludo_cpp.apply_move(state, action)
        except Exception as e:
            print(f'  [skip] apply_move failed for #{i}: {e}')
            continue
        # Re-fetch a fresh state because apply_move mutates and we need
        # the pre-move state intact for the penalty fn.
        state = reconstruct(positions, cp, dice)

        total, bd = compute_bias_penalties(state, next_state, cp, ctx)
        fired = fired_breakdown(bd)
        if fired:
            fired_count += 1
            for k in fired:
                per_penalty[k] += 1
        print(f'{i:>2} {move_count:>4} {dice:>4} T{action}    T{pref}    {total:>+7.4f}  '
              f'{",".join(fired) if fired else "(none)"}')

    n = len(flagged)
    print(f'\nFired on at least one penalty: {fired_count}/{n} ({100*fired_count/n:.0f}%)')
    print(f'Per-penalty trigger counts: {dict(per_penalty)}')
    return fired_count, n, per_penalty


# =============================================================================
# Test 2: baseline false-positive rate
# =============================================================================
def test_baseline(n_sample=2000, seed=42):
    if not DB_PATH.exists():
        print(f'[err] {DB_PATH} not found.')
        return None

    print(f'\n=== Test 2: Baseline false-positive rate ===')
    print(f'Sampling {n_sample} multi-legal decisions from {DB_PATH.name}\n')

    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()

    # Need: dice, current_player, own_pos, opp_pos, legal_mask, action_chosen.
    # Multi-legal = legal_mask has >= 2 ones. Filter at SQL level for speed.
    # Sample: random ordering, limit n_sample. Use id % stride for speed.
    glen = dict(cur.execute('SELECT game_id, n_moves FROM games').fetchall())
    stride = max(1, 16_000_000 // (n_sample * 8))
    rows = cur.execute(f'''
        SELECT game_id, move_idx, current_player, dice,
               own_pos, opp_pos, legal_mask, action_chosen
        FROM decisions
        WHERE id % {stride} = 0
          AND (length(replace(legal_mask, '0', '')) >= 2)
        ORDER BY RANDOM()
        LIMIT {n_sample}
    ''').fetchall()
    print(f'Pulled {len(rows)} candidate rows.')

    fired_count = 0
    per_penalty = Counter()
    move_count_buckets = Counter()
    fired_by_bucket = Counter()
    skipped = 0

    for gid, midx, cp, dice, own_str, opp_str, lm_str, action in rows:
        nm = glen.get(gid)
        if not nm:
            continue
        own = [int(x) for x in own_str.split(',')]
        opp_pos = [int(x) for x in opp_str.split(',')]
        legal = [t for t in range(4) if lm_str[t] == '1']
        if len(legal) < 2:
            skipped += 1
            continue

        # Build positions dict (own_pos is current_player's tokens, opp_pos
        # is the other player). In 2P, opponent of cp is (cp + 2) % 4.
        opp_player = (cp + 2) % 4
        positions = {str(cp): own, str(opp_player): opp_pos}

        try:
            state = reconstruct(positions, cp, dice)
            next_state = ludo_cpp.apply_move(state, action)
            state = reconstruct(positions, cp, dice)
        except Exception:
            skipped += 1
            continue

        ctx = {
            'dice': dice,
            'legal_moves': legal,
            'action': action,
            'move_count': midx,
        }
        total, bd = compute_bias_penalties(state, next_state, cp, ctx)
        fired = fired_breakdown(bd)

        # Bucket by progress decile
        prog = midx / max(1, nm)
        bucket = min(9, int(prog * 10))
        move_count_buckets[bucket] += 1

        if fired:
            fired_count += 1
            fired_by_bucket[bucket] += 1
            for k in fired:
                per_penalty[k] += 1

    n_actual = len(rows) - skipped
    print(f'\nProcessed {n_actual} multi-legal decisions ({skipped} skipped)')
    if n_actual == 0:
        return None
    print(f'Fired on at least one penalty: {fired_count}/{n_actual} '
          f'({100 * fired_count / n_actual:.1f}%)')
    print(f'\nPer-penalty trigger counts:')
    for k in PENALTY_KEYS:
        c = per_penalty.get(k, 0)
        pct = 100 * c / n_actual
        print(f'  {k:<22} {c:>6,} ({pct:5.2f}%)')

    print(f'\nTrigger rate by progress decile:')
    print(f'{"prog":>6} {"n":>8} {"fired":>8} {"rate":>8}')
    for b in range(10):
        n = move_count_buckets.get(b, 0)
        f = fired_by_bucket.get(b, 0)
        rate = 100 * f / n if n else 0
        print(f'  {b/10:.1f}-{(b+1)/10:.1f} {n:>8,} {f:>8,} {rate:>7.1f}%')
    return fired_count, n_actual, per_penalty


# =============================================================================
# Verdict
# =============================================================================
def main():
    res1 = test_manual_flags()
    res2 = test_baseline(n_sample=2000)

    print('\n' + '=' * 70)
    print('VERDICT')
    print('=' * 70)
    if res1 and res2:
        flagged_fire_rate = res1[0] / res1[1]
        baseline_fire_rate = res2[0] / res2[1]
        print(f'Manual-flag fire rate:        {100*flagged_fire_rate:5.1f}%  (target ≥ 70%)')
        print(f'Baseline false-positive rate: {100*baseline_fire_rate:5.1f}%  (target ≤ 25%)')
        print()
        verdict = []
        if flagged_fire_rate >= 0.70:
            verdict.append('✓ Captures hand-flagged failures')
        else:
            verdict.append(f'✗ Misses too many flagged failures '
                           f'({int(res1[0])}/{res1[1]}) — penalties may be too narrow')
        if baseline_fire_rate <= 0.25:
            verdict.append('✓ Specificity is acceptable')
        else:
            verdict.append(f'✗ False-positive rate too high ({100*baseline_fire_rate:.1f}%) '
                           f'— penalties will over-correct during training')
        for v in verdict:
            print('  ' + v)


if __name__ == '__main__':
    main()
