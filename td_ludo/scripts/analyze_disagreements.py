"""Aggregate ALL decision logs → filter to human-vs-AI disagreements →
annotate same-cell-stack false positives → sort by interest_score.

A disagreement is "real" when V12.2's argmax token differs from the
token the human picked AND the human's pick was NOT interchangeable
with the AI's pick (i.e., they don't sit on the same board cell, in
which case moving either is functionally identical).

Outputs:
  play/decision_logs/all_decisions.jsonl       - merged, every game in time order
  play/decision_logs/disagreements_real.jsonl  - filtered, sorted by interest_score
  play/decision_logs/disagreements_summary.md  - human-readable top-30 with context

Usage:
  python scripts/analyze_disagreements.py
"""
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO / 'play' / 'decision_logs'

ALL_OUT      = LOGS_DIR / 'all_decisions.jsonl'
DISAGREE_OUT = LOGS_DIR / 'disagreements_real.jsonl'
SUMMARY_MD   = LOGS_DIR / 'disagreements_summary.md'

# Constants matching td_ludo encoder convention
BASE_POS = -1
HOME_POS = 99
AI_PLAYER = 2
HUMAN_PLAYER = 0


def load_all_decisions():
    """Yield decisions from every decisions_*.jsonl in time-order."""
    files = sorted(glob.glob(str(LOGS_DIR / 'decisions_*.jsonl')))
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def is_interchangeable(d):
    """True if the human's picked token sits on the same board cell as
    the AI's picked token (or any other legal token). Moving either is
    functionally identical, so the disagreement is a UI artifact, not
    a real strategic choice."""
    cp = d.get('current_player', HUMAN_PLAYER)
    own_positions = d.get('positions', {}).get(str(cp), [])
    if len(own_positions) < 4:
        return False
    legal = set(d.get('legal_tokens', []))
    human_pick = d.get('human_token')
    ai_pick = d.get('v12_argmax')
    if human_pick is None or ai_pick is None or human_pick == ai_pick:
        return False
    # Both picks must be at non-base, non-home positions to be on same "cell"
    h_pos = own_positions[human_pick]
    a_pos = own_positions[ai_pick]
    if h_pos == BASE_POS or h_pos == HOME_POS:
        return False
    if a_pos == BASE_POS or a_pos == HOME_POS:
        return False
    return h_pos == a_pos


def stack_count_at_human_cell(d):
    """How many own tokens are stacked at the human's picked cell."""
    cp = d.get('current_player', HUMAN_PLAYER)
    own_positions = d.get('positions', {}).get(str(cp), [])
    if len(own_positions) < 4:
        return 0
    human_pick = d.get('human_token')
    if human_pick is None:
        return 0
    h_pos = own_positions[human_pick]
    if h_pos == BASE_POS or h_pos == HOME_POS:
        return 0
    return sum(1 for p in own_positions if p == h_pos)


def fmt_pos(p):
    return 'B' if p == BASE_POS else 'H' if p == HOME_POS else str(p)


def aggregate_and_filter():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_human_decisions = 0
    n_disagree = 0
    n_interchangeable = 0
    real_disagreements = []

    with open(ALL_OUT, 'w') as f_all:
        for d in load_all_decisions():
            n_total += 1
            f_all.write(json.dumps(d) + '\n')

            # Restrict to human turns only (cp == HUMAN_PLAYER == 0)
            if d.get('current_player') != HUMAN_PLAYER:
                continue
            n_human_decisions += 1
            if d.get('agree', True):
                continue
            n_disagree += 1
            interchangeable = is_interchangeable(d)
            d['_interchangeable'] = interchangeable
            d['_stack_size'] = stack_count_at_human_cell(d)
            if interchangeable:
                n_interchangeable += 1
                continue
            real_disagreements.append(d)

    # Sort by interest_score descending
    real_disagreements.sort(
        key=lambda r: r.get('interest_score', 0.0), reverse=True
    )

    with open(DISAGREE_OUT, 'w') as f:
        for d in real_disagreements:
            f.write(json.dumps(d) + '\n')

    print(f'Stats:')
    print(f'  Total decisions logged:   {n_total}')
    print(f'  Human decisions:          {n_human_decisions}')
    print(f'  Disagreements (raw):      {n_disagree}')
    print(f'  └ Interchangeable (same-cell stack, dropped): {n_interchangeable}')
    print(f'  └ Real disagreements:     {len(real_disagreements)}')
    print(f'  Wrote {ALL_OUT.relative_to(REPO)}')
    print(f'  Wrote {DISAGREE_OUT.relative_to(REPO)} ({len(real_disagreements)} rows)')

    write_summary_md(real_disagreements, n_total, n_human_decisions,
                       n_disagree, n_interchangeable)
    print(f'  Wrote {SUMMARY_MD.relative_to(REPO)}')


def write_summary_md(rows, n_total, n_human, n_disagree, n_inter):
    lines = []
    lines.append('# V12.2 vs Human — Real Disagreements (Top 30)')
    lines.append('')
    lines.append('Aggregated across all play sessions. Disagreement = AI\'s argmax')
    lines.append('token differs from human\'s pick, AND the human\'s pick is NOT')
    lines.append('on the same board cell as the AI\'s pick (which would make the')
    lines.append('move functionally interchangeable — UI false positive).')
    lines.append('')
    lines.append('Sorted by `interest_score = max(v12_policy) × KL(v12 ‖ human)` —')
    lines.append('high values mean V12.2 was confident AND you picked something')
    lines.append('different. These are the moves most worth re-examining.')
    lines.append('')
    lines.append(f'**Pipeline:** {n_total} total log entries → {n_human} human turns → '
                 f'{n_disagree} raw disagreements → {n_disagree - n_inter} real '
                 f'(after dropping {n_inter} same-cell-stack interchangeables).')
    lines.append('')
    lines.append('| # | Game | Move | Dice | Own positions | Opp positions | AI pick | Human pick | AI conf% | Human conf% | KL | Interest |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')

    for i, d in enumerate(rows[:30], 1):
        cp = d.get('current_player', 0)
        own = d.get('positions', {}).get(str(cp), [-1, -1, -1, -1])
        opp = d.get('positions', {}).get(str((cp + 2) % 4), [-1, -1, -1, -1])
        own_s = '[' + ','.join(fmt_pos(p) for p in own) + ']'
        opp_s = '[' + ','.join(fmt_pos(p) for p in opp) + ']'
        policy = d.get('v12_policy', [0, 0, 0, 0])
        ai_pick = d.get('v12_argmax', -1)
        hum_pick = d.get('human_token', -1)
        ai_conf = policy[ai_pick] * 100 if 0 <= ai_pick < 4 else 0
        hum_conf = policy[hum_pick] * 100 if 0 <= hum_pick < 4 else 0
        kl = d.get('kl_v12_to_human', 0)
        interest = d.get('interest_score', 0)
        game_short = d.get('game_id', '')[-6:]
        move = d.get('move_count', '?')
        dice = d.get('dice', '?')
        lines.append(
            f'| {i} | …{game_short} | {move} | {dice} | {own_s} | {opp_s} | '
            f'**T{ai_pick}** | T{hum_pick} | {ai_conf:.1f}% | {hum_conf:.1f}% | '
            f'{kl:.3f} | {interest:.3f} |'
        )
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## Each disagreement\'s full policy distribution')
    lines.append('')
    for i, d in enumerate(rows[:30], 1):
        cp = d.get('current_player', 0)
        own = d.get('positions', {}).get(str(cp), [-1, -1, -1, -1])
        opp = d.get('positions', {}).get(str((cp + 2) % 4), [-1, -1, -1, -1])
        policy = d.get('v12_policy', [0, 0, 0, 0])
        ai_pick = d.get('v12_argmax', -1)
        hum_pick = d.get('human_token', -1)
        legal = d.get('legal_tokens', [])
        dice = d.get('dice', '?')
        win_prob = d.get('v12_win_prob', 0)
        move = d.get('move_count', '?')

        lines.append(f'### #{i}  game …{d.get("game_id","")[-6:]}  move {move}  dice={dice}  win_prob={win_prob:.3f}')
        lines.append('')
        lines.append(f'  Own (P{cp}): {[fmt_pos(p) for p in own]}    '
                     f'Opp (P{(cp+2)%4}): {[fmt_pos(p) for p in opp]}')
        lines.append(f'  Legal: {legal}')
        lines.append('')
        lines.append('  | Token | Position | V12.2 prob | |')
        lines.append('  |---|---|---|---|')
        for t in range(4):
            tag = ''
            if t == ai_pick:  tag += ' ⬅ AI'
            if t == hum_pick: tag += ' ⬅ HUMAN'
            illegal = '' if t in legal else ' (illegal)'
            lines.append(f'  | T{t}{tag} | {fmt_pos(own[t])}{illegal} | {policy[t]*100:.2f}% | |')
        lines.append('')

    SUMMARY_MD.write_text('\n'.join(lines))


if __name__ == '__main__':
    aggregate_and_filter()
