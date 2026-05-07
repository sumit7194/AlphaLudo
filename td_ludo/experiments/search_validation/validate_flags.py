"""Validate the 'search will help' hypothesis on hand-flagged AI moves.

For each disagreement in play/decision_logs/ai_disagreements.jsonl that
includes a `preferred_token`, we:
  1. Reconstruct the pre-move state in the C++ engine.
  2. Sanity-check that the loaded model's argmax matches the saved
     `ai_chosen` (proves reconstruction is correct).
  3. Run two cheap searches and check whether either one's argmax matches
     the human's `preferred_token`:
       - shallow_lookahead : pick action a maximizing V(state-after-a)
                             from current_player's perspective.
       - depth1_expectimax : same as shallow but average over the next
                             player's dice 1..6, with that player picking
                             the model's argmax response.

Output: per-row table + aggregate match rates.

Run from td_ludo/:
    td_env/bin/python experiments/search_validation/validate_flags.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TD_LUDO_DIR = HERE.parent.parent
PLAY_DIR = TD_LUDO_DIR / 'play'

sys.path.insert(0, str(TD_LUDO_DIR))
sys.path.insert(0, str(PLAY_DIR))

import td_ludo_cpp as ludo_cpp  # noqa: E402

# Reuse the loaded model + helpers from play.server (one-time import cost).
import server as play_server  # noqa: E402
from server import MODEL_VERSION  # noqa: E402

MODEL = play_server.model
DEVICE = play_server.device

DISAGREE_PATH = TD_LUDO_DIR / 'play' / 'decision_logs' / 'ai_disagreements.jsonl'


# ── State plumbing ────────────────────────────────────────────────────
def encode(state):
    if MODEL_VERSION == 'v12_2':
        return ludo_cpp.encode_state_v11(state)
    if MODEL_VERSION in ('v11', 'v12'):
        return ludo_cpp.encode_state_v10(state)
    if MODEL_VERSION == 'v6_3':
        return ludo_cpp.encode_state_v6_3(state, 0)
    return ludo_cpp.encode_state_v6(state)


def reconstruct(positions_dict, current_player, dice):
    """Build a state with the recorded token positions, scores derived
    from positions (anything at 99 counts toward player's score)."""
    g = ludo_cpp.create_initial_state_2p()
    pp = list(g.player_positions)
    for pstr, plist in positions_dict.items():
        pp[int(pstr)] = list(int(x) for x in plist)
    g.player_positions = pp
    # Update scores to match positions (engine cares about this for terminal).
    sc = list(g.scores)
    for pstr, plist in positions_dict.items():
        sc[int(pstr)] = sum(1 for x in plist if int(x) == 99)
    g.scores = sc
    g.current_player = int(current_player)
    g.current_dice_roll = int(dice)
    return g


def model_forward(state, legal_mask=None):
    """Returns (policy[4], value_scalar) at current state."""
    enc = np.asarray(encode(state), dtype=np.float32)
    if legal_mask is None:
        legal = list(ludo_cpp.get_legal_moves(state))
        legal_mask = np.zeros(4, dtype=np.float32)
        for m in legal:
            legal_mask[int(m)] = 1.0
    with torch.no_grad():
        s_t = torch.from_numpy(enc).unsqueeze(0).to(DEVICE, dtype=torch.float32)
        m_t = torch.from_numpy(legal_mask).unsqueeze(0).to(DEVICE, dtype=torch.float32)
        out = MODEL(s_t, m_t)
        policy = out[0].squeeze(0).cpu().numpy()
        try:
            v = float(out[1].squeeze().item())
        except (IndexError, AttributeError):
            v = 0.5
    return policy, v


def value_from_perspective(state, my_player):
    """Win-prob estimate for `my_player` at `state`, evaluated at the
    neutral between-turns snapshot (current_player=my_player, dice=0).
    Mirrors the trick play/server.py uses for the displayed win-chance."""
    if state.is_terminal:
        winner = int(ludo_cpp.get_winner(state))
        return 1.0 if winner == my_player else (0.0 if winner != -1 else 0.5)
    saved_cp = int(state.current_player)
    saved_d = int(state.current_dice_roll)
    state.current_player = int(my_player)
    state.current_dice_roll = 0
    try:
        _, v = model_forward(state)
    finally:
        state.current_player = saved_cp
        state.current_dice_roll = saved_d
    return float(v)


# ── Searches ──────────────────────────────────────────────────────────
def shallow_lookahead(positions, current_player, dice):
    """Argmax over V(apply_move(s, a)) from current_player's POV.

    Returns dict {action: score, ...} plus the argmax action.
    """
    state = reconstruct(positions, current_player, dice)
    legal = list(ludo_cpp.get_legal_moves(state))
    scores = {}
    for a in legal:
        s = reconstruct(positions, current_player, dice)
        s = ludo_cpp.apply_move(s, int(a))
        scores[int(a)] = value_from_perspective(s, current_player)
    if not scores:
        return {}, None
    best = max(scores, key=scores.get)
    return scores, best


def depth1_expectimax(positions, current_player, dice):
    """For each legal action a:
       - Apply a → state s'
       - Determine next_player (engine handles bonus turns)
       - Average over next_player's dice 1..6:
           - That player picks model.argmax response (legal-masked)
           - Apply, evaluate from current_player's POV (neutral snapshot)
       - Score = average
    """
    state = reconstruct(positions, current_player, dice)
    legal = list(ludo_cpp.get_legal_moves(state))
    scores = {}
    for a in legal:
        s = reconstruct(positions, current_player, dice)
        s = ludo_cpp.apply_move(s, int(a))
        if s.is_terminal:
            scores[int(a)] = value_from_perspective(s, current_player)
            continue
        # Average over next dice (1..6, equal weight)
        total = 0.0
        for d in range(1, 7):
            ss = reconstruct(_positions_from_state(s), int(s.current_player), d)
            sub_legal = list(ludo_cpp.get_legal_moves(ss))
            if not sub_legal:
                # Pass turn. Just clear dice and let engine roll over.
                ss.current_dice_roll = 0
                # Skip — value at the resulting position
                total += value_from_perspective(ss, current_player)
                continue
            mask = np.zeros(4, dtype=np.float32)
            for m in sub_legal:
                mask[int(m)] = 1.0
            policy, _ = model_forward(ss, legal_mask=mask)
            sub_a = int(np.argmax(policy))
            if sub_a not in sub_legal:
                sub_a = int(sub_legal[0])
            ss = ludo_cpp.apply_move(ss, sub_a)
            total += value_from_perspective(ss, current_player)
        scores[int(a)] = total / 6.0
    if not scores:
        return {}, None
    best = max(scores, key=scores.get)
    return scores, best


def _positions_from_state(state):
    """Snapshot positions from a live state for re-reconstruction."""
    out = {}
    for p in [0, 2]:
        out[str(p)] = [int(x) for x in state.player_positions[p]]
    return out


# ── Main ─────────────────────────────────────────────────────────────
def load_records():
    if not DISAGREE_PATH.exists():
        print(f'[err] {DISAGREE_PATH} not found.')
        sys.exit(1)
    out = []
    with open(DISAGREE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def dedupe(records):
    """Two records are duplicates if they share (game_id, ai_decision_id)."""
    seen = {}
    for r in records:
        key = (r.get('game_id'), r.get('ai_decision_id'))
        # Keep the one with the most informative comment.
        if key not in seen or len((r.get('comment') or '').strip()) > len(
            (seen[key].get('comment') or '').strip()
        ):
            seen[key] = r
    return list(seen.values())


def main():
    records = load_records()
    records = dedupe(records)
    flagged = [r for r in records if r.get('preferred_token') is not None]
    print(f'\n[info] model: {MODEL_VERSION}')
    print(f'[info] loaded {len(records)} unique records, '
          f'{len(flagged)} with a preferred_token\n')

    sanity_pass = 0
    shallow_match = 0
    d1_match = 0
    d0_match = 0   # baseline: model.argmax matching preferred (should be 0)

    print(f'{"#":>2} {"ply":>4} {"dice":>4} {"cp":>3} '
          f'{"chose":>6} {"pref":>5} {"d0":>5} {"shallow":>8} {"d1ex":>5} '
          f'{"san":>4}')
    print('-' * 70)

    for i, r in enumerate(flagged, 1):
        cp = int(r['current_player'])
        dice = int(r['dice'])
        positions = r['positions']
        chose = int(r['ai_chosen'])
        pref = int(r['preferred_token'])
        legal = r.get('legal_tokens') or []

        # 1) Sanity: model argmax should match saved chose.
        s = reconstruct(positions, cp, dice)
        mask = np.zeros(4, dtype=np.float32)
        for m in legal:
            mask[int(m)] = 1.0
        policy, _ = model_forward(s, legal_mask=mask)
        recomputed_argmax = int(np.argmax(policy))
        sane = (recomputed_argmax == chose)
        if sane:
            sanity_pass += 1

        # 2) d0 (baseline): model.argmax — should match `chose` not `pref`
        d0_pick = recomputed_argmax
        if d0_pick == pref:
            d0_match += 1

        # 3) Shallow 1-step value lookahead
        sl_scores, sl_pick = shallow_lookahead(positions, cp, dice)
        if sl_pick == pref:
            shallow_match += 1

        # 4) Depth-1 expectimax
        d1_scores, d1_pick = depth1_expectimax(positions, cp, dice)
        if d1_pick == pref:
            d1_match += 1

        san = '✓' if sane else '✗'
        print(f'{i:>2} {r.get("move_count","?"):>4} {dice:>4} P{cp:>1}  '
              f'T{chose}    T{pref}    T{d0_pick}    T{sl_pick}     T{d1_pick}    {san}')

        # Detail for non-sane rows
        if not sane:
            print(f'    [WARN] reconstruction off: model now picks T{recomputed_argmax} '
                  f'with policy {[round(float(p),3) for p in policy]} (saved chose=T{chose})')

    n = len(flagged)
    print()
    print(f'sanity (model argmax matches saved):    {sanity_pass}/{n}')
    print(f'd0 (current behavior matches preferred): {d0_match}/{n}  ← baseline, expect 0')
    print(f'shallow lookahead matches preferred:    {shallow_match}/{n}')
    print(f'depth-1 expectimax matches preferred:   {d1_match}/{n}')
    print()
    print('Per-row score detail:')
    for i, r in enumerate(flagged, 1):
        cp = int(r['current_player'])
        dice = int(r['dice'])
        positions = r['positions']
        legal = r.get('legal_tokens') or []
        sl_scores, sl_pick = shallow_lookahead(positions, cp, dice)
        d1_scores, d1_pick = depth1_expectimax(positions, cp, dice)
        chose = int(r['ai_chosen'])
        pref = int(r['preferred_token'])
        sl_str = ' '.join(f'T{a}={v:.3f}' for a, v in sorted(sl_scores.items()))
        d1_str = ' '.join(f'T{a}={v:.3f}' for a, v in sorted(d1_scores.items()))
        print(f'\n[{i}] dice={dice} cp=P{cp} chose=T{chose} pref=T{pref}')
        print(f'    shallow: {sl_str}  → pick T{sl_pick}')
        print(f'    d1exp:   {d1_str}  → pick T{d1_pick}')
        c = (r.get('comment') or '').strip()
        if c:
            print(f'    💬 {c}')


if __name__ == '__main__':
    main()
