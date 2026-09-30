"""Unit tests for consequence_targets — hand-computed capture probabilities.

Run: python td_ludo/game/test_consequence_targets.py
P0 = own (abs = rel). P2 = opp (abs = (rel+26)%52). Safe abs squares:
{0,8,13,21,26,34,39,47}.
"""
import numpy as np

from td_ludo.game.consequence_targets import (
    capture_prob_next_turn, compute_per_token_targets, compute_ranked_targets,
    progress_lost_if_captured,
)

_PASS = 0
_FAIL = 0


class DummyState:
    def __init__(self, positions, active=None):
        # positions: list of 4 lists (4 tokens each), token-id indexed.
        self.player_positions = [list(p) for p in positions]
        self.active_players = active or [True, False, True, False]


def _opp_rel_for_abs(target_abs):
    """P2 relative position that maps to a given absolute cell."""
    return (target_abs - 26) % 52


def check(name, got, want, tol=1e-6):
    global _PASS, _FAIL
    if abs(got - want) <= tol:
        _PASS += 1
        print(f"  ✓ {name}: {got:.4f}")
    else:
        _FAIL += 1
        print(f"  ✗ {name}: got {got:.4f} want {want:.4f}")


# P0 token at pos 20 (abs 20) throughout unless noted.
def base_state(opp_positions):
    return DummyState([[20, -1, -1, -1], [-1] * 4, opp_positions, [-1] * 4])


print("capture_prob_next_turn:")
# 1. no opp on track → 0
check("no opp on track", capture_prob_next_turn(base_state([-1, -1, -1, -1]), 0, 20), 0.0)
# 2. one opp exactly 3 behind (abs 17) → 1/6
check("one opp 3 behind", capture_prob_next_turn(base_state([_opp_rel_for_abs(17), -1, -1, -1]), 0, 20), 1/6)
# 3. two opp at 2 and 5 behind → 2/6
check("two opp at 2 and 5", capture_prob_next_turn(
    base_state([_opp_rel_for_abs(18), _opp_rel_for_abs(15), -1, -1]), 0, 20), 2/6)
# 4. two opp at SAME distance (both 3 behind) → 1/6 (distinct dice)
check("two opp same dist", capture_prob_next_turn(
    base_state([_opp_rel_for_abs(17), _opp_rel_for_abs(17), -1, -1]), 0, 20), 1/6)
# 5. opp 7 behind → out of range → 0
check("opp 7 behind (out of range)", capture_prob_next_turn(
    base_state([_opp_rel_for_abs(13), -1, -1, -1]), 0, 20), 0.0)
# 6. all six dice covered (opp at 1..6 behind) → 6/6
check("opp at all of 1..6 behind", capture_prob_next_turn(
    DummyState([[20, -1, -1, -1], [-1]*4,
                [_opp_rel_for_abs(19), _opp_rel_for_abs(18), _opp_rel_for_abs(17),
                 _opp_rel_for_abs(16)], [-1]*4]), 0, 20),
    4/6)  # only 4 opp tokens → at most 4 distinct dice (19,18,17,16 = dist 1,2,3,4)

print("safe-square / stack / off-track immunity:")
# 7. own on safe square (abs 21) with opp 3 behind → 0
safe = DummyState([[21, -1, -1, -1], [-1]*4, [_opp_rel_for_abs(18), -1, -1, -1], [-1]*4])
check("own on safe square", capture_prob_next_turn(safe, 0, 21), 0.0)
# 8. own stacked (two at pos 20) with opp 3 behind → 0
stacked = DummyState([[20, 20, -1, -1], [-1]*4, [_opp_rel_for_abs(17), -1, -1, -1], [-1]*4])
check("own stacked (2 tokens)", capture_prob_next_turn(stacked, 0, 20), 0.0)
# 9. token at base → 0
check("token at base", capture_prob_next_turn(base_state([_opp_rel_for_abs(17), -1, -1, -1]), 0, -1), 0.0)
# 10. inactive opp ignored
inactive = DummyState([[20, -1, -1, -1], [-1]*4, [_opp_rel_for_abs(17), -1, -1, -1], [-1]*4],
                      active=[True, False, False, False])
check("inactive opp ignored", capture_prob_next_turn(inactive, 0, 20), 0.0)

print("cells_at_risk + per-token + ranked:")
# 11. progress_lost: pos 30 → 31
check("progress_lost(30)", progress_lost_if_captured(30), 31.0)
# 12. per-token cells_at_risk: token at 30, opp 3 behind → (1/6 * 31)/51
st = DummyState([[30, -1, -1, -1], [-1]*4, [_opp_rel_for_abs(27), -1, -1, -1], [-1]*4])
cap, car, valid = compute_per_token_targets(st, 0)
check("per-token cap[0]", float(cap[0]), 1/6)
check("per-token car[0] norm", float(car[0]), (1/6 * 31) / 51.0)
check("per-token valid[0]", float(valid[0]), 1.0)
check("per-token valid[1] (base)", float(valid[1]), 0.0)

# 13. ranked targets: tokens at [40, 20, -1, -1]; opp 2 behind the pos-40 token
#     rank0 = pos 40 (most advanced), rank1 = pos 20, rank2 = base (valid 0)
opp_behind_40 = _opp_rel_for_abs(38)   # 2 behind abs 40
rk = DummyState([[40, 20, -1, -1], [-1]*4, [opp_behind_40, -1, -1, -1], [-1]*4])
rcap, rcar, rvalid = compute_ranked_targets(rk, 0)
check("ranked cap[0] (pos40, opp 2 behind)", float(rcap[0]), 1/6)
check("ranked cap[1] (pos20, no opp near)", float(rcap[1]), 0.0)
check("ranked valid[0]", float(rvalid[0]), 1.0)
check("ranked valid[1]", float(rvalid[1]), 1.0)
check("ranked valid[2] (base)", float(rvalid[2]), 0.0)

print(f"\n{_PASS}/{_PASS + _FAIL} passed" + (" ✓" if _FAIL == 0 else " ✗ FAILURES"))
import sys
sys.exit(1 if _FAIL else 0)
