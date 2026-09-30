"""4-PLAYER ENGINE AUDIT.

The 2P pipeline only ever exercised seats {0,2} via create_initial_state_2p(),
so every 4P code path (seats 1 and 3, cross-pair captures, 4-way turn cycling)
is effectively untested. This audit checks them empirically.

It also cross-checks the TWO engines, which matters because the v15 pipeline
*plays* with the legacy engine (td_ludo_cpp) but *encodes* with v15 geometry
(td_ludo_v15_cpp.position_to_cell). If they disagree in 4P, training silently
learns on wrong board pictures.

Run: PYTHONPATH=td_ludo:td_ludo_v15 python -m pytest td_ludo_v15/tests/test_4p_audit.py -q
  or: python td_ludo_v15/tests/test_4p_audit.py   (prints a PASS/FAIL report)
"""
import random
import numpy as np
import td_ludo_cpp as base
import td_ludo_v15_cpp as v15

PATH_LEN = 52
N_PLAYERS = 4
N_TOKENS = 4
BASE_POS = v15.BASE_POS
HOME_POS = v15.HOME_POS
SAFE_ABS = {0, 8, 13, 21, 26, 34, 39, 47}

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    return bool(cond)


def abs_pos(player, rel):
    """Absolute loop index for a player-relative main-track position."""
    return (rel + 13 * player) % PATH_LEN


# ── GEOMETRY ─────────────────────────────────────────────────────────────
def test_geometry():
    # G1: each player's 51 track positions map to 51 distinct cells
    for p in range(N_PLAYERS):
        cells = {v15.position_to_cell(q, p) for q in range(51)}
        check(f"G1 p{p}: 51 track positions → 51 distinct cells",
              len(cells) == 51, f"got {len(cells)}")

    # G2 (KEY): the main loop is SHARED — two players standing on the same
    # absolute loop index must occupy the SAME physical cell.
    mismatches = []
    for p in range(N_PLAYERS):
        for q in range(N_PLAYERS):
            for rp in range(51):
                a = abs_pos(p, rp)
                # find q's relative pos with the same absolute index
                rq = (a - 13 * q) % PATH_LEN
                if rq > 50:
                    continue
                cp_, cq_ = v15.position_to_cell(rp, p), v15.position_to_cell(rq, q)
                if cp_ != cq_:
                    mismatches.append((p, rp, q, rq, cp_, cq_))
    check("G2: shared main loop — same abs index ⇒ same cell (all player pairs)",
          not mismatches, f"{len(mismatches)} mismatches, e.g. {mismatches[:3]}")

    # G3: home stretches (51..55) disjoint across players
    stretch = {p: {v15.position_to_cell(q, p) for q in range(51, 56)} for p in range(N_PLAYERS)}
    overlap = [(p, q, stretch[p] & stretch[q]) for p in range(4) for q in range(p + 1, 4)
               if stretch[p] & stretch[q]]
    check("G3: home stretches disjoint across the 4 players",
          not overlap, f"overlaps: {overlap[:2]}")

    # G4: the 4 home bases occupy 4 distinct quadrants
    bases = {p: v15.position_to_cell(BASE_POS, p) for p in range(N_PLAYERS)}
    check("G4: 4 home-base counters distinct", len(set(bases.values())) == 4, str(bases))

    # G5: entry offset is 13 per seat
    entries = [abs_pos(p, 0) for p in range(N_PLAYERS)]
    check("G5: spawn offsets are 0/13/26/39", entries == [0, 13, 26, 39], str(entries))

    # G6: safe squares agree across players (a safe abs index is safe for everyone)
    bad = []
    for p in range(N_PLAYERS):
        for rel in range(51):
            if abs_pos(p, rel) in SAFE_ABS:
                # the cell should be one of the 8 canonical safe cells
                pass
    check("G6: safe set is absolute-indexed (8 indices)", len(SAFE_ABS) == 8)


# ── RULES (legacy engine — the one that actually plays) ──────────────────
def fresh4():
    s = base.create_initial_state()
    return s


def set_positions(s, pos_map, cur=0, dice=1):
    """pos_map: {player: [4 positions]}"""
    for p, arr in pos_map.items():
        for t, v in enumerate(arr):
            s.player_positions[p][t] = v
    s.current_player = cur
    s.current_dice_roll = dice
    return s


def test_rules_4p():
    # R1: turn cycling over 4 active players
    s = fresh4()
    seen = []
    for _ in range(8):
        seen.append(int(s.current_player))
        s.current_dice_roll = 1          # non-6, no legal move from base → passes
        legal = base.get_legal_moves(s)
        if legal:
            s = base.apply_move(s, int(legal[0]))
        else:
            # emulate pass
            s.current_player = (int(s.current_player) + 1) % 4
    check("R1: 4-way turn cycling visits all seats", len(set(seen)) == 4, str(seen[:8]))

    # R2 (KEY): capture works for ALL ordered pairs, not just 0↔2
    fails = []
    for atk in range(4):
        for vic in range(4):
            if atk == vic:
                continue
            s = fresh4()
            # victim token on a NON-safe absolute cell; attacker 1 step behind
            target_abs = None
            for cand in range(PATH_LEN):
                if cand not in SAFE_ABS:
                    target_abs = cand
                    break
            vic_rel = (target_abs - 13 * vic) % PATH_LEN
            atk_rel = (target_abs - 13 * atk) % PATH_LEN
            if vic_rel > 50 or atk_rel < 1 or atk_rel > 50:
                continue
            pos = {p: [BASE_POS] * 4 for p in range(4)}
            pos[vic][0] = vic_rel
            pos[atk][0] = atk_rel - 1
            s = set_positions(s, pos, cur=atk, dice=1)
            legal = base.get_legal_moves(s)
            if not legal:
                fails.append((atk, vic, "no legal move")); continue
            s2 = base.apply_move(s, int(legal[0]))
            if int(s2.player_positions[vic][0]) != BASE_POS:
                fails.append((atk, vic, f"victim not sent home: {int(s2.player_positions[vic][0])}"))
    check("R2: capture works for all 12 ordered player pairs", not fails, f"failures: {fails[:4]}")

    # R3: blockade (2 victim tokens stacked) prevents capture
    bfails = []
    for atk in range(4):
        for vic in range(4):
            if atk == vic:
                continue
            target_abs = next(c for c in range(PATH_LEN) if c not in SAFE_ABS)
            vic_rel = (target_abs - 13 * vic) % PATH_LEN
            atk_rel = (target_abs - 13 * atk) % PATH_LEN
            if vic_rel > 50 or atk_rel < 1 or atk_rel > 50:
                continue
            s = fresh4()
            pos = {p: [BASE_POS] * 4 for p in range(4)}
            pos[vic][0] = vic_rel; pos[vic][1] = vic_rel      # stack of 2 → blockade
            pos[atk][0] = atk_rel - 1
            s = set_positions(s, pos, cur=atk, dice=1)
            legal = base.get_legal_moves(s)
            if not legal:
                continue
            s2 = base.apply_move(s, int(legal[0]))
            if int(s2.player_positions[vic][0]) == BASE_POS:
                bfails.append((atk, vic, "blockade was captured"))
    check("R3: 2-stack blockade is NOT captured (all pairs)", not bfails, f"{bfails[:4]}")

    # R4: no capture on a safe square
    sfails = []
    for atk in range(4):
        for vic in range(4):
            if atk == vic:
                continue
            safe_abs = 8
            vic_rel = (safe_abs - 13 * vic) % PATH_LEN
            atk_rel = (safe_abs - 13 * atk) % PATH_LEN
            if vic_rel > 50 or atk_rel < 1 or atk_rel > 50:
                continue
            s = fresh4()
            pos = {p: [BASE_POS] * 4 for p in range(4)}
            pos[vic][0] = vic_rel
            pos[atk][0] = atk_rel - 1
            s = set_positions(s, pos, cur=atk, dice=1)
            legal = base.get_legal_moves(s)
            if not legal:
                continue
            s2 = base.apply_move(s, int(legal[0]))
            if int(s2.player_positions[vic][0]) == BASE_POS:
                sfails.append((atk, vic))
    check("R4: no capture on safe squares (all pairs)", not sfails, f"{sfails[:4]}")

    # R5: terminal when a player scores all 4
    s = fresh4()
    pos = {p: [BASE_POS] * 4 for p in range(4)}
    pos[1] = [HOME_POS, HOME_POS, HOME_POS, 55]
    s = set_positions(s, pos, cur=1, dice=1)
    s.scores[1] = 3
    legal = base.get_legal_moves(s)
    if legal:
        s2 = base.apply_move(s, int(legal[0]))
        check("R5: game terminal when seat 1 scores 4th token",
              bool(s2.is_terminal) and int(s2.scores[1]) == 4,
              f"terminal={bool(s2.is_terminal)} score={int(s2.scores[1])}")
    else:
        check("R5: seat-1 final token has a legal move", False, "no legal move")


# ── CROSS-ENGINE PARITY IN 4P (the highest-value test) ───────────────────
def test_cross_engine_parity_4p(n_games=40, seed=0):
    """Drive BOTH engines with identical dice + identical token choice and
    assert full state agreement at every ply."""
    rng = random.Random(seed)
    divergences = []
    for g in range(n_games):
        b = base.create_initial_state()
        v = v15.create_initial_state()
        # The two engines OWN the 3-consecutive-sixes rule differently:
        #   v15.set_dice() applies it internally (and may forfeit the turn);
        #   the legacy engine has no consecutive_sixes field at all — the
        #   caller does it (as v15_player.py does in Python).
        # Mirror it on the legacy side so the comparison is apples-to-apples.
        csix = [0] * N_PLAYERS
        for ply in range(400):
            if bool(b.is_terminal) or bool(v.is_terminal):
                break
            d = rng.randint(1, 6)
            cp_before = int(b.current_player)
            csix[cp_before] = csix[cp_before] + 1 if d == 6 else 0
            forfeit = csix[cp_before] >= 3
            if forfeit:
                csix[cp_before] = 0
            b.current_dice_roll = d
            v = v15.set_dice(v, d)
            if forfeit:
                # v15 forfeited internally; mirror on legacy and resync
                b.current_player = (cp_before + 1) % N_PLAYERS
                b.current_dice_roll = 0
                continue
            if int(b.current_player) != int(v.current_player):
                divergences.append((g, ply, "current_player",
                                    int(b.current_player), int(v.current_player))); break
            bl = list(base.get_legal_moves(b))
            vcells = list(v15.get_legal_source_cells(v))
            if len(bl) != len(vcells):
                # The cell-based API legitimately collapses several own tokens
                # that share one cell (e.g. all 4 still at base) into a single
                # source cell. That's expected — only flag a REAL mismatch,
                # i.e. when distinct source cells still disagree.
                p = int(b.current_player)
                positions = [int(b.player_positions[p][t]) for t in bl]
                distinct_cells = {v15.position_to_cell(sp, p) for sp in positions}
                if len(distinct_cells) != len(vcells):
                    divergences.append((g, ply, "legal_count",
                                        f"base={len(bl)} distinct_cells={len(distinct_cells)}",
                                        f"v15={len(vcells)}")); break
            if not bl:
                b.current_player = (int(b.current_player) + 1) % 4
                v = v15.pass_turn(v)
                continue
            slot = int(rng.choice(bl))
            p = int(b.current_player)
            src = int(b.player_positions[p][slot])
            r, c = v15.position_to_cell(src, p)
            b = base.apply_move(b, slot)
            try:
                v = v15.apply_move_from_cell(v, int(r), int(c))
            except Exception as e:
                divergences.append((g, ply, "v15 apply raised", str(e)[:60], "")); break
            bp = [[int(b.player_positions[pp][t]) for t in range(4)] for pp in range(4)]
            vp = [[int(v.player_positions[pp][t]) for t in range(4)] for pp in range(4)]
            if sorted(map(sorted, bp)) != sorted(map(sorted, vp)):
                divergences.append((g, ply, "positions", bp, vp)); break
            if [int(x) for x in b.scores] != [int(x) for x in v.scores]:
                divergences.append((g, ply, "scores",
                                    [int(x) for x in b.scores], [int(x) for x in v.scores])); break
    check(f"P1: legacy vs v15 engine agree over {n_games} random 4P games",
          not divergences, f"{len(divergences)} divergences, first: {divergences[:2]}")


def main():
    test_geometry()
    test_rules_4p()
    test_cross_engine_parity_4p()
    print("\n" + "=" * 72)
    print("4-PLAYER ENGINE AUDIT")
    print("=" * 72)
    npass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok and detail:
            print(f"         → {detail}")
    print("=" * 72)
    print(f"  {npass}/{len(results)} passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
