"""Unit tests for V13.5 symmetry contract.

What we are validating:
  1. encoder_v18_symmetric: invariant under any permutation of own (or
     opp) token-IDs at the input.
  2. rank_mapping.state_to_rank_mapping: depends only on the multiset of
     positions, not on the token-ID labelling.
  3. aggregate_token_policy_to_ranks: correctly sums per-token probs
     into per-rank probs and matches the rank ordering.
  4. legal_mask_per_rank: a rank is legal iff any token-ID in that rank
     group is legal.
  5. rank_to_token_id: maps back to a legal token at the requested rank.
  6. permute_own_tokens: physical state preserved (positions multiset),
     token-IDs reshuffled, opp tokens untouched.
  7. Round-trip property: V13.2's per-token policy aggregated to per-rank
     is the SAME under any token-ID permutation of V13.2's input — the
     symmetrization contract that justifies our random-permutation
     augmentation in distillation.

Run:
    ./td_env/bin/python experiments/v135/test_v135_symmetry.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v18_symmetric import (
    encode_state_v18_symmetric,
    V18_CHANNELS,
)
from td_ludo.game.rank_mapping import (
    HOME_POS,
    MAX_RANK_SLOTS,
    state_to_rank_mapping,
    aggregate_token_policy_to_ranks,
    legal_mask_per_rank,
    rank_to_token_id,
    permute_own_tokens,
)


PASS = 0
FAIL = 0


def assert_(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")


# ── Test fixtures ─────────────────────────────────────────────────────────
def make_test_state(cp=0, dice=4, n_advance_moves=4):
    """Build an asymmetric state by playing a few moves, so different
    token-IDs end up at different positions."""
    g = ludo_cpp.create_initial_state_2p()
    g.current_player = cp
    g.current_dice_roll = 6
    legal = ludo_cpp.get_legal_moves(g)
    g = ludo_cpp.apply_move(g, legal[0])  # token 0 leaves home
    # Advance same token a few more steps with various dice
    for d in (4, 3, 5, 2)[:n_advance_moves - 1]:
        g.current_player = cp
        g.current_dice_roll = d
        legal = ludo_cpp.get_legal_moves(g)
        if not legal:
            continue
        # Always pick the legal move that touches the token we just moved
        # (which is token 0). If not available, just take legal[0].
        chosen = legal[0]
        g = ludo_cpp.apply_move(g, chosen)
    # Now also free up token 1 to a different cell so we have 2 distinct
    # board positions for own tokens.
    g.current_player = cp
    g.current_dice_roll = 6
    legal = ludo_cpp.get_legal_moves(g)
    if legal:
        # Try to leave home (action that creates a new on-board token)
        for a in legal:
            ng = ludo_cpp.apply_move(g, a)
            # Check that the number of on-board own tokens went up
            old_on = (g.player_positions[cp] >= 0).sum()
            new_on = (ng.player_positions[cp] >= 0).sum()
            if new_on > old_on:
                g = ng
                break
    g.current_player = cp
    g.current_dice_roll = dice
    return g


# ── 1. Encoder is invariant under own-token-ID permutation ───────────────
def test_encoder_invariant_under_own_perm():
    print("\n== encoder_v18: invariant under own-token-ID permutation ==")
    g = make_test_state(cp=0)
    base = encode_state_v18_symmetric(g)
    n_invariant = 0
    n_total = 0
    for perm in itertools.permutations([0, 1, 2, 3]):
        n_total += 1
        gp = permute_own_tokens(g, list(perm))
        enc = encode_state_v18_symmetric(gp)
        if np.allclose(enc, base, atol=1e-6):
            n_invariant += 1
    assert_(n_invariant == n_total,
            f"all {n_total} own-token permutations produce identical encoding ({n_invariant}/{n_total})")


# ── 2. Encoder is invariant under opp-token-ID permutation ───────────────
def test_encoder_invariant_under_opp_perm():
    """Symmetry should hold for opp tokens too. We test by manually
    permuting the opponent's row in player_positions."""
    print("\n== encoder_v18: invariant under opp-token-ID permutation ==")
    g = make_test_state(cp=0)
    # Move some opp tokens out so they have asymmetric positions
    g.current_player = 2
    g.current_dice_roll = 6
    legal = ludo_cpp.get_legal_moves(g)
    if legal:
        g = ludo_cpp.apply_move(g, legal[0])
    g.current_player = 0
    g.current_dice_roll = 4
    base = encode_state_v18_symmetric(g)
    opp = 2
    n_inv = 0
    n_tot = 0
    for perm in itertools.permutations([0, 1, 2, 3]):
        gp = ludo_cpp.create_initial_state_2p()
        gp.current_player = int(g.current_player)
        gp.current_dice_roll = int(g.current_dice_roll)
        gp.player_positions[:] = g.player_positions[:]
        gp.active_players[:] = g.active_players[:]
        gp.player_positions[opp] = g.player_positions[opp][np.array(perm)]
        if np.allclose(encode_state_v18_symmetric(gp), base, atol=1e-6):
            n_inv += 1
        n_tot += 1
    assert_(n_inv == n_tot, f"all {n_tot} opp-token permutations produce identical encoding ({n_inv}/{n_tot})")


# ── 3. Encoder DOES change with the physical state (sanity) ──────────────
def test_encoder_changes_with_physical_state():
    print("\n== encoder_v18: DOES change for genuinely different physical states ==")
    g1 = make_test_state(cp=0, dice=4, n_advance_moves=2)
    g2 = make_test_state(cp=0, dice=4, n_advance_moves=4)
    e1 = encode_state_v18_symmetric(g1)
    e2 = encode_state_v18_symmetric(g2)
    assert_(not np.allclose(e1, e2),
            f"different physical states produce different encodings (|diff|={np.abs(e1 - e2).sum():.2f})")


# ── 4. Rank mapping is invariant under token-ID permutation ──────────────
def test_rank_mapping_invariant():
    print("\n== rank_mapping: rank → position invariant under token-ID perm ==")
    g = make_test_state(cp=0)
    cp = int(g.current_player)
    base_positions, base_tokens = state_to_rank_mapping(g.player_positions[cp])
    n_inv = 0
    n_tot = 0
    for perm in itertools.permutations([0, 1, 2, 3]):
        gp = permute_own_tokens(g, list(perm))
        positions_p, tokens_p = state_to_rank_mapping(gp.player_positions[cp])
        # rank_positions should be identical (same multiset of positions
        # → same canonical rank ordering)
        if positions_p == base_positions:
            n_inv += 1
        n_tot += 1
    assert_(n_inv == n_tot,
            f"all {n_tot} permutations give identical rank → position mapping ({n_inv}/{n_tot})")


# ── 5. Rank ordering is descending (most-advanced first) ────────────────
def test_rank_ordering_descending():
    print("\n== rank_mapping: rank ordering is descending (most-advanced first) ==")
    pp = np.array([10, -1, 25, 5])  # token positions
    positions, tokens = state_to_rank_mapping(pp)
    # Expected: 25 (most), 10, 5, -1 (home, last)
    expected = [25, 10, 5, -1]
    assert_(positions == expected,
            f"positions sorted descending: got {positions}, expected {expected}")
    # Token IDs at each rank
    expected_tokens = [[2], [0], [3], [1]]
    assert_(tokens == expected_tokens,
            f"token IDs at each rank: got {tokens}, expected {expected_tokens}")


# ── 6. Stacks share a single rank ────────────────────────────────────────
def test_stacks_share_rank():
    print("\n== rank_mapping: stacked tokens share a single rank slot ==")
    pp = np.array([10, 10, 25, -1])  # tokens 0 + 1 stacked at pos 10
    positions, tokens = state_to_rank_mapping(pp)
    assert_(len(positions) == 3,
            f"3 unique positions: got {len(positions)} ({positions})")
    # rank 0: pos 25, just token 2
    # rank 1: pos 10, tokens 0 and 1 (sorted ascending)
    # rank 2: pos -1, just token 3
    assert_(positions == [25, 10, -1], f"rank positions: {positions}")
    assert_(tokens == [[2], [0, 1], [3]],
            f"rank token-IDs: got {tokens}")


# ── 7. Aggregation: per-token probs sum to per-rank ──────────────────────
def test_aggregation_correctness():
    print("\n== rank_mapping: aggregate_token_policy_to_ranks correctness ==")
    pp = np.array([10, 10, 25, -1])
    positions, tokens = state_to_rank_mapping(pp)
    # Token probs (token-ID indexed)
    token_probs = [0.10, 0.20, 0.50, 0.20]   # sum 1.0
    rank_probs = aggregate_token_policy_to_ranks(token_probs, tokens)
    # Expected:
    #   rank 0 (pos 25, token 2): 0.50
    #   rank 1 (pos 10, tokens 0+1): 0.10 + 0.20 = 0.30
    #   rank 2 (pos -1, token 3): 0.20
    #   rank 3: unused → 0.0
    expected = np.array([0.50, 0.30, 0.20, 0.0], dtype=np.float32)
    assert_(np.allclose(rank_probs, expected),
            f"rank probs: got {rank_probs.tolist()}, expected {expected.tolist()}")
    assert_(abs(rank_probs.sum() - 1.0) < 1e-6,
            f"rank probs sum to 1: got {rank_probs.sum():.4f}")


# ── 8. Aggregation INVARIANT to token-ID permutation ─────────────────────
def test_aggregation_invariant_under_perm():
    """Crucial property: aggregate(perm·probs, perm·tokens) == aggregate(probs, tokens).

    This is what justifies random-permutation augmentation in distillation —
    the aggregation washes out token-ID-dependent biases.
    """
    print("\n== rank_mapping: aggregation invariant under token-ID permutation ==")
    g = make_test_state(cp=0)
    cp = int(g.current_player)
    # Start with an ARBITRARY teacher per-token policy
    base_probs = np.array([0.10, 0.40, 0.30, 0.20], dtype=np.float32)
    _, base_tokens = state_to_rank_mapping(g.player_positions[cp])
    base_rank_probs = aggregate_token_policy_to_ranks(base_probs, base_tokens)

    n_inv = 0; n_tot = 0
    for perm in itertools.permutations([0, 1, 2, 3]):
        # Apply perm to: state's token-IDs AND the per-token probs
        # in the same way (so the policy is consistent with the relabelled state).
        gp = permute_own_tokens(g, list(perm))
        # In the permuted state, token at slot i is the OLD token at slot perm[i].
        # So the per-token policy in the permuted state is:
        #   new_probs[i] = old_probs[perm[i]]   (the prob of "moving the same physical token")
        permuted_probs = base_probs[np.array(perm)]
        _, perm_tokens = state_to_rank_mapping(gp.player_positions[cp])
        rank_probs = aggregate_token_policy_to_ranks(permuted_probs, perm_tokens)
        if np.allclose(rank_probs, base_rank_probs, atol=1e-6):
            n_inv += 1
        n_tot += 1
    assert_(n_inv == n_tot,
            f"aggregation invariant under all {n_tot} permutations ({n_inv}/{n_tot})")


# ── 9. Legal-mask per rank ───────────────────────────────────────────────
def test_legal_mask_per_rank():
    print("\n== rank_mapping: legal_mask_per_rank correctness ==")
    pp = np.array([10, 10, 25, -1])
    positions, tokens = state_to_rank_mapping(pp)
    # Suppose only tokens 0 and 2 are legal
    legal_token_ids = [0, 2]
    mask = legal_mask_per_rank(legal_token_ids, tokens)
    # rank 0: token 2 legal → 1.0
    # rank 1: token 0 legal → 1.0  (token 1 also at this rank, not legal)
    # rank 2: token 3 not legal → 0.0
    # rank 3: unused → 0.0
    expected = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert_(np.allclose(mask, expected),
            f"legal mask: got {mask.tolist()}, expected {expected.tolist()}")


# ── 10. rank_to_token_id round-trip ──────────────────────────────────────
def test_rank_to_token_roundtrip():
    print("\n== rank_mapping: rank_to_token_id picks a LEGAL token at the rank ==")
    pp = np.array([10, 10, 25, -1])
    positions, tokens = state_to_rank_mapping(pp)
    legal_token_ids = [0, 2, 3]
    # rank 0 (pos 25): token 2 is legal → returns 2
    assert_(rank_to_token_id(0, legal_token_ids, tokens) == 2,
            f"rank 0 → token 2 (got {rank_to_token_id(0, legal_token_ids, tokens)})")
    # rank 1 (pos 10): tokens 0 and 1 here, only 0 is legal → returns 0
    assert_(rank_to_token_id(1, legal_token_ids, tokens) == 0,
            f"rank 1 → token 0 (got {rank_to_token_id(1, legal_token_ids, tokens)})")
    # rank 2 (pos -1): token 3 is legal → returns 3
    assert_(rank_to_token_id(2, legal_token_ids, tokens) == 3,
            f"rank 2 → token 3 (got {rank_to_token_id(2, legal_token_ids, tokens)})")


# ── 11. permute_own_tokens preserves physical state ──────────────────────
def test_permute_preserves_physical_state():
    print("\n== rank_mapping: permute_own_tokens preserves multiset of positions ==")
    g = make_test_state(cp=0)
    cp = int(g.current_player)
    base_multiset = sorted(g.player_positions[cp].tolist())
    opp_row_before = g.player_positions[(cp + 2) % 4].copy()
    n_pres = 0; n_tot = 0
    for perm in itertools.permutations([0, 1, 2, 3]):
        gp = permute_own_tokens(g, list(perm))
        new_multiset = sorted(gp.player_positions[cp].tolist())
        opp_row_after = gp.player_positions[(cp + 2) % 4]
        if new_multiset == base_multiset and np.array_equal(opp_row_after, opp_row_before):
            n_pres += 1
        n_tot += 1
    assert_(n_pres == n_tot,
            f"all {n_tot} permutations preserve own-token multiset AND opp row ({n_pres}/{n_tot})")


# ── 12. End-to-end: V13.2 teacher → symmetrized rank policy ──────────────
def test_e2e_v132_to_symmetrized_rank_policy():
    """The whole point of all this. We feed V13.2 the SAME physical state
    with several different token-ID permutations of own tokens. The
    aggregated per-rank policy should be similar across permutations (it
    won't be exactly identical because V13.2 isn't perfectly invariant —
    that's WHY we want to symmetrize via averaging)."""
    print("\n== e2e: V13.2 per-token output → per-rank policy across permutations ==")
    import torch
    from experiments.distillation_14ch.model_14ch import MinimalCNN14
    import td_ludo.game.encoder_v17 as enc_v17

    # Try to find a V13.2 checkpoint
    cands = [
        "/Users/sumit/Github/AlphaLudo/checkpoint_backups/v132_20260506_015608/model_latest.pt",
        "/Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/v132/model_latest.pt",
    ]
    teacher_path = next((p for p in cands if os.path.exists(p)), None)
    if teacher_path is None:
        print("  SKIP  no V13.2 teacher checkpoint found — skipping e2e test")
        return
    device = torch.device("cpu")
    teacher = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=17)
    sd = torch.load(teacher_path, map_location=device, weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    teacher.load_state_dict(sd, strict=False)
    teacher.eval()

    g = make_test_state(cp=0)
    cp = int(g.current_player)
    legal = ludo_cpp.get_legal_moves(g)
    legal_mask = np.zeros(4, dtype=np.float32)
    for a in legal:
        legal_mask[a] = 1.0

    # Run V13.2 on the original state and on each token-ID permutation,
    # aggregate to per-rank, see how invariant V13.2 actually is.
    rank_probs_per_perm = []
    for perm in itertools.permutations([0, 1, 2, 3]):
        gp = permute_own_tokens(g, list(perm))
        # NOTE: V13.2's input is V17 = V14_minimal(14) + 3 statics. The
        # token-ID permutation in `gp.player_positions` will reflect in
        # V14_minimal's ch0..3 (own tokens). The legal mask must also be
        # permuted: a legal token-ID `a` in original state corresponds to
        # the row `i` in the permuted state where perm[i] == a.
        enc = enc_v17.encode_state_v17(gp)
        x = torch.from_numpy(enc).unsqueeze(0).float()
        # Build permuted legal mask
        # In the permuted state, slot i holds the OLD token perm[i].
        # The OLD legal mask is `legal_mask` (token-ID indexed).
        # So new_mask[i] = legal_mask[perm[i]].
        permuted_legal_mask = legal_mask[np.array(perm)]
        m = torch.from_numpy(permuted_legal_mask).unsqueeze(0).float()
        with torch.no_grad():
            policy, _, _ = teacher(x, m)
        token_probs = policy.squeeze(0).numpy()  # (4,) — over PERMUTED token-IDs
        # Aggregate using the PERMUTED state's rank mapping
        _, perm_tokens = state_to_rank_mapping(gp.player_positions[cp])
        rank_probs = aggregate_token_policy_to_ranks(token_probs, perm_tokens)
        rank_probs_per_perm.append(rank_probs)
    rp = np.stack(rank_probs_per_perm, axis=0)
    mean = rp.mean(axis=0)
    std = rp.std(axis=0)
    max_dev = np.abs(rp - mean).max()
    print(f"  rank_probs across 24 permutations:")
    print(f"    mean: {mean.round(4).tolist()}")
    print(f"    std:  {std.round(4).tolist()}")
    print(f"    max deviation from mean: {max_dev:.4f}")
    # If V13.2 were perfectly token-invariant, std would be 0. We don't
    # expect that — the whole point is that V13.2 isn't invariant. We
    # just verify the variance is bounded (a sanity check on plumbing).
    assert_(std.sum() < 0.5,
            f"per-rank std across permutations is bounded (sum_std={std.sum():.4f})")
    # Mean is the symmetrized target; verify it sums to ≈ 1 modulo the
    # active rank slots.
    n_unique = len(state_to_rank_mapping(g.player_positions[cp])[0])
    active_mass = mean[:n_unique].sum()
    assert_(abs(active_mass - 1.0) < 1e-3,
            f"symmetrized rank policy sums to ~1 over active slots (got {active_mass:.4f})")


# ── Run all ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("V13.5 symmetry contract — unit tests")
    print("=" * 70)
    test_encoder_invariant_under_own_perm()
    test_encoder_invariant_under_opp_perm()
    test_encoder_changes_with_physical_state()
    test_rank_mapping_invariant()
    test_rank_ordering_descending()
    test_stacks_share_rank()
    test_aggregation_correctness()
    test_aggregation_invariant_under_perm()
    test_legal_mask_per_rank()
    test_rank_to_token_roundtrip()
    test_permute_preserves_physical_state()
    test_e2e_v132_to_symmetrized_rank_policy()
    print("\n" + "=" * 70)
    print(f"RESULT: {PASS} pass, {FAIL} fail")
    print("=" * 70)
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
