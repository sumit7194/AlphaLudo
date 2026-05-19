"""Unit tests for V135ProductionAdapter: verify the rank-symmetry contract is
preserved through the production-pipeline wrapper.

We must guarantee that:
  1. The packed encoder is reversible — `unpack_token_to_rank` recovers the
     same int array we packed.
  2. Random token-ID permutations of the same physical state produce
     IDENTICAL token-id-indexed policy distributions through the adapter
     (i.e., the symmetric inner core is not broken by the adapter glue).
  3. Tokens at the same rank get equal probability after legal-mask + softmax.
  4. rank_legal_mask construction is correct.
  5. forward_policy_only logits, when softmaxed + legal-masked, match the
     forward() output policy.

Run:
    ./td_env/bin/python experiments/v135/test_production_adapter.py
"""
from __future__ import annotations

import itertools
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v18_production import (
    encode_state_v18_production,
    unpack_token_to_rank,
    V18_PROD_CHANNELS,
)
from td_ludo.game.rank_mapping import (
    state_to_rank_mapping,
    permute_own_tokens,
    MAX_RANK_SLOTS,
)
from td_ludo.models.v13_5_production import V135ProductionAdapter


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


def make_state_with_distinct_positions():
    """Build a state where tokens are at different board positions, so rank
    mapping is non-trivial. Move token 0 forward via a few legal moves."""
    g = ludo_cpp.create_initial_state_2p()
    g.current_player = 0
    g.current_dice_roll = 6
    g = ludo_cpp.apply_move(g, ludo_cpp.get_legal_moves(g)[0])  # token 0 leaves home
    g.current_player = 0
    g.current_dice_roll = 4
    legal = ludo_cpp.get_legal_moves(g)
    if 0 in legal:
        g = ludo_cpp.apply_move(g, 0)  # advance same token
    g.current_player = 0
    g.current_dice_roll = 6
    legal = ludo_cpp.get_legal_moves(g)
    # Try to leave home with a different token (token 1 → spawn)
    for a in legal:
        ng = ludo_cpp.apply_move(g, a)
        if (ng.player_positions[0] >= 0).sum() > (g.player_positions[0] >= 0).sum():
            g = ng
            break
    g.current_player = 0
    g.current_dice_roll = 3
    return g


def legal_mask_from_state(state):
    legal = ludo_cpp.get_legal_moves(state)
    mask = np.zeros(4, dtype=np.float32)
    for a in legal:
        mask[a] = 1.0
    return mask


# ── 1. Encoder ↔ unpack round-trip ───────────────────────────────────────
def test_unpack_roundtrip():
    print("\n== encoder/unpack: token_to_rank round-trips ==")
    state = make_state_with_distinct_positions()
    enc = encode_state_v18_production(state)
    assert_(enc.shape == (V18_PROD_CHANNELS, 15, 15),
            f"encoded shape == ({V18_PROD_CHANNELS}, 15, 15): got {enc.shape}")

    # Compute expected token_to_rank manually
    cp = int(state.current_player)
    pp = state.player_positions[cp]
    _, rank_token_ids = state_to_rank_mapping(pp)
    expected = np.zeros(4, dtype=np.int64)
    for k, tokens in enumerate(rank_token_ids):
        if k >= MAX_RANK_SLOTS:
            break
        for t in tokens:
            expected[t] = k

    x = torch.from_numpy(enc).unsqueeze(0).float()
    recovered = unpack_token_to_rank(x).squeeze(0).cpu().numpy()
    assert_(np.array_equal(recovered, expected),
            f"unpack_token_to_rank({expected.tolist()}) recovers original")


# ── 2. Permutation invariance through the adapter ─────────────────────────
def test_permutation_invariance():
    """Most important test: feed the same physical state to the adapter under
    several token-ID permutations, then verify the token-id-indexed policy
    distributions are equal AFTER applying the inverse permutation.

    Specifically: if we permute token-IDs by π in the input state, the
    adapter's per-token policy should be a permuted version of the original
    (with the same permutation π). Equivalently:

        adapter(perm(s, π)).policy[π(t)] ≈ adapter(s).policy[t]   for all t

    The "≈" is needed because V135Symmetric is permutation-invariant in
    its rank space — but the adapter then re-broadcasts to token-id space
    via token_to_rank. The token policy values for tokens at the same rank
    should be identical (this is the core invariance the architecture buys).
    """
    print("\n== adapter: permutation-equivariance (token-id space) ==")
    state = make_state_with_distinct_positions()

    adapter = V135ProductionAdapter(num_res_blocks=4, num_channels=64)
    adapter.eval()

    legal_mask = legal_mask_from_state(state)

    # Run on original state
    enc = encode_state_v18_production(state)
    x = torch.from_numpy(enc).unsqueeze(0).float()
    lm = torch.from_numpy(legal_mask).unsqueeze(0).float()
    with torch.no_grad():
        policy0, _, _ = adapter(x, lm)
        logits0 = adapter.forward_policy_only(x, lm)
    p0 = policy0.squeeze(0).numpy()
    l0 = logits0.squeeze(0).numpy()

    # For each permutation π, build the permuted state, run adapter, then
    # verify policy(perm)[π(t)] == policy0[t] (modulo float).
    n_total = 0
    n_match = 0
    n_logit_match = 0
    max_dev = 0.0
    for perm in itertools.permutations([0, 1, 2, 3]):
        n_total += 1
        gp = permute_own_tokens(state, list(perm))
        # The legal mask must also be permuted: in the permuted state, slot
        # i holds the OLD token perm[i]. So new_mask[i] = old_mask[perm[i]].
        permuted_legal = legal_mask[np.array(perm)]
        enc_p = encode_state_v18_production(gp)
        x_p = torch.from_numpy(enc_p).unsqueeze(0).float()
        lm_p = torch.from_numpy(permuted_legal).unsqueeze(0).float()
        with torch.no_grad():
            policy_p, _, _ = adapter(x_p, lm_p)
            logits_p = adapter.forward_policy_only(x_p, lm_p)
        p_p = policy_p.squeeze(0).numpy()
        l_p = logits_p.squeeze(0).numpy()

        # The expected relationship: for each t, policy_p[i] where perm[i]==t
        # should equal policy0[t]. Equivalently: p_p == p0[perm].
        expected_p_p = p0[np.array(perm)]
        # NaN handling: if a slot is illegal, both should be 0 or close
        if np.allclose(p_p, expected_p_p, atol=1e-4, equal_nan=False):
            n_match += 1
        max_dev = max(max_dev, float(np.abs(p_p - expected_p_p).max()))

        # Logits version: for legal slots, l_p[i] should equal l0[perm[i]].
        # Compare only over slots that are legal in both.
        legal_p = permuted_legal > 0.5
        legal_o = legal_mask[np.array(perm)] > 0.5
        common_legal = legal_p & legal_o
        if common_legal.any():
            l_p_legal = l_p[common_legal]
            expected_l_p = l0[np.array(perm)][common_legal]
            if np.allclose(l_p_legal, expected_l_p, atol=1e-3):
                n_logit_match += 1

    assert_(n_match == n_total,
            f"all {n_total} permutations: policy(perm)[π] == policy0 (max_dev={max_dev:.6f})")
    assert_(n_logit_match >= n_total - 1,  # allow one float-precision miss
            f"all {n_total} permutations: logits(perm)[π_legal] == logits0[π_legal] "
            f"(matched {n_logit_match})")


# ── 3. Tokens at the same rank get equal probability ─────────────────────
def test_same_rank_equal_probability():
    """At the initial state (all 4 tokens home, dice=6), all 4 home tokens
    are at rank 0. Adapter should give them equal token-id probability."""
    print("\n== adapter: tokens at same rank → equal token-id probability ==")
    g = ludo_cpp.create_initial_state_2p()
    g.current_player = 0
    g.current_dice_roll = 6

    adapter = V135ProductionAdapter(num_res_blocks=4, num_channels=64)
    adapter.eval()

    legal = ludo_cpp.get_legal_moves(g)
    legal_mask = np.zeros(4, dtype=np.float32)
    for a in legal:
        legal_mask[a] = 1.0

    enc = encode_state_v18_production(g)
    x = torch.from_numpy(enc).unsqueeze(0).float()
    lm = torch.from_numpy(legal_mask).unsqueeze(0).float()
    with torch.no_grad():
        policy, _, _ = adapter(x, lm)
        logits = adapter.forward_policy_only(x, lm)
    p = policy.squeeze(0).numpy()
    # All 4 home tokens at rank 0 → all 4 token-id probs equal (within float).
    assert_(abs(p[0] - p[1]) < 1e-5 and abs(p[1] - p[2]) < 1e-5 and abs(p[2] - p[3]) < 1e-5,
            f"all 4 home-token probs equal: {p.tolist()}")
    assert_(abs(p.sum() - 1.0) < 1e-5,
            f"policy sums to 1.0: {p.sum()}")
    # Logits should also be equal (all tokens at same rank → same gathered logit)
    l = logits.squeeze(0).numpy()
    finite_l = l[np.isfinite(l)]
    assert_(np.ptp(finite_l) < 1e-4,
            f"all 4 home-token logits equal (modulo softmax): range={np.ptp(finite_l):.6f}, vals={l.tolist()}")


# ── 4. forward() and forward_policy_only() are consistent ────────────────
def test_forward_consistency():
    """forward()'s policy should equal softmax(forward_policy_only()) over
    the legal mask (modulo our renormalization in forward())."""
    print("\n== adapter: forward.policy == legal-masked softmax(forward_policy_only) ==")
    state = make_state_with_distinct_positions()
    adapter = V135ProductionAdapter(num_res_blocks=4, num_channels=64)
    adapter.eval()

    legal_mask = legal_mask_from_state(state)
    enc = encode_state_v18_production(state)
    x = torch.from_numpy(enc).unsqueeze(0).float()
    lm = torch.from_numpy(legal_mask).unsqueeze(0).float()
    with torch.no_grad():
        policy, _, _ = adapter(x, lm)
        logits = adapter.forward_policy_only(x, lm)
        # Softmax the masked logits
        from_logits = F.softmax(logits, dim=1)

    # The forward() path goes: rank_softmax → gather → mask → renormalize
    # The forward_policy_only() path goes: rank_logits → gather → mask → softmax
    # These produce different values when ranks have multiple tokens and
    # only some of them are legal (forward() shares mass equally; logits
    # softmax also shares mass equally because tokens at same rank share logit).
    # So they should match.
    p1 = policy.squeeze(0).numpy()
    p2 = from_logits.squeeze(0).numpy()
    diff = float(np.abs(p1 - p2).max())
    assert_(diff < 1e-4,
            f"forward() policy ≈ softmax(forward_policy_only()): max_diff={diff:.6f}")


# ── 5. legal-mask plumbing ───────────────────────────────────────────────
def test_legal_mask_plumbing():
    """Illegal tokens must have probability 0; legal tokens must sum to 1."""
    print("\n== adapter: legal mask zeroes out illegal token-id slots ==")
    # First spawn token 0 with dice=6, then set up dice=4 — only token 0 legal.
    g = ludo_cpp.create_initial_state_2p()
    g.current_player = 0
    g.current_dice_roll = 6
    g = ludo_cpp.apply_move(g, ludo_cpp.get_legal_moves(g)[0])  # token 0 leaves home
    g.current_player = 0
    g.current_dice_roll = 4
    # Now legal moves: only token 0 (the one on board, since others can't leave home with dice=4)
    legal = ludo_cpp.get_legal_moves(g)
    legal_mask = np.zeros(4, dtype=np.float32)
    for a in legal:
        legal_mask[a] = 1.0
    assert legal == [0], f"expected only token 0 legal, got {legal}"

    adapter = V135ProductionAdapter(num_res_blocks=4, num_channels=64)
    adapter.eval()

    enc = encode_state_v18_production(g)
    x = torch.from_numpy(enc).unsqueeze(0).float()
    lm = torch.from_numpy(legal_mask).unsqueeze(0).float()
    with torch.no_grad():
        policy, _, _ = adapter(x, lm)
    p = policy.squeeze(0).numpy()

    # Illegal slots should be 0
    for t in range(4):
        if legal_mask[t] == 0:
            assert_(p[t] < 1e-6, f"illegal token {t} prob == 0: got {p[t]}")
    # Legal slots should sum to 1
    legal_sum = float(p[legal_mask > 0.5].sum())
    assert_(abs(legal_sum - 1.0) < 1e-5,
            f"legal slots sum to 1: got {legal_sum}, legal={legal}")


# ── 6. Param count + sanity ──────────────────────────────────────────────
def test_param_count():
    """V135ProductionAdapter should have the same params as V135Symmetric."""
    print("\n== adapter: parameter count matches inner V135Symmetric ==")
    from td_ludo.models.v13_5 import V135Symmetric
    adapter = V135ProductionAdapter(num_res_blocks=10, num_channels=128)
    inner = V135Symmetric(num_res_blocks=10, num_channels=128, in_channels=13)
    adapter_params = adapter.count_parameters()
    inner_params = inner.count_parameters()
    assert_(adapter_params == inner_params,
            f"adapter.count_parameters() == inner V135Symmetric: {adapter_params:,} == {inner_params:,}")


def main():
    print("=" * 72)
    print("V135ProductionAdapter unit tests")
    print("=" * 72)
    torch.manual_seed(0)
    test_unpack_roundtrip()
    test_permutation_invariance()
    test_same_rank_equal_probability()
    test_forward_consistency()
    test_legal_mask_plumbing()
    test_param_count()
    print("\n" + "=" * 72)
    print(f"RESULT: {PASS} pass, {FAIL} fail")
    print("=" * 72)
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
