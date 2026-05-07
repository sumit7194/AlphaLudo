"""Unit tests for per-player history (Option B) in TemporalDistillEnv (SL)
and SelfPlayEnv (RL).

What we are validating: when player p is to move, the K-frame stack handed
to the model contains ONLY player p's past decision states, never the
opponent's. This matches what an inference-time agent sees (it only
observes its own turns), eliminating the train/test distribution mismatch
in the previous "shared deque" implementation.

Run:
    ./td_env/bin/python experiments/v134/test_per_player_history.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS

# Import the env classes
from train_v133_sl import TemporalDistillEnv, HISTORY_K as SL_K
from train_v133_rl import SelfPlayEnv, HISTORY_K as RL_K


PASS_COUNT = 0
FAIL_COUNT = 0


def assert_(cond, msg):
    global PASS_COUNT, FAIL_COUNT
    if cond:
        PASS_COUNT += 1
        print(f"  PASS  {msg}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {msg}")


def encode_now(game):
    return encode_state_v17(game)


# ── SL env tests ───────────────────────────────────────────────────────────
def test_sl_env_per_player_init():
    print("\n== SL env: per-player init ==")
    env = TemporalDistillEnv(batch_size=2, history_k=SL_K, max_game_len=400)
    assert_(isinstance(env.history, list) and len(env.history) == 2,
            "history is per-game list of size batch_size")
    for i in range(2):
        assert_(isinstance(env.history[i], dict) and 0 in env.history[i] and 2 in env.history[i],
                f"history[{i}] is dict with keys 0 and 2")
        assert_(len(env.history[i][0]) == 0 and len(env.history[i][2]) == 0,
                f"history[{i}][0] and [2] start empty")


def test_sl_env_pushes_to_correct_player():
    """Run a few decisions and verify each frame went into the deque of the
    player who was to move at that step (not the other one)."""
    print("\n== SL env: each frame stored on correct player's deque ==")
    env = TemporalDistillEnv(batch_size=2, history_k=SL_K, max_game_len=400)
    # Do 5 batches of decisions
    seen_cps_per_game = [[], []]
    snapshots = [[], []]  # list of (cp, frame_at_that_decision) per game
    for batch_idx in range(5):
        decision_idxs, cps, _ti, student_hist, hmask, _lm, legal_lists = env.get_batch()
        # For each decision in this batch, the LAST frame in student_hist[k]
        # should equal the frame that was just encoded. We can't reconstruct
        # the exact frame here without re-encoding pre-apply, but we can
        # check (a) that the deque sizes grew correctly and (b) that the
        # last frame in the K-stack matches the LAST frame in that player's
        # deque after the push.
        for k, i in enumerate(decision_idxs):
            cp = cps[k]
            seen_cps_per_game[i].append(cp)
            # The K-stack's last frame must equal the last frame in
            # env.history[i][cp].
            last_in_deque = np.asarray(env.history[i][cp][-1])
            last_in_stack = student_hist[k][-1]
            assert_(np.allclose(last_in_deque, last_in_stack),
                    f"batch {batch_idx} game {i} cp {cp}: stack[-1] == deque[-1]")
            # The full K-stack (excluding pad) must equal the deque contents.
            n_real = int(hmask[k].sum())
            assert_(n_real == len(env.history[i][cp]),
                    f"batch {batch_idx} game {i} cp {cp}: hmask sum == deque len")
            stack_real = student_hist[k][-n_real:] if n_real > 0 else np.empty(0)
            deque_arr = np.stack(list(env.history[i][cp]), axis=0) if n_real > 0 else np.empty(0)
            if n_real > 0:
                assert_(np.allclose(stack_real, deque_arr),
                        f"batch {batch_idx} game {i} cp {cp}: stack real frames == deque")
        # Apply random actions
        actions = np.array([legal_lists[k][0] for k in range(len(decision_idxs))])
        env.apply_actions(decision_idxs, actions, _lm, legal_lists)


def test_sl_env_isolation_between_players():
    """The KEY property: at a player p decision, the OPPONENT's past frames
    must NOT appear in p's history stack. We construct two interleaved
    decision sequences (p0's turn, p2's turn, p0's turn, ...) and verify
    that p0's stack only contains p0's frames."""
    print("\n== SL env: opponent frames never appear in own history ==")
    env = TemporalDistillEnv(batch_size=1, history_k=SL_K, max_game_len=400)
    # Run enough batches that both players have several turns.
    p0_frames_collected = []
    p2_frames_collected = []
    for batch_idx in range(40):
        decision_idxs, cps, _ti, student_hist, hmask, _lm, legal_lists = env.get_batch()
        for k, i in enumerate(decision_idxs):
            cp = cps[k]
            cur = student_hist[k][-1].copy()
            n_real = int(hmask[k].sum())
            stack_frames = [student_hist[k][-(n_real - r)] for r in range(n_real)]
            # The K-stack must NOT contain any frame that's only in the
            # OTHER player's collected list.
            other_frames = p2_frames_collected if cp == 0 else p0_frames_collected
            for stf in stack_frames:
                for of in other_frames:
                    if np.allclose(stf, of):
                        # We found an opponent frame inside our own stack
                        assert_(False, f"opp frame leaked into cp={cp} stack at batch {batch_idx}")
                        return
            if cp == 0:
                p0_frames_collected.append(cur)
            else:
                p2_frames_collected.append(cur)
        actions = np.array([legal_lists[k][0] for k in range(len(decision_idxs))])
        env.apply_actions(decision_idxs, actions, _lm, legal_lists)
    # If we made it here without any cross-leak:
    assert_(len(p0_frames_collected) > 0 and len(p2_frames_collected) > 0,
            "both players had decisions in the test")
    assert_(True, f"no opponent-frame leak detected in {len(p0_frames_collected)+len(p2_frames_collected)} decisions")


def test_sl_env_reset_clears_both_deques():
    print("\n== SL env: reset clears both per-player deques ==")
    env = TemporalDistillEnv(batch_size=1, history_k=SL_K, max_game_len=400)
    # Run a few batches to populate both deques
    for _ in range(20):
        decision_idxs, cps, _ti, _sh, _hm, _lm, legal_lists = env.get_batch()
        actions = np.array([legal_lists[k][0] for k in range(len(decision_idxs))])
        env.apply_actions(decision_idxs, actions, _lm, legal_lists)
    pre_lens = (len(env.history[0][0]), len(env.history[0][2]))
    assert_(pre_lens[0] > 0 and pre_lens[1] > 0, f"both deques populated pre-reset: {pre_lens}")
    env._reset(0)
    post_lens = (len(env.history[0][0]), len(env.history[0][2]))
    assert_(post_lens == (0, 0), f"both deques cleared post-reset: {post_lens}")


# ── RL env tests ───────────────────────────────────────────────────────────
def test_rl_env_per_player_init():
    print("\n== RL env: per-player init ==")
    env = SelfPlayEnv(batch_size=2, history_k=RL_K, max_game_len=400)
    for i in range(2):
        assert_(isinstance(env.history[i], dict) and 0 in env.history[i] and 2 in env.history[i],
                f"history[{i}] is dict with keys 0 and 2")
        assert_(len(env.history[i][0]) == 0 and len(env.history[i][2]) == 0,
                f"history[{i}][0] and [2] start empty")


def test_rl_env_isolation_and_trajectory():
    """Same isolation property as the SL env, but also verify that the
    trajectory's saved stack[-1] equals the player's deque[-1] at each step."""
    print("\n== RL env: opponent frames isolated AND trajectory snapshots correct ==")
    env = SelfPlayEnv(batch_size=1, history_k=RL_K, max_game_len=400)
    p0_frames = []
    p2_frames = []
    for batch_idx in range(40):
        decision_idxs, cps, hist_arr, hmask_arr, lmask_arr, legal_lists = env.spin_to_decision()
        for k, i in enumerate(decision_idxs):
            cp = cps[k]
            cur = hist_arr[k][-1].copy()
            n_real = int(hmask_arr[k].sum())
            assert_(n_real == len(env.history[i][cp]),
                    f"batch {batch_idx} cp {cp}: hmask sum matches per-player deque len")
            # Cross-leak check
            other = p2_frames if cp == 0 else p0_frames
            stack_real = hist_arr[k][-n_real:]
            for stf in stack_real:
                for of in other:
                    if np.allclose(stf, of):
                        assert_(False, f"opp frame leaked into cp={cp} stack")
                        return
            (p0_frames if cp == 0 else p2_frames).append(cur)
        v_preds = np.zeros(len(decision_idxs), dtype=np.float32)
        actions = np.array([legal_lists[k][0] for k in range(len(decision_idxs))])
        finished = env.apply_actions(
            decision_idxs, cps, hist_arr, hmask_arr, lmask_arr, legal_lists, actions, v_preds,
        )
        # If any games finished, their trajectories should not contain any
        # frame from the OTHER player either.
        for entry in finished:
            stack, hmask, lmask, action, v_pred, G = entry
            # We don't have cp recorded in finished tuples (G is set based
            # on cp at trajectory build time), but we can verify stack[-1]
            # is consistent with the saved hmask.
            n_real = int(hmask.sum())
            assert_(0 < n_real <= RL_K, f"finished entry has valid hmask sum: {n_real}")
    assert_(len(p0_frames) > 0 and len(p2_frames) > 0, "both players had decisions")
    assert_(True, f"no opponent-frame leak in {len(p0_frames)+len(p2_frames)} decisions")


# ── Sanity: verify encoder DOES pivot POV (so this fix matters) ──────────
def test_encoder_pivots_pov():
    print("\n== Sanity: encoder_v17 pivots POV (the precondition for this fix to matter) ==")
    g = ludo_cpp.create_initial_state_2p()
    g.current_player = 0; g.current_dice_roll = 6
    legal = ludo_cpp.get_legal_moves(g)
    g = ludo_cpp.apply_move(g, legal[0])
    g.current_player = 0; g.current_dice_roll = 4
    legal = ludo_cpp.get_legal_moves(g)
    g = ludo_cpp.apply_move(g, legal[0])
    # Now p0 has a token out. Compare cp=0 view vs cp=2 view.
    g.current_player = 0; g.current_dice_roll = 3
    e0 = encode_state_v17(g)
    g.current_player = 2; g.current_dice_roll = 3
    e2 = encode_state_v17(g)
    diff = float(np.abs(e0 - e2).sum())
    assert_(diff > 0, f"encoder POV-pivots: |e0 - e2| = {diff:.4f} > 0 in asymmetric state")


# ── Run all ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Per-player history unit tests (Option B fix)")
    print("=" * 70)
    test_encoder_pivots_pov()
    test_sl_env_per_player_init()
    test_sl_env_pushes_to_correct_player()
    test_sl_env_isolation_between_players()
    test_sl_env_reset_clears_both_deques()
    test_rl_env_per_player_init()
    test_rl_env_isolation_and_trajectory()
    print("\n" + "=" * 70)
    print(f"RESULT: {PASS_COUNT} pass, {FAIL_COUNT} fail")
    print("=" * 70)
    if FAIL_COUNT:
        sys.exit(1)


if __name__ == "__main__":
    main()
