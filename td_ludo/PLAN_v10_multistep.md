# V10 Multi-Step Planning — Investigation Plan

Date: 2026-04-23. All 4 architectures (V6/V6.1/V6.3/V10) converge at 77-79% WR and share the same blind spot: no multi-turn reasoning. Human benchmark (Exp 13e) + V10 mech interp (77.7% dice flip rate) confirm the models are pure reactive classifiers.

This plan proposes three ordered experiments, each dependent on the outcome of the previous.

---

## Problem (precise)

All current models compute `f(state, dice) → token`. Mech interp shows:
- **77.7% of states flip preferred action** when dice changes → model is almost entirely dice-determined
- **Linear probe `eventual_win` caps at 71.5%** → backbone encodes current advantage but cannot simulate future
- **No temporal input** — each decision is independent of history
- **Bonus-turn blind spot**: model sees "dice = 6" on ch24 but doesn't reason "therefore I will get another turn to do Y"

The game-theoretic core: Ludo has two sources of branching — my actions (factor 1-4) and dice (factor 6, uniform-random). MCTS treats dice as adversarial (wrong); expectimax treats dice as chance nodes (correct).

---

## Order 1 — Inference-time Expectimax-2 (low-cost plateau break attempt)

**Hypothesis**: V10's calibrated `win_prob` head (Brier 0.17, calibration buckets well-aligned) can drive a **2-ply expectimax search** where V6.3's uncalibrated value head failed. Shallow search avoids the exponential blow-up that killed MCTS.

**Why past attempts failed (and why this is different)**:

| Attempt | Why it failed | V10 advantage |
|---|---|---|
| Exp 9 MCTS training | AlphaZero-style, 200 sims in branching-6 tree = noise; terminal z labels too noisy | We don't train, just use search at inference |
| Exp 13c inference MCTS | V6.1 value head unbounded, `torch.clamp(-1,1)` lossy for pUCT; adversarial dice treatment | V10 win_prob bounded [0,1] natively, **expectimax** handles dice correctly |
| V6.3 1-ply value search | Value head = normalized returns, not P(win) → 27.5% WR | V10 win_prob trained with BCE on actual outcomes |

**Algorithm (2-ply expectimax, MAX-CHANCE-MAX structure)**:
```
1. For each of my legal actions a_i:
   2. Apply a_i → state S_i (opponent's turn, their dice unknown)
   3. For each opp dice d_j ∈ {1..6}:
      4. For each opp legal action b_k (weighted by opp's policy π_opp):
         5. Apply b_k → leaf state L_ijk
         6. Evaluate: win_prob(L_ijk), perspective-flipped to my frame
   7. Expected value of a_i = (1/6) * Σ_j Σ_k π_opp(b_k | S_i, d_j) * value(L_ijk)
8. Pick a_i with max expected value
```

**Optimizations to minimize Python-C++ traffic**:
- Single batched forward pass per decision — collect all leaf encodings into one numpy array, one `model(tensor)` call
- Enumerate leaves sequentially in Python but batch the heavy work (encode + forward)
- Typical decision: ~3 my actions × 6 dice × ~3 opp actions = **~54 leaves** → one batched forward
- Expected decision latency: ~3-5ms on MPS (vs ~0.5ms for greedy policy)
- **Full 500-game eval: ~5-10 minutes** (vs ~2 min for greedy)

**Success criterion**:
- 500-game eval vs bot mix (same bots as evaluate_v10.py)
- **+3pp over V10 greedy baseline** = meaningful gain → proceed to Order 2
- **Equal or worse** = search can't extract more signal from V10's value → abandon search path, go to Order 3
- **Partial** (+1 to +2pp) = helpful but not game-changing — try 3-ply expectimax; if still weak, go to Order 3

**Cost**: ~1 day to implement + evaluate.

---

## Order 2 — Expectimax distillation into policy (AlphaZero-lite)

**Activated only if Order 1 shows ≥ 3pp improvement.**

**Hypothesis**: If expectimax policy is better than greedy policy, we can train a standalone policy network to **imitate the expectimax choices** directly. Result: expectimax-level play at greedy-level inference speed.

**Algorithm**:
1. Run expectimax on a large set of decision states (e.g., 500K states from V10 self-play)
2. For each state, record the expectimax-preferred action + its value estimate
3. Train a fresh V10 model via SL:
   - Policy loss: KL(expectimax_policy || student_policy)
   - Value loss: BCE(win_prob, expectimax_value) — tighter than terminal outcome labels
4. Evaluate distilled V11 vs V10 (both greedy)
5. Optionally iterate: distilled model → new expectimax teacher → new distillation

**Why this should work better than SL-from-V6.1**:
- Teacher (expectimax) is strictly stronger than the base policy
- Training target includes search-derived value estimates, not just outcomes → better value head calibration
- Iterative bootstrap possible

**Success criterion**: distilled model ≥ expectimax-v10 at ~same inference speed → sustained plateau break.

**Cost**: ~2-3 days (data gen + training + eval).

---

## Order 3 — V11 Temporal Context (architectural rewrite)

**Activated only if both Order 1 and Order 2 fail or produce < 2pp gain.**

**Hypothesis**: The 77-79% ceiling is due to **reactive-frame architecture itself**. A model that sees K turns of history can detect opponent patterns (stalling, stack-building, threat persistence) that single-frame CNNs cannot.

**Architecture sketch** (from journal's V7 spec):
- Input: sequence of last K=8 game states, each as 28-channel frame
- Backbone options:
  - (A) **Temporal CNN**: Conv3D over (K, H, W) with channel dim
  - (B) **Frame-embedding + Transformer**: each frame → dense embedding via shared CNN, then self-attention across frames
  - (C) **Frame-embedding + GRU**: same but recurrent aggregation (cheaper)
- Heads: same 3-head (policy + win_prob + moves_remaining)
- Training:
  - SL: V6.1 teacher self-play, record last K states per decision
  - RL: PPO as before, but trajectory step stores K-frame window

**Success criterion**: V11 evals beat V10 plateau by ≥ 5pp sustained.

**Cost**: ~1-2 weeks (new data pipeline, new model, new training loop, new eval).

**Risk**: Ludo is Markov — in theory you don't need history. But practically, good play requires inferring opponent intent across turns, which is a cheap proxy for expectimax-style contingency planning.

---

## Success gate matrix

| Outcome | Conclusion |
|---|---|
| Order 1 works (≥ +3pp) | V10 has untapped capability. Go to Order 2. |
| Order 1 fails, but win_prob IS calibrated | Search can't beat the reactive policy on this state distribution. Skip to Order 3. |
| Order 1 fails AND win_prob shows miscalibration under search | Fundamental value-function problem — retrain SL with calibrated-value loss, then retry Order 1. |
| Order 1 partial (+1-2pp) | Try 3-ply search. If still weak, go to Order 3. |
| Order 2 beats Order 1 at inference | V11-class model via distillation. Stop. |
| Order 2 matches but doesn't exceed Order 1 | Search was the value-add, not distillation. Keep search at inference, ship as-is. |
| Order 3 works | Architectural fix confirmed. Temporal context was the missing piece. |
| Order 3 fails too | Accept 77-79% as the Ludo game ceiling. Human benchmark + qualitative analysis to conclude. |

---

## Implementation notes

### Python ↔ C++ communication overhead

Each `ludo_cpp.apply_move(state, action)` call crosses the Python-C++ boundary. For expectimax with ~54 leaves per decision:
- Naive: 54 × 2 `apply_move` calls + 54 `encode_state_v10` calls = 162 round-trips per decision
- Batched: enumerate leaves in Python, batch the expensive work (encoding + forward pass):
  - 162 small C++ calls is actually fine — each is ~10μs = 1.6ms total
  - ONE batched forward pass on 54 states = ~1-2ms on MPS
  - **Total per decision: ~3-4ms** — acceptable

**Potential C++-side optimization** (only if needed):
- Add `enumerate_expectimax_leaves(state, depth)` binding that returns all leaf state tensors in one call
- Saves the 162 round-trips → 1 round-trip
- Gain: maybe 1-2ms per decision
- **Deferred**: not worth the complexity unless benchmark shows it's a bottleneck

### Checkpoint to use

`checkpoints/ac_v10/model_latest.pt` (game 297,018, 77.6% best eval in that run). Not `model_best.pt` — per user's argument, `latest` has 52K more games of training.

### Evaluation harness

`evaluate_v10_expectimax.py` — mirrors `evaluate_v10.py` signature for drop-in comparison. Returns same dict shape so results are directly comparable.

---

## Next action

Implement `evaluate_v10_expectimax.py` with:
- Depth=1 (value search) and Depth=2 (expectimax) as CLI flag
- Batched forward pass per decision
- Proper perspective flip on opponent turns
- Output compatible with existing eval scripts
- Run 500 games each: greedy (baseline), depth-1 search, depth-2 expectimax

Commit plan + script separately so the thinking is preserved even if the experiment fails.
