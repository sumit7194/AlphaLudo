# Post-V13.2 Experiments — MCTS and Temporal Context

State as of 2026-05-06. V13.2 is the strongest model in the codebase
(beats V12.2 head-to-head 52.4%, 10K games, p < 0.0001). V13.2,
V13, V14_scalar, V12.2 — four trained architectures — all hit the
same 80-83% eval plateau. This document records the next-experiment
plan.

---

## State of the project

Three architectures, one encoder family, all distilled from the V12.2
teacher (or directly trained as V12.2):

| Model | Architecture | Input | Params | Best Eval | H2H rank |
|---|---|---|---|---|---|
| V12.2 | 3 ResBlocks × 128 + 2 attn × 4 heads | 33ch (V11 engineered) | 1.36M | 82.7% | 2 |
| V13.2 | 10 ResBlocks × 128, pure CNN | 17ch (V14 + 3 V11 static) | 3M | 83.8% | **1** |
| V14_scalar | DeepSets MLP | 73 scalar features | 226K | 80.0% | 3 |
| V13 | 10 ResBlocks × 128, pure CNN | 14ch (raw) | 3M | 79.5% | (not in tournament) |

3-way tournament (10K games per pair, p < 0.0001 each pair):
- V13.2 > V12.2 by 4.8pp (+17 Elo)
- V12.2 > V14_scalar by 5.8pp (+20 Elo)
- V13.2 > V14_scalar by 7.8pp (+27 Elo)

Ranking is transitive. V13.2 is the new strongest model in the project.

## What this convergence implies (and what it does NOT)

**What it implies:**
1. Architecture choice (engineered features + attention vs raw input + deep
   CNN vs scalar features + DeepSets) is not the binding constraint at
   this performance level.
2. The recipe — SL distill from V12.2-bias teacher + RL with bias
   penalties + curriculum gating — is reproducible across architectures.
3. The plateau is structural in the supervised signal + reward shaping +
   opponent ladder.

**What it specifically does NOT imply (corrected by post-tournament
analysis):**
1. *That architecture experiments are exhausted.* All three architectures
   are stateless and single-frame. They share that limitation. The
   "three architectures converge" finding is partly evidence about a
   *shared blind spot* (no temporal context across turns), not just
   evidence that architecture variety doesn't matter.
2. *That MCTS won't help.* Prior MCTS attempts (Exp 9, 13c, 17b, V6.3
   1-ply) used the value head as the dominant leaf evaluator and got
   eaten by dice noise. They were not AlphaZero-style "visit-counts as
   policy target" experiments. AlphaZero's policy improvement comes
   from visit-distribution distillation, which is largely decoupled
   from value-head accuracy.
3. *That the teacher ceiling is fundamental.* V12.2's policy is the
   common ancestor of every model in the project. Self-play with
   policy improvement (MCTS visit counts → new policy target) can
   discover moves the V12.2 teacher never made. That is the actual
   mechanism by which a teacher-bound plateau gets broken.

## The two open hypotheses

### Hypothesis A — MCTS unlocks the teacher ceiling

**Claim:** V13.2's plateau is largely set by V12.2's distilled policy
distribution. Self-play with AlphaZero-style MCTS produces visit
distributions that are *policy improvements* over the network's prior.
Distilling those visit distributions back into the network gives a
policy V12.2 never had.

**Why prior failures are not relevant evidence:**
- Used value head as the dominant leaf evaluator (eaten by dice noise).
- Cold-started from random init (AlphaZero starts random, but with
  thousands of TPU-days of compute we don't have).
- Several were tested before BASE_COORDS / encoder rotation / value-head
  inversion bugs were fixed.

**Why now is different:**
- V13.2 has a calibrated value head (82-83% bot-eval, sigmoid+BCE-trained
  on real outcomes).
- Encoder symmetry is fixed (canonical post-fix encoder, P0 view ≡ P2
  view at the same physical state).
- We have four converged checkpoints to seed search, not one
  random-init network.
- Modern AlphaZero loss formulation: visit counts → policy KL target,
  outcomes → value BCE target. No PPO mixed in.

**Cost:** at 50 simulations per move, training throughput drops from
~200 GPM to ~5-10 GPM. ~40-80× slower. 100K MCTS games on the L4 VM ≈
10-14 days wall clock.

**Failure mode (audit before committing):** if V13.2's value head is
miscalibrated on advanced self-play states (where the model spends
most of its inference budget), MCTS exploration is systematically
misled by bad leaf estimates and we plateau lower than V13.2.
*Phase 0 calibration audit is required.*

### Hypothesis B — Temporal context unlocks state-tracking

**Claim:** Every architecture in the project is stateless (sees only
the current state). Information that lives across turns — opponent's
recent dice luck, what they just moved, your own consecutive-sixes
streak, whether the opponent has been ditching their laggard, etc —
must be re-derived from current state every forward pass. A model with
last-K turns of history fed in could exploit temporal patterns the
stateless models structurally cannot.

**Why prior temporal attempts are not strong evidence:**
- V8 (only prior temporal architecture) was on V5-era weak base. Tested
  before encoder fixes, before V11 idle/streak channels, before V12.2
  recipe. Not a clean test of "temporal helps at strong base."

**What we'd test:** transformer encoder over K=8 past turns, each turn
encoded with V13.2's 17ch frame, distilled from V13.2 (so the
non-temporal baseline is V13.2 itself), then RL with same recipe.

**Cost:** more implementation effort than MCTS — needs new
history-tracking encoder, SL data regen with history, RL buffer changes.
But training throughput is comparable to current models (no per-move
search overhead). 1-2 weeks of work + standard training time.

## Decision: MCTS first, transformer second

**Reasons MCTS goes first:**
1. **Directly attacks the teacher ceiling** — the architecture
   experiments don't.
2. **Infrastructure mostly exists** — Exp 24 search code, C++ MCTSEngine
   from Exp 9 era can be revived with bug fixes.
3. **Bounded downside** — if it fails clean with all the bug fixes in
   place, MCTS is definitively ruled out for this codebase. That's
   itself a valuable result.
4. **AlphaZero is the project's stated inspiration** — closing this
   loop is structurally important regardless of outcome.

**Reasons transformer goes second:**
1. **High information value** but more implementation work.
2. **Conditional on MCTS outcome:**
   - If MCTS unlocks the plateau, the temporal experiment becomes a
     follow-up question (do we still need history?) rather than a
     primary plateau-breaker.
   - If MCTS fails clean, temporal becomes the obvious next move
     (everything else holds at the plateau).

## The four-step MCTS plan

### Step 0 — Value-head calibration audit (Day 0, ~1 hour)

Before committing 10-14 days of GPU compute to MCTS RL, verify that
V13.2's value head is well-calibrated on the kinds of states MCTS
will explore.

**Procedure:**
1. Take `checkpoint_backups/v132_*/model_latest.pt` (G=257K, best eval
   83.8%). Freeze it.
2. Run 1000 self-play games of V13.2 vs V13.2 with the SAME
   weights. Random dice. Greedy moves.
3. For every state, record `(V_network, eventual_outcome)` from the
   current player's POV.
4. Bin V_network into 10 deciles. Plot empirical win-rate per bin.
5. Compute calibration error: `mean(|V_predicted - WR_empirical|)`.

**Pass criterion:** calibration error ≤ 5pp across all 10 deciles. So
when V=0.3, empirical WR ≈ 25-35%. When V=0.7, empirical WR ≈ 65-75%.

**If pass:** value head is fit for MCTS use. Proceed to Step 1.

**If fail:** value head is badly miscalibrated on self-play states.
Either retrain the value head only (freeze backbone, train value head
on more diverse self-play data) or accept that MCTS quality will be
bounded by value-head noise.

This is a 1-hour Python script. Worth running before any of the bigger
commits.

### Step 1 — Search-augmented distillation as cheap MCTS proxy (Days 1-3, local Mac)

Before running full AlphaZero-style MCTS RL on the VM, test a much
cheaper variant: 1-ply expectimax + V13.2's value head as the search
target.

**Why this works as a proxy:** AlphaZero's mechanism is "search
produces policy improvement, distill the improvement back into the
network." 1-ply expectimax is the simplest version of that mechanism.
If 1-ply gives a clear lift in H2H vs V13.2, full MCTS is likely to
give more. If 1-ply gives 0 lift, the expensive MCTS RL is unlikely
to either — kills the experiment cheaply.

**Procedure:**
1. Take V13.2-best. Generate 1M self-play states. For each state, run
   1-ply expectimax: enumerate all 4 own moves, weight each by
   `1/6 × Σ_dice V13.2.value(after_move_with_dice)`. Take argmax as
   `search_action` and the weighted value as `search_value`.
2. Train a fresh student on `(state, search_action, search_value)`
   tuples. Use V13.2 architecture (10×128 17ch). SL distillation,
   ~5M states, batch 1024, lr 1e-3 → 1e-4.
3. Run 10K-game H2H tournament: 1-ply-distilled student vs V13.2.

**Pass criterion:** student wins ≥ 53% of H2H (>+10 Elo, p < 0.001
at 10K games). That's a meaningful lift over V13.2's distillation
of the V12.2 teacher.

**If pass:** strong evidence that search-improved targets break the
teacher ceiling. Proceed to Step 2 with high confidence.

**If fail:** 1-ply doesn't extract teacher-improvement. Full MCTS is
unlikely to either. Skip to Step 3 (transformer).

This is 2-3 days of work on the local Mac. Cost: cheap. Information
value: high.

### Step 2 — Full AlphaZero-style MCTS RL (Days 4-18, VM)

Conditional on Step 1 passing.

**Setup:**
- Warm-start from V13.2 `model_latest.pt` (G=257K) — NOT from
  `model_sl.pt`. The RL-trained checkpoint has a calibrated value head
  worth seeding the tree with.
- Add Dirichlet noise at root: `α=0.3, ε=0.25` (AlphaZero's chess
  defaults). Compensates for the lower entropy of an RL-trained policy
  during MCTS exploration.
- 50 simulations per move minimum. If signal is clear at 50, ramp to
  100, then 200. Don't start at 800 — that's a 16× slower training
  run before knowing the recipe works.
- Dice as chance nodes: enumerate all 6 dice values at each chance
  node, weight backed-up value by `1/6 × Σ V(child)`. More
  sample-efficient than dice sampling at small simulation budgets.
- AlphaZero loss: `α_policy · KL(π_search || π_network) + α_value ·
  BCE(V_network, outcome)`. Both weights = 1.0. **PPO disabled.**
  Pure search-improvement training.
- Bias penalties: **kept ON for Phase 1**, dropped for Phase 2 (see
  below).

**Two-phase training:**
- **Phase 1 (~30K games, ~3-4 days):** MCTS + bias penalties + AlphaZero
  loss. Goal: validate that the policy actually diverges from V13.2's
  initial distribution. **Diagnostic:** measure
  `KL(π_mcts || π_v132_initial)` averaged over states. If > 0.1,
  search is finding new moves. If < 0.05, MCTS is just reproducing
  the prior — abort.
- **Phase 2 (~70K games, ~7-10 days):** if Phase 1 shows divergence
  AND eval WR is maintained or improved, drop bias penalties and
  continue. Now the model is doing pure search-driven policy
  improvement.

**Comparison protocol (after each phase):**
- 10K-game H2H tournament: MCTS-trained model vs V13.2.
- σ at 10K games ≈ 0.5pp. A real lift is ≥ +3pp / +10 Elo.

**Failure modes:**
- *KL stays low (< 0.05) in Phase 1:* MCTS isn't finding new moves.
  Either sims/move is too low, or the value head is poor enough that
  the tree always returns to prior. Either way, this is the cleanest
  "MCTS doesn't help here" signal.
- *KL diverges but WR drops:* MCTS is finding moves that look good to
  the noisy value head but lose actual games. Calibration audit was
  too lenient — value head needs improvement before MCTS can help.
- *KL diverges and WR holds at Phase 1 but drops at Phase 2:* bias
  penalties were structural, MCTS isn't a full replacement for them.
  Keep penalties, accept a smaller plateau-break.

### Step 3 — Transformer with K-turn history (Days 19+, only if MCTS doesn't break the plateau)

If Steps 1-2 don't yield a meaningful lift, we've ruled out search and
the next bet is temporal context.

**Setup (sketched, not fully designed):**
- Encoder: K=8 past turns, each encoded as V13.2's 17ch frame.
  History buffer fed to transformer encoder with causal masking.
- Architecture: 4 transformer layers, 128-dim, 4 heads. ~1-2M params.
  Per-turn 17ch goes through a small CNN front-end, then transformer
  attends across the K turns.
- SL distillation: V13.2 generates self-play with full history. Train
  on `(history_K, current_state, V13.2.policy, V13.2.value, outcome)`.
- RL: same recipe as V13.2 (bias penalties + curriculum + PPO), but
  the trainer now needs to maintain history buffers per game.

**What to test for:**
- Does the temporal model beat V13.2 in H2H? (10K-game tournament).
- Does it specifically improve on multi-turn-pattern states (e.g.
  "opponent has rolled 6 twice in a row, what should I do")?

This step is more design work and is conditional on the MCTS outcome.
The detailed design becomes a separate document if/when we get here.

## Risks and open questions

### Methodology risks

- **Bot-eval saturation we keep hitting:** going forward, all
  architecture comparisons should use H2H tournaments (10K+ games per
  pair), not bot-eval WR. Bot evals at 80-83% can hide ≥ 5pp real
  skill differences.
- **Stochastic environment + small N comparisons:** 50-game previews
  routinely give numbers that swing 10pp. The discipline is to never
  draw conclusions from < 1000 games per pair, and to use 10K for any
  release-grade comparison.
- **Reproducibility of the recipe:** the "SL distill V12.2 → RL with
  bias penalties + curriculum" recipe has now produced 4 models in the
  same plateau. Any new architecture experiment should use the same
  recipe and seek to *break* the plateau, not match it.

### Open questions

- *Is V12.2 even the right teacher anymore?* V13.2 is now the strongest
  model. Future distillation experiments should consider V13.2-best as
  the teacher. The first MCTS-trained model would be a candidate to
  replace V13.2 as the teacher for future SL.
- *What's the ceiling of the bot ladder itself?* If a god-tier policy
  beats Heuristic 99% but Expert 95%, the bot evaluations have a
  hard ceiling around 95-98%. We don't know empirically what that
  ceiling is. Worth measuring once per generation.
- *Pure-RL-from-scratch as a control for MCTS:* if MCTS shows a lift
  over V13.2, is it because of search or because of more training
  time? Need a control: V13.2-recipe extended for the same wall-clock
  as MCTS. If pure-RL also lifts comparably, MCTS isn't doing the
  work.

### Compute budget

Across all four steps, rough estimates:
- Step 0: 1 hour Python. Local.
- Step 1: 2-3 days local Mac (CPU/MPS).
- Step 2: 10-14 days L4 VM.
- Step 3 (if reached): 2 weeks + design.

Total project time-to-decision on plateau-break: ~4 weeks.

If the budget is tighter, Steps 0 + 1 alone (4 days, all local) give a
strong pre-commit signal on whether MCTS is worth the L4 commit.

## Document scope

This document captures the post-V13.2 plan as of the V12.2 vs V13.2 vs
V14_scalar tournament conclusion (2026-05-06). The next document
should be either:
- An MCTS Step 0 + Step 1 results report (if those steps are pursued
  next), OR
- An updated post-V13.2 plan if priorities shift.

The MODEL_HISTORY.md and training_journal.md remain the long-form
records of what was actually done. This file is the forward-looking
plan.
