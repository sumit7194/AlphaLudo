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

---

# Locked-in plan (2026-05-06 decisions)

The above sections are the strategic rationale. This section is the
**actionable spec** with every design choice committed, post planning
session.

## Audit of existing C++ MCTS code (`src/mcts.cpp/h`, ~518 LOC)

Reviewed in detail before deciding to use it. Concept is correct
(chance-node averaging, adversarial perspective flip in UCB, Dirichlet
noise, virtual loss for parallelism). But found **4 specific bugs that
would have killed previous MCTS attempts**:

1. **Hardcoded V6.1 24-channel encoder in `get_leaf_tensors`** (lines
   397, 414): `write_state_tensor_v6` is called with a hardcoded 24×15×15
   leaf tensor size. Fatal for V13.2 (needs 17ch) and wrong for V12.2
   (needs 33ch). The MCTS engine produces wrong inputs to the value
   head regardless of which model is loaded.
2. **Chance-node value averaging weights by visited children only**, not
   by true 1/6 over all dice outcomes. Biased estimate when not all 6
   dice values have been explored at a chance node.
3. **Ambiguous policy input format.** Comment says "Already a
   probability, don't exp()!" implying it expects softmaxed probs from
   the network. Bug-prone — if a caller passes logits, search returns
   garbage.
4. **6-roll bonus turn handling unclear.** When dice == 6 in Ludo, the
   same player gets another turn (`current_player` doesn't change
   across `apply_move`). The perspective-flip logic checks parent vs
   child `current_player`, which is correct for that case (no flip),
   but the chance-node trigger checks `current_dice_roll == 0`,
   assuming that means "next roll needed" — the state-machine
   assumption is fragile.

**Decision: ignore the existing C++ MCTS, write fresh Python code.**

Reasons:
- These bugs explain why prior MCTS experiments (Exp 9, 13c, 17b) all
  failed without obvious root cause.
- We are not speed-constrained (VM has L4 GPU, training time per game
  isn't the bottleneck for first MCTS RL run).
- Python MCTS with proper batched inference and `VectorGameState` for
  parallel games is fast enough to run 100K games in 2 weeks on the L4.
- A correct Python reference implementation is the prerequisite for
  any future C++ port (Step 2.5, optional, only if results justify
  production-grade speed).

The existing C++ MCTS stays in the tree as historical record.

## Device assignments

| Workload | Device | Why |
|----------|--------|-----|
| Code development + tests | Local Mac (MPS / CPU) | Fast iteration |
| Step 0 calibration audit | VM (L4 GPU) | Vectorized 200-game-batch inference; ~5-15 min on L4 vs ~30 min serial on Mac CPU |
| Step 1 search-data generation | VM (L4 GPU) | 96 V-evals per state batched into one forward → 96M total for 1M states; needs GPU |
| Step 1 SL distillation | VM (L4 GPU) | Standard SL training |
| Step 1 H2H tournament | Local Mac CPU | Tournament code is single-game (batch=1 inference); CPU beats GPU due to kernel-launch overhead |
| Step 2 MCTS RL | VM (L4 GPU) | Massive batched leaf inference per simulation |
| Step 2 H2H tournament | Local Mac CPU | Same reason as Step 1 H2H |
| Step 2.5 C++ port (optional) | Local for dev, VM for run | Standard C++ workflow |

**Latest over best:** all model references in this plan use V13.2's
`model_latest.pt` (most recent training step), not `model_best.pt`
(highest eval-WR snapshot). Operational reality is we ship latest.

## Step 0 — Calibration audit (locked spec)

**Subject:** V13.2 `model_latest.pt` pulled from VM after pausing the
ongoing RL run.

**Procedure:**
- Run **5,000 self-play games** of V13.2 vs V13.2 with stochastic
  policy (`τ=1.0`, sample from softmax). Vectorized via
  `VectorGameState(batch_size=200)`.
- Record `(V_pred from current player POV, current_player)` at every
  decision state. Discard the first 10 turns of each game (game-start
  states are too easy and dominate the data).
- Walk back from each game's outcome and label every recorded state
  with `eventual_outcome` from the recorder's POV.
- Bin V_pred into 10 equal-frequency deciles. Per bin:
  - Count of samples (need ≥ 1000 per bin for tight CI)
  - Empirical WR = `mean(eventual_outcome)`
  - Mean V_pred
- Compute Brier score = `mean((V_pred - outcome)²)`.
- Compute ECE = `Σ |WR_bin - V_bin| × n_bin / N`.
- Generate plot: empirical WR vs V_pred per bin, with `y=x` reference.

**Pass criteria:**
- ECE ≤ 5pp
- No bin > 10pp deviation from `y=x`
- Brier ≤ 0.20 (random-chance baseline is 0.25)

**If pass:** value head is fit for MCTS use → proceed to Step 1.

**If fail (ECE > 10pp or Brier > 0.22):** retrain value head only.
Freeze backbone, train value head on 1M+ self-play states with BCE on
outcomes. ~1-2 days. Then re-audit.

**If marginal (5pp < ECE ≤ 10pp):** proceed to Step 1 with caution.
Note in results that value-head quality is suboptimal.

## Step 1 — Search-augmented distillation (locked spec)

**Hypothesis:** 2-ply expectimax search produces policy improvements
over V13.2's prior. Distilling those improvements into a fresh student
yields a stronger model than vanilla V13.2 distillation.

**Cost:** 2-3 days total wall time.

### Generator (`generate_search_data.py`)

- Load V13.2-latest frozen.
- Self-play games via `VectorGameState(batch_size=200)`. V13.2 vs V13.2
  (matches what Step 2 MCTS will explore).
- Stochastic move selection during data collection (`τ=1.0`) so the
  data covers diverse states.
- At each visited state, run **2-ply expectimax** (own action → opp
  dice → opp action → V at leaf):
  ```
  for own_a in legal:
    for dice_d in 1..6:
      apply own_a, set dice_d
      legal_opp = get_legal_moves(state_after_own_d)
      best_opp_q = max over opp_a [V(state_after(own_a, d, opp_a))]
                  // opp picks max-V from opp's POV = max for opp = bad for us
      sum += best_opp_q
    Q[own_a] = -sum / 6  // negate because best_opp_q is opp POV
  search_action = argmax_a Q[own_a]
  search_value = max_a Q[own_a]
  search_policy = softmax(Q / 0.5)  // τ=0.5 sharpens slightly
  ```
- Batch all leaf evaluations: 96 V-evals per state into one forward
  pass on the L4. ~10-20 ms per state amortized.
- Total: **1M states**.
- Dump to parquet/npz: `(state_v17_tensor, search_policy, search_value,
  search_action, eventual_outcome)`.

### Trainer (`train_search_distill.py`)

- Clone of `train_v132_sl.py`, swap teacher targets for search targets.
- Fresh V13.2-architecture student (10×128, 17ch input).
- Loss:
  ```
  α_p · KL(student.π || search_policy)
  + α_v · MSE(student.V, search_value)
  + α_o · BCE(student.V, eventual_outcome)
  ```
  with `α_p = 1.0, α_v = 0.5, α_o = 0.5` (anchors the value head to
  terminal truth in case `search_value` is biased).
- 5 epochs over the 1M-state buffer = 5M training steps.
- Adam, lr 1e-3 → 1e-4 cosine. Same as V13.2 SL.
- Output: `checkpoints/mcts_v1_step1_distill/model_latest.pt`.

### Evaluation

H2H tournament: distilled student vs V13.2-latest, **25K games on Mac
CPU**, seat-balanced, greedy moves. σ ≈ 0.32pp.

**Pass criteria for Step 1 (commit to Step 2):**
- Student wins ≥ 53% (≥ +10 Elo, p < 0.001).

**Marginal (51-53%):** review per-state-type breakdown. If student is
clearly better in tactical positions but tied in early game, that's
a pass. If wash everywhere, that's fail.

**Fail (≤ 51%):** abandon MCTS hypothesis, pivot to transformer
(temporal context as the next plateau-break lever).

## Step 2 — Full MCTS RL (locked spec, executes only if Step 1 passes)

### MCTS engine (`mcts_engine.py`)

Fresh Python implementation. Designed for clarity and correctness, not
for speed. Will be ported to C++ in Step 2.5 if results justify.

**Algorithm:** AlphaZero-style PUCT MCTS with explicit chance nodes
for Ludo dice. Per-move structure:

1. From current state s_root, run **N=100 simulations**:
   a. **Selection:** walk down tree from root.
      - At an action node (player to move): pick action via PUCT:
        ```
        a* = argmax [Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))]
        ```
      - At a chance node (dice not yet rolled): sample uniform from 6
        children (each dice value 1-6 has equal prior probability).
      - At a terminal node: return the win/loss outcome.
   b. **Expansion:** when reaching an unexpanded leaf:
      - If state.is_terminal: leaf value = +1 (current player wins) or
        -1 (current player loses).
      - Otherwise: query network for `(P(s, ·), V(s))`. P initializes
        action priors. V initializes the leaf's expected value.
      - Create children for each legal action.
      - At chance nodes: create 6 children, one per dice value.
   c. **Backup:** propagate leaf value up the path.
      - Action nodes: `Q ← (N·Q + v) / (N+1); N += 1`. Flip sign when
        crossing player boundaries.
      - Chance nodes: average over visited children, weighted by visit
        count of each (which converges to 1/6 each as N→∞).
2. After N simulations, derive `π_search(a|s) = N(s_root, a) /
   Σ_a N(s_root, a)`.
3. Sample action from `π_search` with temperature τ.

**MCTS hyperparameters (locked):**
- N (simulations per move): **100** (start). Ramp to 200 if early
  signal is clear and L4 has spare time. AlphaZero used 800 on chess.
- c_puct: **1.5** (AlphaZero chess default).
- Dirichlet noise at root: **α=0.3, ε=0.25** (AlphaZero chess
  defaults). Mixed into root's prior `P(s_root, ·)` to encourage
  exploration of moves the trained policy underweights.
- Move temperature τ: **1.0 for first 30 moves of each game** (sample
  from `π_search`), **0.001 thereafter** (effectively argmax). Standard
  AlphaZero schedule for game-start exploration + game-end precision.
- Chance-node behavior: **enumerate all 6 dice children**, weight backup
  by visit count of each (asymptotically uniform).
- 6-roll bonus turn: **same player keeps current_player flag**, no sign
  flip in backup. Verified explicitly in unit tests.

### Self-play + training loop (`train_mcts_rl.py`)

- Load V13.2-latest as warm start.
- Vectorize K=64 parallel games via `VectorGameState`.
- For each game step:
  - For each game in parallel: run MCTS with N=100 sims, derive
    `π_search`, sample action with current temperature.
  - Apply actions, advance games.
  - Record `(state, π_search, current_player)` per move.
- When a game terminates: walk back, label every recorded move with
  `eventual_outcome` from that move's player POV. Add to replay buffer.
- After every K=200 games: run training step on accumulated buffer.
  - Loss: `α_p · KL(network.π || π_search) + α_v · BCE(network.V,
    eventual_outcome)` with `α_p = 1.0, α_v = 1.0`. Plus L2 weight
    decay = 1e-4.
  - **No PPO, no bias penalties (in Phase 2).** Pure AlphaZero loss.
  - Adam, lr 5e-5 (an order of magnitude lower than SL — small RL
    updates).

### Two-phase training

**Phase 1 — Validation (~30K games, ~3-5 days on L4):**
- MCTS + bias penalties + AlphaZero loss.
- Goal: validate that policy diverges from V13.2's initial
  distribution.
- **Diagnostic:** every 5K games, measure `KL(π_search || π_v132_initial)`
  averaged over states. If `KL ≥ 0.10`, search is finding new moves —
  PROCEED. If `KL < 0.05` after 30K games, search isn't differentiating
  — ABORT.
- Eval cadence: every 5K games, 2.5K games per eval round.

**Phase 2 — Pure search-driven RL (~70K games, ~10-12 days on L4):**
- Drop bias penalties.
- Continue MCTS + AlphaZero loss only.
- Goal: see if pure search-improvement training breaks the V13.2
  plateau.
- Eval cadence: every 5K games.

### Evaluation

After Phase 1: H2H tournament vs V13.2-latest, 25K games. Just to
confirm we haven't regressed.

After Phase 2: H2H tournament vs V13.2-latest, 25K games on Mac CPU.

**Pass criteria for Step 2 (declare MCTS the new SOTA):**
- Final model wins ≥ +5pp / +17 Elo over V13.2-latest in 25K-game H2H
  (p < 0.0001 at 25K).

**Failure modes and responses:**
- *KL stays below 0.05 in Phase 1:* MCTS isn't finding new moves at 100
  sims. Either ramp to 200 sims and re-run Phase 1, or abort.
- *KL diverges but eval WR drops below 80%:* search is finding moves
  the value head likes but lose actual games. Value head is the
  bottleneck. Fix value head first (Step 0 redux), retry.
- *KL diverges, eval WR holds in Phase 1, drops in Phase 2:* bias
  penalties were structural, MCTS doesn't replace them. Settle for
  Phase 1's smaller plateau-break, document as such.
- *Phase 2 finishes with student tied or worse than V13.2:* MCTS
  doesn't break the plateau in this codebase. Definitive negative
  result, pivot to transformer.

## File layout (locked)

All new code lives in `td_ludo/experiments/mcts_v1/`:

```
experiments/mcts_v1/
├── README.md                    # operational spec, runs end-to-end
├── calibration_audit.py         # Step 0
├── generate_search_data.py      # Step 1 generator
├── train_search_distill.py      # Step 1 trainer
├── mcts_engine.py               # Step 2 — fresh Python MCTS
├── train_mcts_rl.py             # Step 2 trainer
├── test_mcts_engine.py          # unit tests for MCTS
└── run_mcts_pipeline.sh         # combined runner for Step 2
```

Checkpoints:
- Step 1 student: `checkpoints/mcts_v1_step1_distill/`
- Step 2 model: `checkpoints/mcts_v1_step2_rl/`

Logging: reuse `live_stats.json` + `training_metrics.json` pattern;
existing dashboards work without changes.

## Compute timeline (committed)

| Phase | Where | Time | Cumulative |
|-------|-------|------|-----------|
| Code development | Mac local | 1-2 days | 1-2 days |
| Pause + backup VM | — | 30 min | 1-2 days |
| Step 0 calibration audit | VM | 15 min | 1-2 days |
| Step 1 search-data generation | VM | 2-4 hrs | 1-2 days + 4 hrs |
| Step 1 SL distillation training | VM | 2-3 hrs | 1-2 days + 7 hrs |
| Step 1 H2H tournament | Mac CPU | 2 hrs | 2-3 days |
| **Decision: commit to Step 2 or pivot** | — | — | 2-3 days |
| Step 2 Phase 1 (validation, 30K games) | VM | 3-5 days | 1 week |
| Step 2 Phase 1 H2H | Mac CPU | 2 hrs | 1 week |
| Step 2 Phase 2 (pure RL, 70K games) | VM | 10-12 days | 2.5-3 weeks |
| Step 2 Phase 2 H2H | Mac CPU | 2 hrs | 2.5-3 weeks |
| **Final go/no-go on MCTS as plateau-breaker** | — | — | 2.5-3 weeks |

V13.2 RL on VM stays paused for the entire 2.5-3 week window. Accepted
as the cost of finding out whether MCTS works.

## What this experiment ultimately answers

One of three outcomes by end of Step 2:

1. **MCTS works.** Final model beats V13.2 by ≥ +5pp. AlphaZero recipe
   confirmed for this codebase. New strongest model. Step 2 model
   becomes the new teacher; future SL distillations use it.

2. **MCTS partially works.** Phase 1 shows policy divergence and small
   WR lift, Phase 2 doesn't materialize. Bias penalties were
   structural; settle for marginal gain, document.

3. **MCTS doesn't work.** Definitive: search doesn't break the plateau
   in this codebase, even with all the bug fixes in place. Pivot to
   transformer with temporal context (Step 3 in original plan, becomes
   the new primary).

Each outcome is informative. The 3-week investment is worth it because
the answer is otherwise unknowable.

---

## Update — 2026-05-06/07: outcomes so far

Plan above was executed in compressed form (1-2 days, not 2-3 weeks)
because Step 1 came back so decisively negative that the longer path
wasn't justified. Summary of what landed:

### Step 1 — MCTS search-distillation: NEGATIVE

Built fresh Python MCTS (`experiments/mcts_v1/`) with 6 unit tests
catching the two real bugs (state aliasing in dice loop, bonus-turn
sign error). Generated 901K-state buffer, distilled into fresh V13.2-arch
student. H2H against V13.2_latest (25K-game tournament, killed at 13.8K
when verdict was clear):

- v1 (with bugs): V13.2 92.2 / Step1 7.8
- v2 (bugs fixed): V13.2 89.6 / Step1 10.4

Bug fixes moved the result by +2.6pp — the bugs were real but the
binding constraint is **2-ply expectimax over the same teacher cannot
meaningfully improve targets.** Shelved Step 1 / Step 2 until we have a
stronger leaf evaluator than V13.2.

### Step 3 (originally optional) — Temporal transformer: ESCALATED

Pivoted to transformer-with-temporal-context experiments.

**V13.3 mini** (418K params, cnn 4×64 + transformer L=2 d=64): SL plateau
at 82% (same band), RL collapsed under vanilla REINFORCE (82→30), RL v2
with KL anchor stabilized but still drifted (82→70). H2H: V13.3 lost
43.4 / 56.6 to V13.2.

**V13.4** (3.79M params, cnn 10×128 + transformer L=4 d=128 — V13.2-
comparable scale): Discovered a train/test history mismatch bug in
the V13.3 envs (mixed-POV in training, own-POV-only in inference,
because encoder rotates board to current_player's POV). Fixed via
per-player history deques in both SL and RL envs. 86 unit tests verify
no opponent-frame leak.

V13.4 SL plateau: also 80-82%. RL stable at 80% so far (200K of 1.5M
states). **H2H verdict pending — phase 3 fires after RL completes.**

### Methodology update

The 4-way H2H (V13.2 / V13.3_SL / V13.3_RL / Step1_Distill, 500 games
per pair, mirrored seeds) revealed a critical methodological lesson:
**bot-eval ceilings around 80-83% across all architectures** mask real
H2H differences of ±5-10pp. Any plateau-break claim must be validated
by H2H, not bot eval.

### Open questions, ranked by promise

1. **Token-symmetry encoder fix** (proposed by user 2026-05-07). Current
   encoders treat the 4 own / 4 opp tokens as 4 distinct channels each,
   even though the rules make them permutation-symmetric. The model has
   likely "specialized" token IDs (token 0 usually further along the
   board). Capture-and-return events break this distribution. Proposal:
   collapse to 1 channel "all my tokens" + 1 channel "all opp tokens"
   (with multiplicity). Lower-risk than search; potentially additive
   with all other improvements.

2. **League / population-based RL.** Train V13.2 (or V13.4) against a
   diverse pool of frozen older checkpoints rather than only against
   itself. Gives meaningful advantage signal that vanilla self-play
   REINFORCE lacks.

3. **Bigger plain V13.2** (more channels, deeper trunk). Tests whether
   the plateau is capacity-bound, independent of temporal information.

4. **MCTS revisit, but only after we have a stronger leaf evaluator.**
   Whatever wins from (1)/(2)/(3) becomes the new leaf; then 2-ply
   search has actual headroom to extract.
