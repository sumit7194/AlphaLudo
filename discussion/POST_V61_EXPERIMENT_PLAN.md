# Post-V6.1 Plateau Experiment Plan

> **Created**: 2026-04-10
> **Author**: Claude + Sumit, after V6.1 resume training plateaued at 0.788 best eval
> **Status**: Step 1 pending execution
> **Owning doc**: This is the live execution plan. Update checkbox status as each step runs. Results summaries go here too, not in `training_journal.md` until the experiment completes.

---

## 1. Context — Why this plan exists

We have conclusively plateaued around **78-79% eval vs Expert bot** across three architecture generations:

| Model | Architecture | Peak eval | Games to peak | Notes |
|---|---|---|---|---|
| V6 | 17ch CNN, 128ch, 10 ResBlocks | 77.4% | ~170K | `ac_v6_big/backups/model_best_v6_77pct_170k.pt` |
| V6.1 | +7 strategic channels (24ch) | **78.8%** | ~157K | `ac_v6_1_strategic/model_best.pt` — current production candidate |
| V6.2 | V6.1 + 4-layer transformer (ReZero fix) | ~79.2% (first eval), plateaued ~76% | ~35K | Transformer learned but added no capability; pivoted off |

**Key observations from the journal and current run:**
- 62K+ games of V6.1 resume training produced **+0.8pp** (0.78 → 0.788). Essentially zero.
- V6.1's Elo oscillates 1400-1600 across 30-min windows with no directional drift
- Expert training-distribution WR stuck at 62-68% regardless of games played
- Mech interp (Exp 10-12) concluded: "V6 is a sophisticated reactive player... zero temporal reasoning"
- V6.2's transformer confirmed there's no useful temporal structure for attention to exploit
- Prior MCTS training attempt (Exp 9, AlphaZero-style) **failed catastrophically** at 34.7% WR vs base PPO — but it was training-time MCTS, not inference-time

**The hypothesis we're testing next**: V6.1's policy priors are good but it has no planning capability. Inference-time MCTS search can amplify a decent policy without any retraining risk. Prior MCTS failure was specific to training; inference-time search is a fundamentally different attack surface.

---

## 2. Decision log — Why this plan and not others

| Considered option | Why deferred or rejected |
|---|---|
| **Keep training V6.1 longer** | 62K games gave +0.8pp. Another 100K will give ~0pp. Optimizer is in its basin. |
| **Wider/deeper CNN (V6.3)** | Mech interp suggested size isn't the bottleneck. Skipping. |
| **Training-time MCTS (AlphaZero-style)** | Already tried in Exp 9. Failed. Root cause (noisy terminal values over 400-move games) has not changed. |
| **Auxiliary loss on V6.2 transformer** | V6.2 run showed the transformer has no meaningful signal to learn in this domain. Adding an aux loss just gives it a different flavor of noise. |
| **Larger SL dataset** | Source data is also from inferior bots. Ceiling problem, not data-volume problem. |
| **Human benchmark** | ← Step 4. Running in parallel with the above, depends only on user time. |
| **Reward shaping revisit** | ← Step 2. Deferred until MCTS results inform whether policy has more room. |
| **Stochastic MuZero port** | ← Step 3. Biggest engineering lift, reserved for if Step 1 validates search helps. |

**Primary bet**: Inference-time MCTS wrapping raw V6.1, no training changes. Low effort, low risk, reuses all existing infra.

---

## 3. Step 1 — Inference-time MCTS sweep

### 3.1 Goal

Determine whether Monte Carlo Tree Search at inference time, using V6.1 as policy prior and leaf value estimator, meaningfully beats raw V6.1 against Expert. If yes, find the sweet spot of simulations per move.

### 3.2 Primary model

- **`/home/sumit/td_ludo/checkpoints/ac_v6_1_strategic/model_latest.pt`** (239K games, AlphaLudoV5, 24ch, ~3M params)
- Backed up as:
  - GCP: `backups/v6_1_latest_239k_best788_20260410.pt`
  - Local: `gcp_snapshots/v61_final_20260410_0137/model_latest.pt`
- This is the post-resume most-trained state. **Not** the cherry-picked `model_best.pt` — we want production model, not best snapshot.
- Weights are frozen for this entire experiment. No training.

### 3.3 Architecture of MCTS wrapper

**Must-do (from research findings):**
1. **Afterstate decomposition** — the move selection and dice chance events must be separated in the tree. This was missing/broken in the old Exp 9 implementation. Decision nodes select *which piece to move* given known dice; chance nodes sample *next dice roll* from true distribution.
2. **V6.1 as both policy prior AND leaf value estimator** — pUCT selection uses the policy, leaf values come from the value head. No rollouts.
3. **Batched neural inference** — the C++ engine from Exp 9 called the model one state at a time. On GPU, that's <1% utilization. Must batch sim expansions to leverage the T4.
4. **Legal move masking** — same as used during training, applied to the policy prior.
5. **Dirichlet noise at root** — standard AlphaZero trick for exploration. α=0.3, ε=0.25.

**Nice-to-have (from research, include if time allows):**
- **CHANCEPROBCUT-style chance node pruning** — skip sampling dice outcomes when the value window makes the current decision unlikely to change.
- **Dynamic simulation stopping** — early-terminate search when one action dominates by a large margin.

**Explicitly skip (don't over-engineer v1):**
- Tree reuse across moves (would help speed, adds complexity)
- Transposition tables
- Progressive widening

### 3.4 Eval protocol

All matches are **2-player, P0 vs P2**, consistent with training config. Use `evaluate_v6_1.py` or equivalent.

**Phase A — Raw baseline (sanity check current model state)**
- [ ] V6.1 raw vs Expert: **1,000 games**, greedy (temp=0)
  - Expected: ~76-79% (noisy, matches recent evals)

**Phase B — MCTS sweep vs Expert**
- [ ] V6.1 + MCTS(25) vs Expert: **2,000 games**
- [ ] V6.1 + MCTS(50) vs Expert: **2,500 games**
- [ ] V6.1 + MCTS(100) vs Expert: **2,500 games**
- [ ] V6.1 + MCTS(200) vs Expert: **1,500 games**

**Phase C — Head-to-head (cleanest signal, no bot noise)**
- [ ] V6.1 + MCTS(50) vs V6.1 raw: **1,500 games**
- [ ] V6.1 + MCTS(100) vs V6.1 raw: **1,500 games**
- [ ] V6.1 + MCTS(200) vs V6.1 raw: **1,000 games**

**Phase D — Cross-model sanity check**
- [ ] V6.1 + MCTS(100) vs V6.1 `model_best.pt` @ 157K: **500 games**
- [ ] V6.1 + MCTS(100) vs V6.2 `model_latest.pt` @ 35K: **500 games** (optional, only if V6.2 is already loaded)

**Total**: ~14,500 eval games across 11 matchups.

### 3.5 Runtime budget

Expected runtime on GCP T4, assuming batched inference and average game length ~175 moves:

| Matchup | Games | Est. GPM | Est. time |
|---|---:|---:|---:|
| V6.1 raw vs Expert | 1,000 | 130 | ~8 min |
| MCTS(25) vs Expert | 2,000 | 20 | ~100 min |
| MCTS(50) vs Expert | 2,500 | 12 | ~210 min |
| MCTS(100) vs Expert | 2,500 | 6 | ~420 min |
| MCTS(200) vs Expert | 1,500 | 3 | ~500 min |
| MCTS(50) H2H | 1,500 | 12 | ~125 min |
| MCTS(100) H2H | 1,500 | 6 | ~250 min |
| MCTS(200) H2H | 1,000 | 3 | ~335 min |
| Cross-model (500+500) | 1,000 | 6 | ~170 min |
| **Total** | **14,500** | — | **~35 hours** |

**Design target: 24+ hours of runtime.** We'll exceed that comfortably. If something breaks early we still have usable data from the first 2-3 completed matchups.

If this proves too long in practice, Phase D is optional and Phase B MCTS(200) can be shortened to 1,000 games.

### 3.6 Success / failure criteria

**Clear win**: MCTS(100) beats raw V6.1 by ≥3pp head-to-head, OR overall WR vs Expert crosses 81%.
→ Ship V6.1+MCTS(100 or whatever sim count wins) as production model. Skip Steps 2 and 3.

**Marginal**: MCTS shows +1-2pp over raw V6.1 but inference cost is painful.
→ Evaluate whether the cost/benefit is worth it. Consider tightening just the top-performing N. Proceed to Step 2.

**Failure**: MCTS at all N values is within ±1pp of raw V6.1 head-to-head.
→ Policy priors are already at local optimum. Search can't amplify what isn't there. Proceed to Step 2 (reward shaping) with lowered expectations. Consider Step 4 human benchmark as the definitive ceiling test.

### 3.7 Gotchas to avoid (from journal)

- **Input channel mismatch**: Exp 9 had a C++ MCTS engine hardcoded to 21 channels while the model expected 17. We're on 24 channels now (V6.1). **Verify C++ engine input dim matches before running anything.**
- **Value head calibration**: V6.1 uses return normalization. MCTS leaf values from `model.forward(...)[1]` will be in normalized space. Need to de-normalize before backing up through the tree, OR train value head without normalization. **Easier: use the model's raw value output and let UCB handle relative ordering — absolute scale doesn't matter for search.**
- **Dice distribution**: chance nodes must sample from 1-6 uniform. Don't weight by "what the model expects."
- **Stale model_best.pt**: the 157K best.pt might be better calibrated than the 239K latest.pt due to the mild drift. Phase D comparison tells us.

### 3.8 Execution checklist

- [ ] Verify existing C++ MCTS engine (`td_ludo_cpp::MCTSEngine`) compiles with 24-channel input
- [ ] Write Python wrapper `mcts_eval.py` that loads V6.1 + batched inference + afterstate MCTS
- [ ] Unit test: single move with MCTS(25), verify legal move mask applied, verify output policy sums to 1
- [ ] Smoke test: V6.1+MCTS(25) vs Expert, 20 games, verify no crashes
- [ ] Launch full sweep on GCP T4 in a screen session with logging
- [ ] Wait 24-36 hours
- [ ] Collect results into this document under Section 3.9

### 3.9 Results (filled in after execution)

_Empty pending execution._

---

## 4. Step 2 — Reward shaping revisit (conditional on Step 1 results)

### 4.1 Goal

If Step 1 shows V6.1's policy priors have more headroom when amplified by search, test whether *training-time* reward tweaks can push the base policy upward without needing MCTS at inference.

### 4.2 Hard rules from the journal (do not violate)

- ❌ Do NOT use PBRS in >100 move games (Exp 1 proved γ-leak is fatal)
- ❌ Do NOT scale any intermediate reward below 0.05 (Exp 4, 6 both collapsed)
- ❌ Do NOT change rewards mid-training without clearing the PPO buffer
- ❌ Do NOT remove capture/kill rewards (Exp 8 proved this kills credit assignment)
- ❌ Do NOT train PPO without return normalization (V6 drift was caused by missing this)

### 4.3 Proposed experiment (ONE change only)

**Config**: Resume V6.1 from `model_latest.pt` @ 239K.

**Single change**: Reduce all intermediate rewards by **50%** (not 80% like failed v2/v3).
- Spawn: 0.05 → 0.025
- Forward: 0.005 → 0.0025
- Home stretch: 0.10 → 0.05
- Score: 0.40 → 0.20
- Capture: 0.20 → 0.10
- Killed: −0.20 → −0.10
- All v2.2 strategic rewards halved too

**Rationale**: Journal hypothesis was that rewards saturate the value head. 50% keeps them "loud enough" per the hard rule (largest reward stays at 0.20 > 0.05 floor) but reduces saturation. This is a conservative version of the failed v2/v3 experiments, informed by their failure modes.

**Stop criteria:**
- Eval WR drops >2pp in first 10K games → abort, revert
- Eval WR flat for 20K games → abort, revert (no effect)
- Eval WR rising → extend to 50K games, track trend

**Buffer**: clear on resume (mandatory per hard rule).

### 4.4 Execution checklist (Step 2)

- [ ] Confirm Step 1 results justify Step 2
- [ ] Back up current reward_shaping.py config as `reward_shaping_v2_2.py.bak`
- [ ] Edit reward magnitudes
- [ ] Clear buffer flag in resume command
- [ ] Launch for 30K games, monitor every 2 hours
- [ ] Decide at 10K, 20K, 30K checkpoints

### 4.5 Results (Step 2)

_Empty — conditional on Step 1 completion._

---

## 5. Step 3 — Stochastic MuZero port (far future, only if Steps 1+2 fail)

### 5.1 Goal

Full port of DeepMind's Stochastic MuZero to Ludo. Decouple decision nodes from chance nodes with learned chance distribution (VQ-VAE codebook). Train value and policy networks from scratch with proper afterstate handling.

### 5.2 Effort estimate

**2-3 weeks of focused work.** Not a weekend project. Only commit to this if:
- Step 1 proves search helps (validates the "planning is the bottleneck" hypothesis)
- Step 2 confirms reward-side has no more room
- You have dedicated time and motivation for a multi-week project

### 5.3 References

- [Planning in Stochastic Environments with a Learned Model (DeepMind 2022)](https://openreview.net/pdf?id=X6D9bAHhBQ1)
- [DHDev0/Stochastic-muzero (PyTorch impl we can adapt)](https://github.com/DHDev0/Stochastic-muzero)

### 5.4 Execution checklist (Step 3)

- [ ] Conditional on Steps 1+2 failing/plateauing
- [ ] Clone reference impl, run their backgammon demo to verify it works
- [ ] Adapt state/action encoding to Ludo
- [ ] Integrate 24-channel V6.1 representation as initial encoder
- [ ] Initial training on CPU before committing to long GPU runs
- [ ] Comparison eval vs V6.1 baseline at matching game counts

---

## 6. Step 4 — Human benchmark (parallel, ongoing)

### 6.1 Goal

Determine whether V6.1's 78% ceiling is a training/architecture limit or a fundamental Ludo ceiling (i.e., the game is too luck-heavy for meaningful separation beyond this point).

### 6.2 Protocol

- [ ] Sumit plays **20-50 games** vs V6.1 `model_latest.pt` using the Android app or local evaluator
- [ ] Record: win/loss, game length, subjective notes on model mistakes
- [ ] If Sumit wins <50%: model is at or above human expert level, other work is cosmetic
- [ ] If Sumit wins 60%+: there's real room above 78% and all above work is justified
- [ ] If 50-60%: the model is strong but beatable, Step 1 MCTS is likely to be worth it

### 6.3 When

Whenever Sumit is free. Does not block any other step. Feed results back here.

---

## 7. Current checkpoint inventory (as of 2026-04-10)

| File | Location | Description | Games | Best eval |
|---|---|---|---:|---:|
| `v6_1_best_157k_wr788_20260410.pt` | GCP backups/ | V6.1 peak snapshot | 157K | 0.788 |
| `v6_1_latest_239k_best788_20260410.pt` | GCP backups/ | V6.1 post-resume final | 239K | 0.788 |
| `model_latest.pt` (V6.2) | GCP ac_v6_2_transformer/ | V6.2 ReZero final | 35K | ~0.79 first eval |
| `gcp_snapshots/v62_final_20260409_1445/` | Local | Full V6.2 code + checkpoints | — | — |
| `gcp_snapshots/v61_final_20260410_0137/` | Local | V6.1 post-resume snapshot | 239K | 0.788 |
| `td_ludo/checkpoints/ac_v6_1_strategic/model_latest.pt` | Local | V6.1 pre-resume | 155K | 0.78 |
| `td_ludo/checkpoints/ac_v6_big/backups/model_best_v6_77pct_170k.pt` | Local | V6 peak | 170K (V6) | 0.774 |

**Rule**: do not delete any of the above. Any experiment that overwrites should first copy the previous state to `backups/` with a date-tagged name.

---

## 8. Logistics

- **Execution environment**: GCP T4 (`alphaludo-gpu-test`, zone `asia-east1-c`). Current V6.1 training is **paused** as of 2026-04-10 ~20:07 UTC. Do not restart training until Step 1 completes.
- **Monitoring**: Hourly cron from the existing Claude session is disabled once Step 1 launches (no point monitoring an eval run the same way as a training run). Use a lighter check — every 4 hours, grep eval progress from the sweep log.
- **Local snapshots**: continue dropping full GCP snapshots into `/Users/sumit/Github/AlphaLudo/gcp_snapshots/` at every major experiment boundary.
- **Results propagation**: once Step 1 completes and we pick a direction, a concise summary goes into `td_ludo/training_journal.md` as "Experiment 13: Inference-time MCTS sweep". Full details stay in this plan doc.

---

## 9. Update log

| Date | Change | By |
|---|---|---|
| 2026-04-10 | Plan created after V6.1 plateau discussion | Claude + Sumit |
| 2026-04-28 | Plan extended with §10–§13: post-V11.1 status, RLHF idea for future | Claude + Sumit |

---
---

## 10. Status as of 2026-04-28 (post-V11.1, mid-V12)

The original plan (Steps 1–6 from §3) was executed through 2026-04. Outcomes summarized here so this document doubles as "where we are now" without needing to grep the journal.

| Step | What | Result |
|---|---|---|
| 1: MCTS@inference sweep | V6.1 + 25/50/100/200 sims vs Expert | **FAILED** — every sim count *worse* than raw policy. Dice branching dilutes UCB, value head not calibrated. Confirmed in journal Exp 13c. |
| 2: Reward shaping 50% reduction | V6.1 RL with halved dense rewards | **NEUTRAL-NEGATIVE** — drifted -2pp, never exceeded 78.8%. Journal Exp 13d. |
| 3: Reward shaping baseline restoration | Reverted to v2.2 dense rewards | OK, used as baseline for V6.3+ work |
| 4: Human benchmark | 3 games vs V6.1 best | **Decisive finding** — V6.1 plays strong tactics, zero multi-turn planning. Motivated V6.3 capture-prediction head. Journal Exp 13e. |
| 5: V6.3 capture-prediction head | 3 new channels + aux capture head | Neutral by 1000-game H2H (50/50 vs V6.1). Journal Exp 14, 14b. |
| 6: V10 slim multi-task | 6×96 CNN, 28ch input, 3 calibrated heads | 1/3 the params, matched V6.1's eval ceiling. Journal Exp 15, 16, 18. |

After V6 family was exhausted, we ran:
- **Exp 19** — exploiter self-play (PSRO-lite). Same-family exploiter from V10 SL plateaued at 47.9% vs frozen V10.2 in 84K games. **Confirmed plateau is robust to behavioral exploitation within the V10 family.** Did not pursue V6.1-init heterogeneous exploiter (user push: V6.1 ≈ V10 architecturally).
- **Exp 20** — V11 ResTNet (CNN + 2 attn over 225 cells). OOM on 16GB Mac. V11.1 (1 attn × 2 heads × dim 64) trained successfully. Best 79.05% on 2000-game eval (project record), mean 77.4% across last 10 evals (+5pp band shift over V10.2). **Did not break the 80% gate.** Gameplay analysis revealed leader-greedy + capture-blind failure modes traceable to "attention over cells, not entities".
- **Exp 21** — V12 token-entity attention (IN PROGRESS). 8-token attention over the actual game pieces. SL parity passed (95.9% val acc, 73.5% bot WR). Per-bot profile flipped from V11.1 — strong vs Aggressive/Random, weak vs Expert/Heuristic. RL launched on L4, early evals 76.4–77.5% at G=10–30K. Verdict pending ~24h training.

**The 78–79% asymptote is stable across 4 architecture families.** V11.1 showed it's marginally above V6.1 (with much better methodology) but the strict 80% gate hasn't fallen. V12's qualitative gameplay difference is more interesting than its raw eval number.

---

## 11. What's left in the plateau-break toolkit

After V12 finishes (success or plateau), the remaining unexplored angles, ranked by EV:

### 11.1 Reward shaping rerun on V12

V11.1's failure mode (capture-blind, leader-greedy) was clearly **reward-induced**, not architecture-induced — the model learned exactly what `+0.40 per scored token + ±1 terminal` rewards. With V12's token-entity attention now in place, a *reread* of explicit capture/danger rewards may behave differently:

- V6.3-era dense capture reward (+0.20 capture, -0.20 got-killed) was abandoned because it distorted V6.1's policy. But V12's architecture is fundamentally different.
- Proposal: V12.1 = V12 + add `+0.10 per capture, -0.10 per got-captured` to existing sparse rewards. Smaller magnitude than v1 dense rewards (which V6.1 used at +0.20).
- Risk: similar to V6.1's failure mode — reward distorts the learned behavior. But token-entity attention may absorb the signal more cleanly.

### 11.2 RLHF — Reinforcement Learning from Human Feedback (idea for future)

User-flagged for future: **make AlphaLudo human-aligned, not just bot-eval-optimized**. The 78-79% bot eval ceiling may be the actual limit of "playing well against this bot mix", but **subjectively good play** (rich strategy, satisfying gameplay for human opponents) may not correlate with bot-WR at all.

**Why this matters for our project specifically**:
- User's gameplay reports describe V11.1 as "boring, predictable, leader-greedy". The eval WR (best 79.05%) doesn't capture this.
- A model that plays at 76% WR but exhibits **diverse strategy choice, dynamic capture, multi-turn planning** would be strictly better as an opponent than one at 79% WR with mechanical leader-pushing.
- The bot-mix evaluation framework (Random + Heuristic + Aggressive + Defensive + Racing + Expert) doesn't penalize predictability or boredom.

**Possible RLHF flavors for AlphaLudo** (to discuss when V12 settles):

A. **Human preference ranking on game pairs**:
   - Generate 2 game trajectories from V12 (different temperatures or different ghost opponents)
   - Show user the games (or summary of moves), get binary preference
   - Train a reward model on preferences, RL the policy against that reward
   - Reference: Christiano et al. 2017 "Deep RL from Human Preferences"
   - Cost: ~100-1000 user-labeled comparisons, then standard PPO with new reward
   - Pros: directly optimizes for "fun to play against"
   - Cons: requires hours of user feedback collection

B. **Behavior cloning from human play**:
   - User plays N=100+ games (against bots, not against V12)
   - Train a model that imitates user's move distribution
   - Use this as an additional opponent in V12's training mix
   - Effect: V12 learns to handle "human-like" strategies, not just bot patterns
   - Pros: zero subjective rating burden — just play normally
   - Cons: needs significant gameplay logging infra

C. **Reward shaping based on observed gameplay critiques**:
   - User flags specific moves as "bad" in past gameplay (we're already doing this with the log analysis)
   - Convert critiques to programmatic reward components: e.g., "missed capture" → -0.05, "kept token idle 10+ turns" → -0.03
   - Cheaper than full RLHF; uses your structured analysis from gameplay.
   - Pros: builds on existing analysis pipeline (game log parser)
   - Cons: relies on us inferring what critiques generalize

D. **Best-of-N sampling at inference**:
   - Train a separate "fun" reward head via small-scale RLHF (≤500 labels)
   - At play time, sample N=10 candidate moves from policy, pick the one with highest "fun" reward
   - No RL retraining needed; just inference-time selection
   - Pros: cheapest, easiest to add to play server
   - Cons: bounded by policy's existing diversity (can't introduce moves the policy gives ≈0 prob to)

**Recommendation for when we revisit**:
- Start with **(C) — programmatic reward shaping from observed critiques**. We have the gameplay analysis infrastructure already (`/tmp/analyze_games.py`). We have specific failure modes documented. Translate them to reward components and re-run V12 RL with the augmented reward. Cheapest experiment with the most direct connection to "what we observed was wrong".
- If (C) doesn't move the needle on subjective gameplay quality, escalate to **(A) — true RLHF**. Costs more user time but is the principled approach.
- (B) behavior cloning is interesting but probably needs >100 logged games of user play to be useful, and currently we have ~5.

**Not recommended**: (D) inference-time best-of-N. The policy is already very narrow (V11 puts 99% on one move, 0% on others); sampling N=10 from a near-deterministic policy gives you 1 unique move 99% of the time.

### 11.3 Token-entity attention with explicit relational features (V12.1 / V13)

If V12 underperforms the 78-79% asymptote, the next architectural variant would add explicit **relational features** between token pairs:
- For each pair of tokens (own, opp): include scalar features like "distance_modulo_52" or "in_capture_range_with_dice_X"
- Pass relational features as bias to attention scores
- Reference: graph attention networks (GAT), pair-wise relational reasoning

This makes capture-detection a built-in inductive bias rather than something the model must learn from scratch. Risk: brittle to game variants, more code complexity. Defer until V12 verdict.

### 11.4 Stochastic MuZero (the moonshot)

Still on the table per research synthesis, but ~6 weeks of careful engineering. Defer until everything cheaper is exhausted and we still want to push past 80%.

---

## 12. Updated artifact ledger (2026-04-28)

| Path | Location | Description | Games | Best eval |
|---|---|---|---|---|
| `td_ludo/checkpoints/ac_v6_1_strategic/model_best.pt` | Local | V6.1 strategic | ~157K | 78.8% (500-game) |
| `td_ludo/checkpoints/ac_v6_3_capture/model_best.pt` | Local | V6.3 SL+RL | ~156K | 77.8% (500-game) |
| `td_ludo/checkpoints/ac_v10/model_v102_frozen_g302k.pt` | Local | V10.2 final | 302K | 75.15% (2000-game) |
| `td_ludo/checkpoints/_archive/ac_v11_1_premac_*` | Local | V11.1 final, multiple snapshots | up to 770K | **79.05% (2000-game)** ★ project best |
| `td_ludo/play/model_weights/model_v11.pt` | Local play server | V11.1 best ckpt for human-vs-AI | (G=650K, 79.05%) | for play |
| `~/AlphaLudo/td_ludo/checkpoints/ac_v12/` | GCP L4 | V12 (in training) | growing | TBD |
| `gs://[no bucket yet]` | GCS | (none — direct VM-to-VM transfers) | — | — |

**Rule**: V11.1 final ckpts archived. Do not delete `_archive/ac_v11_1_premac_*` directories. They represent the strongest CNN+attention model the project produced and are needed for any A/B comparison vs V12 outcomes.

---

## 13. Open decision queue

These are pending discussions, listed so we don't lose them:

1. **If V12 RL plateaus near 78-79%**: pursue 11.1 (capture reward shaping) before declaring done. ETA: 1-2 days of V12.1 RL.
2. **If V12 RL breaks 80%**: stop, document, ship V12 as final. Update play server.
3. **RLHF roll-out** (§11.2): pause until V12 settles. Then revisit with the recommendation in §11.2.
4. **Architecture moonshot**: Stochastic MuZero on the table but not justified until 11.1 + 11.3 also fail.
