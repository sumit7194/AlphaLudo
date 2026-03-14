# AlphaLudo Training Journal

> **Purpose**: Document every experiment, parameter change, and its measured impact on training.
> This prevents repeating the same mistakes and provides a clear reference for future decisions.

---

## Architecture & Setup
no
| Parameter | Value |
|---|---|
| Model | AlphaLudoV4 (32 channels, 3 ResNet blocks) |
| Run Name | `td_v3_small` |
| Mode | 2-Player (P0 vs P2) |
| Batch Size | 512 parallel games |
| Buffer Size | 200,000 transitions |
| Replay | PER (α=0.6, β=0.4→1.0), 24 steps, every 5 games |
| Throughput | ~160 GPM (907 samples/s) |
| Resume Command | `TD_LUDO_RUN_NAME=td_v3_small td_env/bin/python train.py --resume --model v4` |

---

## Experiment Log

### Experiment 1: PBRS (Potential-Based Reward Shaping)
- **Games**: 0 → 155K
- **Config**: `R = R_raw + γ·Φ(s') - Φ(s)`, scale=0.15, γ=0.995
- **Results**:
  - V-mean: **-0.776** (extremely pessimistic)
  - WR vs Random: **35%** (losing to random bots)
  - Eval WR: **14%**
  - TD Error: 0.007 (stable but useless)
- **Root Cause**: The formula `γ·Φ(s') - Φ(s)` creates a systematic negative bias ("living penalty") because γ < 1. Every turn the opponent moves, the model receives a negative reward of approximately `-0.0015` even when nothing changes. Over ~145 moves, this accumulates to roughly **-0.22** per game, making the model believe every game is a loss.
- **Lesson**: ⚠️ **PBRS is theoretically elegant but practically toxic in long-horizon games like Ludo.** The `(γ-1)·Φ(s)` term creates an unavoidable negative leak per timestep. Do NOT use PBRS with γ < 1 in games with 100+ moves.

---

### Experiment 2: Dense Direct Rewards v1 (The "Dopamine" Fix)
- **Games**: 155K → 320K (~165K games)
- **Config**:
  | Reward | Value |
  |---|---|
  | Spawn | +0.05 |
  | Forward/step | +0.005 |
  | Home Stretch | +0.10 |
  | Score Token | +0.40 |
  | Capture Enemy | +0.20 |
  | Got Killed | -0.20 |
  | **Win** | **+1.0** |
  | **Loss** | **-1.0** |
- **Results**:
  - V-mean: **+0.557** → **+0.807** (rose rapidly)
  - WR (100g): **52-57%** (healthy)
  - Eval WR: peaked at **55%** at game 282K
  - WR vs Random: **60-67%** ✅
  - WR vs Heuristic: **30-32%**
  - Elo: **1,820** (above Random 1,741)
  - TD Error: 0.010-0.015 (healthy)
- **V-mean Concern**: V-mean at +0.80 was flagged as "too optimistic." In a ~145-move game, the model accumulates ~+0.7 to +1.2 in intermediate rewards regardless of outcome, meaning wins and losses look similar in total reward.
- **Lesson**: ✅ **v1 rewards produced the best training era.** The model was actively learning strategy, beating Random at 60%+, and climbing Elo. The V-mean being high was a cosmetic concern, not a functional one.

---

### Experiment 3: Phase 2 Curriculum Shift
- **Games**: ~161K → ~320K (concurrent with Exp 2)
- **Config**: Changed `GAME_COMPOSITION` from Phase 1 to Phase 2:
  | Opponent | Phase 1 | Phase 2 |
  |---|---|---|
  | SelfPlay | 70% | 65% |
  | Random | 20% | 5% |
  | Heuristic | 10% | 20% |
  | Aggressive | 0% | 10% |
- **Results**: Model maintained ~50% WR against the harder mix, which is expected. The critical metric (vs Heuristic) held at ~30%.
- **Lesson**: ✅ **Curriculum shift was appropriately timed.** Introducing Aggressive bots didn't crash training. However, Heuristic/Aggressive WR plateaued at ~30%, suggesting the model hit a capability ceiling.

---

### Experiment 4: Dense Direct Rewards v2 (5x Scale-Down) ❌ FAILED
- **Games**: 320K → 340K (with old buffer), then 340K → 445K (with cleared buffer)
- **Config**: All intermediate rewards reduced by 5x:
  | Reward | v1 | v2 |
  |---|---|---|
  | Spawn | +0.05 | +0.01 |
  | Forward/step | +0.005 | +0.001 |
  | Home Stretch | +0.10 | +0.02 |
  | Score Token | +0.40 | +0.08 |
  | Capture | +0.20 | +0.04 |
  | Got Killed | -0.20 | -0.04 |
- **Hypothesis**: v1 rewards were "too loud" and drowning out the terminal signal. Reducing by 5x would let +1/-1 dominate.
- **Results**:
  - V-mean: **-0.554** (swung back to pessimistic)
  - WR (100g): **29%** (catastrophic decline)
  - Eval WR: **24.5%** (below random chance)
  - WR vs Random: **33%** (down from 67%)
  - WR vs Aggressive: **3.9%** (nearly zero)
  - Elo: **1,662** (below Random at 1,693)
  - The model declined continuously for **125K games straight** — never recovered
- **Root Cause**: The intermediate rewards became so faint (+0.01 to +0.08) that the model couldn't distinguish good moves from bad during normal play. In a 145-move game with tiny per-step rewards, the signal-to-noise ratio was too low. The terminal signal alone (+1/-1) is too sparse and delayed to drive learning in such a long game.
- **Buffer Contamination**: Initially ran v2 rewards with a buffer full of v1 data (200K old transitions). This created chaotic mixed-magnitude replay. Clearing the buffer at game ~340K helped briefly but the v2 rewards themselves were the fundamental problem.
- **Lesson**: ❌ **Never scale intermediate rewards below 0.05 for the largest reward.** The model needs rewards that are clearly distinguishable from noise. The terminal signal is too sparse (once per ~145 steps) to be the primary learning driver. A 5x reduction killed all learning.

---

### Experiment 5: v1 Restored + Clean Buffer
- **Games**: 445K → 515K (~70K games)
- **Config**: v1 reward magnitudes restored, buffer cleared via `--clear-buffer`
- **Results**:
  - V-mean: **+0.802** (immediately returned to v1 levels)
  - WR (100g): **49.4%** (recovered from 29% v2 floor)
  - Eval WR: **32%** (reproduced v1 era range)
  - WR vs Random: **57.6%** ✅
  - WR vs Heuristic: **30.2%** (right back to 30% ceiling)
  - WR vs Aggressive: **28.3%**
  - Elo: **1,745** (above SelfPlay, near Random)
- **Conclusion**: Perfectly reproduced the v1 era. v1 rewards are functional; v2 was the problem.
- **BUT**: We are back at the same V-mean ceiling (+0.80) and the same Heuristic plateau (30%). The model learned everything it can learn under this configuration.

---

### Experiment 6: v3 Tanh-Calibrated Rewards (The "Goldilocks" Attempt) ❌ FAILED
- **Games**: 515K → 530K (~15K games)
- **Config**: Scaled intermediate rewards to total ~+0.25 per game.
  - Spawn: +0.02, Forward: +0.002, Capture: +0.08, Score: +0.15, Killed: -0.08
  - Buffer was cleared.
- **Hypothesis**: By scaling total intermediate rewards to ~+0.25, they fit perfectly inside the value head's `tanh` [-1, +1] range. This gives the model room to differentiate positions without saturating against the +1.0 ceiling or -1.0 floor.
- **Results**:
  - V-mean: **-0.626** (Swung deeply negative)
  - WR (100g): **32.2%** (Catastrophic decline)
  - Eval WR: **19.5%**
  - WR vs Random: **54.7%**
  - WR vs Heuristic: **19.3%**
- **Root Cause**: While v3 was *mathematically* correct for the `tanh` output range, it was *functionally* too weak to overcome environmental noise. Ludo is highly stochastic (dice rolls). The variance of a delayed terminal reward (+1.0 or -1.0) backed up over ~145 steps completely overpowers small intermediate rewards (+0.02). The model abandons the weak local signals and gets overwhelmed by the noisy terminal signals.
- **Lesson**: ❌ **In highly stochastic, long-horizon games, intermediate rewards MUST be loud enough to overpower terminal noise.** "Saturating" the value function (like v1 did) is actually a necessary feature, not a bug — it acts as a strong, immediate dopamine hit that forces the policy to learn local tactics (capturing, scoring) when long-term tracking is obfuscated by dice rolls.

---

## Key Learnings & Rules

### 🔴 DO NOT
1. **Do NOT use PBRS** in games with >100 moves per episode. The `(γ-1)·Φ(s)` leak is unavoidable.
2. **Do NOT worry about V-mean saturation (+0.80) if win rates are healthy.** In high-variance games like Ludo, "loud" intermediate rewards that saturate the V-head are necessary to provide strong local gradients. 
3. **Do NOT change reward magnitudes without clearing the buffer.** Mixed-magnitude data in replay creates chaotic gradients.
4. **Do NOT scale intermediate rewards so small that terminal noise dominates.** If the largest intermediate reward is <0.20, local tactics learning collapses.

### 🟢 DO
1. **DO use loud, "dopamine-heavy" intermediate rewards (like v1).** They are the only way to cut through the extreme stochastic noise of dice rolls.
2. **DO clear the buffer** whenever reward shaping parameters change significantly.
3. **DO let experiments run for at least 15K games** before concluding they've failed.
4. **DO track per-opponent WR trends** (not just overall WR) since the game composition affects the aggregate number.
5. **DO document every parameter change** with before/after metrics in this journal.

### 📊 Healthy Metric Ranges (With Loud v1 Rewards)
| Metric | Healthy Range | Warning |
|---|---|---|
| V-mean | +0.6 to +0.85 | Below 0 = pessimistic, model not learning |
| WR (100g) | 45-60% | Below 40% = something is wrong |
| TD Error | 0.008-0.020 | Spike >0.05 = reward instability |
| vs Random | >55% | Below 50% = model hasn't learned basics |
| vs Heuristic | >25% | Below 15% = model is broken |
| Elo | >1,750 | Below Random Elo = model is failing |

---

## Timeline Summary (V4/V5 Era)

```
Game 0     → 155K   PBRS era (V=-0.77, WR=35%) ❌
Game 155K  → 320K   Dense v1 era (V=+0.80, WR=52%) — structurally saturated, but functionally optimal ✅
Game 320K  → 340K   v2 rewards + old buffer (shock, WR crashed) ❌
Game 340K  → 445K   v2 rewards + clean buffer (too quiet, failed) ❌
Game 445K  → 515K   v1 restored (reproduced v1 era exactly, V=+0.80) ✅
Game 515K  → 530K   v3 calibrated rewards (too quiet, failed) ❌
```

---
---

# V6 Big Brain Era (AlphaLudoV6-Big: 128ch, 10res, 2.99M params)

## Architecture & Setup

| Parameter | Value |
|---|---|
| Model | AlphaLudoV5 class ("V6 Big Brain": 128 channels, 10 ResNet blocks) |
| Run Name | `ac_v6_big` |
| Parameters | 2,990,917 (~3M) |
| Mode | 2-Player (P0 vs P2) |
| Algorithm | PPO (3 epochs, 64-game buffer, 256 minibatch) |
| Batch Size | 512 parallel games |
| LR | 1e-5 |
| Entropy Coeff | 0.005 |
| Temperature | 1.1 → 0.95 (decay 20k games) |
| Throughput | ~80-120 GPM |
| Device | MPS (Apple Silicon) |

---

### Experiment 7: V6 Big Brain — Full RL Run with v1 Dense Rewards

- **SL Baseline**: Trained 1 epoch on bot game data, 86% validation accuracy
- **SL Eval vs Bots**: ~58% win rate (1000 games)
- **Rewards**: v1 dense rewards (same as Experiment 2, proven scale)
- **Games**: 0 → 323K

#### Phase 1: Rapid Learning (0 → 100K games)
- **Eval WR**: 56% → 70%
- **Entropy**: 0.23 → 0.51 (model exploring beyond SL)
- **Matchmaking**: SelfPlay 40%, Heuristic 20%, Aggressive 10%, Defensive 10%, Racing 5%, Random 15%

#### Phase 2: Matchmaking Shift (at ~118K games)
- Changed matchmaking to: SelfPlay 50%, Heuristic 25%, Aggressive 10%, Defensive 10%, Racing 0%, Random 5%
- Rationale: Model already dominating Random (82%) and Racing (65%). Focused on harder opponents.

#### Phase 3: Peak & Plateau (100K → 323K games)
- **Best Eval WR**: 77.4% (at ~170K games)
- **Plateau**: 73% ± 2% for last 120K games
- **Policy Loss went negative** (-0.0007): PPO clipping preventing further change
- **Value Loss stuck** at ~0.80: Critic reached its prediction ceiling for stochastic game

#### Final 1000-Game Evaluation (at 323K)
| Opponent | Latest Model | Best Checkpoint |
|---|---|---|
| **Overall** | **75.0%** | **73.9%** |
| vs Random | 94.5% | 91.5% |
| vs Racing | 73.3% | 71.3% |
| vs Defensive | 71.3% | 67.0% |
| vs Aggressive | 70.5% | 69.0% |
| vs Heuristic | 65.5% | 71.0% |

#### Plateau Analysis
- Model converged to strategies that mirror bot heuristics
- Shaped rewards (+0.20 capture, -0.20 kill) may be creating a ceiling
- Forward progress, scoring, base-leaving rewards are "potential-based" (mathematically safe)
- Capture/kill rewards are NOT potential-based (impose strategy bias)

#### Backup
- `checkpoints/ac_v6_big/backups/model_latest_323k_shaped.pt`
- `checkpoints/ac_v6_big/backups/model_best_323k_shaped.pt`

---

### Experiment 8: Surgical Reward Unbias — Remove Capture/Kill Rewards

- **Start**: ~323K games (resume from latest checkpoint)
- **Change**: Remove ONLY the biased rewards:
  - ❌ Capture enemy: +0.20 → **0.00** (removed)
  - ❌ Got killed: -0.20 → **0.00** (removed)
  - ✅ Spawn: +0.05 (kept — potential-based)
  - ✅ Forward/step: +0.005 (kept — potential-based)
  - ✅ Home stretch: +0.10 (kept — potential-based)
  - ✅ Score token: +0.40 (kept — always objectively good)
- **Hypothesis**: The capture/kill rewards impose a bias that says "capturing is always worth +0.20." This may prevent the model from discovering advanced strategies where NOT capturing (e.g., blocking, sacrificing, holding position) is optimal. Removing them lets the model learn the true value of captures from game outcomes alone, while safe potential-based rewards still provide enough signal for PPO through dice noise.
- **Risk**: Medium. The remaining rewards (spawn, forward, home, score) still provide ~60% of the original reward density. Unlike Experiments 4/5/6, we are NOT reducing all rewards — just removing the biased ones.
- **Results**: **Failure (Regression).** Tracked from 323K to 427K games. 
  - **Win rate trend:** Dropped steadily from ~56.5% peak back down to ~51.2% (approaching random play).
  - **Evaluation:** Win rate against the baseline suite dropped from 74.4% to 69.6%.
  - **Per-opponent Breakdown:** Suffered consistent 1.5% - 2.5% drops against Defensive, Aggressive, and Random bots.
  - **Value Loss:** Increased from 0.733 to 0.740.
- **Conclusion:** The sparse terminal reward of Ludo (+1/-1 over ~175 moves with high dice randomness) creates a **Credit Assignment Nightmare** for standard PPO. The model cannot discern the value of a mid-game capture purely from an outcome that happens 80 moves later. The shaped rewards were crucial priors, not harmful biases.

---

### Phase 8: MCTS Integration (Experiment 9 - AlphaZero Style)

- **Rationale**: PPO with a single forward pass struggles with the variance of Ludo. To learn complex tactics without manual reward shaping, we need explicit search. MCTS reduces variance by simulating thousands of futures per move.
- **Architecture Shift**: 
  - Pivoted from PPO actor-critic to an AlphaZero-style loop.
  - **C++ MCTS Engine**: Leveraged the existing `td_ludo_cpp::MCTSEngine` (Expecti-MCTS with chance nodes for dice rolls, UCB, Dirichlet noise).
  - **Fixes**: Resolved a critical bug where the C++ engine hardcoded 21-channel states instead of AlphaLudoV5's 17-channel architecture.
- **Training Loop (`train_mcts.py`)**:
  1.  **Self-Play:** Run 64 parallel games. For every move, use the current V5 model to guide 50 MCTS simulations. MCTS returns refined policy probabilities.
  2.  **Data Collection:** Store `(state, pi, winner)` triples.
  3.  **Optimization:** Train the V5 model to predict the MCTS policy (Cross-Entropy) and the eventual game outcome (MSE).
- **Warm Start**: Resuming from the `323K` model (our peak shaped-reward checkpoint). This gives MCTS strong structural priors and value estimates right out of the gate, avoiding the "garbage-in-garbage-out" trap of searching with a clueless policy.

- **Results**: **Failure (Regression).** Tracked over 25 Iterations (100,000 transitions).
  - **Win rate trend:** In a 1000-game head-to-head evaluation against the base `323K PPO` model, the trained MCTS model lost disastrously, achieving only a **34.7% win rate**.
  - **Matchups vs Heuristics:** Plunged to ~28% out of a baseline 65-70%.
- **Root Cause**: Ludo breaks the fundamental assumptions of AlphaZero. 
  - *Extreme Stochasticity (Chance Nodes):* The 1/6 dice roll creates an exponentially branching search tree that dilutes MCTS simulations instantly. 200 simulations look nowhere "deep" into the future.
  - *Value Target Noise:* Training the Value network purely on final terminal states ($z \in {-1, +1}$) over extremely long horizons (~400 moves) causes the network to unlearn strategy, since a 99% winning board position can result in a -$1.0$ label due to bad end-game dice luck.
- **Conclusion & Pivot**: Pure MCTS self-play training is inefficient/ineffective for high-variance games like Ludo. 
  - We have **abandoned MCTS training** and reverted `model_latest.pt` back to the **323K PPO Baseline**.
  - We also **restored the Capture (+0.20) and Kill (-0.20) shaped rewards** in `reward_shaping.py`, as Experiment 8 proved that trying to surgically un-bias PPO just destroys its ability to assign credit in sparse terminal environments.

---

### Phase 9: Expert Bot Curriculum Enhancement

- **Start**: 323K games (resumed from `model_latest_323k_shaped.pt` — shaped rewards fully restored after Exp 8/9 failures)
- **End**: 382K games (~59K games run)
- **Change**: Added `Expert` bot to game composition. Expert bot is a hand-coded heuristic combining forward progress + capture aggression + safe-square awareness — harder than individual Aggressive/Defensive/Racing bots.
- **Matchmaking (Phase 9)**: SelfPlay 40%, Expert 25%, Heuristic 15%, Aggressive 10%, Defensive 10%

#### Results

| Metric | Value |
|--------|-------|
| **Total Games** | **382,010** |
| **Total PPO Updates** | **407,502** |
| **Final Eval WR** | **70.6%** |
| Best Eval WR (all time) | 77.4% (at ~170K games) |
| Rolling WR (100g) | ~53% |
| Policy Entropy | 0.46 |
| ELO (main model) | 1,581.9 |
| Value Loss | 0.817 (stable, at ceiling) |
| Policy Loss | ~0.0 (PPO fully clipped) |

#### Per-Opponent Lifetime Win Rates (Final)

| Opponent | Win Rate | Games |
|----------|----------|-------|
| vs Random | 82.5% | 33,663 |
| vs Expert | 60.6% | 14,873 |
| vs Heuristic | 54.7% | 110,786 |
| vs Aggressive | 54.6% | 60,299 |
| vs Defensive | 54.4% | 60,587 |
| vs Racing | 54.3% | 17,433 |
| vs SelfPlay | 49.8% | 180,839 |

#### Analysis

- **Plateau confirmed.** Model settled into 70-73% eval WR oscillation with no signs of breaking through. Expert curriculum added useful data (60.6% WR vs Expert) but did not shift the capability ceiling.
- **PPO fully converged.** Policy loss ~0, clip fraction ~10% — policy barely changing. Optimizer at local optimum.
- **Ghost ELO parity.** Highest-ELO agents are now past ghosts (ghost_422276 @ 1660, ghost_366230 @ 1639) sitting *above* the current model (1581). Sign of mild regression/forgetting — common PPO failure mode at end of training.
- **Value Loss ceiling.** Locked at ~0.81 — irreducible stochastic noise floor of Ludo dice. Critic cannot compress variance of a 175-move, 1/6 probability game any further with this architecture.

#### Checkpoint State at Close of V6 Era

- `backups/model_final_v6_382k_70pct.pt` — final checkpoint before V7 overhaul
- `backups/model_best_v6_77pct_170k.pt` — all-time best (77.4% eval WR, ~170K games)
- See `CHECKPOINT_README.md` for full file inventory

---

---

# V6 Era — Final Summary & Archive (March 2026)

> **Status: ARCHIVED. V7 architecture overhaul begins.**

## Overall Training Timeline

```
-- SL Warm Start ----------------  86% val acc, ~58% WR vs bots
-- Exp 7: PPO Rapid Learning ----  0 to 100K    |  WR: 56% to 70% (rising)
-- Exp 7: Peak ------------------  ~170K        |  WR: 77.4% (ALL-TIME BEST)
-- Exp 7: Plateau ---------------  100K to 323K  |  WR: 73% +/- 2%
-- Exp 8: Unbias Attempt --------  323K to 427K  |  WR: 75% to 69.6% FAILED
-- Exp 9: MCTS Integration ------  25 iters     |  WR: 34.7% vs baseline ABANDONED
-- Phase 9: Expert Curriculum ---  323K to 382K  |  WR: 70-73%, plateau continues
```

## What Worked

- **Dense v1 rewards** — the dopamine-heavy shaped reward set is the backbone of all learning
- **PPO with ghost self-play pool** — ghosts prevented catastrophic forgetting and provided diverse opponents
- **SL warm start** — starting from a supervised policy (86% val acc) dramatically sped up early RL learning
- **Large architecture** (128ch, 10 res blocks) — enabled more complex strategy representation vs earlier v4/v5 models

## What Didn't Work

- **PBRS** (Exp 1) — unavoidable gamma-leak in long games
- **Reward scaling down** (Exp 4, 6) — signal-to-noise collapses with dice variance
- **Removing capture/kill rewards** (Exp 8) — credit assignment breaks without dense shaped rewards in sparse terminal environments
- **MCTS/AlphaZero-style training** (Exp 9) — extreme stochasticity kills MCTS signal; 200 sims ~= 6 ply in a 4-branch dice tree
- **PPO alone past 73%** — policy gradient near zero; architecture hitting expressivity ceiling

## Hard Capability Ceiling Hypothesis

The model plateaued at 73-77% win rate. Likely causes:

1. **Input representation** — 17 channels capture token positions but lack higher-order spatial reasoning (relative distances, blocking formations, threat maps)
2. **Policy head bottleneck** — 4-logit output (one per token) loses move-type semantics; cannot distinguish "move A to safe square" vs "move A into danger"
3. **Reward bias** — v1 rewards incentivize capture/scoring greedily; model may be stuck in a locally optimal aggressive strategy
4. **PPO variance ceiling** — even with batch=512 games, a 175-move stochastic game creates variance that flat-lr PPO cannot overcome

## Recommendations for V7

1. **Richer input channels** — path-distance features, threat fields, safe-path highlighting
2. **Move-type-aware policy head** — encode action semantics rather than pure 4-token logits
3. **Value decomposition** — per-token value heads or auxiliary prediction tasks to improve credit assignment
4. **Consider SAC** — natural entropy regularization suited to stochastic environments; handles dice variance better than standard PPO
5. **Stronger curriculum opponents** — rule-based agents with even 1-ply look-ahead (using dice averaging) would be meaningfully harder than current heuristics

---
---

# Post-V6 Mechanistic Interpretability (March 2026)

> **Purpose**: Diagnose the 73-77% ceiling by probing what the V6 model actually learned internally.
> Full results and visualizations in `discussion/` directory.

---

### Experiment 10: Channel Ablation Study

- **Method**: Zeroed each of 17 input channels individually on 500 random states + curated tactical buckets (200 states each). Measured Policy KL divergence and Critic MAE vs baseline.
- **Key Findings**:
  - **Policy** most sensitive to own token positions: Ch 0 (My Token 0) → KL = 0.384, Ch 1 → KL = 0.188
  - **Critic** dominated by opponent locked % (Ch 10) → MAE = 1.807 — massively above all others
  - **Dice channels** globally washed out (KL < 0.02) but **extremely strong when conditioned on specific rolls** (Roll 6: Ch 16 Policy KL = 0.398)
  - **Captures** are a combined spatial + dice phenomenon: opponent density (Ch 4 KL = 1.033) dominates in capture-available states, with multiple dice channels contributing
- **Implication**: Model treats dice channels as broadcast modifiers, not integrated tactical signals. Processing is reactive, not anticipatory.
- Full results: `discussion/RESULTS_ablation.md`

---

### Experiment 11: Dice Sensitivity Analysis

- **Method**: For 300 states, swept dice roll 1-6, measuring policy shift (JS divergence, action flips) in both masked (legal moves enforced) and unmasked (raw preference) modes.
- **Key Findings**:
  - **295/300 states** flip preferred action across dice rolls (unmasked) — model's behavior is almost entirely dice-determined
  - Roll 6 dominates action concentration: top-prob 0.525 vs 0.31-0.33 for rolls 1-5
  - Masked JS divergence (0.139) much higher than unmasked (0.035) — legal move constraints amplify dice sensitivity
  - **Capture-available states** show highest sensitivity across all metrics
  - **Home stretch 2+ states**: masked policies become extremely confident on many rolls
- **Implication**: The model is a sophisticated lookup table: `f(board, dice) → action`. It has no evidence of multi-turn planning, threat anticipation, or velocity-based reasoning.
- Full results: `discussion/RESULTS_diceSensitivity.md`

---

### Experiment 12: Linear Probing on GAP Features

- **Method**: Extracted 128-dim GAP vector from backbone, trained logistic regression probes on 2,500 decision states for 5 game concepts.
- **Results**:

| Concept | Balanced Accuracy | Majority Baseline |
|---|---:|---:|
| can_capture_this_turn | 0.881 | 0.956 |
| leading_token_in_danger | 0.739 | 0.937 |
| home_stretch_count | 0.731 | 0.683 |
| **eventual_win** | **0.787** | **0.531** |
| closest_token_to_home | 0.577 | 0.377 |

- **Key Findings**:
  - **Eventual winner strongly decodable** (0.787 bal acc) — backbone carries an explicit global advantage signal
  - Progress-to-home concepts are easier than exact token identity — model encodes race progress more cleanly than positional specifics
  - Tactical concepts (capture, danger) are present but sparse/imbalanced in training data
- **Implication**: The backbone *knows* who's winning but the reactive architecture can't translate this into multi-step planning. The advantage signal exists but is trapped behind single-step decision-making.
- Full results: `discussion/RESULTS_linearProbing.md`

---

### Mech Interp Conclusion

The V6 CNN is a **sophisticated reactive player**: it reads the current board + dice, consults internal representations that encode global advantage and tactical opportunities, and selects a locally optimal move. But it has **zero temporal reasoning** — no velocity tracking, no threat persistence memory, no multi-turn setup capability.

This explains the 73-77% ceiling: reactive heuristic-like play is sufficient to beat scripted bots ~75% of the time, but cannot discover higher-order strategies that require looking at *how the board has been evolving* rather than just *what the board currently shows*.

---
---

# V7 Architecture Decision — Sequence Transformer (March 2026)

> **Status: DESIGN PHASE.** Architecture and training plan finalized. Implementation begins.

## Why Transformer + 1D Sequence

The mech interp results (Experiments 10-12) conclusively showed:
- CNN processes the board reactively with no temporal context
- The backbone encodes global advantage but can't use it for planning
- The 15×15 spatial grid is a wasteful encoding for what is fundamentally a 1D track game
- Strategy in Ludo requires temporal reasoning (velocity, threat persistence, multi-turn setups) that a single-frame CNN cannot provide

A Transformer with self-attention over a context window of past turns directly addresses all four limitations.

## V7 Architecture Spec

| Parameter | Value |
|---|---|
| **Input (per turn)** | 1D state vector: 8 token positions (int 0-58) + 3 global scalars + 6-dim dice one-hot + 1 historical action |
| **Position Encoding** | nn.Embedding(60, embed_dim) for token positions; nn.Embedding(5, embed_dim) for actions |
| **Tactical Flags** | None — raw positions only, let attention learn spatial relationships |
| **Context Window** | K=16 past turns (sliding window over game trajectory) |
| **Model** | Transformer, embed_dim=128, 4 attention heads, 4 layers |
| **Policy Head** | Linear(embed_dim → 4) from CLS token or last-turn embedding |
| **Value Head** | Linear(embed_dim → 1, tanh) from same |
| **Estimated Params** | ~800K-1M (vs V6's 3M) |
| **Target Device** | 16GB M4 Mac Mini (MPS) |

### What Attention Enables

- **Cross-token reasoning**: Self-attention between all 8 tokens learns threat relationships (my token near opponent token), blocking formations, and race competition
- **Cross-turn reasoning**: Context window K=16 turns provides velocity information (which tokens are advancing/stalled), threat persistence (opponent lingering near my piece for N turns), and game phase detection
- **Dice-conditioned planning**: Rather than reactively flipping action on dice roll (as V6 does), the model can learn "if I get a 6 next turn" contingency from observing dice patterns

## V7 Training Plan

### Stage 1: SL Warmstart
- Generate 50K-100K bot-vs-bot games in 1D sequence format
- Mix of Heuristic, Aggressive, Defensive, Expert bots
- Store complete game trajectories; slice into K=16 context windows at training time
- Train transformer for 1-2 epochs, target ~80% action prediction accuracy
- Purpose: bootstrap position embeddings, legal move patterns, basic board understanding

### Stage 2: PPO RL (Primary Training)
- Dense v1 rewards (proven effective, do not change reward structure simultaneously with architecture)
- Same curriculum progression as V6: start easy (Random-heavy) → gradually increase difficulty → Expert
- Ghost self-play pool (save model snapshots as past-self opponents)
- Target: 80%+ eval WR (breaking V6's 77.4% ceiling)

### Stage 3: SAC Experiment (V7.1)
- Once PPO validates the architecture at 80%+ WR, experiment with Discrete SAC
- Short replay buffer: last 5K-10K complete game trajectories (not individual transitions)
- Automatic temperature (α) tuning
- Key advantage: sample efficiency (10x+ reuse per game) + natural entropy regularization
- Key risk: distributional shift from off-policy context windows — mitigate with short buffer
- Purpose: determine if SAC can push past PPO's new ceiling

## V7 Design References

- Architecture evolution discussion: `discussion/AlphaLudo Architecture Evolution_ From Reactive CNN to Strategic Sequence Transformer.docx`
- 1D state vector specification: `discussion/The 2-Player 1D State Vector .docx`
