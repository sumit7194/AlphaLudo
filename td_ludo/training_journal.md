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

---

## V6 Strategic Reward Overhaul

### Experiment 10: Strategic Reward Shaping v2.0

- **Date**: March 25, 2026
- **Resume from**: V6 latest checkpoint (382K games, 70.6% eval WR)
- **Architecture**: AlphaLudoV3 (128ch, 10 ResBlocks, 17ch input, ~3M params)
- **Pipeline**: train.py + game_player.py (V6 pipeline, ~64 GPM)

#### Motivation

Mech interp (Experiments 10-12 in `discussion/`) showed V6 is a "sophisticated lookup table" — reads board + dice, makes locally optimal moves, but has zero strategic reasoning. The root cause is the reward function: it rewards outcomes (captures, scoring) but not setup behaviors (chasing, safety-seeking, blocking).

V9 (CNN + Transformer) attempted to fix this with temporal context, but the weaker CNN backbone (80ch vs 128ch) and insufficient training meant it never matched V6's spatial perception. The hypothesis now: **better rewards on V6's strong CNN will teach strategic play without needing temporal context**.

#### Changes: 6 New Strategic Rewards (added on top of all v1.1 rewards)

| Reward | Trigger | Magnitude | Design |
|--------|---------|-----------|--------|
| **Chase target** | Moved token gains new capturable opp in dice range (1-6) | +0.06 per target | Delta-based, skip safe/stacked targets |
| **Safety transition** | Token escapes danger (opp 1-6 behind) to safe/stacked position | +0.08 | Only when token was actually endangered |
| **Danger reduction** | Fewer own tokens in danger after move | +0.06 per token | Delta-based, max(delta, 0) |
| **Stack formed** | New 2+ token stack on main track | +0.07 per stack | Delta-based, main track only |
| **Leader capture** | Capture the score leader | +0.08 bonus | On top of existing +0.20 |
| **Endgame urgency** | Score token when opp has 3+ scored | +0.15 bonus | On top of existing +0.40 |

#### Design Principles
1. All existing v1.1 rewards preserved exactly (Exp 8 proved removal hurts)
2. Delta-based: reward state transitions, not static states (avoids PBRS γ-leak from Exp 1)
3. Magnitudes ≥ 0.05 (Exp 4/6 showed weaker signals drown in dice noise)
4. max(delta, 0) pattern: only reward improvement, never penalize lack of it
5. Safety transition requires actual danger (opponent 1-6 behind), not just any unsafe→safe

#### Smoke Test Results (200 random games)
- Avg reward/game: ~9.7 (was ~7.5 with v1.1 only)
- New rewards add ~2.0-2.5 per game on top of existing ~7.0
- Terminal ±1.0 still dominates returns over full episode (γ=0.999, ~44 player moves)

#### Monitoring Plan
- Run 15K+ games minimum before judging (per journal rules)
- Watch: entropy > 0.35, policy loss < 0.05, eval WR trend
- Key question: does the model start chasing/blocking/seeking safety?
- Backup: `checkpoints/ac_v6_big/backups/model_latest_382k_pre_strategic_rewards.pt`

#### Results (v2.0 — never deployed)
The v2.0 strategic rewards were committed (Mar 25) but **never actually deployed to training**. The local working copy retained v1.1 rewards throughout, so the ac_v6_big run from 382K→632K trained entirely on v1.1. The v2.0 code existed only in git history.

---

### Experiment 10: Empirical Reward Decomposition & v2.2 Noise-Pruned Rewards
**Date**: 2026-03-28 | **Games at start**: ~632K (v1.1 running)

#### Motivation
Before deploying strategic rewards, ran a 300-game decomposition analysis to verify which rewards actually correlate with winning. Played current model (AlphaLudoV6-Big, 128ch/10res) vs Expert bot, tracked per-game reward totals split by win/loss.

#### Key Finding: v1.1 Rewards vs Strategic Rewards

| Category | Win avg | Loss avg | Delta | Signal strength |
|----------|---------|----------|-------|-----------------|
| **v1.1 total** | +5.34 | +3.67 | **+1.67** | Strong |
| **Strategic total** | +0.43 | +0.37 | **+0.058** | 29x weaker |

#### Per-Component Win-Loss Correlation

| Reward | Win avg | Loss avg | Delta | Verdict |
|--------|---------|----------|-------|---------|
| score | +1.60 | +0.84 | +0.76 | WIN-CORR (strongest) |
| forward | +2.31 | +1.72 | +0.59 | WIN-CORR |
| capture | +0.69 | +0.47 | +0.22 | WIN-CORR |
| home_stretch | +0.39 | +0.24 | +0.14 | WIN-CORR |
| danger_reduction | +0.12 | +0.09 | **+0.024** | WIN-CORR (best strategic) |
| leader_capture | +0.04 | +0.03 | **+0.011** | WIN-CORR |
| endgame_urgency | +0.04 | +0.03 | **+0.011** | WIN-CORR |
| safety_transition | +0.05 | +0.04 | **+0.009** | WIN-CORR |
| **chase_target** | +0.13 | +0.12 | **+0.005** | **NEUTRAL (noise)** |
| **stack_formed** | +0.06 | +0.06 | **-0.002** | **NEUTRAL (noise)** |

#### Decision: Deploy v2.2 (noise-pruned)
- **Removed**: chase (+0.02) and stack (+0.02) — verified noise, no win correlation
- **Kept**: safety (+0.025), danger_reduction (+0.02), leader_capture (+0.025), endgame_urgency (+0.05)
- Magnitudes kept at current levels; will increase in a future experiment if the model plateaus again

#### Additional Context: Training State at 632K
- Eval WR: ~66-69% (rolling avg), plateau since ~460K
- Value loss: 0.92 (stable but elevated from 0.78 pre-382K)
- Entropy: 0.51 (slowly rising)
- Game length: ~190 moves (up from 176 pre-382K)
- vs Expert: trending down from 60.6% (300-382K) to 51.2% (600-640K)
- vs Heuristic: trending down from 56.2% to 50.0%

#### Monitoring Plan
- Watch value loss: should stay ≤0.92 or ideally decrease (fewer noise sources)
- Watch eval WR: looking for recovery above 69% plateau
- Watch vs Expert/Heuristic WR: should stabilize or improve
- If no improvement after 50K games, consider increasing magnitudes of remaining 4 rewards
- This is the first time strategic rewards are actually deployed to training

---

### Experiment 11: Return Normalization Fix & V6 Degradation Root Cause
**Date**: 2026-03-31 | **Model**: V6 (ac_v6_big)

#### Discovery: PPO Value Head Drift
Analysis of V6 training history revealed the model peaked at 77.4% eval (170K games) then declined monotonically to 57% (820K games). Root cause investigation:

1. **Value head positive bias**: Returns from dense rewards are always positive (+1 to +7), even for losses. The value head learned to predict ~+3.5 for all states, providing poor advantage estimates.
2. **Missing return normalization**: Standard PPO implementations normalize returns to zero-mean/unit-std. This was missing from the trainer, causing the value head to operate on large, always-positive targets.
3. **Original system worked differently**: The pre-rewrite trainer (pre-300K) used TD(0) with tanh-bounded values and PBRS rewards (zero-mean by design), which implicitly avoided this issue.

#### Fix Applied
Added running mean/std return normalization to `src/trainer.py`:
```python
all_returns = (all_returns_raw - running_mean) / (running_std + 1e-8)
```
Running stats use exponential moving average (α=0.01) and persist in checkpoints.

#### Results
- Value loss dropped from 0.92 → 0.27 (massive improvement)
- Model stopped degrading — held steady at 70-71% eval from 262K→522K
- However, did not surpass the old 77.4% peak

#### Championship Tournament (500 games per pair)
Ran round-robin between V6 snapshots to measure actual strength:

| Rank | Player | Overall WR |
|------|--------|-----------|
| 1 | V6_170k (best) | 61.5% |
| 2 | V6_382k | 60.8% |
| 3 | Ghost_570k | 55.6% |
| 4 | V6_latest (675K) | 51.2% |
| 5 | Expert_bot | 35.6% |
| 6 | SL_pretrain | 35.3% |

**Key finding**: V6_170k beats V6_latest 60-40%. More training made the model weaker.

---

### Experiment 12: V6.1 Strategic Input Encoding (24 channels)
**Date**: 2026-04-03 | **Model**: V6.1 (ac_v6_1_strategic)

#### Motivation
Mech interp on V6 (522K) revealed information gaps in the 17-channel encoding:
- Opponent tokens lumped as density (Ch 4) — can't distinguish stacked vs spread
- No explicit danger information — CNN must compute threat distance from spatial correlation
- No capture opportunity signal — requires cross-referencing token positions with dice
- Score diff (Ch 8) completely unused by the model (0.000 ablation impact)

#### Architecture: 24 Channels (17 existing + 7 new)

| Channel | Content | Justification |
|---------|---------|---------------|
| 0-16 | Original V6 encoding | Unchanged, proven |
| **17-20** | Individual opponent tokens (1.0 each) | Resolves density ambiguity |
| **21** | Danger map (1.0 at endangered own tokens) | Direct threat signal |
| **22** | Capture opportunity map (1.0 at capturable positions) | Direct tactical signal |
| **23** | Safe landing map (1.0 at safe reachable positions) | Direct safety signal |

**Model**: AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24) — ~3M params
**Encoding**: `encode_state_v6()` implemented in C++ (game.cpp)

#### Mech Interp Findings (informing V6.1 design)

**Channel Ablation (522K V6)**:
- Score diff (Ch 8): Policy KL = 0.000, Value MAE = 0.000 — completely ignored
- Opp density (Ch 4): Value MAE = 0.216 — critical for value head but ambiguous
- Dice 6 (Ch 16): Policy KL = 0.127 — special, much higher than dice 1-5

**Linear Probes (128-dim GAP features)**:
- game_phase: 96.2% balanced acc — perfectly decodable
- can_capture_this_turn: 82.3% — model already detects captures (but not perfectly)
- leading_token_in_danger: 81.4% — model already detects danger (but 18% gap)
- eventual_win: 69.8% — moderate win prediction

**CKA Similarity**:
- Blocks 7-9: CKA > 0.98 in all phases — redundant
- Blocks 3-5: CKA drops to 0.52-0.63 in late game — doing real work
- Decision: Keep all 10 blocks for V6.1 (new inputs might activate previously redundant blocks)

#### Training Setup
- **SL pretraining**: Knowledge distillation from V6 teacher (522K model)
  - V6 plays games (17ch encoding) → V6.1 encodes same states (24ch) → trains on V6's soft policy
  - 500K states generated from 5,615 games, 4 epochs, 84.5% action prediction accuracy
- **RL training**: PPO with v2.2 rewards + return normalization
- **Rewards**: v2.2 noise-pruned (safety, danger_reduction, leader_capture, endgame_urgency)

#### Results (as of 132K games)

| Games | Eval WR | Value Loss | Entropy | Notes |
|-------|---------|-----------|---------|-------|
| 5K | 72.2% | 0.277 | 0.508 | SL baseline, first RL eval |
| 25K | 76.8% | 0.262 | 0.421 | Rapid improvement |
| 50K | **78.0%** | 0.261 | 0.395 | **New all-time best** (V6 peak was 77.4%) |
| 75K | 75.4% | 0.259 | 0.389 | Slight dip |
| 100K | 73.6% | 0.257 | 0.382 | Settled |
| 130K | 74.8% | 0.257 | 0.387 | Plateau at ~74-76% |

**Last 10 eval avg: 74.6%**

#### Key Takeaways
1. **V6.1 reached 78% in 50K games** — V6 needed 170K for 77.4%. SL distillation + strategic channels = 3.4x faster
2. **New all-time best**: 78.0% eval WR surpasses V6's 77.4%
3. **Value loss stable at 0.26** — return normalization working perfectly
4. **Plateaued at ~74-76%** after the initial peak — similar pattern to V6 but at a higher baseline
5. **Open question**: Is ~78% the theoretical ceiling for 2-player Ludo against Expert bot, or can we push further?

#### Comparison: V6 vs V6.1

| Metric | V6 (best) | V6.1 (best) |
|--------|-----------|-------------|
| Peak eval WR | 77.4% | **78.0%** |
| Games to peak | 170K | **50K** |
| Value loss at peak | 0.80 | **0.26** |
| Params | 2.99M | 3.00M |
| Input channels | 17 | 24 |
| SL pretrained | Yes (bot imitation) | Yes (V6 distillation) |

---

## Hard Rules (updated)

- ❌ Don't use PBRS in >100 move games
- ❌ Don't scale intermediate rewards below 0.05
- ❌ Don't change rewards mid-training without clearing buffer
- ❌ Don't train PPO without return normalization
- ❌ Don't remove tanh on value head without return normalization
- ❌ Don't edit training code in worktrees — always edit main repo directly
- ❌ Don't rebuild `td_ludo_cpp` without first deleting `td_ludo/td_ludo_cpp*.so` — Python prefers the local `.so` (editable install leftover) over the freshly installed site-packages wheel, silently using the stale build. Symptom: code changes don't take effect after `pip install .`
- ❌ Don't change model input channels without auditing `src/mcts.cpp::get_leaf_tensors()` and `src/bindings.cpp::get_leaf_tensors` lambda — both hardcode the channel count. Currently set to 24 for V6.1; previously stuck at 17 which broke Exp 9 silently
- ✅ DO use loud rewards
- ✅ DO normalize returns (running mean/std) before advantage computation
- ✅ DO use knowledge distillation from stronger model for SL pretraining
- ✅ DO validate strategic rewards with empirical win-loss decomposition before deploying
- ✅ DO let experiments run 15K+ games minimum
- ✅ DO track per-opponent WR trends
- ✅ DO run mech interp to inform architecture decisions

---

---

### Experiment 13a: V6.1 Resume Training (2026-04-09)
**Run**: ac_v6_1_strategic | **Games**: 155,464 → 239,513 (+84K) | **Platform**: GCP T4 (CUDA)

Resumed V6.1 from its old peak (155K, eval 0.78) to test whether the plateau was an artifact of the earlier MPS-only training or a real ceiling. Ran for ~18 hours.

**Result**: **+0.8pp improvement** (0.78 → 0.788) over 84K additional games. Plateau confirmed.

- Best eval WR: **0.788** (hit once, never beaten across 17+ subsequent 500-game evals)
- Eval moving avg: bounced in 74-76% band the entire run
- Expert training WR stayed in 62-68% band across the run (no directional drift)
- Elo oscillated 1440-1650 with no persistent advantage over self-generated ghosts
- PPO metrics all healthy: vl≈0.26, entropy≈0.39, kl≈0.011, clip≈0.08, no NaN
- GPM 130-140 on T4 (4× the old MPS throughput)

Training paused at game 239,513. Checkpoints preserved:
- `checkpoints/ac_v6_1_strategic/backups/v6_1_best_157k_wr788_20260410.pt` (peak snapshot)
- `checkpoints/ac_v6_1_strategic/backups/v6_1_latest_239k_best788_20260410.pt` (post-resume)

**Conclusion**: V6.1 architecture has genuinely plateaued at ~78-79% eval. Same conclusion as V6 at 77.4% and V6.2 at ~76%. Three architecture generations in the same neighborhood = real ceiling under PPO self-play, not a training bug.

### Experiment 13b: Inference-Time MCTS Sweep (2026-04-10, in progress)
**Run**: `checkpoints/mcts_eval/` | **Platform**: GCP T4

First results of the MCTS-amplification experiment (Step 1 in `POST_V61_EXPERIMENT_PLAN.md`).

**Phase A completed**: V6.1 raw vs Expert, 1000 games → **70.0% WR** (700W/300L). This is the clean baseline — lower than the noisy 74-76% moving average from training evals because of the larger sample and fewer per-bot variations.

**Phase B+ running**: MCTS(25/50/100/200) vs Expert, plus head-to-head vs raw. Early signal: MCTS(25) at ~950 games (from the aborted first run) was showing 73% WR — +3pp over raw baseline.

Full results will land here when the sweep completes (~25-35 hours ETA). In-flight dashboard at `http://35.201.209.164:8788/`.

---

### Experiment 13c: Inference-Time MCTS Sweep — COMPLETED (2026-04-10)
**Run**: `checkpoints/mcts_eval/` | **Platform**: GCP T4 | **Games**: 14,500 across 9 matchups

Tested whether MCTS search at inference time could amplify V6.1's policy priors.

**Result: DECISIVE FAILURE.** More sims = worse play at every level tested.

| Matchup | Sims | WR | Verdict |
|---|---:|---:|---|
| Raw vs Expert | 0 | **70.0%** | baseline |
| MCTS(25) vs Expert | 25 | 69.8% | flat |
| MCTS(50) vs Expert | 50 | 57.1% | −13pp |
| MCTS(100) vs Expert | 100 | 51.0% | −19pp |
| MCTS(200) vs Expert | 200 | 48.4% | −22pp |
| MCTS(50) vs Raw (H2H) | 50 | 36.9% | raw wins 63% |
| MCTS(100) vs Raw (H2H) | 100 | 36.6% | raw wins 63% |
| MCTS(200) vs Raw (H2H) | 200 | 31.8% | raw wins 68% |

Root cause: V5 value head is unbounded (no tanh), `torch.clamp(-1,1)` is lossy for pUCT. Ludo's 6× dice branching dilutes simulation budget into noise. MCTS amplifies noise when value estimates are themselves noisy.

---

### Experiment 13d: Reward Shaping 50% Reduction — COMPLETED (2026-04-11)
**Run**: ac_v6_1_strategic (modified rewards) | **Games**: 239K → 269K (+30K) | **Platform**: GCP T4

Halved all intermediate reward magnitudes (score 0.40→0.20, capture 0.20→0.10, etc.) to test whether reward saturation was the plateau cause.

**Result: NEUTRAL-TO-MILDLY-HARMFUL.** Eval MA drifted from ~76% to ~74% (−2pp). Never exceeded pre-experiment best of 78.8%. No collapse (vl stayed <0.30, entropy healthy at 0.41), but zero improvement.

**Conclusion**: The journal's v1 reward magnitudes are the optimal operating point. Saturation is NOT the plateau's cause. Rewards reverted to v2.2 after experiment.

---

### Experiment 13e: Human Benchmark (2026-04-11)
Sumit played 3 games vs V6.1's best model (78.8% eval) via the Play web UI. Model lost all 3, but marginally — close games with only a few suboptimal moves.

**Key observation**: the model's blind spot is **multi-turn planning**. It doesn't know that rolling 6 gives a bonus turn, so it can't plan 2-move sequences (chase-then-capture, spawn-then-advance). A human naturally does this ("if I roll a 6, I get another chance to catch that opponent 10 ahead").

This observation directly motivated V6.3.

---

### Experiment 14: V6.3 — Bonus-Turn Awareness + Capture Prediction (2026-04-11, IN PROGRESS)
**Run**: ac_v6_3_capture | **Base**: V6.1 model_best (56K games, eval 0.78) | **Architecture**: AlphaLudoV63

V6.3 adds 3 new input channels (24→27) and 1 auxiliary prediction head to V6.1:

**New input channels:**
- Ch 24 `bonus_turn_flag`: broadcast 1.0 if dice=6 (player will get another turn)
- Ch 25 `consecutive_sixes`: broadcast 0.0/0.5/1.0 for 0/1/2 consecutive 6s (warns of triple-6 penalty)
- Ch 26 `two_roll_capture_map`: 1.0 at positions where opponents are capturable in a 6+X two-roll sequence (7-12 squares ahead)

**Auxiliary capture prediction head:**
- Linear(128→64→1) with sigmoid, branching from GAP features
- Target: "did this player capture an opponent within the next 5 of their own turns?"
- Loss: BCE at 0.1× weight, added to PPO loss
- Purpose: forces the model to learn multi-step threat awareness from single-state snapshots

**Weight transfer**: V6.1 stem conv zero-padded from (128,24,3,3) → (128,27,3,3). All other weights copied exactly. Aux head random-init. Model starts as V6.1-equivalent (init parity verified at diff=0.0).

**Hypothesis**: the bonus-turn channels give the CNN explicit information it currently lacks (dice=6 → I get another action), and the aux head forces the backbone features to encode multi-step capture opportunities that improve both the aux prediction AND the policy.

**Success criteria**: eval WR > 80% sustained (breaks the 78.8% all-time best).

**Phase 1 results (0→140K games, aux ON with backbone gradient flow):**
- Eval WR never surpassed initial 78.4% — averaged ~73.5%, declining trend
- Entropy rose monotonically: 0.374 → 0.417
- **Bug found**: aux head collapsed to predicting ~0.93 everywhere (true positive rate ~9%). The BCE loss with 0.1× weight was backpropagating bad gradients into the shared backbone, corrupting the learned V6.1 representations
- **Fix**: `trainer_v63._ppo_update` now passes `detach_aux=True` to model forward — aux gradients only update aux head weights, not backbone. Added `pos_weight=10` to BCE for class imbalance.
- Also fixed `recent_clip_fracs` → `recent_clip_fractions` attribute name bug in trainer

**Phase 2 results (140K→170K games, aux detached + pos_weight):**
- VM preempted at 170K. Eval WR still flat ~73%
- Entropy continued rising

**Phase 3 results (170K→310K games, aux detached + pos_weight, resumed after preemption):**
- 148 evals total over 310K games. Best remained 78.4% (the very first eval from V6.1 init)
- Eval WR trajectory: started 78.4%, declined to avg ~73%, latest readings 67-70%
- Entropy: 0.374 → 0.462 (rising monotonically across all 310K games)
- **Bug found again**: aux head re-collapsed to 0.90 mean despite detach. The `pos_weight=10` was overweighting rare positives, causing the head to predict 1 everywhere. Even though detached, the aux params consumed optimizer state and gradient norm budget.
- **Key finding**: entropy at 308K (0.43) is actually LOWER than V6.1 init entropy (0.82). The "rising" trend was recovering from over-compressed policy, not diverging. This suggests the model may still be adapting.
- BatchNorm running stats drifted significantly (running_var up to 0.42 in deep blocks) — expected with new channel distributions propagating through the network.
- New channel stem weights learned slowly: Ch24 drift=0.003, Ch25=0.004, Ch26=0.006 (vs old channels ~0.01). Weak gradient signal from sparse new features.

**Phase 4 results (310K→395K games, aux DISABLED):**
- Disabled aux loss entirely (`aux_loss_coeff=0.0`). Pure PPO training with 27-channel input.
- Rationale: aux head repeatedly collapsed and polluted optimizer. The 3 new channels should learn from policy/value gradients alone if they carry useful information.
- Result: ~85K games of pure PPO training improved mean eval WR by only +1.5% (71.1% → 72.8%). Best eval WR never surpassed 78.4% (initial V6.1 weights). Plateau confirmed — V6.1-transferred backbone could not recover or exceed V6.1's performance.
- Conclusion: weight transfer approach fundamentally broken. The backbone learned to use 24 channels first; adding 3 more channels post-hoc does not get integrated properly.
- Archived this run to `checkpoints/_archive/v6_3_27ch_failed_2026_04_14/` for reference.

**Phase 5 (2026-04-14, FRESH ATTEMPT: SL distillation + RL):**

Different strategy after Phase 1-4 failure: train V6.3 from scratch via supervised learning on V6.1 self-play games, then RL from that SL baseline. This way the 27-channel architecture learns to use ALL channels from day one, not as zero-padded additions.

**Step 1 — SL data generation (`generate_sl_data_v6_3.py`)**:
- V6.1 best model plays 5,250 self-play games on GCP (8 minutes, batched 512 parallel games)
- Records 1,000,345 (state_27ch, V6.1_policy_distribution, game_outcome) tuples
- Pure V6.1 self-play (no heuristic bots) — clean teacher signal
- V6.1 plays at temperature 1.0 with `encode_state_v6` (24ch); we save `encode_state_v6_3` (27ch) for the same state alongside V6.1's policy probs
- Output: 101 NPZ chunks, 100 MB total

**Step 2 — SL training (`train_sl_v6_3.py`)**:
- Random-init V6.3 model (3,010,758 params)
- Loss: `KL(V6.1_policy || student_policy) + 0.5 * smooth_l1(value, game_outcome)`
- 250K samples in RAM (6 GB float32, fit on 14GB VM); 10 epochs, batch 512, AdamW lr=1e-3 with cosine schedule
- 38 minutes total
- **Final epoch: 94.6% val action-match accuracy** vs teacher, KL=0.019. Successful distillation.
- **SL-only eval vs Expert (200 games): 74.0% WR** — only 4.8% below V6.1 (78.8%), much better than V6.1-transfer's 72-73% plateau

**Step 3 — RL from SL checkpoint (`train_v6_3.py`, no `--resume`)**:
- Loads `model_sl.pt` as starting weights, runs PPO with 27ch input, no aux head, return normalization on
- Trained 156,040 games (205,182 PPO updates) over ~21 hours on T4
- 77 evals (every 2K games)
- **Best eval WR: 77.8%** (1.0% below V6.1's 78.8%)
- **Mean eval WR: 73.7%**
- **First 10 evals avg: 73.9%** (matches SL baseline 74.0%)
- **Last 10 evals avg: 74.6%** (only +0.7% over the entire RL run)
- Entropy stable at 0.40 (no collapse, no blow-up — much healthier than Phase 1-4)
- vs Expert (recent): 59.4% WR

**Final verdict on V6.3 (2026-04-16)**:

V6.3 is a **failed hypothesis**, but cleanly:
- The 27-channel architecture itself is fine (SL achieves 74% from random init)
- Pure RL from a clean SL baseline trained healthy without instability
- BUT 156K RL games could not break past V6.1's 78.8% ceiling — best eval was 77.8%
- The 3 new channels (`bonus_turn_flag`, `consecutive_sixes`, `two_roll_capture_map`) provide **no measurable value** beyond what V6.1 already learned implicitly from the 24-channel encoding
- V6.1 likely already encodes "dice=6 → bonus turn" through the dice channel + self-play learning. The explicit features are redundant.

**Compared to all V6.3 attempts:**
| Run | Method | Games | Best Eval | Mean Eval |
|---|---|---|---|---|
| V6.1 baseline | RL only (24ch) | 56K | 78.8% | ~76% |
| V6.3 attempt 1 | V6.1 transfer + RL | 395K | 78.4% (init only) | 71-73% |
| V6.3 attempt 2 | SL distill + RL | 156K | **77.8%** | **73.7%** |

V6.3 is abandoned. The 78.8% V6.1 ceiling appears to be a fundamental property of the game/architecture combo, not an artifact of missing features.

---

### Experiment 14b: 1000-game robust head-to-head (2026-04-17)

**Motivation**: Human play tests of V6.3 (Phase 5 attempt 2) showed strong tactical play with no obvious mistakes — losses traced to dice variance, not strategic errors. This contradicted the "failed" Phase 5 verdict, prompting a robust statistical comparison.

**Setup**: 1000 games each, CPU device, against the 4 strong heuristic bots (Expert, Heuristic, Aggressive, Defensive) — no SelfPlay, no ghosts, no Random/Inactive padding. Apples-to-apples measurement.

| Opponent | V6.1 best (24ch) | V6.3 latest (27ch SL+RL) | Δ |
|---|---|---|---|
| Heuristic | 77.6% (190/245) | **77.9%** (176/226) | +0.3 |
| Aggressive | **73.1%** (174/238) | 70.1% (169/241) | -3.0 |
| Defensive | 67.6% (165/244) | **68.1%** (194/285) | +0.5 |
| Expert | **68.9%** (188/273) | 66.1% (164/248) | -2.8 |
| **Overall** | **71.7%** (717/1000) | **70.3%** (703/1000) | **-1.4** |

**Statistical analysis**: At n=1000, std error ≈ ±1.45pp. The 1.4pp gap is at exactly 1 SE — **within noise**. V6.3 and V6.1 are statistically equivalent.

**Reconciling with prior numbers**:
- The 78.8% V6.1 / 77.8% V6.3 "best eval" figures from training were cherry-picked maxima of 500-game evals (high variance) using a broader bot mix that included Inactive/SelfPlay/Random — inflating apparent strength
- This 1000-game strategic-bots-only measurement is the true skill ceiling: ~71-72% for both architectures

**Revised verdict (replaces "V6.3 failed" framing)**:

V6.3 is **neutral, not failed**. The data shows:
1. SL distillation pipeline works (94.6% action match → 70.3% vs strong bots, in same band as V6.1)
2. Channel expansion to 27ch is non-destructive (model reaches V6.1-equivalent strength)
3. The 3 new channels (`bonus_turn_flag`, `consecutive_sixes`, `two_roll_capture_map`) provide **no measurable lift** but also **no degradation**
4. Human-play observations (strong tactical play, losses to dice variance only) confirm V6.3 plays at V6.1-equivalent skill

V6.3 shipped as a **viable production alternative** to V6.1, not a failed experiment. Use either model interchangeably; V6.3 is preferred only if a future feature requires the extra channels for analysis/explainability (e.g., the auxiliary capture-prediction head could be revived for value-debugging UIs).

The real ceiling discovery: **~71% vs strategic bots is the architectural plateau** for this CNN family on 2-player Ludo, not the 78.8% number we'd been chasing. Future work should target architecture changes (e.g., wider/deeper, attention, MCTS at inference) to break this true ceiling.

---

### Experiment 15: V10 — Slim joint-SL architecture (2026-04-21, IN PROGRESS)

**Motivation**: After the 71% plateau confirmed, two failed intermediate experiments re-pointed at the root cause:

1. **V6.3 calibrated heads on frozen backbone** (train_heads_v6_3.py): froze V6.3 after PPO, trained dedicated `win_prob` (BCE) + `moves_remaining` (MSE) heads on a 250K-state SL dataset. Val accuracy stuck at 65.9% from epoch 1 onward — the backbone features, optimized by PPO for action selection, don't encode the outcome-prediction signal the heads need.
2. **V6.3 1-ply value search** (evaluate_v6_3_value_search.py): used V6.3's existing value head for 1-ply lookahead. Result: 27.5% WR vs 67.5% pure-policy baseline. The value head was trained on normalized returns, not win probabilities — so `V(s)` is not `P(win|s)` and lookahead collapses.

Both failures point to the same underlying issue: **you cannot retrofit calibrated outcome prediction onto a backbone that wasn't trained to produce it**.

V10 inverts the assumption: train the backbone from scratch with a multi-task loss that forces it to learn features useful for policy, win probability, *and* pace-of-game — all at once.

**Architecture** (driven by V6.3 mech interp findings):

| Dimension | V6.3 | V10 | Rationale |
|---|---|---|---|
| Residual blocks | 10 | **6** | Last 3 V6.3 layers were near-identical in CKA and contributed ~0 in layer knockout |
| Channel width | 128 | **96** | Most of 128 were active, modest shrink |
| Input channels | 27 | **28** | Drop V6.3 ch25 (consecutive_sixes, KL=0.0000 — dead); add 2 strategic |
| Parameters | ~3M | **1.04M** | 3× smaller, fast enough to train locally |
| Heads | policy + value | **policy + win_prob + moves_remaining** | Drop `value`; `win_prob` is the value proxy (2·p−1 ∈ [−1,1]) |

Two new input channels targeting the "option value" blind spot found during human play:
- **ch26 `non_home_tokens_frac`**: (tokens not yet home) / 4, broadcast. Signals "forced mode" (0.25 when 3/4 scored) and endgame proximity.
- **ch27 `my_leader_progress`**: max(token_progress_01), broadcast. Top token's position divided by 56.

**Pipeline** (local, Mac MPS):
1. `generate_sl_data_v10.py` — V6.1 teacher self-play (2000 states/s on MPS), emits (state28, teacher_policy, won, own_moves_remaining, legal_mask) per decision. Target 150K states.
2. `train_sl_v10.py` — joint loss `1.0·KL + 0.5·BCE + 0.02·MSE` from scratch.
3. `eval_v10_sl.py` — policy WR vs heuristic bots + Brier score + reliability buckets + moves MAE.
4. `pipeline_dashboard.py` — live HTTP dashboard on :8788 surfacing all three stages.

**Target signal** for declaring V10 worth investing RL into:
- Policy WR ≥ 60% vs Expert (SL alone, no RL)
- **Brier < 0.20** (V6.3-on-frozen-backbone was ~0.23; this is the go/no-go)
- Moves MAE < 10 own-turns

**SL training results** (3 epochs, 150K states, 1,036,278 params):

| Metric | E1 | E2 | E3 |
|---|---|---|---|
| Val policy acc | 60.3% | 60.3% | 60.2% |
| Val win acc | 58.1% | 64.2% | 64.2% |
| Val moves MAE | 10.7 | 10.7 | 10.4 |

Policy acc plateaus at 60% (V6.3 SL reached 94.6% — V10 at 1/5 the epochs is badly undercooked). Win acc climbed E1→E2 (+6.1pp) then stalled. Moves MAE barely moved.

**Final eval (200 games vs bot mix)**:

| Metric | Value |
|---|---|
| Overall policy WR | **27.5%** |
| vs Expert | 23.3% |
| vs Heuristic | 18.6% |
| vs Aggressive | 19.4% |
| vs Defensive | 32.4% |
| vs Random | **53.8%** (SE ~10pp — indistinguishable from 50%) |
| Brier score | **0.1948** (crossed <0.20 threshold) |
| Moves MAE | 12.94 |

**Calibration buckets** — the interesting result:

| Predicted range | N (of 13,220) | Pred mean | Actual mean |
|---|---|---|---|
| [0.0, 0.2) | 1,388 (10.5%) | 0.132 | 0.020 |
| [0.2, 0.4) | 3,244 (24.5%) | 0.314 | 0.150 |
| [0.4, 0.6) | 6,656 (50.3%) | 0.500 | 0.349 |
| [0.6, 0.8) | 1,648 (12.5%) | 0.682 | **0.664** |
| [0.8, 1.0) | 284 (2.1%) | 0.848 | **0.842** |

**Interpretation** (Brier-pass ≠ win):

The Brier score passed the 0.20 threshold only because 50% of predictions cluster at ~50% — hedging is cheap MSE. The real signal is in the buckets:

- **At high confidence (>0.6, 14.6% of decisions): calibration is excellent.** 0.848 → 0.842 is near-perfect, far better than V6.3's frozen-backbone attempt (stuck at 65.9% ~ equivalent to Brier ~0.23).
- **At low/mid confidence (<0.6, 85.4% of decisions): systematic over-optimism** by 11-16pp. The model thinks random play wins half the time when it actually loses ⅔ of the time.

Architectural takeaway: **joint training does push the backbone toward outcome-predictive features** (proven by the high-confidence buckets). But with only 3 epochs of 150K samples, the policy is too weak for those features to matter in most positions, so the model hedges and calibration at mid-range suffers.

**Revised verdict**:

Do **not** build V10 RL trainer yet. The Brier-< 0.20 threshold was mechanically met but for the wrong reason (hedging, not skill). The architecture shows real promise at the calibration extremes, but needs:

- **More epochs** (15 instead of 3) — match V6.3's SL budget
- **Possibly more data** (500K states back on the table)

Plan: retrain with 15 epochs on same 150K data first (cheapest signal). If policy acc climbs from 60% toward 85%+ and bucket calibration spreads (fewer predictions stuck at 50%), re-eval and reconsider RL. If policy stays flat at 60%, regenerate 500K states and retrain.

Secondary finding: **cosine LR schedule hit 0.00000 at end of epoch 3** because the scheduler was configured for 3 epochs total. When retraining with more epochs, the scheduler will properly anneal over the full run.

---

## Active Experiment Plan (post-V6.1 plateau)

As of 2026-04-11. Steps 1 (MCTS) and 2 (reward shaping) completed and failed. Step 4 (human benchmark) completed — identified multi-turn blindness. V6.3 experiment in progress.

**→ `/Users/sumit/Github/AlphaLudo/discussion/POST_V61_EXPERIMENT_PLAN.md`**
