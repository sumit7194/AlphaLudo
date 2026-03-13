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

### Phase 9: Return to PPO with Enhanced Heuristics (Planned)

- **Start**: Resuming from `323K` checkpoint with v1 dense combat rewards fully enabled.
- **Goal**: Research and implement stronger heuristic bot algorithms globally to force the PPO agent to learn deeper tactics through harder curriculum opponents, before unpausing PPO training.

