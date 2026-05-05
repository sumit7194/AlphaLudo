# mcts_v1 — Search-augmented training experiments

Operational spec for the MCTS plateau-break experiment. Strategic
context lives in `td_ludo/post_v13_2_experiments.md`; this file is the
"how to actually run it" doc.

## Goal

Break the 80-83% plateau that V12.2 / V13.2 / V14_scalar all hit, by
introducing search-improved policy targets. Either find the lift or
definitively rule out search for this codebase.

**Pass condition:** a model that beats V13.2-latest by ≥ +5pp / +17 Elo
in 25K-game H2H tournament (p < 0.0001).

## Three steps

| Step | What it tests | Time | Device |
|------|---------------|------|--------|
| Step 0 | Is V13.2's value head calibrated enough to seed MCTS? | ~15 min | VM (L4) |
| Step 1 | Does 2-ply expectimax distillation lift over vanilla V13.2 distillation? | ~2-3 days | VM (gen+train) + Mac (H2H) |
| Step 2 | Does full AlphaZero-style MCTS RL break the plateau? | ~2.5 weeks | VM (RL) + Mac (H2H) |

Each step is gated on the previous one passing.

## Files (will be created in this directory)

```
experiments/mcts_v1/
├── README.md                    # this file
├── calibration_audit.py         # Step 0
├── generate_search_data.py      # Step 1 — search-augmented data generator
├── train_search_distill.py      # Step 1 — SL trainer on search targets
├── mcts_engine.py               # Step 2 — fresh Python MCTS implementation
├── train_mcts_rl.py             # Step 2 — AlphaZero-style RL trainer
├── test_mcts_engine.py          # unit tests for MCTS (chance nodes, perspective flips, etc)
└── run_mcts_pipeline.sh         # combined runner for Step 2
```

Checkpoints live outside this dir (alongside other training runs):
- Step 1 distilled student: `td_ludo/checkpoints/mcts_v1_step1_distill/`
- Step 2 MCTS-RL model: `td_ludo/checkpoints/mcts_v1_step2_rl/`

## Step 0 — Calibration audit

**Subject:** V13.2 `model_latest.pt` from VM (pull post-pause).

**Procedure:**
1. Run 5,000 self-play games of V13.2 vs V13.2 with stochastic policy
   (`τ=1.0`). Vectorized via `VectorGameState(batch_size=200)`.
2. Record `(V_pred, current_player)` at every decision state. Discard
   first 10 turns of each game.
3. Walk back from each game's outcome, label every state with
   `eventual_outcome` from POV.
4. Bin V_pred into 10 deciles. Compute per-bin empirical WR.
5. Compute Brier score, Expected Calibration Error (ECE).
6. Output: calibration plot, per-bin metrics JSON, pass/fail verdict.

**Pass:** ECE ≤ 5pp, no bin > 10pp deviation, Brier ≤ 0.20.

**Marginal (5-10pp ECE):** proceed with caution.

**Fail (> 10pp ECE):** retrain value head only (1-2 days), re-audit.

Run:
```bash
python -m experiments.mcts_v1.calibration_audit \
  --model checkpoints/v132/model_latest.pt \
  --num-games 5000 \
  --output runs/mcts_v1_calibration.json
```

## Step 1 — Search-augmented distillation

**Subject:** V13.2-latest as data generator. Fresh V13.2-architecture
student.

### Generate search-improved training data

For each visited state, run **2-ply expectimax** (own move → opp dice →
opp move → V at leaf). Output:
- `search_action` = argmax over my actions of expected Q.
- `search_value` = max Q.
- `search_policy` = softmax(Q / 0.5) — soft target.

Run:
```bash
python -m experiments.mcts_v1.generate_search_data \
  --teacher checkpoints/v132/model_latest.pt \
  --target-states 1000000 \
  --batch-size 200 \
  --output runs/mcts_v1_search_buffer.npz
```

Cost: ~96 V-evals per state, batched. ~2-4 hrs on L4.

### Train student on search targets

Fresh V13.2-architecture student (10×128, 17ch input). Loss:
```
α_p · KL(student.π || search_policy)
+ α_v · MSE(student.V, search_value)
+ α_o · BCE(student.V, eventual_outcome)
```
with `α_p = 1.0, α_v = 0.5, α_o = 0.5`. Adam, lr 1e-3 → 1e-4 cosine,
5 epochs over the 1M-state buffer.

Run:
```bash
TD_LUDO_RUN_NAME=mcts_v1_step1_distill \
python -m experiments.mcts_v1.train_search_distill \
  --buffer runs/mcts_v1_search_buffer.npz \
  --epochs 5 \
  --batch-size 1024 \
  --lr 1e-3 \
  --port 8794
```

Cost: ~2-3 hrs on L4.

### Evaluate

H2H tournament: distilled student vs V13.2-latest, **25K games on Mac
CPU**, seat-balanced, greedy.

```bash
python -m experiments.tournament.run \
  --add-model V13_2:v132:checkpoints/v132/model_latest.pt \
  --add-model Step1_Distill:v132:checkpoints/mcts_v1_step1_distill/model_latest.pt \
  --games-per-pair 25000 \
  --device cpu \
  --output runs/tournament_step1_vs_v132.json
```

**Pass:** student wins ≥ 53%. Commit to Step 2.

**Marginal (51-53%):** review per-state-type breakdown.

**Fail (≤ 51%):** abandon MCTS, pivot to transformer.

## Step 2 — Full MCTS RL

**Only execute if Step 1 passes.**

### MCTS engine design

Fresh Python implementation, optimized for clarity over speed.

**Algorithm:** AlphaZero-style PUCT MCTS with explicit chance nodes
for Ludo dice.

**Hyperparameters (locked):**
- N (simulations per move): 100. Ramp to 200 if early signal is clear.
- c_puct: 1.5
- Dirichlet noise at root: α=0.3, ε=0.25
- Move temperature: τ=1.0 first 30 moves, τ=0.001 thereafter
- Chance nodes: enumerate all 6 dice children
- 6-roll bonus: same player keeps current_player, no perspective flip

**Tested invariants** (in `test_mcts_engine.py`):
- Single MCTS run from a known position produces reproducible visit
  counts.
- Perspective flips are correct across player boundaries (not flipped
  on 6-roll bonus).
- Chance node averaging converges to true 1/6 expectation as N → ∞.
- Terminal states return ±1 directly (no value-head call).

### RL training loop

- Load V13.2-latest as warm start.
- Vectorize K=64 parallel games via `VectorGameState`.
- For each game step:
  - Run MCTS with N=100 sims per game (parallel batched leaf evaluation).
  - Sample action from `π_search`.
  - Apply, advance.
- On terminal: walk back, label with `eventual_outcome`.
- Every 200 games: training step on accumulated buffer.
  - Loss: `α_p · KL(network.π || π_search) + α_v · BCE(network.V,
    eventual_outcome)` with `α_p = 1.0, α_v = 1.0`. L2 = 1e-4.
  - **No PPO. No bias penalties (in Phase 2).**
  - Adam, lr 5e-5.

### Two-phase schedule

**Phase 1 — Validation (~30K games, ~3-5 days on L4):**
- MCTS + bias penalties + AlphaZero loss.
- Diagnostic every 5K games: `KL(π_search || π_v132_initial)` averaged
  over states.
  - **KL ≥ 0.10:** search is finding new moves → PROCEED.
  - **KL < 0.05 after 30K games:** ABORT (search isn't differentiating).
- Eval every 5K games, 2.5K games per eval round.

**Phase 2 — Pure search-driven RL (~70K games, ~10-12 days on L4):**
- Drop bias penalties.
- Continue MCTS + AlphaZero loss only.
- Eval every 5K games.

### Evaluation (after each phase)

H2H tournament vs V13.2-latest, 25K games on Mac CPU.

**Final pass:** Phase 2 model beats V13.2-latest by ≥ +5pp.

## Risks and known unknowns

- **Value head calibration on advanced self-play states.** Step 0
  audits the bot-mix-similar distribution; MCTS will explore further.
  Mitigation: re-audit periodically during Step 2 Phase 1.
- **MCTS compute cost.** 100 sims/move × 64 parallel games × ~5 ms per
  V-eval ≈ 32 ms per training game step. Should give ~50-100 GPM on
  L4. If slower than ~30 GPM, drop to 50 sims/move.
- **Replay buffer size.** Need to size for ~K games of recent
  experience. Start with 50K-game ring buffer, tune.
- **Exploration collapse.** If Dirichlet noise isn't enough and PUCT
  converges to a tight policy quickly, we lose the AlphaZero
  improvement signal. Mitigation: `c_puct` tuning, increase Dirichlet
  α.

## Why fresh code, not the existing C++ MCTS

`src/mcts.cpp` exists but has 4 specific bugs (see
`post_v13_2_experiments.md` audit section) including a hardcoded
24-channel encoder that would feed wrong inputs to V13.2's value head.
Rather than debug 500 lines of C++ to find them all, we write a fresh
Python implementation that's simpler to audit. If Step 2 results
justify production-grade speed, we port to C++ in Step 2.5.
