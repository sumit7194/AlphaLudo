# V12.2 RL Pipeline — Throughput Optimization Plan

**Branch:** `claude/perf-opt-exp24` (off `claude/new-session-83Q8f`)
**Goal:** Speed up the V12.2 RL training loop on the L4 without changing
algorithm semantics. All changes must preserve correctness verified by
existing unit tests + a per-tier delta check on training metrics.

## Why this is worth doing now

- A single Exp 24 retry burns ~10h at current 333 GPM on the L4.
- Each opt is independent of the experiment direction — speedup
  applies whether we keep search-during-training, pivot to a wider
  value head, or run any other future experiment.
- Sandbox CPU benchmarks let us validate Tier 1 cheaply before
  deploying to L4. CPU-bound bottlenecks (Python loops, pybind
  crossings) get the same speedup ratio whether on torch-CPU or
  torch-CUDA.

## Pipeline anatomy (per `play_step`, BATCH_SIZE=512)

```
┌─ Python loop ×512 (per game) ──────────────────────────────┐
│   env.get_game(i)         pybind call                      │
│   get_legal_moves(g)      pybind call                      │
│   encode_state_v11(g)     pybind call + numpy alloc        │
│   build legal_mask        Python                           │
└────────────────────────────────────────────────────────────┘
   ↓
np.stack([... 512 ...])    Python memcpy
torch.from_numpy(..).to()  H2D copy (blocking by default)
   ↓
model.forward_policy_only  GPU, batched
   ↓
sampled.cpu().numpy()      D2H sync (BLOCKING)
   ↓
Python loop ×512 (trajectory append)
   ↓
[Exp 24] _maybe_run_search Python enum × ~128 games
   ↓
env.step(actions)          C++ batch (serial loop inside)
   ↓
Python loop ×512 (rewards, deaths, results)
```

## Strong evidence pipeline is CPU-bound

- Sandbox 4-core CPU: 50–60 GPM at PROD batch=512.
- L4 GPU: 333 GPM with search (526 without).
- Ratio ~6–9× — far below typical CPU-vs-GPU gap (would be 30×+ if
  GPU were the bottleneck).
- The big work each turn is Python/pybind overhead, not the forward
  pass. Forward is one batched call; Python loops are 4× of them.

## Optimization tiers

### Tier 1 — high-EV, low-effort (target: +15–20% GPM)

**1a. Batched encoder in C++.**
- Add `td_ludo_cpp.encode_states_v11_batch(VectorGameState) → np.ndarray (B, 33, 15, 15)`.
- Single contiguous buffer, single pybind call, no per-game allocation,
  no `np.stack`.
- Expected speedup: 10–15% per turn (eliminates 512 small allocations
  + the stack copy).
- Correctness: unit test compares per-game encoding to batched encoding
  for 100 random states.
- Files: `td_ludo/src/bindings.cpp`, `td_ludo/td_ludo/game/players/v11.py`.

**1b. Strip pre_step snapshot to train-player turns only.**
- Currently `play_step` builds `pre_step_states` for ALL 512 games
  even though only train_player turns need rewards.
- Build only the indices that will need step_reward.
- Expected speedup: 3–5%.
- Files: `td_ludo/td_ludo/game/players/v11.py`.

**1c. Pinned memory + non_blocking H2D.**
- `torch.from_numpy(arr).pin_memory().to(device, non_blocking=True)`
  on the encoder→model transfer.
- Lets the H2D copy overlap with the next CPU-side work.
- Expected speedup: 1–3% on GPU; **near-zero on sandbox CPU**.
- Files: `td_ludo/td_ludo/game/players/v11.py`.

### Tier 2 — medium-EV, medium-effort (target: +10–15% GPM under search)

**2d. Vectorize search depth-1 enumeration (Python-side).**
- Currently each searched state does ~17 sequential `cpp.apply_move`
  calls in Python, each a pybind boundary crossing.
- Refactor: generate all (game_idx, first_action, dice, second_action)
  tuples upfront; do a single tight loop over them.
- This doesn't reduce the number of `apply_move` calls but reduces
  Python interpreter overhead per call.
- Expected speedup: ~50% search time → ~10–15% overall GPM under search.
- Files: `td_ludo/td_ludo/training/search_policy_target.py`.

### Tier 3 — speculative, deferred

**3e. C++ helper `expand_depth_1`.**
- Replaces Tier 2d's Python enumeration with a single pybind call that
  returns batched leaf states + ownership flags.
- 2–3× over Tier 2d. Worth doing only if we keep depth-1 search in the
  long run.

**3f. Async actor-learner split.**
- Decouple env.step + trajectory accumulation (CPU thread) from forward
  pass + PPO update (GPU). AlphaZero-style.
- 1.5–2× speedup ceiling; 1+ week of work; out of scope for this branch.

**3g. OpenMP in `env.step` C++ loop.**
- ~2% overall improvement; cheap. Skip unless we're squeezing the last
  drop.

## Validation protocol (every tier)

For each tier we validate **correctness** before benchmarking:

1. Run `td_ludo.training.test_search_policy_target` — must pass.
2. Run a 100-step TEST-mode run with the V12.2 weights — no crashes,
   pi_search fill rate matches expected fraction.
3. **For 1a only:** unit test comparing per-game vs batched encoding
   on 100 random states (should be bit-identical).

Then **benchmark** with `scripts/bench_throughput.py`:
- 30s warmup + 90s measurement window
- Report marginal GPM, step latency, search diagnostics
- Two configs: search OFF, search ON

## Benchmark methodology

Single source of truth: `td_ludo/scripts/bench_throughput.py`.

- TD_LUDO_MODE=PROD (BATCH_SIZE=512)
- V12.2 weights from `td_ludo/play/model_weights/v12_2/model_latest.pt`
- 30s warmup (lets JIT/cache stabilize)
- 90s measurement window (large enough to drown variance)
- CPU only (sandbox); 4 torch threads

Sandbox absolute numbers do NOT predict L4 absolute numbers. Sandbox
**deltas** between tiers DO predict L4 deltas for CPU-bound bottlenecks.
Tier 1c is the exception — pinned memory speedup only manifests on a
real GPU, so we'll note "expected on L4, not measured here."

## Tracking table (filled in as we go)

| Tier | Description | Sandbox GPM (search OFF) | GPM (search ON) | Δ vs baseline | Notes |
|------|-------------|--------------------------|-----------------|---------------|-------|
| 0    | Baseline (current code) | TBD | TBD | — | |
| 1a   | Batched encoder | TBD | TBD | TBD | |
| 1b   | + strip pre_step | TBD | TBD | TBD | |
| 1c   | + pinned + non_blocking | TBD | TBD | TBD | sandbox CPU likely shows no benefit |
| 2d   | + vectorize search enum | TBD | TBD | TBD | |

## Decision after all tiers

After Tier 2d benchmark:

- If cumulative gain is **< 10%**: not worth the merge cost. Discard
  the branch.
- If gain is **10–25%**: merge Tier 1 only (1a, 1b, 1c). Skip 2d if
  we're not sure search-during-training will continue past alpha=0.25.
- If gain is **> 25%**: merge everything.

The Tier 1 ceiling is bounded by Amdahl: removing all the Python loop
overhead can only get us to GPU-saturated, which on L4 with V12.2
inference is probably ~1500 GPM. Realistic Tier 1 target is +15–20%
landing us around 380–400 GPM with search ON, 600–630 without.
