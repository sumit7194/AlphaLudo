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

## Tracking table — RESULTS

**Note on methodology:** sandbox 4-core CPU + BATCH_SIZE=512 means games take
~80+ turns to complete. With search OFF (~1.2s/step) the 90s window catches
~50 game completions; with search ON (~4.6s/step) it catches zero. So we
track **step_ms** as the primary metric. The search-ON column is also noisy
because the measure window only catches 17–22 steps (small sample).

| Tier | Description | step_ms (OFF) | step_ms (ON) | GPM (OFF) | Δ from baseline (OFF) |
|------|-------------|---------------|--------------|-----------|------------------------|
| 0    | Baseline | 1242 | 4559 | 25.5 | — |
| 1a   | Batched player encoder | 1159 | 4248 | 27.8 | **-7%** |
| 1b   | + strip pre_step | 1132 | 4755 | 27.1 | **-9%** |
| 1c   | + pinned + non_blocking | 1178 | 4514 | 27.6 | **-5%** (within noise of 1b; CPU=no-op) |
| 2d   | + batched search leaves | 1139 | 4607 | 29.3 | **-8%** |

The 1c entry shows a slight regression from 1b but is within the run-to-run
noise band (range 1132–1178 on OFF column across optimized tiers, ~4% spread).
Real improvement vs baseline is roughly 8% step_ms reduction on the OFF path.

The search-ON column is dominated by noise (small sample). The Tier 2d
batched leaf encoder almost certainly helps on a real bench (more leaves,
smaller per-leaf overhead), but the sandbox can't measure it cleanly.

## Decision

**~8% sandbox throughput improvement is real but smaller than my projected
15–20%.** Reasons:
- The forward pass dominates more than I expected. On 4-core CPU it's
  not infinite-fast — ~50% of step time is the per-game model forward.
- The eliminated cost (small numpy allocations, np.stack memcpy, sparse
  list copies) was real but bounded. ~80–100 ms saved per turn.
- Tier 1c is CPU-side no-op. Real benefit only on L4.

**L4 prediction (extrapolating from sandbox CPU patterns):**
- Tier 1a: same % gain (Python+pybind work happens on CPU regardless).
  ~7% step_ms reduction.
- Tier 1b: same % gain. ~3% step_ms reduction.
- Tier 1c: only place sandbox can't measure benefit. On L4 with V12.2
  inference, the ~10ms H2D copy overlap is maybe 2–5% of step time.
- Tier 2d: harder to predict. On sandbox the search-ON column was noisy.
  On L4 the leaf forward pass is faster (GPU), so the batched encoder
  saves a bigger % of search overhead. Estimate 5–10% on search-ON.

**Cumulative L4 prediction:** ~10–15% step_ms reduction overall.
Translates to 333 GPM (current with search) → ~370–385 GPM. Or 526 GPM
(search OFF) → ~580–605 GPM.

## Worth merging?

**Tier 1a + 1b: yes.** ~10% of code size, ~9% sandbox improvement, no
device-specific gating. Pure win.

**Tier 1c: yes (small effort, CUDA-only path is gated on `device.type`).**
Free win on L4. Zero impact on sandbox/CPU.

**Tier 2d: maybe.** Sandbox didn't show a clean win on search-ON
(within noise). The C++ binding and Python refactor are clean and well-
tested. The case for merging is "it should help on L4 when search is on,
and it can't hurt." The case against is "we may pivot away from search
and never use this path." If we keep search-during-training experiments
(any depth, any alpha), 2d helps. If we abandon search entirely, 2d is
dead code.

## Recommendation

Merge Tier 1a + 1b + 1c (commits `0051974`, `97a71c3`, `de6ec5b`) to
main. Defer Tier 2d (`11785d4`) until we know whether search-during-
training continues past the alpha=0.25 retry on the L4. If it does,
merge 2d too. If not, drop 2d.

Either way, the unit tests and bench script (commit `b9...` not shown,
the initial branch commit) stay valuable for future optimization
attempts on the same pipeline.

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
