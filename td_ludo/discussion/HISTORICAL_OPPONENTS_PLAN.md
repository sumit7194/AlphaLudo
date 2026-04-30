# Historical Models as RL Opponents — Design

**Branch:** `claude/historical-opponents` (off `main`)
**Status:** Phases 1 + 2 done. Phase 3 (ELO + dashboard) and Phase 4
(L4 deploy) still ahead. Roster reduced from 6 planned to 4 active —
see "Roster outcome" below.

## Motivation

The current bot mix (Heuristic, Aggressive, Defensive, Expert) is
saturated against V12.2:

| Opponent | V12.2 recent WR (G≈1M) |
|---|---|
| Expert | 77.9% |
| Heuristic | 75.0% |
| Defensive | 72.7% |
| Aggressive | 70.6% |

Wins against these add little gradient — the model already crushes them.
Meanwhile we have a stack of historical model checkpoints that span a
wide range of strengths and *defect profiles* (V6.3 with token-stickiness,
V10 with no transformer, V11 with attention, V12 ancestral). Replacing
the bot mix with these is a higher-quality curriculum.

## Goal

Replace the bot mix in the v122 game composition with a fixed-roster of
historical V-model opponents. Keep self-play and ghost-of-current-run
mechanisms unchanged. Optionally retain `Random` at 1–2% as a
catastrophic-collapse floor.

Key differences vs the existing ghost mechanism:
- Ghosts are checkpoints from the *current* run, all V12.2 architecture,
  loaded with `model_factory()` (V12.2 architecture).
- Historicals are checkpoints from *prior* generations, with
  *different architectures and encoders*. Need per-opponent dispatch.

## Architectures and encoders we need to support

| Tag | Architecture | Encoder | In channels | Comments |
|---|---|---|---|---|
| `Hist_V6_3` | `AlphaLudoV63` | `encode_state_v6_3` | 27 | takes `consecutive_sixes` arg |
| `Hist_V10` | `AlphaLudoV10` | `encode_state_v10` | 28 | 3-head output |
| `Hist_V11` | `AlphaLudoV11` | `encode_state_v11` | 33 | with token attention |
| `Hist_V12` | `AlphaLudoV12Legacy` (or `AlphaLudoV12`) | `encode_state_v11` | 33 | drop-in V11 encoder |

V6.2 is a stepping stone; skip unless we want yet more diversity.
V3/V4/V5 (17ch) are too weak to be useful — like beating Heuristic.

## Memory cost

| Tag | Params | GPU memory |
|---|---|---|
| Hist_V6_3 | ~3M | ~12 MB |
| Hist_V10 | ~640K | ~2.5 MB |
| Hist_V11 | ~951K | ~4 MB |
| Hist_V12 | ~951K | ~4 MB |
| **Total** | **~5.5M** | **~22 MB** |

Plus 4 V12.2 ghosts already in cache: ~22 MB. Plus main model: ~5 MB.
Grand total ~50 MB on L4's 22GB. Trivial.

## Inference cost

The existing `play_step` decision-grouping batches by
`(controller, controller_id)`. Adding historical opponents adds more
groups, but each group runs ONE batched forward pass per turn. Per-turn
forward count grows by O(distinct opponents in this batch), which is
bounded by the number of historical tags (≤ 4). Sub-linear in
`BATCH_SIZE`.

## Game composition (proposed v123 mix)

```python
GAME_COMPOSITION_V123 = {
    "SelfPlay":  0.65,    # V12.2 vs V12.2 (with active-ghost subbed in
                          # at SELFPLAY_GHOST_FRACTION rate, unchanged)
    "Hist_V11":  0.10,    # recent strong opponent
    "Hist_V12":  0.10,    # parallel-architecture opponent
    "Hist_V10":  0.08,    # different-architecture opponent
    "Hist_V6_3": 0.05,    # old-style opponent (different defects)
    "Random":    0.02,    # sanity floor — guards against collapse
}
```

Bot mix gone. Self-play still dominant (the strongest gradient signal).

## ELO tracking

Each `Hist_V*` registers as a named entity in the ELO tracker. Lets us
track:
- Main model's ELO trajectory against fixed historical reference points.
- "Did we cross 1700?" "Did we beat V11 majority?" — concrete external
  benchmarks, not just self-relative.

## Implementation effort

### Phase 1 — registry (this branch, today)

A self-contained `opponent_registry.py` that maps tag →
`(model_factory, encoder_fn, in_channels, ckpt_path, kwargs)`. Loads
on demand, caches per-process. Exposes a `select_action(tag, game)`
helper that handles encoder dispatch + forward pass + legal-mask
filtering. Player loop calls into this.

### Phase 2 — player wiring (next branch)

Modify `td_ludo/td_ludo/game/players/v11.py`:
- `_random_composition` samples from the new mix, assigns
  `controller=Hist_*` to each non-self-play slot.
- The decision-groups loop in `play_step` adds branches for
  `controller.startswith("Hist_")` — calls into the registry's
  batched `select_action`.

### Phase 3 — config + ELO + dashboard

- `src/config.py`: add `GAME_COMPOSITION_V123` mode.
- `train_v12.py`: add `--game-composition v123`.
- ELO tracker: register historical tags as opponents.
- Dashboard: show win rates per historical opponent (existing per-bot
  WR section just needs new tags).

### Phase 4 — smoke + deploy

- TEST mode 100-game shakedown — verify all historical models load,
  encode correctly, produce legal moves.
- 10K-game L4 shakedown — verify GPM doesn't drop catastrophically.
- Full deploy alongside next major experiment.

## Risks

1. **Architecture loading mismatch.** V11/V12 share the V11 encoder
   (33ch) but V12 has the token attention layer. Loading a V12
   checkpoint into `AlphaLudoV11` would fail. Registry must use the
   correct architecture class per tag.

2. **Training-time leakage from weak opponents.** V6.3's mistakes
   (token-stickiness, no bonus-turn awareness) could distort the
   training signal. Mitigation: keep V6.3 weight low (5%); main signal
   from self-play + V10/V11/V12.

3. **Curriculum decay.** As V12.2 strengthens, all four historical
   opponents become "easy" too. Self-play vs ghosts of the current run
   remains the long-term curriculum (V12.2 vs better-V12.2). The
   historical mix is for *strategic diversity*, not endless ELO climb.

4. **Stale ELO tracking.** If we keep training V12.2, but historical
   tags have fixed weights, their ELO drifts as they only ever play
   against the strengthening main. Acceptable: their "strength relative
   to current main" is the metric we care about.

## Phase 1 deliverable (DONE — `40dd1f2`)

`td_ludo/td_ludo/game/players/opponent_registry.py` — registry +
on-demand loading + batched `select_action`. Plus a unit test verifying
all historical tags load cleanly and produce legal moves on a few
sample states.

## Phase 2 deliverable (DONE — `de55f4b`)

Player-loop wiring (`td_ludo/td_ludo/game/players/v11.py`):
- `VectorACGamePlayer.__init__` takes `historical_opponents_enabled`,
  instantiates the registry on first use.
- `play_step` adds a `hist_pending` dict alongside `decision_groups`.
  Routing: `Hist_*` controllers go to `hist_pending`, get resolved in
  one `registry.select_actions_batched()` call after the model loop.

CLI flag `--game-composition v123` in `train_v12.py` activates the new
mix and enables historical opponents.

End-to-end TEST-mode 30-game smoke: 126 PPO updates, no crashes,
44 GPM (vs 22 GPM bot-mix smoke at same scale).

## Roster outcome (real checkpoints inspected — `6e505b3`)

Cherry-picked the user's backup batch (commit `83c5298` from
`claude/perf-opt-exp24`) and inspected each `.pt` file's state-dict
shape. Several discrepancies vs the Phase 1 plan:

| Tag | Status | Notes |
|---|---|---|
| `Hist_V6_big` | ✓ Active | NEW — V5-era 17ch encoder, 10×128, no attn |
| `Hist_V6_1` | ✓ Active | NEW — 24ch encoder, AlphaLudoV5 10×128 |
| `Hist_V6_3` | ✓ Active | As planned |
| `Hist_V10` | ✓ Active | As planned |
| `Hist_V11` | ✗ Deferred | Checkpoint uses `attn_dim=64` (V11.1 era), not the default 96. Loading is solvable (single line in spec) but per user direction: deferred. |
| `Hist_V12` | ✗ Skipped | Not in the backup batch. V12.2 lives in the main slot via SelfPlay/Ghost mechanism — no need as opponent. |
| `Hist_V6_2` | ✗ Deferred | Temporal transformer over K=16 past states. Needs per-game history tracking outside the registry's stateless interface. Separate effort. |

Path resolution bug fixed: `_resolve_ckpt` was off by one dirname,
landed inside the inner `td_ludo` package instead of the run-dir root.
All 4 active paths now resolve correctly.

## Final v123 mix (no Random — `d401551`)

```
SelfPlay 67%  (main + ghost-of-current-run, unchanged mechanism)
Hist_V10 18%  (strongest available historical)
Hist_V6_3 8%
Hist_V6_1 4%
Hist_V6_big 3%
```

Random is intentionally absent. Trained models all crush Random ~95%,
so games against it carry zero gradient signal. The historical-opponent
WRs (especially Hist_V10) act as the real collapse-detector if the
policy ever degrades — Random as a "sanity floor" is redundant.

## Tournament infrastructure (paired addition — `62a0c94`)

Added `td_ludo/experiments/tournament/`. Round-robin tournament between
arbitrary mix of historical V-models (via `--hist`), bots (via
`--bots`), and custom checkpoints (via `--add-model NAME:ARCH:PATH`).
Seven architecture presets supported; first 15K-game sandbox run
launched 2026-04-30 21:40.

Used to:
1. Anchor relative strength of historicals (informs v123 weighting).
2. Benchmark new architectures (e.g., MinimalCNN14 distill student)
   against a calibrated reference set.

## What's next

- **Phase 3** — ELO tracker + dashboard wiring. Each `Hist_*` registers
  as a named entity in the ELO tracker; main-model ELO trajectory
  becomes meaningful relative to fixed historical reference points.
  Dashboard adds per-historical-opponent WR display.
- **Phase 4** — L4 deploy. Graceful-kill current run; pull this branch
  (or merge first); rebuild C++ extension; restart with
  `--game-composition v123`.
- **Optional: V11 re-enable.** Add `attn_dim=64` to the registry spec.
  Single-line fix; deferred per user.
- **Optional: V6.2 support.** Temporal-context wrapper that maintains
  per-game history. Larger lift; not currently planned.
