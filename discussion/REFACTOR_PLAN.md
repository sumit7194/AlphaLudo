# AlphaLudo Refactor Plan (Phase 2)

> **Status**: Iteration 1 of 3 (planning, not yet executing)
> **Created**: 2026-04-10 by `.refactor_worker.py` cron
> **Predecessor**: `discussion/REFACTOR_PHASE1_REPORT.md` (cleanup phase)
> **Successor**: phase 9 of refactor worker (small commits to main, local only)

This plan describes how to restructure the AlphaLudo codebase from its
current ad-hoc shape into a clean, modular Python package — without
breaking the running GCP MCTS sweep, without disturbing production
training, and without losing any history.

---

## 1. Constraints & guardrails (read first)

These are the hard rules every refactor step must obey. They override
any "best practice" the planner might otherwise apply.

1. **GCP MCTS sweep** is running on `alphaludo-gpu-test` and consumes
   the following files at runtime. **Never touch any of these locally
   in a way that would change their public API or import path** until
   the sweep finishes:
   - `td_ludo/mcts_eval_sweep.py`
   - `td_ludo/src/model.py` (AlphaLudoV5)
   - `td_ludo/src/model_v6_2.py`
   - `td_ludo/src/heuristic_bot.py`
   - `td_ludo/src/mcts.cpp`, `mcts.h`
   - `td_ludo/src/bindings.cpp`
   - `td_ludo/src/game.cpp`, `game.h`
2. **No file in `td_ludo/checkpoints/ac_v6_1_strategic/`** may be moved,
   renamed, or deleted. This is the production checkpoint dir; keep
   path stability.
3. **No commits to `origin/main` from the refactor cron.** Local commits
   on `main` only. The user pushes manually when satisfied.
4. **Backward compatibility**: every refactor step must end with a
   working `import` graph for the surviving entrypoints
   (`train_v6_1.py`, `train_v6_2_fast.py`, `evaluate_v6_1.py`,
   `evaluate_v6_2.py`, `mcts_eval_sweep.py`, `init_v62_from_v61.py`,
   `generate_sl_data_v6_1.py`, `tune_heuristic.py`,
   `td_ludo/play/server.py`).
5. **Per-step verification**: every commit must pass
   `python -c "import ast; ast.parse(open(<edited_file>).read())"`
   for each Python file touched, and
   `python -c "import td_ludo.src.<module>"` smoke test where applicable.
   Compiled C++ extension is not rebuilt by the refactor — only Python
   reorganization.
6. **One small step per cron tick.** No mass rewrites. If a step would
   touch >5 files, split it.
7. **No deletes.** Phase 1 already handled deletions. Phase 9 only moves
   and edits.

---

## 2. Current structure (post-Phase-1 cleanup)

```
AlphaLudo/                              ← repo root
├── README.md
├── ARCHITECTURE.md
├── TESTS.md
├── Input_Tensor_Architecture_v5.md
├── async_training_plan.md
├── config.json                         ← active training config
├── setup.py                            ← root, protected (legacy ludo_cpp)
├── pyproject.toml
├── .gitignore
├── .refactor_worker.py                 ← scratch, gitignored
├── .refactor_state.json                ← scratch, gitignored
├── .refactor_cron_prompt.md            ← scratch, gitignored
├── .mcts_monitor_cron_prompt.md        ← scratch, gitignored
├── .RESTART_AFTER_POWER_LOSS.md        ← scratch, gitignored
│
├── apps/                               ← Android app (UI)
│   └── android/
│       ├── models_mobile/*.ptl         ← committed mobile binaries
│       └── (other UI files)
│
├── AlphaLudo Dashboard Design/         ← React/TS dashboard prototype
│   └── src/
│       ├── App.tsx
│       ├── components/                 ← 36 .tsx files
│       └── ui/                         ← shadcn-ish primitives
│
├── discussion/                         ← documentation (sacred, never delete)
│   ├── POST_V61_EXPERIMENT_PLAN.md
│   ├── REFACTOR_PHASE1_REPORT.md
│   ├── REFACTOR_PLAN.md                ← this file
│   ├── 01_channel_ablation/
│   ├── 02_dice_sensitivity/
│   ├── 03_linear_probes/
│   ├── 04_layer_knockout/
│   ├── 05_channel_activation/
│   └── 06_cka_similarity/
│
├── gcp_snapshots/                      ← gitignored historical archives
│   ├── v61_final_20260410_0137/
│   └── v62_final_20260409_1445/
│
├── v62_gcp_snapshot/                   ← gitignored, older snapshot
│
└── td_ludo/                            ← MAIN production tree
    ├── README, journals, plans
    ├── training_journal.md
    ├── PERFORMANCE_OVERHAUL.md
    ├── V7_ARCHITECTURE.md
    ├── setup.py                        ← td_ludo_cpp build (current)
    ├── pyproject.toml
    │
    ├── src/                            ← LIBRARY: 28 files
    │   ├── __init__.py
    │   ├── # core C++ extension
    │   ├── game.cpp, game.h
    │   ├── mcts.cpp, mcts.h
    │   ├── bindings.cpp
    │   │
    │   ├── # config
    │   ├── config.py
    │   │
    │   ├── # models
    │   ├── model.py                    (AlphaLudoV5: V6, V6.1)
    │   ├── model_v6_2.py               (V6.2 = V5 + transformer)
    │   │
    │   ├── # game players (per-version)
    │   ├── game_player.py              (base)
    │   ├── game_player_v6_1.py
    │   ├── game_player_v7.py           (legacy, kept by rule)
    │   ├── game_player_v8.py           (legacy, kept by rule)
    │   ├── game_player_v9.py           (legacy, kept by rule)
    │   │
    │   ├── # actor / learner / inference
    │   ├── fast_actor.py               (V9-era base)
    │   ├── fast_actor_v62.py           (V6.2 fork)
    │   ├── fast_learner.py             (V9-era base)
    │   ├── fast_learner_v62.py         (V6.2 fork)
    │   ├── inference_server.py         (V9-era)
    │   ├── inference_server_v6.py      (current V6.1 server)
    │   │
    │   ├── # trainer
    │   ├── trainer.py                  (single-process AC trainer)
    │   ├── training_utils.py
    │   ├── tensor_utils.py
    │   │
    │   ├── # game support
    │   ├── heuristic_bot.py
    │   ├── reward_shaping.py
    │   ├── elo_tracker.py
    │   └── game_db.py
    │
    ├── # entry-point scripts (top-level of td_ludo/)
    ├── train_v6_1.py                   ← single-process trainer
    ├── train_v6_1_fast.py              ← multi-process variant (broken import)
    ├── train_v6_2_fast.py              ← V6.2 fast trainer
    ├── train_sl_v6_1.py                ← SL pretraining for V6.1
    ├── generate_sl_data_v6_1.py
    ├── init_v62_from_v61.py
    ├── evaluate_v6_1.py
    ├── evaluate_v6_2.py
    ├── mcts_eval_sweep.py              ← inference-MCTS eval (running on GCP)
    ├── check_v62_parity.py
    ├── debug_gameplay.py
    ├── test_gameplay.py
    ├── tune_heuristic.py
    │
    ├── play/                           ← interactive play mode (gameplay)
    │   ├── server.py
    │   ├── model.py
    │   └── static/                     ← html/js front-end
    │
    ├── manual_test/                    ← manual gameplay tests
    │   └── runner.py
    │
    ├── checkpoints/                    ← gitignored, kept locally
    │   ├── ac_v6_1_strategic/          ← PROTECTED (current production)
    │   ├── ac_v6_2_transformer/        (V6.2 archive)
    │   ├── ac_v6_big/                  (V6 archive)
    │   ├── ac_v5/, ac_v5_11ch/         (legacy)
    │   ├── ac_v7_transformer/          (legacy)
    │   ├── ac_v8_cnn_transformer/      (legacy)
    │   ├── ac_v9/, ac_v9_slim_transformer/  (legacy)
    │   └── td_v2_11ch/, td_v3_small/   (legacy)
    │
    ├── gcp/                            ← deploy scripts
    │   ├── deploy.sh
    │   ├── setup_vm.sh
    │   └── start_v6.sh
    │
    └── td_env/                         ← venv (gitignored)
```

### Pain points (what makes the current layout hard to navigate)

1. **No package boundary**: `td_ludo/src/` is a flat module dump.
   It mixes config, models, trainers, game players, actors, learners,
   inference servers, utilities, game logic and bot code in one folder.
2. **Version sprawl**: Both `model.py` and `model_v6_2.py` exist as
   peers; same for `fast_actor.py` / `fast_actor_v62.py`,
   `fast_learner.py` / `fast_learner_v62.py`,
   `inference_server.py` / `inference_server_v6.py`,
   `game_player.py` / `_v6_1` / `_v7` / `_v8` / `_v9`. There is no
   single answer to "which one runs in production right now".
3. **Entry-point scripts at top level**: `td_ludo/train_*.py`,
   `evaluate_*.py`, `generate_sl_data_*.py`, `init_*.py` all live next
   to the project README, with no `scripts/` or `cmd/` directory.
   Discoverability is poor.
4. **`td_ludo` is itself a subdir of `AlphaLudo/`**, which is also a
   git repo, which also contains `apps/`, `discussion/`,
   `AlphaLudo Dashboard Design/`, etc. The Python package boundary is
   ambiguous: importing `from src.model import ...` works only when
   CWD is `td_ludo/`. There is no `td_ludo` Python package on
   `sys.path`; the existing imports rely on cwd-relative `sys.path`
   manipulation in each entry script.
5. **Two `setup.py` files** (root + td_ludo) — only `td_ludo/setup.py`
   builds the current `td_ludo_cpp` extension. Root `setup.py` is a
   legacy stub.
6. **Mixed concerns**: `mcts_eval_sweep.py` is both an inference
   harness AND a dashboard server AND a sweep runner AND a partial
   checkpoint manager. Should be split.
7. **No tests directory** at the project level. The previous
   `tests/` was full of mastery-era cruft and was deleted in phase 1.
   We need a fresh `tests/` for current code.

---

## 3. Target structure (proposed)

The target is a single importable Python package, with clearly named
subpackages, plus a `scripts/` directory for entry points and a
`tests/` directory for fresh tests.

```
AlphaLudo/                              ← repo root (unchanged top-level layout)
├── README.md
├── pyproject.toml                      ← single source of truth for build
├── setup.py                            ← thin shim to pyproject
├── .gitignore
├── apps/                               ← unchanged (Android UI)
├── AlphaLudo Dashboard Design/         ← unchanged (React dashboard)
├── discussion/                         ← unchanged (docs)
├── gcp_snapshots/                      ← unchanged (gitignored archives)
│
├── td_ludo/                            ← Python package root + everything else
│   ├── pyproject.toml                  ← td_ludo_cpp build
│   ├── setup.py                        ← thin shim
│   ├── README.md
│   ├── training_journal.md
│   │
│   ├── td_ludo/                        ← THE PYTHON PACKAGE (new dir, same name)
│   │   ├── __init__.py
│   │   │
│   │   ├── _native/                    ← C++ extension sources
│   │   │   ├── game.cpp, game.h
│   │   │   ├── mcts.cpp, mcts.h
│   │   │   └── bindings.cpp
│   │   │
│   │   ├── config.py                   ← single config module
│   │   │
│   │   ├── models/                     ← all model architectures
│   │   │   ├── __init__.py
│   │   │   ├── v5.py                   (was: model.py, AlphaLudoV5)
│   │   │   └── v6_2.py                 (was: model_v6_2.py)
│   │   │
│   │   ├── game/                       ← gameplay primitives (Python side)
│   │   │   ├── __init__.py
│   │   │   ├── player.py               (consolidates game_player*.py)
│   │   │   ├── heuristic_bot.py
│   │   │   ├── reward_shaping.py
│   │   │   └── tensor_utils.py
│   │   │
│   │   ├── training/                   ← training-time logic
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py              (was: trainer.py)
│   │   │   ├── fast_actor.py
│   │   │   ├── fast_actor_v62.py
│   │   │   ├── fast_learner.py
│   │   │   ├── fast_learner_v62.py
│   │   │   ├── inference_server.py
│   │   │   ├── inference_server_v6.py
│   │   │   └── utils.py                (was: training_utils.py)
│   │   │
│   │   ├── eval/                       ← evaluation harnesses
│   │   │   ├── __init__.py
│   │   │   ├── elo_tracker.py
│   │   │   ├── evaluate_v6_1.py        (importable, with main())
│   │   │   ├── evaluate_v6_2.py
│   │   │   ├── mcts_sweep.py           (was: mcts_eval_sweep.py, refactored)
│   │   │   └── parity.py               (was: check_v62_parity.py)
│   │   │
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   └── game_db.py
│   │   │
│   │   └── play/                       ← interactive play (web)
│   │       ├── __init__.py
│   │       ├── server.py
│   │       ├── model.py
│   │       └── static/                 (unchanged front-end)
│   │
│   ├── scripts/                        ← entry-point scripts
│   │   ├── train_v6_1.py
│   │   ├── train_v6_1_fast.py
│   │   ├── train_v6_2_fast.py
│   │   ├── train_sl_v6_1.py
│   │   ├── generate_sl_data_v6_1.py
│   │   ├── init_v62_from_v61.py
│   │   ├── evaluate_v6_1.py            (thin: from td_ludo.eval.evaluate_v6_1 import main; main())
│   │   ├── evaluate_v6_2.py
│   │   ├── mcts_eval_sweep.py
│   │   ├── tune_heuristic.py
│   │   ├── debug_gameplay.py
│   │   └── check_v62_parity.py
│   │
│   ├── tests/                          ← fresh tests (added in phase 9)
│   │   ├── __init__.py
│   │   ├── test_game_engine.py         (smoke: td_ludo_cpp loads, basic ops)
│   │   ├── test_models.py              (load each V5/V6.2 weight, forward pass)
│   │   ├── test_heuristic_bot.py
│   │   ├── test_eval_v6_1.py
│   │   └── test_mcts_smoke.py
│   │
│   ├── gcp/                            ← deploy scripts (unchanged)
│   │   ├── deploy.sh, setup_vm.sh, start_v6.sh
│   │
│   ├── manual_test/                    ← unchanged
│   ├── checkpoints/                    ← unchanged (gitignored)
│   └── td_env/                         ← unchanged (venv, gitignored)
```

### Key principles in the target layout

- **Single Python package** named `td_ludo` (the inner `td_ludo/td_ludo/`).
- **Clear subpackages**: `models`, `game`, `training`, `eval`, `data`,
  `play`, `_native`. Each has a single responsibility.
- **Thin scripts**: every entry point in `scripts/` is a 5-15 line
  shim that imports from the package and calls `main()`. Easy to
  test, easy to relocate, easy to deprecate.
- **Tests come back**: a new `tests/` directory with smoke tests for
  the things we actually use today. No mastery-era leftovers.
- **No file is duplicated**: each file lives in exactly one place.
- **Legacy versions kept**: `game_player_v7.py`, `game_player_v8.py`,
  `game_player_v9.py` stay (per phase-1 keep rule) but move under
  `td_ludo/game/legacy/` to signal they're not active.

---

## 4. Step-by-step refactor plan (phase 9 of worker)

Each step is bounded so it can run in a single cron tick, leave the
codebase in a working state, and be committed locally on main.

### Pre-step: defer until MCTS sweep finishes

The first batch of steps (Steps 1-3) reorganizes files used by the
running GCP sweep. **These steps must wait until the sweep completes**
(or until the user explicitly tells us the sweep is done). Phase 9
starts with safe steps that don't touch sweep-critical files.

### Stage A — Safe restructuring (can run while sweep runs)

**Step A1**: Create the new `td_ludo/td_ludo/` package skeleton.
- Make directory `td_ludo/td_ludo/`
- Add empty `__init__.py`
- Add empty subpackage dirs: `models/`, `game/`, `training/`, `eval/`,
  `data/`, `play/`, `_native/`, each with empty `__init__.py`
- Touches no existing files. Commit.

**Step A2**: Add `tests/` directory at `td_ludo/tests/` with one
smoke test that does `import td_ludo` and asserts the package exists.
This validates the new package layout without depending on any moved
files yet. Commit.

**Step A3**: Move `td_ludo/src/elo_tracker.py` →
`td_ludo/td_ludo/eval/elo_tracker.py`. Update 0 imports (this file
isn't yet used by the new package). Add a backward-compat shim at
`td_ludo/src/elo_tracker.py` that does
`from td_ludo.eval.elo_tracker import *`. Commit.

**Step A4**: Move `td_ludo/src/game_db.py` →
`td_ludo/td_ludo/data/game_db.py`, with shim. Commit.

**Step A5**: Move `td_ludo/src/reward_shaping.py` →
`td_ludo/td_ludo/game/reward_shaping.py`, with shim. Commit.

**Step A6**: Move `td_ludo/src/tensor_utils.py` →
`td_ludo/td_ludo/game/tensor_utils.py`, with shim. Commit.

**Step A7**: Move `td_ludo/src/training_utils.py` →
`td_ludo/td_ludo/training/utils.py`, with shim
`from td_ludo.training.utils import *`. Commit.

**Step A8**: Add a `td_ludo/setup.cfg` (or `pyproject.toml`)
configuration that registers `td_ludo` as a discoverable package.
Verify `pip install -e .` works in the venv. Commit.

### Stage B — Defer until sweep finishes

The following steps touch files used by the running MCTS sweep on GCP
(`mcts_eval_sweep.py`, `model.py`, `model_v6_2.py`, `heuristic_bot.py`,
fast_actor*, fast_learner*, game.cpp/h, mcts.cpp/h, bindings.cpp).
**These steps DO NOT run until the user signals "sweep complete"**.

**Step B1**: Move `td_ludo/src/heuristic_bot.py` →
`td_ludo/td_ludo/game/heuristic_bot.py`, with shim.

**Step B2**: Move `td_ludo/src/model.py` →
`td_ludo/td_ludo/models/v5.py`, with shim.

**Step B3**: Move `td_ludo/src/model_v6_2.py` →
`td_ludo/td_ludo/models/v6_2.py`, with shim.

**Step B4**: Move `td_ludo/src/game_player*.py` →
`td_ludo/td_ludo/game/players/`. Each file gets its own module.
Keep shim for the v6_1 one (active production import).

**Step B5**: Move `td_ludo/src/trainer.py` →
`td_ludo/td_ludo/training/trainer.py`, with shim.

**Step B6**: Move `td_ludo/src/fast_actor.py`, `fast_actor_v62.py`,
`fast_learner.py`, `fast_learner_v62.py`,
`inference_server.py`, `inference_server_v6.py` →
`td_ludo/td_ludo/training/`, with shims.

**Step B7**: Move `td_ludo/src/{game,mcts,bindings}.{cpp,h}` →
`td_ludo/td_ludo/_native/`, update `td_ludo/setup.py` Extension
`sources` paths accordingly, **rebuild and verify** the extension
loads. This is the only step that requires a rebuild.

**Step B8**: Move all entry-point scripts at `td_ludo/*.py`
(train_v6_1, train_v6_2_fast, evaluate_v6_*, etc.) to `td_ludo/scripts/`.
Each becomes a thin shim that calls a `main()` function in the
corresponding package module.

**Step B9**: Delete the empty `td_ludo/src/` directory once all shims
have been live for at least one full re-eval cycle and confirmed not
to break.

### Stage C — Tests + cleanup

**Step C1-C5**: Add fresh smoke tests under `td_ludo/tests/`:
- `test_game_engine.py`: import `td_ludo._native`, init game state, get legal moves
- `test_models.py`: load V5 + V6.2 from disk, forward pass with random tensor
- `test_heuristic_bot.py`: ExpertBot.select_move() doesn't crash
- `test_evaluate_v6_1.py`: 10-game eval against Expert returns valid result dict
- `test_mcts_sweep_smoke.py`: 5-game MCTS(5) run completes

**Step C6**: Update `README.md` and `td_ludo/training_journal.md` with
the new package layout. Add migration notes.

---

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Refactor breaks the running MCTS sweep on GCP | All Stage B steps defer until sweep completes. Stage A is read-only relative to sweep-critical files. |
| Backward compat shim doesn't capture all imports | Each step grep-checks for callers before moving. Smoke test runs after every commit. |
| `td_ludo_cpp` C++ extension fails to rebuild after path change (Step B7) | Skip B7 if any other step is in progress. Run `pip install --no-cache-dir .` and verify with `python -c "import td_ludo_cpp; td_ludo_cpp.create_initial_state_2p()"`. |
| Entry-point scripts have hardcoded `sys.path.insert(0, ...)` and break on relocation | Each script gets a one-line `import td_ludo` at the top after move; the shim removes the manual sys.path. |
| `td_ludo/checkpoints/ac_v6_1_strategic/` accidentally becomes inaccessible due to import path change | Checkpoint paths are absolute strings; not affected by Python package layout. Verified by smoke test. |
| Cron worker dies mid-step, leaving repo in half-committed state | Each step is one git commit; resume picks up next step. State file tracks step number under `notes`. |

---

## 6. Detailed file move map

Every file that exists in the post-Phase-1 tree, mapped to its target
location. Every move includes the exact shim contents that will live
at the old path until the deprecation window closes (Step B9).

### 6.1 td_ludo/src/ → td_ludo/td_ludo/

| Old path | New path | Shim required? | Step |
|---|---|:---:|:---:|
| `td_ludo/src/__init__.py` | `td_ludo/td_ludo/__init__.py` | yes | A1 |
| `td_ludo/src/config.py` | `td_ludo/td_ludo/config.py` | yes | A8 |
| `td_ludo/src/elo_tracker.py` | `td_ludo/td_ludo/eval/elo_tracker.py` | yes | A3 |
| `td_ludo/src/game_db.py` | `td_ludo/td_ludo/data/game_db.py` | yes | A4 |
| `td_ludo/src/reward_shaping.py` | `td_ludo/td_ludo/game/reward_shaping.py` | yes | A5 |
| `td_ludo/src/tensor_utils.py` | `td_ludo/td_ludo/game/tensor_utils.py` | yes | A6 |
| `td_ludo/src/training_utils.py` | `td_ludo/td_ludo/training/utils.py` | yes | A7 |
| `td_ludo/src/heuristic_bot.py` | `td_ludo/td_ludo/game/heuristic_bot.py` | yes | B1 |
| `td_ludo/src/model.py` | `td_ludo/td_ludo/models/v5.py` | yes | B2 |
| `td_ludo/src/model_v6_2.py` | `td_ludo/td_ludo/models/v6_2.py` | yes | B3 |
| `td_ludo/src/game_player.py` | `td_ludo/td_ludo/game/players/base.py` | yes | B4 |
| `td_ludo/src/game_player_v6_1.py` | `td_ludo/td_ludo/game/players/v6_1.py` | yes | B4 |
| `td_ludo/src/game_player_v7.py` | `td_ludo/td_ludo/game/players/legacy_v7.py` | no | B4 |
| `td_ludo/src/game_player_v8.py` | `td_ludo/td_ludo/game/players/legacy_v8.py` | no | B4 |
| `td_ludo/src/game_player_v9.py` | `td_ludo/td_ludo/game/players/legacy_v9.py` | no | B4 |
| `td_ludo/src/trainer.py` | `td_ludo/td_ludo/training/trainer.py` | yes | B5 |
| `td_ludo/src/fast_actor.py` | `td_ludo/td_ludo/training/fast_actor.py` | yes | B6 |
| `td_ludo/src/fast_actor_v62.py` | `td_ludo/td_ludo/training/fast_actor_v62.py` | yes | B6 |
| `td_ludo/src/fast_learner.py` | `td_ludo/td_ludo/training/fast_learner.py` | yes | B6 |
| `td_ludo/src/fast_learner_v62.py` | `td_ludo/td_ludo/training/fast_learner_v62.py` | yes | B6 |
| `td_ludo/src/inference_server.py` | `td_ludo/td_ludo/training/inference_server.py` | yes | B6 |
| `td_ludo/src/inference_server_v6.py` | `td_ludo/td_ludo/training/inference_server_v6.py` | yes | B6 |
| `td_ludo/src/game.cpp` | `td_ludo/td_ludo/_native/game.cpp` | n/a (rebuild) | B7 |
| `td_ludo/src/game.h` | `td_ludo/td_ludo/_native/game.h` | n/a | B7 |
| `td_ludo/src/mcts.cpp` | `td_ludo/td_ludo/_native/mcts.cpp` | n/a | B7 |
| `td_ludo/src/mcts.h` | `td_ludo/td_ludo/_native/mcts.h` | n/a | B7 |
| `td_ludo/src/bindings.cpp` | `td_ludo/td_ludo/_native/bindings.cpp` | n/a | B7 |

### 6.2 td_ludo/*.py (entry points) → td_ludo/scripts/

| Old path | New path | Step |
|---|---|:---:|
| `td_ludo/train_v6_1.py` | `td_ludo/scripts/train_v6_1.py` | B8 |
| `td_ludo/train_v6_1_fast.py` | `td_ludo/scripts/train_v6_1_fast.py` | B8 |
| `td_ludo/train_v6_2_fast.py` | `td_ludo/scripts/train_v6_2_fast.py` | B8 |
| `td_ludo/train_sl_v6_1.py` | `td_ludo/scripts/train_sl_v6_1.py` | B8 |
| `td_ludo/generate_sl_data_v6_1.py` | `td_ludo/scripts/generate_sl_data_v6_1.py` | B8 |
| `td_ludo/init_v62_from_v61.py` | `td_ludo/scripts/init_v62_from_v61.py` | B8 |
| `td_ludo/evaluate_v6_1.py` | `td_ludo/scripts/evaluate_v6_1.py` | B8 |
| `td_ludo/evaluate_v6_2.py` | `td_ludo/scripts/evaluate_v6_2.py` | B8 |
| `td_ludo/mcts_eval_sweep.py` | `td_ludo/scripts/mcts_eval_sweep.py` | B8 |
| `td_ludo/check_v62_parity.py` | `td_ludo/scripts/check_v62_parity.py` | B8 |
| `td_ludo/debug_gameplay.py` | `td_ludo/scripts/debug_gameplay.py` | B8 |
| `td_ludo/test_gameplay.py` | `td_ludo/tests/test_gameplay.py` | C1 |
| `td_ludo/tune_heuristic.py` | `td_ludo/scripts/tune_heuristic.py` | B8 |

### 6.3 td_ludo/play/ → td_ludo/td_ludo/play/

| Old path | New path | Shim? | Step |
|---|---|:---:|:---:|
| `td_ludo/play/server.py` | `td_ludo/td_ludo/play/server.py` | yes | B8 |
| `td_ludo/play/model.py` | `td_ludo/td_ludo/play/model.py` | yes | B8 |
| `td_ludo/play/static/*` | `td_ludo/td_ludo/play/static/*` | n/a | B8 |

### 6.4 Shim template (Python)

For every Python module move that needs a shim, the old path is
replaced with this content:

```python
# DEPRECATED: this module has moved.
# Old path: td_ludo/src/<old_name>.py
# New path: td_ludo/td_ludo/<subpackage>/<new_name>.py
#
# This shim re-exports the public symbols so existing imports still
# work. Remove after Step B9 (one full re-eval cycle confirms safety).

import warnings
warnings.warn(
    "td_ludo.src.<old_name> is deprecated; "
    "import from td_ludo.<subpackage>.<new_name> instead",
    DeprecationWarning,
    stacklevel=2,
)

from td_ludo.<subpackage>.<new_name> import *  # noqa: F401,F403
```

### 6.5 Entry-point script template

For every old top-level script in `td_ludo/*.py` that moves to
`td_ludo/scripts/`, the new file becomes a thin shim:

```python
#!/usr/bin/env python3
"""Entry point for <name>. Real implementation lives in
td_ludo.<subpackage>.<module>:main."""
from td_ludo.<subpackage>.<module> import main

if __name__ == "__main__":
    main()
```

The corresponding library module exposes a `main()` function that
takes no arguments and reads sys.argv internally. This requires
extracting the existing top-level `if __name__ == "__main__"` block
into a `def main():` block — a mechanical edit.

---

## 7. Cross-cutting concerns

These are issues that don't belong to a single step but affect every
step. Address them once, in the right order.

### 7.1 sys.path manipulation

Every entry-point script today starts with:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

This works only because the script lives at `td_ludo/` root and the
modules it needs are at `td_ludo/src/`. After the refactor, scripts
live at `td_ludo/scripts/` and the modules they need are at
`td_ludo/td_ludo/`. The sys.path hack must be replaced with one of:

- **Option A (preferred)**: install the package via `pip install -e
  td_ludo/` once, then scripts just `import td_ludo` directly. No
  sys.path needed.
- **Option B (fallback)**: every script header becomes
  `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))`.
  Functional but ugly.

Decision: **Option A**. Step A8 sets this up. Step B8 removes the
sys.path hacks during script relocation.

### 7.2 td_ludo_cpp Extension build path

`td_ludo/setup.py` currently has:

```python
ext_modules = [
    Extension(
        "td_ludo_cpp",
        ["src/bindings.cpp", "src/game.cpp", "src/mcts.cpp"],
        include_dirs=[pybind11.get_include(), "src"],
        ...
    ),
]
```

After Step B7 (move .cpp/.h to `_native/`), this becomes:

```python
ext_modules = [
    Extension(
        "td_ludo_cpp",
        [
            "td_ludo/_native/bindings.cpp",
            "td_ludo/_native/game.cpp",
            "td_ludo/_native/mcts.cpp",
        ],
        include_dirs=[pybind11.get_include(), "td_ludo/_native"],
        ...
    ),
]
```

After the change, **mandatory rebuild**:

```bash
cd td_ludo
rm -f td_ludo_cpp*.so   # see hard rule from Phase 1: stale .so kills you
rm -rf build *.egg-info
./td_env/bin/pip install --force-reinstall --no-deps --no-cache-dir .
./td_env/bin/python3 -c "import td_ludo_cpp; print(td_ludo_cpp.create_initial_state_2p())"
```

If this verification fails, Step B7 reverts the path change and stays
at Stage B7 stuck. The plan won't proceed until the rebuild works.

### 7.3 Logging strategy (post-refactor)

Currently every script uses bare `print(..., flush=True)`. This is
fine for stdout-only training jobs but makes it hard to filter, route,
or persist anywhere else.

Plan (deferred to a future iteration, NOT executed in phase 9):

- Add `td_ludo/logging.py` with `get_logger(name)` returning a
  pre-configured stdlib logger
- Replace `print` calls in library modules (not entry-point scripts)
  with `logger.info` / `logger.warning` calls
- Entry-point scripts keep `print` for the human-readable progress
  banner; library modules use the logger

This is a multi-step change and is **explicitly out of scope for
phase 9**. Add it as a follow-up issue in `discussion/`.

### 7.4 Configuration

`td_ludo/src/config.py` defines `MODE`, `RUN_NAME`, `CHECKPOINT_DIR`,
etc. via env vars + a `CONF` dict. This works but the env-var-driven
mode switching is fragile (hard to override per-script).

Plan: leave `config.py` alone in phase 9 (move only, no behavior
change). A future cleanup can convert it to dataclass-based config
with explicit `from_env()` constructors.

### 7.5 Type hints

Almost no module has type hints. Adding them is valuable but is
**out of scope for phase 9**. Add as a follow-up.

### 7.6 Tests for currently-running sweep

The MCTS sweep on GCP uses the **old** `td_ludo/src/...` import paths.
The shims must remain functional throughout phase 9 — never break
backward compatibility for any module that might be imported by the
running sweep. The shim's `from new.path import *` re-export is
sufficient: any old import that worked before the shim still works
after.

---

## 8. Stage A sanity check

Walking through Stage A one more time looking for ordering issues,
missing dependencies, or shim circularity.

### Step A1 — package skeleton

Creates `td_ludo/td_ludo/{models,game,training,eval,data,play,_native}/__init__.py`.
- ✅ No existing files touched.
- ✅ No imports broken.
- ⚠️ Concern: `td_ludo/td_ludo/` is the same name as the parent directory.
  Python will resolve `import td_ludo` to whichever is on `sys.path` first.
  Once we install editable in step A8, the inner `td_ludo` becomes the
  canonical package and the outer `td_ludo/` directory is just a host.
  Until A8 runs, attempting to `import td_ludo` from within the outer
  directory may pick up a non-package directory (no `__init__.py` at the
  outer level).
  - **Fix**: outer `td_ludo/` already has no `__init__.py` at its top
    level (it's a project directory not a package). Confirmed safe.

### Step A2 — tests/

Adds `td_ludo/tests/__init__.py` + a single `test_package_imports.py`
that does `import td_ludo` and asserts a `__file__` attribute. Run with
`python -m pytest td_ludo/tests/` from inside `td_ludo/`.
- ✅ Doesn't depend on any moved file. Safe to run before A3.
- ⚠️ Pytest needs to be in the venv. Verify with
  `td_ludo/td_env/bin/pip list | grep pytest`. If not present, add
  `pytest` to `requirements.txt` and install.
  - **Fix**: defer pytest install — for the first test we can use a
    plain `python -m unittest` or even just `python -c "import td_ludo;
    assert td_ludo"` from inside Stage A2 commit.

### Step A3-A7 — leaf module moves

Each step moves a file that has at most a handful of callers and no
inbound dependencies from other moved files in this stage (e.g.
elo_tracker.py is imported by trainer.py and game_db.py is imported by
trainer.py — but trainer.py isn't being moved in Stage A).
- ✅ No circular imports introduced (each leaf is moved on its own).
- ✅ Shims preserve old import paths.
- ⚠️ The shim does `from td_ludo.eval.elo_tracker import *`. This works
  IF `td_ludo` is importable as a package. Until A8, that requires the
  outer `td_ludo/` to have a venv with `td_ludo` installed editable.
  - **Fix**: reorder. **A8 must run BEFORE A3-A7.** Updated step
    sequence: A1 → A2 → A8 → A3 → A4 → A5 → A6 → A7.

### Step A8 — editable install

Creates a minimal `td_ludo/pyproject.toml` (or extends the existing
one) so `pip install -e td_ludo/` works.

```toml
[build-system]
requires = ["setuptools>=61", "pybind11>=2.10"]
build-backend = "setuptools.build_meta"

[project]
name = "td_ludo"
version = "0.0.2"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "numpy",
    "psutil",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["td_ludo*"]
exclude = ["scripts*", "tests*", "checkpoints*", "td_env*", "play.static*"]
```

After this file exists, run:
```bash
cd td_ludo
./td_env/bin/pip install -e . --no-deps
./td_env/bin/python3 -c "import td_ludo; print(td_ludo.__file__)"
```

If the import succeeds, A8 is done. The shims in A3-A7 will then work.

- ⚠️ The existing `td_ludo/setup.py` builds the C++ extension
  `td_ludo_cpp`. We must NOT regress that build. The new pyproject.toml
  must coexist with setup.py — setuptools picks up both. Test:
  ```bash
  cd td_ludo && ./td_env/bin/python3 -c "import td_ludo_cpp; print('cpp ok')"
  ```
  must still pass after A8.

### Updated Stage A order

1. **A1**: Create package skeleton (empty __init__.py files).
2. **A2**: Add tests/ + smoke test (no pytest dep).
3. **A8**: Add pyproject.toml + editable install + verify both `td_ludo`
   and `td_ludo_cpp` import.
4. **A3**: Move elo_tracker.py.
5. **A4**: Move game_db.py.
6. **A5**: Move reward_shaping.py.
7. **A6**: Move tensor_utils.py.
8. **A7**: Move training_utils.py.

---

## 9. Stage C — Concrete test assertions

Each test file under `td_ludo/tests/` should be small, fast, and not
require GPU. Use plain `unittest` to avoid the pytest dependency.

### test_package_imports.py (Step A2)

```python
import unittest

class TestPackageImports(unittest.TestCase):
    def test_root_package(self):
        import td_ludo
        self.assertTrue(hasattr(td_ludo, '__file__'))

    def test_subpackages_exist(self):
        import td_ludo.models
        import td_ludo.game
        import td_ludo.training
        import td_ludo.eval
        import td_ludo.data
        import td_ludo.play

if __name__ == '__main__':
    unittest.main()
```

### test_game_engine.py (Step C1)

```python
import unittest

class TestGameEngine(unittest.TestCase):
    def test_cpp_extension_imports(self):
        import td_ludo_cpp
        self.assertTrue(hasattr(td_ludo_cpp, 'create_initial_state_2p'))

    def test_initial_state(self):
        import td_ludo_cpp
        s = td_ludo_cpp.create_initial_state_2p()
        self.assertFalse(s.is_terminal)
        self.assertIn(s.current_player, (0, 1, 2, 3))

    def test_legal_moves_from_dice_6(self):
        import td_ludo_cpp
        s = td_ludo_cpp.create_initial_state_2p()
        s.current_dice_roll = 6
        moves = td_ludo_cpp.get_legal_moves(s)
        self.assertIsInstance(moves, list)
        # On dice 6 from initial state, at least one base token can spawn
        self.assertGreater(len(moves), 0)

if __name__ == '__main__':
    unittest.main()
```

### test_models.py (Step C2)

```python
import unittest
import torch

class TestModels(unittest.TestCase):
    def test_v5_model_constructs(self):
        from td_ludo.models.v5 import AlphaLudoV5
        m = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad) > 1_000_000, True)

    def test_v5_forward_random(self):
        from td_ludo.models.v5 import AlphaLudoV5
        m = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
        m.eval()
        x = torch.randn(2, 24, 15, 15)
        legal = torch.ones(2, 4)
        with torch.no_grad():
            policy, value = m(x, legal)
        self.assertEqual(policy.shape, (2, 4))
        self.assertEqual(value.shape, (2, 1))

    def test_v6_2_model_constructs(self):
        from td_ludo.models.v6_2 import AlphaLudoV62
        m = AlphaLudoV62(context_length=4, num_res_blocks=10, in_channels=24)
        self.assertTrue(hasattr(m, 'trans_out_proj'))

if __name__ == '__main__':
    unittest.main()
```

### test_heuristic_bot.py (Step C3)

```python
import unittest

class TestHeuristicBot(unittest.TestCase):
    def test_expert_bot_select(self):
        from td_ludo.game.heuristic_bot import ExpertBot
        import td_ludo_cpp
        bot = ExpertBot(player_id=0)
        s = td_ludo_cpp.create_initial_state_2p()
        s.current_dice_roll = 6
        legal = td_ludo_cpp.get_legal_moves(s)
        action = bot.select_move(s, legal)
        self.assertIn(action, legal)

if __name__ == '__main__':
    unittest.main()
```

### test_evaluate_v6_1.py (Step C4)

```python
import unittest, os, torch

CKPT = "checkpoints/ac_v6_1_strategic/model_latest.pt"

class TestEvaluateV6_1(unittest.TestCase):
    @unittest.skipIf(not os.path.exists(CKPT), "ckpt not present locally")
    def test_eval_10_games_smoke(self):
        from td_ludo.eval.evaluate_v6_1 import evaluate_model
        from td_ludo.models.v5 import AlphaLudoV5
        device = "cpu"
        m = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=24)
        sd = torch.load(CKPT, map_location=device, weights_only=False)
        m.load_state_dict(sd.get('model_state_dict', sd))
        m.to(device).eval()
        result = evaluate_model(m, device, num_games=10, verbose=False, bot_types=['Expert'])
        self.assertGreaterEqual(result['win_rate'], 0.0)
        self.assertLessEqual(result['win_rate'], 1.0)

if __name__ == '__main__':
    unittest.main()
```

### test_mcts_smoke.py (Step C5)

```python
import unittest, os, torch

CKPT = "checkpoints/ac_v6_1_strategic/model_latest.pt"

class TestMCTSSmoke(unittest.TestCase):
    @unittest.skipIf(not os.path.exists(CKPT), "ckpt not present locally")
    def test_mcts_5_sims_smoke(self):
        # Just verify mcts_eval_sweep entry point can run a tiny matchup
        from td_ludo.eval.mcts_sweep import run_matchup
        result, paused = run_matchup(
            label="smoke",
            model_path=CKPT,
            num_sims=5,
            opponent_spec={"type": "bot", "bot": "Expert"},
            num_games=3,
            device="cpu",
            seed=42,
            log_path="/tmp/mcts_smoke.log",
        )
        self.assertEqual(result["num_games"], 3)
        self.assertFalse(paused)

if __name__ == '__main__':
    unittest.main()
```

### Test execution

After Stage C is complete, the cron tick that finishes C5 also runs:

```bash
cd td_ludo
./td_env/bin/python3 -m unittest discover -s tests -v
```

If any test fails, that step's commit is reverted (`git reset --hard
HEAD~1`) and the cron stops at phase 9 with an error note for the user.

---

## 10. Status & next iteration

- **Iteration 1**: ✅ done (initial plan)
- **Iteration 2**: ✅ done (file move map + cross-cutting concerns)
- **Iteration 3**: ✅ done (Stage A sanity check + concrete test assertions + pyproject.toml + reordered Stage A so A8 runs before A3-A7)

**Next cron tick**: advance phase 8 → 9 and execute Stage A1 (create
package skeleton). After that, one Stage A step per tick, sequentially,
until we hit the Stage B boundary (which waits on the GCP MCTS sweep).
