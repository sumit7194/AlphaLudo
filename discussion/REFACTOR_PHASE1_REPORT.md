# Refactor Phase 1 Report

**Date**: 2026-04-10
**Worker**: `.refactor_worker.py` driven by in-session cron `e3f03c31` (refactor) + `318bf2cb` (MCTS monitor)
**Plan reference**: see top-level user request and worker source

## Headline numbers

| Metric | Value |
|---|---:|
| Files inventoried | **914** |
| Deletions executed | **384** |
| Gitignore entries added | **23 new** (272 already covered by existing patterns) |
| Files marked `keep` | **235** |
| Files marked `keep_gitignore` total | **295** |
| Disk freed by deletions | **~1.38 GB** |
| Disk now gitignored (preserved locally) | **~1.93 GB** |
| Apply errors | **0 failed, 0 skipped (safety)** |

## What got deleted (categories)

- **Mastery-era code** (entire abandoned generation pre-V6.1): root `src/` directory (~50 .py files), `train_mastery.py`, `train_pbt.py`, `train_specialist.py`, `train_async.py`, `train_league.py`, `cleanup_ghosts.py`, `debug_buffer.py`, `debug_db_stats.py`, `demo_visualizer.py`, `inspect_buffer.py`, `inspect_model.py`, `remote_proxy.py`, `run_remote.sh`, `run_training.sh`, `Start_AlphaLudo_Training.command`, all of `AlphaLudo_Export/` (~28 files), `apps/android/convert_to_tflite*.py`, `apps/android/convert_to_torchscript.py` (all using AlphaLudoV3)
- **V7/V8/V9 abandoned versions**: `td_ludo/src/model_v7.py`, `model_v8.py`, `model_v9.py`, `trainer_v7/8/9.py`, `state_encoder_1d.py`; `td_ludo/evaluate_v7.py`, `evaluate_v8.py`, `evaluate_v9.py`, `train_v7.py`, `train_v8.py`, `train_v9.py`, `train_v9_fast.py`, `train_sl_v7.py`, `train_sl_v9.py`, `generate_sl_data_v7.py`, `generate_sl_data_v9.py`
- **Exp 9 MCTS training (abandoned per journal)**: `td_ludo/train_mcts.py`, `watch_mcts.py`, `mcts_dashboard.py`, `tools/reset_training_state.py`, `td_ludo/checkpoints/mcts_v1/model_iter_*.pt`
- **Old test directories**: `tests/test_augmentation.py`, `tests/test_cpp_mcts.py`, `tests/test_mcts.py`, `tests/test_model.py`, `tests/test_specialist.py`, `tests/test_training.py`, `tests/test_vector.py`, `tests/test_engine.py`, `tests/test_dynamic_config.py`, `tests/test_hybrid_tensor.py`, `tests/repro_*.py`, `tests/debug_*.py`, `tests/verify_pbt_cpuct.py`, plus root `test_coords.py`, `test_persistence.py`, `test_snapshot.py`, td_ludo `test_17ch.py`, `test_coords.py`, `test_data_flow_core.py`, `test_encoder.py`, `test_expert.py`, `test_rotation.py`
- **Stale runtime debug dumps**: `dump.rdb`, `nohup.out`, `td_ludo/nohup.out`, `td_ludo/debug_output.txt`, all `game_debug_*.txt`, `baseline_summary.txt`, `data/pid.txt`, `data/actor_stats.json`, `data/metrics.json`, `data/tuner_history.json`, `snapshot.json`, `kickstart_summary.txt`, `Gemini_Generated_Image_*.png`
- **Stale training logs at root** (16 files): `training_logs_db*.txt`, `training_logs_density_upgrade.txt`, `training_logs_final.txt`, `training_logs_ghost_cleanup.txt`, `training_logs_input_enhanced.txt`, `training_logs_mastery_v*`, `training_logs_mcts_fix*`, `training_logs_rotation_fix.txt`
- **Corrupted SQLite files** (excluding protected ac_v6_1_strategic): `td_ludo/checkpoints/ac_v6_big/game_history.db.corrupted_*`, `ac_v9_slim_transformer/game_history.db.corrupted_*`, `td_v3_small/game_history.db.corrupted_*`, `.recovered`, `.corrupted_manual`
- **Abandoned experiment ghosts and weights**: full `experiments/kickstart/` (debug_glob, generate_data, run_kickstart.sh, stream_trainer, train_kickstart, verify_buffer, model_kickstart.pt, ghosts/, models_to_test/), `experiments/token0_bias/`, `experiments/benchmark_ane.py`, `experiments/run_all_experiments.py`, `models_to_test/model_cycle_100.pt`, `td_ludo/checkpoints/ac_v6_big/ghosts/*.pt` (all), `ac_v7_transformer/ghosts/*.pt`, `ac_v8_cnn_transformer/ghosts/*.pt`, `ac_v9/ghosts/*.pt`, `ac_v9_slim_transformer/ghosts/*.pt`, `td_v2_11ch/ghosts/*.pt`, `td_v3_small/ghosts/*.pt`, `tests/checkpoints_pbt_test/agent_*.pt`, `td_ludo/pretrained/model_kickstart*.pt`, `td_ludo/checkpoints_test/td_test/*.pt`, V7/V9 SL checkpoints
- **Misc**: `td_ludo/migrate_buffer.py`, `migrate_weights.py`, `recover_db.py`, `td_ludo/check_buffer.py`, `compare_v6_v9.py`, `diagnose_rl.py`, `eval_sl.py`, `evaluate.py`, `td_ludo/benchmark_models.py`, `benchmark_throughput.py`, `td_ludo/tournament.py`, `td_ludo/start_training.sh`, `td_ludo/run_v9.sh`, `td_ludo/gcp/start_v9.sh`, `td_ludo/train.py`, `td_ludo/train_sl.py`, `td_ludo/inspect_buffer.py`, `package_for_colab.sh` (Colab), `fork_v2_pure.sh`, `fetch_stats*.py`

## What was kept (235 files)

- **Production training pipeline**: `td_ludo/train_v6_1.py`, `train_v6_1_fast.py`, `train_v6_2_fast.py`, `train_sl_v6_1.py`, `init_v62_from_v61.py`, `evaluate_v6_1.py`, `evaluate_v6_2.py`, `mcts_eval_sweep.py`, `check_v62_parity.py`
- **td_ludo/src/**: `model.py`, `model_v6_2.py`, `trainer.py`, `training_utils.py`, `tensor_utils.py`, `reward_shaping.py`, `config.py`, `elo_tracker.py`, `game_db.py`, `heuristic_bot.py`, `fast_actor.py`, `fast_actor_v62.py`, `fast_learner.py`, `fast_learner_v62.py`, `game_player.py`, `game_player_v6_1.py`, `game_player_v7.py`, `game_player_v8.py`, `game_player_v9.py`, `inference_server.py`, `inference_server_v6.py`
- **C++ extension**: `src/game.cpp`, `game.h`, `mcts.cpp`, `mcts.h`, `bindings.cpp`, `td_ludo/setup.py`, `td_ludo/pyproject.toml`
- **Gameplay code**: `td_ludo/play/model.py`, `play/server.py`, `play/static/game.js`, `td_ludo/test_gameplay.py`, `td_ludo/tune_heuristic.py`, `td_ludo/debug_gameplay.py`, `td_ludo/manual_test/runner.py`
- **GCP deploy scripts**: `td_ludo/gcp/deploy.sh`, `setup_vm.sh`, `start_v6.sh`
- **Mech interp / discussion** (entire `discussion/` folder, 75+ files): all `run_*.py` scripts, all `*_metrics.json`, all PNG result charts, all RESULTS_*.md docs
- **UI / Dashboard**: `AlphaLudo Dashboard Design/` (50+ tsx/ts/css/html files), `apps/android/` mobile model files (kept) and android Python helpers, html dashboards in td_ludo
- **Critical docs**: `README.md`, `td_ludo/training_journal.md`, `discussion/POST_V61_EXPERIMENT_PLAN.md`, `Input_Tensor_Architecture_v5.md`, `TESTS.md`, `td_ludo/PERFORMANCE_OVERHAUL.md`, `td_ludo/android/android_integration_guide.md`, `td_ludo/checkpoints/ac_v6_big/CHECKPOINT_README.md`, `AlphaLudo Dashboard Design/src/Attributions.md`, `AlphaLudo Dashboard Design/src/guidelines/Guidelines.md`
- **Logos**: `td_ludo/logo.png`, `logo_solid_bg.png`, `logo_v1.png`, `logo_v2.png`, `logo_v3.png`
- **Build/dep files**: `setup.py` (root, safety rule), `td_ludo/setup.py`, `pyproject.toml`, `td_ludo/requirements.txt`, `requirements.txt`, `AlphaLudo_Export/requirements.txt`
- **`export_android.py`** (only Python script using current AlphaLudoV5)
- **2 mobile model `.ptl`** files in `apps/android/models_mobile/` (git-tracked, used by Android app)

## What is now gitignored (295 files, ~1.93 GB)

- **All model weights** in `td_ludo/checkpoints/**/model_latest.pt`, `model_best.pt`, `model_sl.pt`, `actor_weights.pt`
- **All ghosts in protected `td_ludo/checkpoints/ac_v6_1_strategic/ghosts/`** (kept locally per safety rule, never committed)
- **gcp_snapshots** (V6.1 + V6.2 historical archives, 2 dirs, ~180 MB code + checkpoints)
- **v62_gcp_snapshot** (~150 KB)
- **SQLite databases** (`game_history.db`, `ludo_db*`, `training_history.db`)
- **Training data** (`.npz` experience buffers, `large_eval.log`)
- **Runtime metrics JSON** (`elo_ratings.json`, `live_stats.json`, `training_metrics.json`, `actor_weights.pt`, etc.)
- **Backup checkpoints** in `td_ludo/checkpoints/ac_v6_big/backups/`, `ac_v9_slim_transformer/backups/`
- **Mobile binaries** (`.ptl` not in apps/, `.pt` in `td_ludo/android/`)
- **`td_ludo/play/model_weights/model.pt`** (gameplay runtime dependency)
- **23 new explicit gitignore entries** appended to `.gitignore` under section `# --- Added by .refactor_worker.py ---`

## Apply log (10 ticks)

| Tick | Deletes done | Cumulative |
|---:|---:|---:|
| 1 | 40 | 40 |
| 2 | 40 | 80 |
| 3 | 40 | 120 |
| 4 | 40 | 160 |
| 5 | 40 | 200 |
| 6 | 40 | 240 |
| 7 | 40 | 280 |
| 8 | 40 | 320 |
| 9 | 40 | 360 |
| 10 | 24 | 384 |

Zero failures, zero safety skips. Worker auto-advanced phase 4 → 5 on tick 10.

## Safety guardrails honored

- ✅ `td_ludo/checkpoints/ac_v6_1_strategic/` — every file kept (gitignored, never deleted)
- ✅ All `discussion/**` content — kept
- ✅ All UI/dashboard files — kept (.tsx, .html, .css, AlphaLudo Dashboard Design/, apps/, logos)
- ✅ All gameplay code — kept (game.cpp/h, heuristic_bot.py, play/, manual_test/)
- ✅ `setup.py`, `pyproject.toml`, `README.md`, `training_journal.md`, `POST_V61_EXPERIMENT_PLAN.md` — untouched
- ✅ `.git/`, `.claude/`, `.agent/` — untouched
- ✅ MCTS sweep on GCP — untouched (separate VM, refactor is local-only)
- ✅ Worker, state file, prompt files (.refactor_*) — gitignored, untouched

## Next: phase 6 (dev branch push)

Worker advances to phase 6 on next tick. Will:
1. `git fetch origin`
2. `git checkout -b dev-refactor-local origin/dev`
3. `git merge main --no-ff`
4. `git add -A` + commit + push to `origin/dev` using temporary token
5. Switch back to `main`
6. Advance to phase 8 (refactor planning)
