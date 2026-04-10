# V9 Training Performance Overhaul

## Results

| Config | GPM | Speedup | Time to 1M games |
|--------|-----|---------|-------------------|
| Before (2 actors x 32, full CNN) | 48 | 1x | ~10 days |
| **After (4 actors x 64, cached PPO)** | **115-134** | **~2.5x** | **~4 days** |

## What Changed

### 1. More Actors, Larger Batches
- `--actors` default: 2 -> 4
- `--actor-batch` default: 32 -> 64
- `torch.set_num_threads(1)` per actor (was 2) to avoid over-subscription
- Uses 4 of 10 CPU cores for game simulation + inference

### 2. Cached CNN in PPO Updates (the big win)
**File: `src/fast_learner.py`**

The learner's PPO update previously ran the full CNN backbone (5 ResBlocks, 750K params) on every minibatch in every epoch. With 4096 steps, 256 batch, 3 epochs = 48 CNN passes + 1 for advantages = **49 total**.

Now:
- CNN features pre-computed once before PPO update (`compute_cnn_features`)
- Advantage computation uses `forward_cached()` (skips CNN)
- PPO epochs 1-2 use `forward_cached()` (skips CNN, ~3x faster per epoch)
- PPO epoch 3 (last) uses full `forward()` so **CNN backbone still gets gradient updates**
- Net: CNN runs ~17 times instead of 49 = **65% less CNN compute**

### 3. Python 3.12 Environment
Created `td_env_312/` with Python 3.12 for Core ML compatibility.
- Core ML benchmark: CNN backbone runs **26x faster** on Neural Engine vs CPU
- Not yet integrated into training pipeline (future work)

## How to Run

```bash
# Use Python 3.12 venv (has coremltools support)
cd td_ludo
./td_env_312/bin/python3 train_v9_fast.py --resume --context-length 8

# Or with original Python 3.14 venv (no Core ML but training works)
./td_env/bin/python3 train_v9_fast.py --resume --context-length 8
```

### Optional: GPU Inference Server (experimental)
```bash
./td_env_312/bin/python3 train_v9_fast.py --resume --context-length 8 --gpu-actors
```
GPU actors generate 3x more games but the learner can't consume fast enough.
Use when learner bottleneck is resolved.

## Hardware Utilization (M4 Mac Mini)

| Resource | Before | After | Ideal (future) |
|----------|--------|-------|-----------------|
| CPU (10 cores) | 4 threads (~40%) | 4 actors (~50%) | 6-8 actors |
| GPU (10 cores) | Learner only (~20%) | Learner (~30%) | + inference server |
| Neural Engine | 0% | 0% | CNN inference (26x) |
| RAM (16GB) | ~2GB | ~4GB | OK |

## Architecture (current)

```
Actor 0 (CPU, 64 games) --+
Actor 1 (CPU, 64 games) --+--> trajectory_queue --> Learner (MPS GPU)
Actor 2 (CPU, 64 games) --+                         |
Actor 3 (CPU, 64 games) --+                         +--> PPO with cached CNN epochs
                                                     +--> checkpoint saves
                                                     +--> weight sync to actors
```

## Future Optimization Paths

1. **Core ML for learner CNN**: Pre-compute `compute_cnn_features` via ANE (26x faster)
   - Blocked: needs gradient-free CNN path, or separate ANE pre-computation process
2. **GPU inference server**: Already built (`src/inference_server.py`), works with `--gpu-actors`
   - Useful once learner is no longer the bottleneck
3. **Shared memory IPC**: Replace mp.Queue with shared memory for zero-copy on unified memory
4. **torch.compile**: Apply to transformer forward for MPS kernel fusion

## Files Modified

- `train_v9_fast.py` — New defaults (4 actors, batch 64), GPU actor support
- `src/fast_actor.py` — Threads 2->1, added `actor_worker_gpu` + `_package_game_standalone`
- `src/fast_learner.py` — Cached CNN for advantages + 2/3 PPO epochs, weight_update_queue

## Files Created

- `src/inference_server.py` — Centralized GPU inference server (experimental)
- `experiments/benchmark_ane.py` — CPU vs MPS vs Core ML benchmark
- `td_env_312/` — Python 3.12 virtual environment with Core ML support

## Benchmark Data

### CNN Backbone Inference (experiments/benchmark_ane.py)
```
Backend  batch=64     Samples/sec    vs CPU
CPU      47.04 ms     1,361/sec      1x
MPS      8.27 ms      7,737/sec      5.7x
ANE      1.82 ms      35,259/sec     25.9x
```
