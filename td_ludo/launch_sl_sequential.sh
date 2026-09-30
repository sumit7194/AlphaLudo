#!/usr/bin/env bash
# Sequential local SL training: V15.2 → V12.3 → V13.6.
#
# Why sequential: parallel MPS contention made each run ~3x slower per step.
# Solo, each gets full GPU.
#
# Order:
#   1. V15.2  — most important + slowest + furthest from done (~15 hr from resume)
#   2. V12.3  — moderate compute, ~7 hr from resume
#   3. V13.6  — fresh restart after _encode_v13_6 bug fix (~7 hr)
#
# V13.6 starts from scratch because its previous checkpoint trained on a
# leaky target (action_rank always = lowest-legal-rank fallback). Bug
# fixed in sl_dataset.py line ~108. Old checkpoint moved to
# checkpoints/v136_sl_before_fix_bug/.
#
# Each trainer now has --resume flag (added 2026-05-24): loads
# model_latest.pt, restores optimizer + step counter + LR scheduler,
# exits at total_steps regardless of epoch.
set -euo pipefail
cd /Users/sumit/Github/AlphaLudo/td_ludo

PYBIN=/Users/sumit/Github/AlphaLudo/td_ludo/td_env/bin/python
export PYTHONPATH=.:../td_ludo_v15
COMMON="--shard-dir /Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/sl_dataset_v1 --epochs 1 --batch-size 256 --num-workers 4 --device mps"

LOG_OVERALL=/Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/sl_sequential.log
echo "=== SEQUENTIAL SL TRAINING STARTED $(date) ===" >> "$LOG_OVERALL"

run_step() {
  local name="$1" script="$2" out="$3" extra="$4"
  echo "--- [$(date)] launching $name ---" | tee -a "$LOG_OVERALL"
  $PYBIN -u "$script" $COMMON --out-dir "$out" $extra \
    >> "$out/train.log" 2>&1
  local rc=$?
  echo "--- [$(date)] $name exited rc=$rc ---" | tee -a "$LOG_OVERALL"
  return $rc
}

# 1. V15.2 (resume)
run_step "V15.2" train_v152_sl.py \
  /Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/v152_sl --resume

# 2. V12.3 (resume)
run_step "V12.3" train_v123_sl.py \
  /Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/v123_sl --resume

# 3. V13.6 (fresh — old ckpt was corrupt due to encoder bug)
run_step "V13.6" train_v136_sl.py \
  /Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/v136_sl ""

echo "=== SEQUENTIAL SL TRAINING DONE $(date) ===" >> "$LOG_OVERALL"
