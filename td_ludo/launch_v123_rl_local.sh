#!/usr/bin/env bash
# V12.3 RL on Mac — caffeinated, storm/restart-resilient.
# Port 8811 (VM dashboard stays on 8790).
#
# 2026-05-27 ROLLBACK: prior run collapsed at G=50K→55K (WR 62%→32%,
# entropy doubled 0.27→0.59). model_best.pt at G=25K (76.4% eval) was
# copied as model_latest.pt; G=76K collapsed ckpts archived to
# ac_v123_rl_collapsed_G76K/.
#
# Safer-than-default hyperparams (vs the failed run):
#   --entropy-coeff 0.005   (was argparse default 0.01 — config PROD is 0.005)
#   --eval-interval 15000   (catch any regression earlier)
#   --eval-games 2000
#
# Located inside the repo (not /tmp) so it survives Mac restarts.
set -uo pipefail
cd /Users/sumit/Github/AlphaLudo/td_ludo
export PYTHONPATH=.:../td_ludo_v15
PYBIN=./td_env/bin/python

LOG=/Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/ac_v123_rl/console.log

RESUME_FLAG=""
if [[ -f checkpoints/ac_v123_rl/model_latest.pt ]]; then
  RESUME_FLAG="--resume"
  echo "[launcher] resuming from checkpoints/ac_v123_rl/model_latest.pt" >> "$LOG"
else
  # Fresh start reads checkpoints/ac_v123_rl/model_sl.pt. train_v12.py does NOT
  # crash if it's missing — it silently falls back to RANDOM weights, which
  # quietly wastes a run. Seed it from the canonical SL final instead.
  INIT=checkpoints/ac_v123_rl/model_sl.pt
  SL_SRC=checkpoints/v123_sl/model_sl.pt
  if [[ ! -f "$INIT" ]]; then
    if [[ -f "$SL_SRC" ]]; then
      cp "$SL_SRC" "$INIT"
      echo "[launcher] seeded model_sl.pt from $SL_SRC" >> "$LOG"
    else
      echo "[launcher] FATAL: no SL baseline — neither $INIT nor $SL_SRC exists" | tee -a "$LOG"
      exit 1
    fi
  fi
  echo "[launcher] fresh start from model_sl.pt" >> "$LOG"
fi

TD_LUDO_RUN_NAME=ac_v123_rl $PYBIN -u train_v12.py \
  --model-arch v132 \
  --num-res-blocks 8 --num-channels 96 \
  --game-composition v123_hard \
  --device mps \
  --port 8811 \
  --eval-interval 15000 \
  --eval-games 2000 \
  --entropy-coeff 0.005 \
  $RESUME_FLAG \
  >> "$LOG" 2>&1
