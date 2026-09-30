#!/usr/bin/env bash
# V12.3 RL — WINNING-RECIPE PARITY build.
# train_v12.py → trainer_v10.py (PPO), the exact champion code path.
# Recipe: dense v1 shaped rewards + bias penalties + γ=0.999 + return/adv
# norm + win-prob BCE value + v13_5_no_bots STRONG-NEURAL opponent pool
# (no scripted bots). MinimalCNN14 8×96 (V17 17ch engineered input).
# Init from the 1-epoch SL checkpoint.
set -uo pipefail
cd /Users/sumit/Github/AlphaLudo/td_ludo
export PYTHONPATH=.:../td_ludo_v15
PYBIN=./td_env/bin/python

export TD_LUDO_RUN_NAME=v123_rl_parity
export LUDO_REWARD_MENU=v1_dense
export LUDO_BIAS_PENALTIES=1

RUN_DIR=checkpoints/v123_rl_parity
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/console.log
# Seed from the 1-epoch SL checkpoint (only if no checkpoint yet).
if [[ ! -f "$RUN_DIR/model_sl.pt" && ! -f "$RUN_DIR/model_latest.pt" ]]; then
  cp checkpoints/ac_v123_rl/model_sl.pt "$RUN_DIR/model_sl.pt"
  echo "[launcher] seeded model_sl.pt from ac_v123_rl/model_sl.pt" >> "$LOG"
fi

PORT="${1:-8821}"
EVAL_INTERVAL="${2:-100000000}"   # default: eval effectively off (GPM probe)

$PYBIN -u train_v12.py \
  --model-arch v132 --num-res-blocks 8 --num-channels 96 \
  --game-composition v13_5_no_bots \
  --device mps --port "$PORT" \
  --eval-interval "$EVAL_INTERVAL" --eval-games 2000 \
  --entropy-coeff 0.005 \
  >> "$LOG" 2>&1
