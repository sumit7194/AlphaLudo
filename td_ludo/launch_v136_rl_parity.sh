#!/usr/bin/env bash
# V13.6 RL — WINNING-RECIPE PARITY build.
# Runs the V135Symmetric arch (6×96, head_hidden 64) through train_v12.py →
# trainer_v10.py (PPO) — the SAME champion code path as V12.3 / V13.5.
# Requires the --v135-num-channels patch (else 96→128 sentinel). Recipe:
# dense v1 shaped + bias penalties + γ=0.999 + return/adv norm + win-prob
# BCE value + v13_5_no_bots STRONG-NEURAL pool (no bots).
# Init from the 1-epoch SL checkpoint (v136_sl/model_latest.pt).
set -uo pipefail
cd /Users/sumit/Github/AlphaLudo/td_ludo
export PYTHONPATH=.:../td_ludo_v15
PYBIN=./td_env/bin/python

export TD_LUDO_RUN_NAME=v136_rl_parity
export LUDO_REWARD_MENU=v1_dense
export LUDO_BIAS_PENALTIES=1

RUN_DIR=checkpoints/v136_rl_parity
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/console.log
if [[ ! -f "$RUN_DIR/model_sl.pt" && ! -f "$RUN_DIR/model_latest.pt" ]]; then
  cp checkpoints/v136_sl/model_latest.pt "$RUN_DIR/model_sl.pt"
  echo "[launcher] seeded model_sl.pt from v136_sl/model_latest.pt" >> "$LOG"
fi

PORT="${1:-8822}"
EVAL_INTERVAL="${2:-100000000}"

$PYBIN -u train_v12.py \
  --model-arch v13_5 --num-res-blocks 6 --v135-num-channels 96 --head-hidden 64 \
  --game-composition v13_5_no_bots \
  --device mps --port "$PORT" \
  --eval-interval "$EVAL_INTERVAL" --eval-games 2000 \
  --entropy-coeff 0.005 \
  >> "$LOG" 2>&1
