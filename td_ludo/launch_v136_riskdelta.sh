#!/usr/bin/env bash
# Exp 53 (2026-06-15) — V13.6 risk-delta reward (representation + INCENTIVE).
# The world-model heads (Exp 52) taught the trunk to PREDICT capture risk
# (corr 0.96) but the behavioral probe showed the policy didn't ACT on it
# (laggard-rescue unchanged: 52% vs 55%) — because nothing REWARDS protecting
# the laggard. This adds the missing incentive: a per-step reward =
# coeff × (own cells-at-risk before − after), engine-computed. Inits from the
# world-model checkpoint so it KEEPS the trained heads (representation) AND
# gets the incentive. Separate run dir → world-model + champion stay safe.
set -uo pipefail
BASE=/home/sumit/AlphaLudo/td_ludo
cd "$BASE"
export PYTHONPATH="$BASE:/home/sumit/AlphaLudo/td_ludo_v15"
export TD_LUDO_RUN_NAME=v136_riskdelta
export LUDO_REWARD_MENU=v1_dense
export LUDO_BIAS_PENALTIES=1

RUN_DIR="$BASE/checkpoints/v136_riskdelta"
INIT="$BASE/checkpoints/v136_worldmodel/model_latest.pt"   # world-model (heads trained)
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"

RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then
  RESUME_FLAG="--resume"
  echo "[launcher] resuming risk-delta run from model_latest.pt" >> "$LOG"
else
  cp "$INIT" "$RUN_DIR/model_sl.pt"
  echo "[launcher] fresh risk-delta run, init from world-model $INIT" >> "$LOG"
fi

CMD="cd $BASE && PYTHONPATH=$BASE:/home/sumit/AlphaLudo/td_ludo_v15 \
TD_LUDO_RUN_NAME=v136_riskdelta LUDO_REWARD_MENU=v1_dense LUDO_BIAS_PENALTIES=1 \
PYTHONUNBUFFERED=1 python3 -u train_v12.py \
  --model-arch v13_5 --num-res-blocks 6 --v135-num-channels 96 --head-hidden 64 \
  --game-composition v13_5_no_bots \
  --consequence-coeff 0.4 --risk-delta-coeff 0.02 \
  --device cuda --port 8795 --no-dashboard \
  --eval-interval 15000 --eval-games 2000 \
  --entropy-coeff 0.005 \
  $RESUME_FLAG \
  >> $LOG 2>&1"

setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 14
PID=$(ps -eo pid,args | grep "[t]rain_v12" | awk '$2=="python3"{print $1}' | head -1)
if [[ -z "$PID" ]]; then echo "FAILED — log tail:"; tail -30 "$LOG"; exit 1; fi
echo "launched V13.6 RISK-DELTA: PID=$PID  (consequence 0.4 + risk-delta 0.02)"
sleep 6
tail -22 "$LOG"
