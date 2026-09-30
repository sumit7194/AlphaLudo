#!/usr/bin/env bash
# V13.6 WORLD-MODEL fine-tune (2026-06-14) — target-B attack angle.
# Continues from the V13.6 champion with the new consequence-prediction aux
# heads (capture-prob + cells-at-risk, token-indexed) trained against
# engine-computed ground truth (consequence_targets.py). Hypothesis: forcing
# the trunk to encode capture-risk cures the laggard-neglect flaw (the model
# was never taught the trailing token is worth protecting). See
# discussion/WORLD_MODEL_ATTACK_ANGLE.md.
#
# Everything else identical to the champion recipe (v13_5_no_bots mix, dense
# reward + bias, 6×96, entropy 0.005). ONLY new variable: --consequence-coeff.
# Separate run dir → champion stays safe. Init from champion (new heads load
# random via the strict=False fix in trainer.load_checkpoint).
set -uo pipefail
BASE=/home/sumit/AlphaLudo/td_ludo
cd "$BASE"
export PYTHONPATH="$BASE:/home/sumit/AlphaLudo/td_ludo_v15"
export TD_LUDO_RUN_NAME=v136_worldmodel
export LUDO_REWARD_MENU=v1_dense
export LUDO_BIAS_PENALTIES=1

RUN_DIR="$BASE/checkpoints/v136_worldmodel"
CHAMP="$BASE/checkpoints/v136_rl_parity/model_latest.pt"   # the champion
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"

RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then
  RESUME_FLAG="--resume"
  echo "[launcher] resuming world-model fine-tune from model_latest.pt" >> "$LOG"
else
  cp "$CHAMP" "$RUN_DIR/model_sl.pt"   # init = champion; new heads → random
  echo "[launcher] fresh world-model fine-tune, init from champion $CHAMP" >> "$LOG"
fi

CMD="cd $BASE && PYTHONPATH=$BASE:/home/sumit/AlphaLudo/td_ludo_v15 \
TD_LUDO_RUN_NAME=v136_worldmodel LUDO_REWARD_MENU=v1_dense LUDO_BIAS_PENALTIES=1 \
PYTHONUNBUFFERED=1 python3 -u train_v12.py \
  --model-arch v13_5 --num-res-blocks 6 --v135-num-channels 96 --head-hidden 64 \
  --game-composition v13_5_no_bots \
  --consequence-coeff 0.4 \
  --device cuda --port 8794 --no-dashboard \
  --eval-interval 15000 --eval-games 2000 \
  --entropy-coeff 0.005 \
  $RESUME_FLAG \
  >> $LOG 2>&1"

setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 14
PID=$(ps -eo pid,args | grep "[t]rain_v12" | awk '$2=="python3"{print $1}' | head -1)
if [[ -z "$PID" ]]; then echo "FAILED — log tail:"; tail -30 "$LOG"; exit 1; fi
echo "launched V13.6 WORLD-MODEL fine-tune: PID=$PID  (consequence-coeff 0.4)"
sleep 6
tail -25 "$LOG"
