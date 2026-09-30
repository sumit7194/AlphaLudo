#!/usr/bin/env bash
# V13.6 RL on the NEW VM (us-east1-d, L4) — winning-recipe parity, RESUMING
# from the 663K-game checkpoint (best 80.5%). Layout base = ~/AlphaLudo/td_ludo.
set -uo pipefail
BASE=/home/sumit/AlphaLudo/td_ludo
cd "$BASE"
export PYTHONPATH="$BASE:/home/sumit/AlphaLudo/td_ludo_v15"
export TD_LUDO_RUN_NAME=v136_rl_parity
export LUDO_REWARD_MENU=v1_dense
export LUDO_BIAS_PENALTIES=1
RUN_DIR="$BASE/checkpoints/v136_rl_parity"
LOG="$RUN_DIR/console.log"
mkdir -p "$RUN_DIR"

RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then
  RESUME_FLAG="--resume"
  echo "[launcher] resuming from $RUN_DIR/model_latest.pt" >> "$LOG"
fi

CMD="cd $BASE && PYTHONPATH=$BASE:/home/sumit/AlphaLudo/td_ludo_v15 \
TD_LUDO_RUN_NAME=v136_rl_parity LUDO_REWARD_MENU=v1_dense LUDO_BIAS_PENALTIES=1 PYTHONUNBUFFERED=1 \
python3 -u train_v12.py \
  --model-arch v13_5 --num-res-blocks 6 --v135-num-channels 96 --head-hidden 64 \
  --game-composition v13_5_no_bots \
  --device cuda --port 8790 \
  --eval-interval 15000 --eval-games 2000 \
  --entropy-coeff 0.005 \
  $RESUME_FLAG \
  >> $LOG 2>&1"

setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 10
PID=$(pgrep -f 'train_v12.py.*v13_5' | head -1 || true)
if [[ -z "$PID" ]]; then echo "FAILED — log tail:"; tail -30 "$LOG"; exit 1; fi
echo "launched V13.6 RL (resume): PID=$PID"
sleep 8
tail -25 "$LOG"
