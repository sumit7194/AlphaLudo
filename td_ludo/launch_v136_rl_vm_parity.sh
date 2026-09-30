#!/usr/bin/env bash
# V13.6 RL on VM (L4 cuda) — WINNING-RECIPE PARITY.
# V135Symmetric 6×96 (head_hidden 64) on train_v12.py → trainer_v10.py (PPO):
# dense v1 shaped + bias penalties + γ=0.999 + return/adv norm + win-prob BCE
# + v13_5_no_bots STRONG-NEURAL pool (no scripted bots). Requires the
# --v135-num-channels patch (uploaded). Inits from the epoch-2 SL model
# (warm-started on 1.5M, val_acc ~93.5%), placed as model_sl.pt in the run dir.
set -uo pipefail
cd /home/sumit/td_ludo
export TD_LUDO_RUN_NAME=v136_rl_parity
export LUDO_REWARD_MENU=v1_dense
export LUDO_BIAS_PENALTIES=1
RUN_DIR=/home/sumit/td_ludo/checkpoints/v136_rl_parity
LOG=$RUN_DIR/console.log
mkdir -p "$RUN_DIR"

RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then
  RESUME_FLAG="--resume"
  echo "[launcher] resuming V13.6 RL from $RUN_DIR/model_latest.pt" >> "$LOG"
else
  echo "[launcher] fresh start from $RUN_DIR/model_sl.pt" >> "$LOG"
fi

CMD="cd /home/sumit/td_ludo && PYTHONPATH=/home/sumit/td_ludo:/home/sumit/td_ludo_v15:. \
TD_LUDO_RUN_NAME=v136_rl_parity LUDO_REWARD_MENU=v1_dense LUDO_BIAS_PENALTIES=1 PYTHONUNBUFFERED=1 \
/usr/bin/python3 -u train_v12.py \
  --model-arch v13_5 --num-res-blocks 6 --v135-num-channels 96 --head-hidden 64 \
  --game-composition v13_5_no_bots \
  --device cuda --port 8790 \
  --eval-interval 15000 --eval-games 2000 \
  --entropy-coeff 0.005 \
  $RESUME_FLAG \
  >> $LOG 2>&1"

setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 8
PID=$(pgrep -f 'train_v12.py.*v13_5' | head -1 || true)
if [[ -z "$PID" ]]; then echo "FAILED — check $LOG"; tail -40 "$LOG"; exit 1; fi
echo "launched V13.6 RL on VM: PID=$PID"
sleep 8
tail -30 "$LOG"
