#!/usr/bin/env bash
# V13.6 RL on VM — GENTLER restart (2026-05-29).
#
# First attempt (G=0→43K) destabilized: entropy ballooned 0.36→0.75,
# model started LOSING to its own ghosts (24% WR) = regressing below
# past selves, first eval dropped 76%(SL)→69.8%. Diagnosis: dense reward
# + entropy 0.02 + mid-hard opps (Expectimax 15 + MinimaxExp 10) was too
# much pressure on the 1M-param model. Destabilized run archived to
# checkpoints/v136_rl_destabilized_G43K/.
#
# Fresh restart from SL with three stabilizing changes:
#   1. opp mix EASED: drop MinimaxExp, halve Expectimax, add 2 easy bots
#      Self 55 + Ghost 25 + Expectimax 10 + Aggressive 5 + Heuristic 5
#   2. entropy_coeff 0.02 → 0.01 (less exploration pressure)
#   3. kl_anchor_coeff 0.1 → 0.2 (stronger pull toward SL teacher)
set -euo pipefail
cd /home/sumit/td_ludo

export TD_LUDO_RUN_NAME=v136_rl
RUN_DIR=/home/sumit/td_ludo/checkpoints/v136_rl
INIT=$RUN_DIR/model_sl.pt
LATEST=$RUN_DIR/model_latest.pt
LOG=$RUN_DIR/console.log
mkdir -p $RUN_DIR

if [[ -f "$LATEST" ]]; then
  echo "[launcher] resuming from $LATEST"
  MODE_FLAG="--resume"
else
  echo "[launcher] FRESH init from $INIT (gentler config)"
  MODE_FLAG="--init $INIT"
fi

CMD="cd /home/sumit/td_ludo && PYTHONPATH=/home/sumit/td_ludo:/home/sumit/td_ludo_v15:. TD_LUDO_RUN_NAME=v136_rl PYTHONUNBUFFERED=1 /usr/bin/python3 -u train_v135_rl.py $MODE_FLAG \
  --kl-teacher $INIT \
  --kl-anchor-coeff 0.2 \
  --use-shaped-reward 1 \
  --num-res-blocks 6 --num-channels 96 --head-hidden 64 \
  --opp-weight-self 50 \
  --opp-weight-ghost 25 \
  --opp-weight-expectimax 10 \
  --opp-weight-aggressive 5 \
  --opp-weight-minimax-expectimax 5 \
  --opp-weight-expert 5 \
  --ghost-save-interval 5000 \
  --max-ghosts 8 \
  --target-states 200000000 \
  --parallel-games 64 \
  --train-chunk 2048 \
  --minibatch-size 256 \
  --train-epochs 2 \
  --lr 1e-5 --lr-end 5e-6 \
  --entropy-coeff 0.01 \
  --value-coeff 0.5 \
  --eval-every-games 15000 \
  --eval-games 2000 \
  --save-every-games 5000 \
  --log-every 10 \
  --temperature 1.1 \
  --device cuda \
  --port 8790 \
  >> $LOG 2>&1"

setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown

sleep 6
PID=$(pgrep -f "train_v135_rl.py" | head -1 || true)
if [[ -z "$PID" ]]; then echo "FAILED to launch — check $LOG"; tail -40 "$LOG"; exit 1; fi
echo "launched V13.6 RL GENTLER (eased opps + ent 0.01 + anchor 0.2): PID=$PID"
sleep 6
echo "--- first lines ---"
tail -30 "$LOG"
