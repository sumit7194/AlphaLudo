#!/usr/bin/env bash
# V12.3 RL on VM (moved from Mac 2026-05-30). The stable run (76.5% eval,
# healthy) gets the GPU. Resumes from the local checkpoint (G≈249K) which
# was uploaded to checkpoints/ac_v123_rl/model_latest.pt.
#
# Same config as the local run was using:
#   - mix C (v123_hard): Self 70 + Expectimax 15 + AggressiveExpectimax 10 + Expert 5
#   - eval every 15000 games × 2000 games
#   - entropy 0.005
#   - device cuda (VM L4)
set -euo pipefail
cd /home/sumit/td_ludo
export TD_LUDO_RUN_NAME=ac_v123_rl
# ── WINNING-RECIPE REWARDS (2026-06-01) ──────────────────────────────────
# Revert from the drifted sparse score-only rewards back to the proven
# champion recipe (V13.5 best / V13.2 → 84.55%): dense v1 shaped rewards
# (score/forward/capture/home/spawn/kill) + bias penalties (laggard at
# base, leader-into-danger). The trainer mechanics (γ=0.999, return+adv
# normalization, SmoothL1) were already correct — only the rewards drifted.
export LUDO_REWARD_MENU=v1_dense
export LUDO_BIAS_PENALTIES=1
RUN_DIR=/home/sumit/td_ludo/checkpoints/ac_v123_rl
LOG=$RUN_DIR/console.log
mkdir -p $RUN_DIR

RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then
  RESUME_FLAG="--resume"
  echo "[launcher] resuming V12.3 from $RUN_DIR/model_latest.pt" >> "$LOG"
fi

CMD="cd /home/sumit/td_ludo && PYTHONPATH=/home/sumit/td_ludo:/home/sumit/td_ludo_v15:. TD_LUDO_RUN_NAME=ac_v123_rl LUDO_REWARD_MENU=v1_dense LUDO_BIAS_PENALTIES=1 PYTHONUNBUFFERED=1 /usr/bin/python3 -u train_v12.py \
  --model-arch v132 --num-res-blocks 8 --num-channels 96 \
  --game-composition v123_hard \
  --device cuda \
  --port 8790 \
  --eval-interval 15000 --eval-games 2000 \
  --entropy-coeff 0.005 \
  $RESUME_FLAG \
  >> $LOG 2>&1"

setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 6
PID=$(pgrep -f 'train_v12.py.*ac_v123_rl' | head -1 || pgrep -f 'train_v12.py' | head -1 || true)
if [[ -z "$PID" ]]; then echo "FAILED — check $LOG"; tail -40 "$LOG"; exit 1; fi
echo "launched V12.3 RL on VM: PID=$PID"
sleep 6
tail -25 "$LOG"
