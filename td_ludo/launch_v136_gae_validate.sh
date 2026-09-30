#!/usr/bin/env bash
# Exp 55a (2026-06-15) — GAE PRACTICAL VALIDATION.
# Runs the NORMAL champion recipe (dense v1 + bias) but with --use-gae, from
# the champion. Purpose: confirm GAE doesn't break normal training in the
# REAL pipeline (trajectory alignment in the live rollout, truncated games,
# BatchNorm) — synthetic unit tests already proved the math (λ=1 ≡ MC). If
# eval HOLDS ~80 over ~30K games, GAE is validated and we proceed to the
# terminal-only+GAE experiment. Separate run dir; champion safe.
set -uo pipefail
BASE=/home/sumit/AlphaLudo/td_ludo
cd "$BASE"
export PYTHONPATH="$BASE:/home/sumit/AlphaLudo/td_ludo_v15"
export TD_LUDO_RUN_NAME=v136_gae_validate
export LUDO_REWARD_MENU=v1_dense
export LUDO_BIAS_PENALTIES=1

RUN_DIR="$BASE/checkpoints/v136_gae_validate"
CHAMP="$BASE/checkpoints/v136_rl_parity/model_latest.pt"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"
RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then RESUME_FLAG="--resume";
else cp "$CHAMP" "$RUN_DIR/model_sl.pt"; echo "[launcher] init from champion" >> "$LOG"; fi

CMD="cd $BASE && PYTHONPATH=$BASE:/home/sumit/AlphaLudo/td_ludo_v15 \
TD_LUDO_RUN_NAME=v136_gae_validate LUDO_REWARD_MENU=v1_dense LUDO_BIAS_PENALTIES=1 \
PYTHONUNBUFFERED=1 python3 -u train_v12.py \
  --model-arch v13_5 --num-res-blocks 6 --v135-num-channels 96 --head-hidden 64 \
  --game-composition v13_5_no_bots \
  --use-gae --gae-lambda 0.95 \
  --device cuda --port 8797 --no-dashboard \
  --eval-interval 10000 --eval-games 2000 \
  --entropy-coeff 0.005 \
  $RESUME_FLAG \
  >> $LOG 2>&1"
setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 14
PID=$(ps -eo pid,args | grep "[t]rain_v12" | awk '$2=="python3"{print $1}' | head -1)
if [[ -z "$PID" ]]; then echo "FAILED — log:"; tail -30 "$LOG"; exit 1; fi
echo "launched GAE VALIDATION (dense+GAE): PID=$PID"
sleep 6
tail -18 "$LOG"
