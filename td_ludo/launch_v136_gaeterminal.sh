#!/usr/bin/env bash
# Exp 55b (2026-06-16) — TERMINAL-ONLY + GAE (the real target-B test).
# The MC terminal-only run (Exp 54) drifted: MC gradients so noisy that
# entropy won. GAE (validated in Exp 55a: dense+GAE held eval ~80 over 240K)
# gives a low-variance advantage, so the small-but-real signal for the
# suppressed lines (chase-on-6, laggard rescue) becomes ESTIMABLE instead of
# drowned. Win/loss ±1 only (no dense artifacts), GAE λ0.95, modest
# exploration (entropy 0.01 — GAE's cleaner gradient needs less than the 0.02
# that drifted), NO anchor (anchor freezes the champion's flaws). Init from
# champion. Judged by the laggard behavior probe, NOT eval. Separate run dir.
set -uo pipefail
BASE=/home/sumit/AlphaLudo/td_ludo
cd "$BASE"
export PYTHONPATH="$BASE:/home/sumit/AlphaLudo/td_ludo_v15"
export TD_LUDO_RUN_NAME=v136_gaeterminal
export LUDO_TERMINAL_ONLY=1
export LUDO_TERMINAL_COEFF=1.0

RUN_DIR="$BASE/checkpoints/v136_gaeterminal"
CHAMP="$BASE/checkpoints/v136_rl_parity/model_latest.pt"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"
RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then RESUME_FLAG="--resume";
else cp "$CHAMP" "$RUN_DIR/model_sl.pt"; echo "[launcher] init from champion" >> "$LOG"; fi

CMD="cd $BASE && PYTHONPATH=$BASE:/home/sumit/AlphaLudo/td_ludo_v15 \
TD_LUDO_RUN_NAME=v136_gaeterminal LUDO_TERMINAL_ONLY=1 LUDO_TERMINAL_COEFF=1.0 \
PYTHONUNBUFFERED=1 python3 -u train_v12.py \
  --model-arch v13_5 --num-res-blocks 6 --v135-num-channels 96 --head-hidden 64 \
  --game-composition v13_5_no_bots \
  --use-gae --gae-lambda 0.95 \
  --device cuda --port 8798 --no-dashboard \
  --eval-interval 10000 --eval-games 2000 \
  --entropy-coeff 0.01 \
  $RESUME_FLAG \
  >> $LOG 2>&1"
setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 14
PID=$(ps -eo pid,args | grep "[t]rain_v12" | awk '$2=="python3"{print $1}' | head -1)
if [[ -z "$PID" ]]; then echo "FAILED — log:"; tail -30 "$LOG"; exit 1; fi
echo "launched TERMINAL-ONLY + GAE: PID=$PID"
sleep 6
tail -18 "$LOG"
