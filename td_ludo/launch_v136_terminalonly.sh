#!/usr/bin/env bash
# Exp 54 (2026-06-15) — TERMINAL-ONLY fine-tune from the champion.
# Win/loss ±1 ONLY, NO dense shaping (LUDO_TERMINAL_ONLY=1 zeroes every
# per-step reward). NO KL anchor (the anchor would just freeze the champion's
# flaws). HIGH exploration (entropy 0.02 + temp 1.1→0.95) so the policy
# SAMPLES the lines the dense reward used to suppress (e.g. chase-on-6 vs
# spawn) and lets the terminal signal reinforce whatever actually wins.
#
# Hypothesis: removing the dense-reward artifacts lets the play flaws
# (spawn-on-6 tilt, laggard neglect) relax. RISK: at the variance ceiling
# advantage≈0, so it may drift (entropy) rather than improve — it's a RACE
# between shedding-flaws and drifting. Watched + snapshotted; judged by the
# behavior probe (laggard rescue), NOT eval. Stop if eval craters.
# Separate run dir → champion stays safe.
set -uo pipefail
BASE=/home/sumit/AlphaLudo/td_ludo
cd "$BASE"
export PYTHONPATH="$BASE:/home/sumit/AlphaLudo/td_ludo_v15"
export TD_LUDO_RUN_NAME=v136_terminalonly
# Terminal-only: NO dense reward menu, NO bias penalties. Terminal ±1 ON.
export LUDO_TERMINAL_ONLY=1
export LUDO_TERMINAL_COEFF=1.0

RUN_DIR="$BASE/checkpoints/v136_terminalonly"
CHAMP="$BASE/checkpoints/v136_rl_parity/model_latest.pt"   # the strongest model
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"

RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then
  RESUME_FLAG="--resume"
  echo "[launcher] resuming terminal-only run from model_latest.pt" >> "$LOG"
else
  cp "$CHAMP" "$RUN_DIR/model_sl.pt"
  echo "[launcher] fresh terminal-only run, init from champion $CHAMP" >> "$LOG"
fi

CMD="cd $BASE && PYTHONPATH=$BASE:/home/sumit/AlphaLudo/td_ludo_v15 \
TD_LUDO_RUN_NAME=v136_terminalonly LUDO_TERMINAL_ONLY=1 LUDO_TERMINAL_COEFF=1.0 \
PYTHONUNBUFFERED=1 python3 -u train_v12.py \
  --model-arch v13_5 --num-res-blocks 6 --v135-num-channels 96 --head-hidden 64 \
  --game-composition v13_5_no_bots \
  --device cuda --port 8796 --no-dashboard \
  --eval-interval 10000 --eval-games 2000 \
  --entropy-coeff 0.02 \
  $RESUME_FLAG \
  >> $LOG 2>&1"

setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 14
PID=$(ps -eo pid,args | grep "[t]rain_v12" | awk '$2=="python3"{print $1}' | head -1)
if [[ -z "$PID" ]]; then echo "FAILED — log tail:"; tail -30 "$LOG"; exit 1; fi
echo "launched V13.6 TERMINAL-ONLY: PID=$PID  (win/loss only, entropy 0.02, no anchor)"
sleep 6
tail -22 "$LOG"
