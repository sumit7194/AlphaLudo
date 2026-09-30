#!/usr/bin/env bash
# Exp 57 (2026-06-17) — LIGHTER-TOUCH DE-BIAS.
# Exp 56 proved the de-bias fixes the laggard flaw (laggard_rescue 55.9->78.3%)
# but priced it too high: ~6-7pp eval (85.5 -> ~78, converged). This run keeps
# the laggard fix but makes it CHEAP:
#   - DROP the spawn cut (LUDO_DEBIAS_SPAWN=0): spawn stays canonical 0.05. The
#     spawn edit targeted flaw #1, not the laggard, and adds general-play tax.
#   - HALF-strength laggard penalty (LUDO_DEBIAS_DANGER_MULT=0.5): the newly
#     covered pos<=35 range costs ~0.042 not ~0.084. The proven advanced-token
#     (>35) penalty is untouched.
# Hypothesis: buy most of the laggard fix for ~2pp eval instead of 6-7pp.
# Init from gaeterminal BEST (85.5%). v1_dense + bias + GAE l0.95, entropy 0.005.
# Judged by laggard probe + H2H vs gaeterminal-best, NOT eval. Separate run dir.
set -uo pipefail
BASE=/home/sumit/AlphaLudo/td_ludo
cd "$BASE"
export PYTHONPATH="$BASE:/home/sumit/AlphaLudo/td_ludo_v15"
export TD_LUDO_RUN_NAME=v136_gae_debias_light
export LUDO_REWARD_MENU=v1_dense
export LUDO_BIAS_PENALTIES=1
export LUDO_DEBIAS_DENSE=1
export LUDO_DEBIAS_SPAWN=0
export LUDO_DEBIAS_DANGER_MULT=0.5

RUN_DIR="$BASE/checkpoints/v136_gae_debias_light"
INIT="$BASE/checkpoint_backups/v136_gaeterminal_2026-06-17/v136_gaeterminal_BEST_eval85.5pct_G200k_2026-06-16.pt"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"
RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then RESUME_FLAG="--resume";
else cp "$INIT" "$RUN_DIR/model_sl.pt"; echo "[launcher] init from gaeterminal BEST (85.5%)" >> "$LOG"; fi

CMD="cd $BASE && PYTHONPATH=$BASE:/home/sumit/AlphaLudo/td_ludo_v15 \
TD_LUDO_RUN_NAME=v136_gae_debias_light LUDO_REWARD_MENU=v1_dense LUDO_BIAS_PENALTIES=1 \
LUDO_DEBIAS_DENSE=1 LUDO_DEBIAS_SPAWN=0 LUDO_DEBIAS_DANGER_MULT=0.5 \
PYTHONUNBUFFERED=1 python3 -u train_v12.py \
  --model-arch v13_5 --num-res-blocks 6 --v135-num-channels 96 --head-hidden 64 \
  --game-composition v13_5_no_bots \
  --use-gae --gae-lambda 0.95 \
  --device cuda --port 8800 --no-dashboard \
  --eval-interval 10000 --eval-games 2000 \
  --entropy-coeff 0.005 \
  $RESUME_FLAG \
  >> $LOG 2>&1"
setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 14
PID=$(ps -eo pid,args | grep "[t]rain_v12" | awk '$2=="python3"{print $1}' | head -1)
if [[ -z "$PID" ]]; then echo "FAILED — log:"; tail -30 "$LOG"; exit 1; fi
echo "launched LIGHT DE-BIAS: PID=$PID"
sleep 6
tail -20 "$LOG"
