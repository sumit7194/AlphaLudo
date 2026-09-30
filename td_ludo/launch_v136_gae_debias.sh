#!/usr/bin/env bash
# Exp 56 (2026-06-17) — GAE + DE-BIASED DENSE (the last high-juice lever).
# Two prior data points: terminal-only+GAE beat champion (54.4%) and touched
# the 85.5% ceiling but is flaw-agnostic; plain dense+GAE (gae_validate) held
# ~80 and didn't push. So dense+GAE only earns its keep if the dense rewards
# are DE-BIASED to aim at the two flaws:
#   - spawn-on-6: REWARD_SPAWN 0.05 -> 0.02 (< forward-6's 0.03) so a 6 only
#     spawns when nothing better exists.
#   - laggard neglect: danger penalty now covers ALL main-track positions
#     (pos>=1, was >35), so leaving the 4th token exposed costs ~0.09.
# Both gated by LUDO_DEBIAS_DENSE=1 (canonical recipe untouched for other runs).
# Dense gives the per-step gradient terminal-only lacks; GAE keeps variance low;
# the de-bias aims the signal AT the flaws. Init from the gaeterminal BEST
# (85.5%) so we start from the new peak, not the old champion. Entropy 0.005
# (dense provides signal -> less exploration needed, guards against drift).
# Judged by the laggard behavior probe + H2H, NOT eval. Separate run dir.
set -uo pipefail
BASE=/home/sumit/AlphaLudo/td_ludo
cd "$BASE"
export PYTHONPATH="$BASE:/home/sumit/AlphaLudo/td_ludo_v15"
export TD_LUDO_RUN_NAME=v136_gae_debias
export LUDO_REWARD_MENU=v1_dense
export LUDO_BIAS_PENALTIES=1
export LUDO_DEBIAS_DENSE=1

RUN_DIR="$BASE/checkpoints/v136_gae_debias"
INIT="$BASE/checkpoint_backups/v136_gaeterminal_2026-06-17/v136_gaeterminal_BEST_eval85.5pct_G200k_2026-06-16.pt"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"
RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then RESUME_FLAG="--resume";
else cp "$INIT" "$RUN_DIR/model_sl.pt"; echo "[launcher] init from gaeterminal BEST (85.5%)" >> "$LOG"; fi

CMD="cd $BASE && PYTHONPATH=$BASE:/home/sumit/AlphaLudo/td_ludo_v15 \
TD_LUDO_RUN_NAME=v136_gae_debias LUDO_REWARD_MENU=v1_dense LUDO_BIAS_PENALTIES=1 LUDO_DEBIAS_DENSE=1 \
PYTHONUNBUFFERED=1 python3 -u train_v12.py \
  --model-arch v13_5 --num-res-blocks 6 --v135-num-channels 96 --head-hidden 64 \
  --game-composition v13_5_no_bots \
  --use-gae --gae-lambda 0.95 \
  --device cuda --port 8799 --no-dashboard \
  --eval-interval 10000 --eval-games 2000 \
  --entropy-coeff 0.005 \
  $RESUME_FLAG \
  >> $LOG 2>&1"
setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 14
PID=$(ps -eo pid,args | grep "[t]rain_v12" | awk '$2=="python3"{print $1}' | head -1)
if [[ -z "$PID" ]]; then echo "FAILED — log:"; tail -30 "$LOG"; exit 1; fi
echo "launched GAE + DE-BIASED DENSE: PID=$PID"
sleep 6
tail -20 "$LOG"
