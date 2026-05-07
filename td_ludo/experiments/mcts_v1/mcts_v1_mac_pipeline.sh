#!/usr/bin/env bash
# Mac-side overnight chain: poll VM for done flag, pull weights, run H2H tournament.
# Designed to be daemonized via Python double-fork. Logs to /tmp/mcts_v1_mac_pipeline.log.
set -uo pipefail

cd /Users/sumit/Github/AlphaLudo/td_ludo || exit 1

LOG=/tmp/mcts_v1_mac_pipeline.log
DONE_FLAG=/tmp/mcts_v1_mac_done.flag
ERROR_FLAG=/tmp/mcts_v1_mac_error.flag

VM_DONE_FLAG=/tmp/mcts_v1_step1_done.flag
VM_ERROR_FLAG=/tmp/mcts_v1_step1_error.flag

# Clean stale flags
rm -f "$DONE_FLAG" "$ERROR_FLAG"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"
}

ssh_check() {
  # Returns "DONE", "ERROR", or "RUNNING"
  out=$(gcloud compute ssh alphaludo-l4 --zone asia-southeast1-a --tunnel-through-iap --command "
    if [ -f $VM_DONE_FLAG ]; then echo DONE
    elif [ -f $VM_ERROR_FLAG ]; then echo ERROR; cat $VM_ERROR_FLAG
    else echo RUNNING
    fi
  " 2>&1 | grep -v "^WARNING\|NumPy\|TCP\|please see\|^$" | head -2)
  echo "$out"
}

log "=== Mac pipeline starting ==="
log "Polling VM every 2 min for done flag..."

# Phase 1: poll until VM training done
ATTEMPTS=0
MAX_ATTEMPTS=480  # 16 hours max
while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
  STATUS=$(ssh_check)
  STATE=$(echo "$STATUS" | head -1 | tr -d '[:space:]')
  if [ "$STATE" = "DONE" ]; then
    log "VM signals DONE"
    break
  fi
  if [ "$STATE" = "ERROR" ]; then
    log "VM signals ERROR: $STATUS"
    echo "VM error: $STATUS" > "$ERROR_FLAG"
    exit 1
  fi
  ATTEMPTS=$((ATTEMPTS+1))
  sleep 120
done

if [ $ATTEMPTS -ge $MAX_ATTEMPTS ]; then
  log "ERROR: timeout waiting for VM done flag (>16 hrs)"
  echo "timeout" > "$ERROR_FLAG"
  exit 1
fi

# Phase 2: pull trained weights
log "Phase 2: pulling trained weights from VM..."
mkdir -p checkpoints/mcts_v1_step1_distill
gcloud compute scp \
  alphaludo-l4:~/AlphaLudo/td_ludo/checkpoints/mcts_v1_step1_distill/model_latest.pt \
  alphaludo-l4:~/AlphaLudo/td_ludo/checkpoints/mcts_v1_step1_distill/model_sl.pt \
  checkpoints/mcts_v1_step1_distill/ \
  --zone asia-southeast1-a --tunnel-through-iap >> "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  log "ERROR: scp pull failed rc=$RC"
  echo "scp failed" > "$ERROR_FLAG"
  exit 1
fi
ls -la checkpoints/mcts_v1_step1_distill/ >> "$LOG" 2>&1

# Phase 3: backup also to checkpoint_backups
TS=$(date +%Y%m%d_%H%M%S)
BK=/Users/sumit/Github/AlphaLudo/checkpoint_backups/mcts_v1_step1_distill_${TS}
mkdir -p "$BK"
cp checkpoints/mcts_v1_step1_distill/model_latest.pt "$BK/" 2>/dev/null
cp checkpoints/mcts_v1_step1_distill/model_sl.pt "$BK/" 2>/dev/null
log "backup → $BK"

# Phase 4: H2H tournament — 25K games per pair, V13.2-latest vs Step1 distill
log "Phase 4: launching 25K-game H2H tournament (CPU)..."
mkdir -p runs

# Use the latest V13.2 backup we pulled this evening
V132_LATEST=$(ls -dt /Users/sumit/Github/AlphaLudo/checkpoint_backups/v132_2026* | head -1)
V132_PATH="${V132_LATEST}/model_latest.pt"
if [ ! -f "$V132_PATH" ]; then
  log "ERROR: V13.2 baseline checkpoint not found at $V132_PATH"
  echo "v132 missing" > "$ERROR_FLAG"
  exit 1
fi
log "V13.2 baseline: $V132_PATH"

./td_env/bin/python -u -m experiments.tournament.run \
  --add-model V13_2_latest:v132:"$V132_PATH" \
  --add-model Step1_Distill:v132:checkpoints/mcts_v1_step1_distill/model_latest.pt \
  --games-per-pair 25000 \
  --device cpu \
  --seed 42 \
  --output runs/tournament_step1_vs_v132_${TS}.json \
  >> "$LOG" 2>&1
RC=$?
log "tournament rc=$RC"

if [ $RC -ne 0 ]; then
  log "ERROR: tournament failed rc=$RC"
  echo "tournament rc=$RC" > "$ERROR_FLAG"
  exit 1
fi

# Done
echo "$(date '+%Y-%m-%d %H:%M:%S')" > "$DONE_FLAG"
log "=== Mac pipeline DONE ==="
log "Tournament results: runs/tournament_step1_vs_v132_${TS}.json"
