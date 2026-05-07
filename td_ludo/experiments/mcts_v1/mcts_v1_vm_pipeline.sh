#!/usr/bin/env bash
# VM-side overnight chain: wait for generation, then train, then mark done.
# Designed to be daemonized on VM. Logs to /tmp/mcts_v1_vm_pipeline.log.
set -uo pipefail

cd "$HOME/AlphaLudo/td_ludo" || exit 1

LOG=/tmp/mcts_v1_vm_pipeline.log
DONE_FLAG=/tmp/mcts_v1_step1_done.flag
ERROR_FLAG=/tmp/mcts_v1_step1_error.flag

# Clean stale flags
rm -f "$DONE_FLAG" "$ERROR_FLAG"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"
}

log "=== VM pipeline starting ==="

# Phase 1: wait for generation to finish
log "Phase 1: waiting for generate_search_data to finish..."
while pgrep -f generate_search_data >/dev/null; do
  sleep 60
done
log "generate_search_data process gone"

# Verify buffer exists
BUFFER=runs/mcts_v1_search_buffer.npz
if [ ! -f "$BUFFER" ]; then
  log "ERROR: buffer $BUFFER missing — generation must have failed"
  echo "buffer missing" > "$ERROR_FLAG"
  exit 1
fi
SIZE=$(stat -c %s "$BUFFER" 2>/dev/null || stat -f %z "$BUFFER")
log "buffer present: $BUFFER ($SIZE bytes)"

# Verify buffer is non-empty (>10MB at minimum for 1M states)
if [ "$SIZE" -lt 10000000 ]; then
  log "ERROR: buffer too small ($SIZE bytes); generation likely truncated"
  echo "buffer too small: $SIZE bytes" > "$ERROR_FLAG"
  exit 1
fi

# Phase 2: launch training
log "Phase 2: launching train_search_distill..."
TD_LUDO_RUN_NAME=mcts_v1_step1_distill ./td_env/bin/python -u -m experiments.mcts_v1.train_search_distill \
  --buffer "$BUFFER" \
  --epochs 5 \
  --batch-size 1024 \
  --lr 1e-3 --lr-end 1e-4 \
  --eval-every 99999999 \
  --save-every 5000000 \
  --no-dashboard \
  --device cuda \
  >> "$LOG" 2>&1
RC=$?
log "training exited with rc=$RC"

if [ $RC -ne 0 ]; then
  log "ERROR: training failed"
  echo "training rc=$RC" > "$ERROR_FLAG"
  exit 1
fi

# Verify output checkpoint
OUT=checkpoints/mcts_v1_step1_distill/model_latest.pt
if [ ! -f "$OUT" ]; then
  log "ERROR: output checkpoint $OUT missing"
  echo "output missing" > "$ERROR_FLAG"
  exit 1
fi
log "output checkpoint saved: $OUT"

# Phase 3: H2H tournament (Step1 distilled student vs V13.2-latest)
# Run on VM CPU — small models, batch=1 inference; CPU beats GPU due to
# kernel-launch overhead. Roughly 200-400 gpm; 25K games ≈ ~80-120 min.
log "Phase 3: launching 25K-game H2H tournament on VM CPU..."
TS=$(date '+%Y%m%d_%H%M%S')
TOURNAMENT_OUT="runs/tournament_step1_vs_v132_${TS}.json"
mkdir -p runs

./td_env/bin/python -u -m experiments.tournament.run \
  --add-model V13_2_latest:v132:checkpoints/v132/model_latest.pt \
  --add-model Step1_Distill:v132:checkpoints/mcts_v1_step1_distill/model_latest.pt \
  --games-per-pair 25000 \
  --device cpu \
  --seed 42 \
  --output "$TOURNAMENT_OUT" \
  >> "$LOG" 2>&1
TRC=$?
log "tournament exited with rc=$TRC"

if [ $TRC -ne 0 ]; then
  log "ERROR: tournament failed"
  echo "tournament rc=$TRC" > "$ERROR_FLAG"
  exit 1
fi

# Verify output
if [ ! -f "$TOURNAMENT_OUT" ]; then
  log "ERROR: tournament output $TOURNAMENT_OUT missing"
  echo "tournament output missing" > "$ERROR_FLAG"
  exit 1
fi
log "tournament results: $TOURNAMENT_OUT"

# Phase 4: mark done (path includes tournament output for the watcher)
echo "$(date '+%Y-%m-%d %H:%M:%S') ${TOURNAMENT_OUT}" > "$DONE_FLAG"
log "=== VM pipeline DONE ==="
