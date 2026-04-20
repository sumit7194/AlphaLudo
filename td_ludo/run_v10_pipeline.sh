#!/usr/bin/env bash
# V10 pipeline: smoke test → data gen → SL (3 epochs) → eval
#
# Usage:
#   ./run_v10_pipeline.sh            # full pipeline from scratch
#   ./run_v10_pipeline.sh --epochs 5 # override SL epochs
#   ./run_v10_pipeline.sh --skip-data  # skip data gen (use existing)
#
# On completion, prints:
#   - SL training log summary
#   - Pure-policy WR vs Expert bots
#   - win_prob calibration report (Brier + reliability buckets)
#   - moves_remaining MAE
#
# If calibration is GOOD (Brier < 0.20) we should build RL next.
# If calibration is WEAK, V10's backbone isn't learning outcome features
# and we need to rethink before investing in RL.

set -euo pipefail

cd "$(dirname "$0")"
PY=./td_env/bin/python

EPOCHS=3
SKIP_DATA=0
MIN_CHUNKS=10        # need at least 10 chunks (100K states) to skip data gen
EVAL_GAMES=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)    EPOCHS="$2"; shift 2 ;;
    --skip-data) SKIP_DATA=1; shift ;;
    --eval-games) EVAL_GAMES="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

LOG_DIR=checkpoints/ac_v10
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo -e "\n═══════════════════════════════════════════════════════════"
  echo "  $1"
  echo -e "═══════════════════════════════════════════════════════════\n"
}

# Tee everything to a log file for later inspection
exec > >(tee -a "$LOG_FILE") 2>&1

log "V10 Pipeline start — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python: $PY"
echo "  Epochs: $EPOCHS"
echo "  Skip data gen: $SKIP_DATA"
echo "  Eval games: $EVAL_GAMES"
echo "  Log: $LOG_FILE"

# ─────────────────────────────────────────────────────────────
# Auto-start dashboard (port 8788) unless already running
# ─────────────────────────────────────────────────────────────
DASH_PORT=8788
if ! lsof -ti :$DASH_PORT >/dev/null 2>&1; then
  nohup $PY pipeline_dashboard.py --port $DASH_PORT --log-dir "$LOG_DIR" \
    > "$LOG_DIR/dashboard.log" 2>&1 &
  DASH_PID=$!
  echo "  Dashboard: http://localhost:$DASH_PORT  (PID $DASH_PID)"
  sleep 1
else
  echo "  Dashboard: http://localhost:$DASH_PORT  (already running)"
fi

# ─────────────────────────────────────────────────────────────
# Stage 0: Smoke test — make sure V10 encoder works
# ─────────────────────────────────────────────────────────────
log "Stage 0: V10 encoder smoke test"
# Capture output instead of piping to tail (pipefail + Python SIGPIPE issues)
SMOKE_OUT=$($PY smoke_test_v10_channels.py 2>&1)
echo "$SMOKE_OUT" | tail -5

# ─────────────────────────────────────────────────────────────
# Stage 1: Data generation
# ─────────────────────────────────────────────────────────────
DATA_DIR=checkpoints/sl_data_v10
# Count existing chunks without failing when dir/pattern is empty
EXISTING_CHUNKS=0
if [[ -d "$DATA_DIR" ]]; then
  EXISTING_CHUNKS=$(find "$DATA_DIR" -maxdepth 1 -name 'chunk_*.npz' -type f 2>/dev/null | wc -l | tr -d ' ')
fi

if [[ $SKIP_DATA -eq 1 ]]; then
  log "Stage 1: SKIPPED (--skip-data)"
  echo "  Using existing $EXISTING_CHUNKS chunks at $DATA_DIR"
elif [[ $EXISTING_CHUNKS -ge $MIN_CHUNKS ]]; then
  log "Stage 1: SKIPPED (found $EXISTING_CHUNKS existing chunks, need $MIN_CHUNKS)"
  echo "  To regenerate: rm -rf $DATA_DIR and rerun"
else
  log "Stage 1: Generate V10 SL data (~40 min for 500K states)"
  $PY generate_sl_data_v10.py
fi

# ─────────────────────────────────────────────────────────────
# Stage 2: Joint SL training
# ─────────────────────────────────────────────────────────────
log "Stage 2: V10 joint SL training ($EPOCHS epochs)"
$PY train_sl_v10.py --epochs "$EPOCHS" --max-states 500000

# ─────────────────────────────────────────────────────────────
# Stage 3: Evaluation
# ─────────────────────────────────────────────────────────────
log "Stage 3: V10 evaluation ($EVAL_GAMES games)"
$PY eval_v10_sl.py --games "$EVAL_GAMES"

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
log "V10 Pipeline complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Checkpoint:  $LOG_DIR/model_sl.pt"
echo "  Full log:    $LOG_FILE"
echo
echo "  Next steps based on eval output:"
echo "    - If Brier < 0.20:  calibration is good → build V10 RL trainer"
echo "    - If Brier 0.20-0.23:  mixed signal → try more SL epochs first"
echo "    - If Brier > 0.23:  calibration weak → rethink architecture before RL"
