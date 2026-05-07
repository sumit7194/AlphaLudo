#!/usr/bin/env bash
# V13.4 chain: SL distillation → RL self-play → H2H tournament → done flag.
#
# Designed to be daemonized on VM. Logs to checkpoints/v134/chain.log.
# Each phase writes its own progress JSON for the unified dashboard.
#
# Usage:
#   ./experiments/v134/chain.sh [--smoke]
#
# --smoke: run with tiny numbers to verify chain end-to-end.
set -uo pipefail

cd "$(dirname "$0")/../.." || { echo "cd failed"; exit 1; }
PROJECT_ROOT="$(pwd)"

# ── Config ──────────────────────────────────────────────────────────────
SMOKE="${1:-}"
RUN_NAME="v134"
CKPT_DIR="checkpoints/${RUN_NAME}"
LOG="${CKPT_DIR}/chain.log"

# V13.4-B architecture (3.79M params)
CNN_BLOCKS=10
CNN_CHANNELS=128
D_MODEL=128
N_LAYERS=4
NHEAD=4
FFN_DIM=512

# Teacher (V13.2)
TEACHER="${TEACHER:-checkpoints/v132/model_latest.pt}"

# Phase budgets
if [[ "$SMOKE" == "--smoke" ]]; then
  SL_TARGET=5000
  SL_BATCH=64
  SL_SAVE=100000     # never fires in smoke
  SL_EVAL=100000     # never fires in smoke
  SL_LOG_EVERY=2

  RL_TARGET=5000
  RL_PARALLEL=32
  RL_CHUNK=512
  RL_MINIBATCH=128
  RL_EPOCHS=1
  RL_SAVE=100000
  RL_EVAL=100000
  RL_LOG_EVERY=1

  H2H_GAMES=20
else
  SL_TARGET=10000000     # 10M states
  SL_BATCH=256           # V13.4 (3.79M params + K=8) is heavy; OOM'd at 1024 on L4
  SL_SAVE=2000000
  SL_EVAL=1000000
  SL_LOG_EVERY=50

  RL_TARGET=1500000      # 1.5M states
  RL_PARALLEL=64         # halved to fit V13.4 in GPU memory
  RL_CHUNK=2048
  RL_MINIBATCH=256
  RL_EPOCHS=2
  RL_SAVE=200000
  RL_EVAL=200000
  RL_LOG_EVERY=1

  H2H_GAMES=500
fi

DEVICE="${DEVICE:-cuda}"
PY="${PY:-${PROJECT_ROOT}/td_env/bin/python}"

# ── Setup ──────────────────────────────────────────────────────────────
mkdir -p "$CKPT_DIR"
DONE_FLAG="${CKPT_DIR}/CHAIN_DONE"
ERROR_FLAG="${CKPT_DIR}/CHAIN_ERROR"
rm -f "$DONE_FLAG" "$ERROR_FLAG"

log() {
  local ts msg
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  msg="$*"
  echo "[$ts] $msg" | tee -a "$LOG"
}

write_chain_status() {
  local phase="$1"
  local detail="$2"
  cat > "${CKPT_DIR}/chain_status.json" <<EOF
{
  "stage": "chain",
  "phase": "$phase",
  "detail": "$detail",
  "smoke": "$([[ "$SMOKE" == "--smoke" ]] && echo true || echo false)",
  "arch": "v134",
  "params": "3.79M",
  "cnn_blocks": $CNN_BLOCKS,
  "cnn_channels": $CNN_CHANNELS,
  "d_model": $D_MODEL,
  "n_layers": $N_LAYERS,
  "ts": $(date +%s)
}
EOF
}

mark_error() {
  local msg="$1"
  log "ERROR: $msg"
  echo "$msg" > "$ERROR_FLAG"
  write_chain_status "error" "$msg"
}

log "=== V13.4 chain starting (smoke=${SMOKE:-no}) ==="
log "  RUN_NAME=$RUN_NAME  device=$DEVICE"
log "  arch: cnn ${CNN_BLOCKS}×${CNN_CHANNELS} + transformer ${N_LAYERS}L×${NHEAD}H d=${D_MODEL} ffn=${FFN_DIM}"
log "  budgets: SL=${SL_TARGET} states, RL=${RL_TARGET} states, H2H=${H2H_GAMES} games/pair"

# ── Phase 1: SL distillation ─────────────────────────────────────────────
write_chain_status "sl_running" "starting SL"
log "Phase 1: SL distillation (V13.2 teacher → V13.4 student)..."

if [ ! -f "$TEACHER" ]; then
  mark_error "teacher not found at $TEACHER"
  exit 1
fi

TD_LUDO_RUN_NAME="$RUN_NAME" "$PY" -u train_v133_sl.py \
  --teacher "$TEACHER" \
  --target-states "$SL_TARGET" \
  --batch-size "$SL_BATCH" \
  --lr 1e-3 --lr-end 1e-4 \
  --save-every "$SL_SAVE" \
  --eval-every "$SL_EVAL" \
  --eval-games 200 \
  --log-every "$SL_LOG_EVERY" \
  --cnn-blocks "$CNN_BLOCKS" \
  --cnn-channels "$CNN_CHANNELS" \
  --d-model "$D_MODEL" \
  --n-layers "$N_LAYERS" \
  --nhead "$NHEAD" \
  --ffn-dim "$FFN_DIM" \
  --device "$DEVICE" \
  --no-dashboard \
  >> "$LOG" 2>&1
SL_RC=$?
log "SL exited with rc=$SL_RC"

if [ $SL_RC -ne 0 ]; then
  mark_error "SL failed rc=$SL_RC"
  exit 1
fi

SL_OUT="${CKPT_DIR}/model_sl.pt"
if [ ! -f "$SL_OUT" ]; then
  mark_error "SL output $SL_OUT missing"
  exit 1
fi
log "SL output: $SL_OUT"

# ── Phase 2: RL self-play (init from SL) ─────────────────────────────────
write_chain_status "rl_running" "starting RL"
log "Phase 2: RL self-play REINFORCE (init from SL output)..."

TD_LUDO_RUN_NAME="$RUN_NAME" "$PY" -u train_v133_rl.py \
  --init "$SL_OUT" \
  --teacher "$TEACHER" \
  --target-states "$RL_TARGET" \
  --parallel-games "$RL_PARALLEL" \
  --train-chunk "$RL_CHUNK" \
  --minibatch-size "$RL_MINIBATCH" \
  --train-epochs "$RL_EPOCHS" \
  --lr 5e-5 --lr-end 5e-6 \
  --entropy-coeff 0.02 \
  --value-coeff 0.5 \
  --kl-anchor-coeff 0.1 \
  --save-every "$RL_SAVE" \
  --eval-every "$RL_EVAL" \
  --eval-games 200 \
  --log-every "$RL_LOG_EVERY" \
  --cnn-blocks "$CNN_BLOCKS" \
  --cnn-channels "$CNN_CHANNELS" \
  --d-model "$D_MODEL" \
  --n-layers "$N_LAYERS" \
  --nhead "$NHEAD" \
  --ffn-dim "$FFN_DIM" \
  --device "$DEVICE" \
  --no-dashboard \
  >> "$LOG" 2>&1
RL_RC=$?
log "RL exited with rc=$RL_RC"

if [ $RL_RC -ne 0 ]; then
  mark_error "RL failed rc=$RL_RC"
  exit 1
fi

RL_OUT="${CKPT_DIR}/model_rl.pt"
if [ ! -f "$RL_OUT" ]; then
  mark_error "RL output $RL_OUT missing"
  exit 1
fi
log "RL output: $RL_OUT"

# ── Phase 3: H2H — V13.2 vs V13.4_SL vs V13.4_RL ────────────────────────
write_chain_status "h2h_running" "starting H2H"
log "Phase 3: H2H tournament (3 agents, ${H2H_GAMES} games per pair)..."

"$PY" -u experiments/v134/h2h_v134.py \
  --games "$H2H_GAMES" \
  --device "$DEVICE" \
  --teacher "$TEACHER" \
  --sl "$SL_OUT" \
  --rl "$RL_OUT" \
  --cnn-blocks "$CNN_BLOCKS" \
  --cnn-channels "$CNN_CHANNELS" \
  --d-model "$D_MODEL" \
  --n-layers "$N_LAYERS" \
  --nhead "$NHEAD" \
  --ffn-dim "$FFN_DIM" \
  --output "${CKPT_DIR}/h2h_results.json" \
  >> "$LOG" 2>&1
H2H_RC=$?
log "H2H exited with rc=$H2H_RC"

if [ $H2H_RC -ne 0 ]; then
  mark_error "H2H failed rc=$H2H_RC"
  exit 1
fi

# ── Done ────────────────────────────────────────────────────────────────
write_chain_status "completed" "all phases done"
echo "$(date '+%Y-%m-%d %H:%M:%S')" > "$DONE_FLAG"
log "=== V13.4 chain DONE ==="
