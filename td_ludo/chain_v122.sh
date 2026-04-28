#!/usr/bin/env bash
# V12.2 chain: data-gen → SL warm-up → RL.
#
# Robust to the V12.1 bug (python ... | tee silently swallowed crash codes):
# `set -euo pipefail` propagates errors through pipes.
#
# Writes chain_status.json at every state transition so the dashboard can
# show "Stage 1 — SL warm-up running, epoch 6/10" etc.
#
# Usage:
#   ./chain_v122.sh [TARGET_STATES] [SL_EPOCHS]
# Defaults: TARGET_STATES=500000, SL_EPOCHS=10
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_STATES="${1:-500000}"
SL_EPOCHS="${2:-10}"
RUN_NAME="ac_v12_2"
TEACHER_PATH="play/model_weights/v12_final/model_latest.pt"
PORT="8790"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/td_env/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

RUNDIR="checkpoints/${RUN_NAME}"
DATA_DIR="checkpoints/sl_data_v122"
mkdir -p "$RUNDIR" "$DATA_DIR"

CHAIN_STATUS="${RUNDIR}/chain_status.json"

write_status() {
  # write_status STAGE PHASE [DETAIL_KEY=VALUE ...]
  local stage="$1"; shift
  local phase="$1"; shift
  local extras=""
  for kv in "$@"; do
    extras+=", \"${kv%%=*}\": \"${kv#*=}\""
  done
  cat > "$CHAIN_STATUS" <<JSON
{
  "stage": "${stage}",
  "phase": "${phase}",
  "run_name": "${RUN_NAME}",
  "ts": $(date +%s)${extras}
}
JSON
}

trap 'write_status "${CURRENT_STAGE:-unknown}" "failed" detail="see logs"' ERR

# ── Stage 0: SL data generation ─────────────────────────────────────────────
CURRENT_STAGE="data"
echo "[chain $(date '+%H:%M:%S')] === STAGE 0: SL data generation (target=${TARGET_STATES}) ==="
write_status "data" "running" "target=${TARGET_STATES}"
"$PY" -u scripts/generate_sl_data_v122.py \
  --teacher "${TEACHER_PATH}" \
  --output-dir "${DATA_DIR}" \
  --target-states "${TARGET_STATES}" \
  --batch-size 256 2>&1 | tee "${RUNDIR}/data.log"
DATA_TOTAL=$(/usr/bin/find "${DATA_DIR}" -name 'chunk_*.npz' | wc -l | tr -d ' ')
write_status "data" "done" "chunks=${DATA_TOTAL}"

# ── Stage 1: SL warm-up ─────────────────────────────────────────────────────
CURRENT_STAGE="SL"
echo "[chain $(date '+%H:%M:%S')] === STAGE 1: SL warm-up (epochs=${SL_EPOCHS}) ==="
write_status "SL" "running" "epochs=${SL_EPOCHS}"
"$PY" -u train_sl_v12.py \
  --data-dir "${DATA_DIR}" \
  --output "${RUNDIR}/model_sl.pt" \
  --epochs "${SL_EPOCHS}" \
  --batch-size 512 \
  --num-res-blocks 3 --num-channels 128 \
  --num-attn-layers 2 --num-heads 4 --ffn-ratio 4 \
  --dropout 0.1 \
  --max-states "${TARGET_STATES}" 2>&1 | tee "${RUNDIR}/sl.log"
write_status "SL" "done"

# ── Stage 2: RL ─────────────────────────────────────────────────────────────
CURRENT_STAGE="RL"
echo "[chain $(date '+%H:%M:%S')] === STAGE 2: RL (V12.2 mix, entropy 0.01) ==="
write_status "RL" "running"
TD_LUDO_RUN_NAME="${RUN_NAME}" "$PY" -u train_v12.py \
  --resume \
  --port "${PORT}" \
  --num-res-blocks 3 --num-channels 128 \
  --num-attn-layers 2 --num-heads 4 --ffn-ratio 4 \
  --game-composition v122 \
  --entropy-coeff 0.01 2>&1 | tee "${RUNDIR}/rl.log"
write_status "RL" "done"

echo "[chain $(date '+%H:%M:%S')] === DONE ==="
