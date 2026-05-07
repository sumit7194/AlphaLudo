#!/usr/bin/env bash
# V13.2 full pipeline: SL distillation → RL with bias penalties + curriculum.
#
# V13.2 vs V13.1:
#   - Student has 17ch input (V14 + 3 V11 static channels) instead of 14ch
#   - Architecture: 10×128 (~3M params) instead of 12×160 (~5.6M)
#   - No aux heads — static board info is in input, no need to predict
#
# Phases:
#   1. SL: train_v132_sl.py distills V12.2-bias teacher into MinimalCNN14 (17ch).
#   2. RL: train_v12.py --resume picks up model_sl.pt. Starts with v122 (bots).
#      When 3 consecutive evals ≥ 80%, curriculum swaps to v122_hist_v2.
#
# Usage:
#   ./run_v132_pipeline.sh                       # default settings
#   SL_TARGET=15000000 ./run_v132_pipeline.sh
#   SL_ONLY=1 ./run_v132_pipeline.sh             # stop after SL
#   SKIP_SL=1 ./run_v132_pipeline.sh             # jump straight to RL
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME=${RUN_NAME:-v132}
TEACHER=${TEACHER:-play/model_weights/v12_2/model_latest.pt}
SL_TARGET=${SL_TARGET:-10000000}
SL_BATCH=${SL_BATCH:-1024}
SL_LR=${SL_LR:-1e-3}
SL_EVAL_EVERY=${SL_EVAL_EVERY:-250000}
SL_EVAL_GAMES=${SL_EVAL_GAMES:-200}
DASHBOARD_PORT=${DASHBOARD_PORT:-8792}
RL_GAME_COMP=${RL_GAME_COMP:-v122}
RL_CURRICULUM=${RL_CURRICULUM:-auto}
RL_CURRICULUM_TARGET=${RL_CURRICULUM_TARGET:-v122_hist_v2}
RL_ENTROPY=${RL_ENTROPY:-0.01}
SL_ONLY=${SL_ONLY:-0}
SKIP_SL=${SKIP_SL:-0}

CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
mkdir -p "$CHECKPOINT_DIR"

PYTHON="$(pwd)/td_env/bin/python"

echo "======================================================================"
echo "V13.2 PIPELINE — ${RUN_NAME}"
echo "======================================================================"
echo "  CHECKPOINT_DIR : ${CHECKPOINT_DIR}"
echo "  TEACHER        : ${TEACHER}"
echo "  SL_TARGET      : ${SL_TARGET} states"
echo "  SL_EVAL_EVERY  : ${SL_EVAL_EVERY} (${SL_EVAL_GAMES} games)"
echo "  RL_GAME_COMP   : ${RL_GAME_COMP}"
echo "  RL_CURRICULUM  : ${RL_CURRICULUM} → ${RL_CURRICULUM_TARGET}"
echo "  DASHBOARD      : http://<host>:${DASHBOARD_PORT}"
echo "  SL_ONLY=${SL_ONLY}, SKIP_SL=${SKIP_SL}"
echo "======================================================================"

# ── Phase 1: SL ──────────────────────────────────────────────────────
if [ "$SKIP_SL" = "0" ]; then
  echo ""
  echo "[Pipeline] Phase 1 — SL distillation"
  TD_LUDO_RUN_NAME="$RUN_NAME" "$PYTHON" -u train_v132_sl.py \
    --teacher "$TEACHER" \
    --target-states "$SL_TARGET" \
    --batch-size "$SL_BATCH" \
    --lr "$SL_LR" \
    --eval-every "$SL_EVAL_EVERY" \
    --eval-games "$SL_EVAL_GAMES" \
    --port "$DASHBOARD_PORT"
  echo "[Pipeline] SL phase complete. model_sl.pt at ${CHECKPOINT_DIR}/model_sl.pt"
else
  echo "[Pipeline] SKIP_SL=1 — skipping Phase 1"
fi

if [ "$SL_ONLY" = "1" ]; then
  echo "[Pipeline] SL_ONLY=1 — done after SL phase."
  exit 0
fi

# ── Phase 2: RL ──────────────────────────────────────────────────────
echo ""
echo "[Pipeline] Phase 2 — RL training with curriculum"

# Sanity: SL output must exist
if [ ! -f "${CHECKPOINT_DIR}/model_sl.pt" ]; then
  echo "[Pipeline] ERROR: ${CHECKPOINT_DIR}/model_sl.pt missing. SL phase did not complete."
  exit 1
fi

# Update chain_status to RL phase
"$PYTHON" -c "
import json, time
with open('${CHECKPOINT_DIR}/chain_status.json', 'w') as f:
    json.dump({'stage': 'RL', 'phase': 'starting', 'run_name': '${RUN_NAME}', 'ts': int(time.time())}, f)
"

# RL — always pass --resume so re-runs of this script (e.g. after a config
# change + restart) pick up model_latest.pt instead of starting fresh from
# model_sl.pt and overwriting the RL progress. On the very first launch,
# load_with_fallback() in train_v12.py falls back to model_sl.pt automatically.
TD_LUDO_RUN_NAME="$RUN_NAME" LUDO_BIAS_PENALTIES=1 "$PYTHON" -u train_v12.py \
  --resume \
  --port "$DASHBOARD_PORT" \
  --model-arch v132 \
  --num-res-blocks 10 \
  --num-channels 128 \
  --game-composition "$RL_GAME_COMP" \
  --curriculum-mode "$RL_CURRICULUM" \
  --curriculum-target "$RL_CURRICULUM_TARGET" \
  --entropy-coeff "$RL_ENTROPY"
