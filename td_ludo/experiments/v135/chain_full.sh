#!/usr/bin/env bash
# V13.5 full-size chain: SL → backup → H2H vs V13.2_latest.
# Mirrors the V13.4 chain shape but for the V13.5 token-symmetric arch.

set -euo pipefail
cd "$(dirname "$0")/../.."  # → td_ludo/

PY=/Users/sumit/Github/AlphaLudo-MechInterp/.venv/bin/python

RUN_NAME="${RUN_NAME:-v135_full}"
CKPT_DIR="checkpoints/${RUN_NAME}"
TEACHER="checkpoints/v132/model_latest.pt"

NUM_RES_BLOCKS=10
NUM_CHANNELS=128
TARGET_STATES=5000000
BATCH=256
EVAL_EVERY=1000000
EVAL_GAMES=200
SAVE_EVERY=500000
H2H_GAMES=500

mkdir -p "$CKPT_DIR"
LOG="$CKPT_DIR/chain.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date +%T)] === V13.5 full-size chain start ==="
echo "  RUN_NAME=$RUN_NAME"
echo "  arch: V135Symmetric ${NUM_RES_BLOCKS}×${NUM_CHANNELS} (~3M params, V13.2-matched)"
echo "  perm-augment: OFF (POC ablation showed -1.6pp H2H penalty for ON)"
echo "  budget: SL=${TARGET_STATES} states · eval=${EVAL_EVERY}/${EVAL_GAMES}g · H2H=${H2H_GAMES}g"

#######################################
# Phase 1: SL distillation
#######################################
echo
echo "[$(date +%T)] Phase 1: SL distillation (V13.2 teacher → V13.5 student)..."
$PY train_v135_sl.py \
    --teacher "$TEACHER" \
    --run-name "$RUN_NAME" \
    --target-states $TARGET_STATES \
    --batch-size $BATCH \
    --num-res-blocks $NUM_RES_BLOCKS --num-channels $NUM_CHANNELS \
    --no-perm-augment \
    --eval-every $EVAL_EVERY --eval-games $EVAL_GAMES \
    --save-every $SAVE_EVERY \
    --log-every 20 \
    --device mps \
    --port 8798
SL_RC=$?
echo "[$(date +%T)] SL exited with rc=$SL_RC"
[[ $SL_RC -ne 0 ]] && { echo "CHAIN_ERROR: SL failed"; touch "$CKPT_DIR/CHAIN_ERROR"; exit 1; }

#######################################
# Phase 2: Backup
#######################################
TS=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="../checkpoint_backups/${RUN_NAME}_${TS}"
mkdir -p "$BACKUP_DIR"
cp "$CKPT_DIR/model_sl.pt" "$BACKUP_DIR/" 2>/dev/null || true
cp "$CKPT_DIR/model_latest.pt" "$BACKUP_DIR/" 2>/dev/null || true
cp "$CKPT_DIR/sl_stats.json" "$BACKUP_DIR/" 2>/dev/null || true
cp "$CKPT_DIR/sl.log" "$BACKUP_DIR/" 2>/dev/null || true
echo "[$(date +%T)] Backed up to $BACKUP_DIR"

#######################################
# Phase 3: H2H vs V13.2_latest
#######################################
echo
echo "[$(date +%T)] Phase 3: H2H vs V13.2_latest (${H2H_GAMES} games, mirrored seeds, greedy)..."
$PY experiments/v135/h2h_v135.py \
    --v132 "$TEACHER" \
    --v135 "$CKPT_DIR/model_latest.pt" \
    --num-res-blocks $NUM_RES_BLOCKS --num-channels $NUM_CHANNELS \
    --games $H2H_GAMES \
    --device mps \
    | tee "$CKPT_DIR/h2h_results.txt"
H2H_RC=$?
echo "[$(date +%T)] H2H exited with rc=$H2H_RC"

#######################################
# Done
#######################################
touch "$CKPT_DIR/CHAIN_DONE"
echo
echo "[$(date +%T)] === V13.5 full-size chain DONE ==="
echo "  SL log:        $CKPT_DIR/sl.log"
echo "  H2H result:    $CKPT_DIR/h2h_results.txt"
echo "  Backup:        $BACKUP_DIR"
echo "  Flag:          $CKPT_DIR/CHAIN_DONE"
