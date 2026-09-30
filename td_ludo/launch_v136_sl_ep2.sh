#!/usr/bin/env bash
# V13.6 SL — EPOCH 2 on the full 1.5M-game dataset.
# Warm-starts from the 1-epoch weights (checkpoints/v136_sl/model_latest.pt)
# with a FRESH optimizer/step/LR schedule (gentle 1e-4 → 1e-5 cosine), then
# runs one full epoch over the grown 1.5M dataset (~440K steps).
#
# - New out-dir (v136_sl_ep2) so the epoch-1 model stays intact.
# - First launch: --init (weights-only warm-start).
# - Restarts: auto --resume from v136_sl_ep2/model_latest.pt (continues
#   step/optimizer/LR), so power-loss / Mac-restart loses only ~2000 steps.
set -uo pipefail
cd /Users/sumit/Github/AlphaLudo/td_ludo
export PYTHONPATH=.:../td_ludo_v15
PYBIN=./td_env/bin/python

OUT=checkpoints/v136_sl_ep2
INIT=checkpoints/v136_sl/model_latest.pt
mkdir -p "$OUT"

# Restart-safe: if a checkpoint already exists in the new dir, resume it;
# otherwise warm-start from the epoch-1 weights.
if [[ -f "$OUT/model_latest.pt" ]]; then
  MODE="--resume"
  echo "[launcher] resuming epoch-2 from $OUT/model_latest.pt"
else
  MODE="--init $INIT"
  echo "[launcher] warm-starting epoch-2 from $INIT"
fi

$PYBIN -u train_v136_sl.py \
  --shard-dir checkpoints/sl_dataset_v1 \
  --out-dir "$OUT" \
  --epochs 1 --batch-size 256 \
  --lr 1e-4 --lr-end 1e-5 \
  --num-res-blocks 6 --num-channels 96 --head-hidden 64 \
  --num-workers 4 --device mps \
  $MODE \
  >> "$OUT/console.log" 2>&1
