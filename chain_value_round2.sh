#!/bin/bash
# Value round 2 (unattended, resumable). Re-running skips finished stages.
#   1. +200k champion self-play games  -> data/selfplay_v152_b (GPU + CPU workers)
#   2. continue the value net from value_v1 on ALL 400k games -> checkpoints/value_v2
#   3. tests vs the champion: (a) 1-ply by the new value (300 games; v1 = 43.0%)
#                             (b) policy-guided search, 400 evals, champion policy explores,
#                                 new value judges, most-visited move (300 games; v1 value era = 48%)
# Pause cleanly: touch stop (repo root).  Resume: ./start_value_round2.sh
cd "$(dirname "${BASH_SOURCE[0]}")"
P=./td_ludo/td_env/bin/python
OUT=checkpoints/value_v2; SP2=data/selfplay_v152_b
mkdir -p "$OUT" "$SP2"
log() { echo "[$(date '+%F %T')] $*" >> "$OUT/chain.log"; }
stopped() { [ -f stop ] && { log "stop file present, exiting"; exit 0; }; }

N=$(ls $SP2/shard_*.bin 2>/dev/null | wc -l | tr -d ' ')
if [ "$N" -lt 40 ]; then
  log "stage 1: self-play into $SP2 ($N/40 shards)"
  rm -f $SP2/*.claim $SP2/*.tmp
  $P -u python/alphaludo/gen_selfplay.py --out $SP2 --games 200000 --concurrency 512 --backend mlx \
      --tag "[gpu] " >> $SP2/console_gpu.log 2>&1 &
  G=$!
  $P -u python/alphaludo/gen_selfplay.py --out $SP2 --games 200000 --concurrency 256 --device cpu \
      --cpu-frac 0 --threads 6 --tag "[cpu] " >> $SP2/console_cpu.log 2>&1 &
  C=$!
  wait $G $C
  stopped
fi
N=$(ls $SP2/shard_*.bin 2>/dev/null | wc -l | tr -d ' ')
[ "$N" -lt 40 ] && { log "self-play incomplete ($N/40), exiting"; exit 1; }

if [ ! -f "$OUT/TRAINED" ]; then
  log "stage 2: value training on data/selfplay_v152 + $SP2 (init value_v1 final)"
  $P -u python/alphaludo/train_value.py --data data/selfplay_v152,$SP2 --out "$OUT" \
      --init checkpoints/value_v1/value_latest.pt --epochs 2 --per-game 20 --micro-batch 256 \
      --lr 5e-5 --min-lr 5e-6 --warmup 300 --device mps >> "$OUT/train.log" 2>&1
  stopped
  grep -q "\[exit\] clean" "$OUT/train.log" && [ ! -f stop ] && touch "$OUT/TRAINED"
fi
[ -f "$OUT/TRAINED" ] || { log "training not finished, exiting"; exit 1; }

if ! grep -q "win rate" "$OUT/test_1ply.log" 2>/dev/null; then
  log "stage 3a: 1-ply by new value vs champion (300 games)"
  $P -u python/alphaludo/net_farm.py h2h --a-kind expecti --max-evals 100 --games 300 --device mps --cpu-frac 0.4 \
      --value-model "$OUT/value_latest.pt" > "$OUT/test_1ply.log" 2>&1
  log "3a done: $(grep 'win rate' "$OUT/test_1ply.log" | tail -1)"
  stopped
fi
if ! grep -q "win rate" "$OUT/test_combined.log" 2>/dev/null; then
  log "stage 3b: policy-guided search (400 evals, by visits) with new value vs champion (300 games)"
  $P -u python/alphaludo/net_farm.py h2h --a-kind expecti --max-evals 400 --by-visits --games 300 --device mps \
      --cpu-frac 0.4 --value-model "$OUT/value_latest.pt" > "$OUT/test_combined.log" 2>&1
  log "3b done: $(grep 'win rate' "$OUT/test_combined.log" | tail -1)"
fi
log "round 2 complete"
