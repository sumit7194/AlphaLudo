#!/bin/bash
# Unattended value phase: wait for self-play generation -> train value net ->
# decisive test (1-ply by the NEW value vs the champion policy).
# Logs: checkpoints/value_v1/chain.log. Stop anything: touch stop (repo root).
cd "$(dirname "${BASH_SOURCE[0]}")"
P=./td_ludo/td_env/bin/python
OUT=checkpoints/value_v1; mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" >> "$OUT/chain.log"; }
log "waiting for gen_selfplay to finish"
while pgrep -f "gen_selfplay.py" > /dev/null; do sleep 60; done
[ -f stop ] && { log "stop file present, exiting"; exit 0; }
log "generation done: $(ls data/selfplay_v152/shard_*.bin | wc -l) shards; training value net"
$P -u python/alphaludo/train_value.py --data data/selfplay_v152 --out "$OUT" --epochs 2 --per-game 20 --micro-batch 256 \
    --device mps >> "$OUT/train.log" 2>&1
[ -f stop ] && { log "stop file present after training, exiting"; exit 0; }
log "training done; decisive test (1-ply by new value vs champion, 300 games)"
$P -u python/alphaludo/net_farm.py h2h --a-kind expecti --max-evals 100 --games 300 --device mps --cpu-frac 0.4 \
    --value-model "$OUT/value_best.pt" >> "$OUT/test.log" 2>&1
log "test done: $(grep 'win rate' "$OUT/test.log" | tail -1)"
