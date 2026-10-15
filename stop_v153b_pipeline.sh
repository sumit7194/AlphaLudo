#!/bin/bash
# Graceful stop for the V15.3b pipeline: generator finishes its current shard,
# trainer saves model_latest.pt and exits. Re-run launch_v153b_pipeline.sh to resume.
cd "$(dirname "${BASH_SOURCE[0]}")"
CKPT="${CKPT:-checkpoints/v15_3b}"
mkdir -p "$CKPT"; touch "$CKPT/stop"
pkill -f "generator_monitor.py" 2>/dev/null || true
echo "🛑 $CKPT/stop created (generator + trainer exit cleanly)."
