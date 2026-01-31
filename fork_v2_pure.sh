#!/bin/bash
# fork_v2_pure.sh - Fork to mastery_v2_pure experiment with Expecti-MCTS
# This script:
# 1. Stops any running training
# 2. Copies model weights from v1 to v2
# 3. Clears the replay buffer (new config requires fresh data)
# 4. Starts training with the new experiment name

set -e

echo "=== AlphaLudo: Forking to mastery_v2_pure ==="

# 1. Stop existing training
echo ">>> Stopping any existing training..."
pkill -f train_async.py 2>/dev/null || true
pkill -f remote_proxy.py 2>/dev/null || true
sleep 2

# 2. Setup directories
SOURCE_DIR="checkpoints_mastery/mastery_v1"
TARGET_DIR="checkpoints_mastery/mastery_v2_pure"

echo ">>> Creating target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR/ghosts"

# 3. Copy model weights (preserve trained intelligence)
if [ -f "$SOURCE_DIR/model_latest.pt" ]; then
    echo ">>> Copying model weights..."
    cp "$SOURCE_DIR/model_latest.pt" "$TARGET_DIR/"
    echo "    ✓ Model copied"
else
    echo "    ⚠ No model found in $SOURCE_DIR, starting fresh"
fi

# 4. Copy Elo ratings (preserve ranking history)
if [ -f "$SOURCE_DIR/elo_ratings.json" ]; then
    echo ">>> Copying Elo ratings..."
    cp "$SOURCE_DIR/elo_ratings.json" "$TARGET_DIR/"
    echo "    ✓ Elo ratings copied"
fi

# 5. DO NOT copy replay buffer (old data incompatible with new MCTS)
echo ">>> Clearing replay buffer (required for Expecti-MCTS)..."
rm -f "$TARGET_DIR/replay_buffer.pkl"
echo "    ✓ Buffer cleared"

# 6. Reset stats
echo ">>> Resetting training stats..."
echo '{"total_games": 0, "fps": 0.0, "buffer_size": 0, "iteration": 0}' > "$TARGET_DIR/wc_stats.json"
echo "    ✓ Stats reset"

# 7. Set environment and start training
echo ">>> Setting RUN_NAME=mastery_v2_pure"
export ALPHALUDO_RUN_NAME=mastery_v2_pure

echo ""
echo "=== Fork Complete ==="
echo "New experiment: $TARGET_DIR"
echo "MCTS Sims: 1000 (5x increase)"
echo "Expecti-MCTS: Enabled (Chance Node averaging)"
echo "Float16: Enabled (2x speedup)"
echo ""
echo ">>> Starting training..."
./run_training.sh
