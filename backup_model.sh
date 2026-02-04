#!/bin/bash
# Backup latest model and stats to Git (Bypassing .gitignore)

echo ">>> Backing up Model & Stats..."

# Define paths
CKPT_DIR="checkpoints_mastery/mastery_v3_prod"

if [ ! -d "$CKPT_DIR" ]; then
    echo "❌ Error: Checkpoint directory $CKPT_DIR not found!"
    exit 1
fi

# Add specific files (FORCE add to override .gitignore)
echo "Adding model_latest.pt..."
git add -f "$CKPT_DIR/model_latest.pt"

echo "Adding stats files..."
git add -f "$CKPT_DIR/elo_ratings.json"
git add -f "$CKPT_DIR/wc_stats.json"

# Commit
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "Committing..."
git commit -m "Backup: Model & Stats at $TIMESTAMP"

# Push
echo ">>> Pushing to origin..."
git push origin main

echo "✅ Backup Complete!"
