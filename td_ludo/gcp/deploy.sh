#!/usr/bin/env bash
# deploy.sh - Create a deployment tarball of AlphaLudo source (run on Mac)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TARBALL="alphaludo_deploy.tar.gz"

echo "============================================"
echo "  AlphaLudo Deployment Packager"
echo "============================================"
echo ""

# Create a temp staging directory with td_ludo/ prefix
STAGING=$(mktemp -d)
mkdir -p "$STAGING/td_ludo/src" "$STAGING/td_ludo/gcp"

# Copy source files
cp "$PROJECT_DIR/train.py" "$STAGING/td_ludo/"
cp "$PROJECT_DIR/train_v9_fast.py" "$STAGING/td_ludo/"
cp "$PROJECT_DIR/setup.py" "$STAGING/td_ludo/"
cp "$PROJECT_DIR/pyproject.toml" "$STAGING/td_ludo/"
cp "$PROJECT_DIR/requirements.txt" "$STAGING/td_ludo/"
cp "$PROJECT_DIR/index.html" "$STAGING/td_ludo/"

# Copy Python source
for f in __init__.py config.py elo_tracker.py fast_actor.py fast_learner.py \
         game_db.py game_player.py game_player_v9.py heuristic_bot.py \
         model.py model_v9.py reward_shaping.py state_encoder_1d.py \
         tensor_utils.py trainer.py trainer_v9.py training_utils.py; do
    [ -f "$PROJECT_DIR/src/$f" ] && cp "$PROJECT_DIR/src/$f" "$STAGING/td_ludo/src/"
done

# Copy C++ source
for f in game.cpp game.h mcts.cpp mcts.h bindings.cpp; do
    [ -f "$PROJECT_DIR/src/$f" ] && cp "$PROJECT_DIR/src/$f" "$STAGING/td_ludo/src/"
done

# Copy GCP scripts
cp "$PROJECT_DIR/gcp/setup_vm.sh" "$STAGING/td_ludo/gcp/"
cp "$PROJECT_DIR/gcp/start_v6.sh" "$STAGING/td_ludo/gcp/"
cp "$PROJECT_DIR/gcp/start_v9.sh" "$STAGING/td_ludo/gcp/"

# Create tarball
cd "$STAGING"
tar czf "$PROJECT_DIR/$TARBALL" td_ludo/
rm -rf "$STAGING"

SIZE=$(du -h "$PROJECT_DIR/$TARBALL" | cut -f1)
echo "Created: $PROJECT_DIR/$TARBALL ($SIZE)"
echo ""
echo "============================================"
echo "  Upload & setup instructions:"
echo "============================================"
echo ""
echo "  1. Upload to your GCP VM:"
echo "     gcloud compute scp td_ludo/$TARBALL VM_NAME:~ --zone=us-central1-a"
echo ""
echo "  2. SSH into the VM:"
echo "     gcloud compute ssh VM_NAME --zone=us-central1-a"
echo ""
echo "  3. Extract and set up:"
echo "     cd ~ && tar xzf $TARBALL"
echo "     bash td_ludo/gcp/setup_vm.sh"
echo ""
echo "  4. Upload checkpoint (run from td_ludo/ on Mac):"
echo "     gcloud compute scp checkpoints/ac_v6_big/model_latest.pt VM_NAME:~/td_ludo/checkpoints/ac_v6_big/ --zone=us-central1-a"
echo ""
echo "  5. Start training:"
echo "     cd ~/td_ludo && bash gcp/start_v6.sh"
echo ""
