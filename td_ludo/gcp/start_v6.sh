#!/usr/bin/env bash
# start_v6.sh - Start V6 TD-learning training inside a screen session
set -euo pipefail

cd "$HOME/td_ludo"
source venv/bin/activate

SESSION="train_v6"

if screen -list | grep -q "$SESSION"; then
    echo "Training session '$SESSION' is already running."
    echo "  Attach: screen -r $SESSION"
    exit 0
fi

echo "Starting V6 training in screen session '$SESSION'..."
screen -dmS "$SESSION" bash -c "
    cd $HOME/td_ludo && \
    source venv/bin/activate && \
    python train.py --resume 2>&1 | tee train_v6.log
"

echo "Training started."
echo "  Attach:  screen -r $SESSION"
echo "  Detach:  Ctrl-A, D"
echo "  Log:     tail -f ~/td_ludo/train_v6.log"
