#!/usr/bin/env bash
# V13.6 RL — FRESH restart on Mac from SL (2026-05-30). Moved off VM (the
# VM run got stuck at ~68% eval after 295K games of broken-normalization
# training drove it into a bad basin; resuming + fixing couldn't recover
# it). Fresh-from-SL with ALL fixes baked in:
#   - return + advantage normalization (champion recipe)
#   - GPM display fix
# Stage A opponents (gentle, user-chosen):
#   Self 50 + Ghost 25 + Expectimax 10 + Aggressive 5 + Heuristic 5 + Expert 5
# dense reward · KL anchor 0.2 · entropy 0.01 · eval 15K×2K · device mps.
set -uo pipefail
cd /Users/sumit/Github/AlphaLudo/td_ludo
export PYTHONPATH=.:../td_ludo_v15
# Champion reward recipe: bias penalties on (laggard + danger-advance).
export LUDO_BIAS_PENALTIES=1
PYBIN=./td_env/bin/python
export TD_LUDO_RUN_NAME=v136_rl_local
RUN_DIR=/Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/v136_rl_local
INIT=$RUN_DIR/model_sl.pt
# Canonical SL final to seed a truly-fresh run dir from. The v136_sl SL trainer
# exited before writing its own model_sl.pt, so its model_latest.pt IS the final
# (byte-identical to the copy already promoted into $INIT). Keep these in sync.
SL_SRC=/Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/v136_sl/model_sl.pt
LOG=$RUN_DIR/console.log
mkdir -p $RUN_DIR

if [[ -f "$RUN_DIR/model_latest.pt" ]]; then
  echo "[launcher] resuming from model_latest.pt" >> "$LOG"
  MODE_FLAG="--resume"
else
  # Fresh start: --init needs $INIT to exist or train_v135_rl.py crashes on
  # torch.load (FileNotFoundError, no guard). Seed it from the SL dir if the
  # run dir was wiped.
  if [[ ! -f "$INIT" ]]; then
    if [[ -f "$SL_SRC" ]]; then
      cp "$SL_SRC" "$INIT"
      echo "[launcher] seeded model_sl.pt from $SL_SRC" >> "$LOG"
    else
      echo "[launcher] FATAL: no init checkpoint — neither $INIT nor $SL_SRC exists" | tee -a "$LOG"
      exit 1
    fi
  fi
  echo "[launcher] FRESH init from SL $INIT" >> "$LOG"
  MODE_FLAG="--init $INIT"
fi

TD_LUDO_RUN_NAME=v136_rl_local $PYBIN -u train_v135_rl.py $MODE_FLAG \
  --kl-teacher $INIT \
  --kl-anchor-coeff 0.2 \
  --use-shaped-reward 1 \
  --num-res-blocks 6 --num-channels 96 --head-hidden 64 \
  --opp-weight-self 50 \
  --opp-weight-ghost 25 \
  --opp-weight-expectimax 15 \
  --opp-weight-minimax-expectimax 5 \
  --opp-weight-expert 5 \
  --ghost-save-interval 5000 --max-ghosts 8 \
  --target-states 200000000 \
  --parallel-games 32 --train-chunk 1024 --minibatch-size 256 --train-epochs 2 \
  --lr 1e-5 --lr-end 5e-6 --entropy-coeff 0.01 --value-coeff 0.5 \
  --eval-every-games 15000 --eval-games 2000 --save-every-games 5000 \
  --log-every 10 --temperature 1.1 \
  --device mps --port 8811 \
  >> "$LOG" 2>&1
