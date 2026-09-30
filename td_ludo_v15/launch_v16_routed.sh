#!/usr/bin/env bash
# V16 ROUTED — two-property connections, routed gradients (a<-RL, b<-aux).
#
# FAITHFUL to launch_v152_gaeterminal_vm.sh (the champion's recipe). Every flag
# below is the champion's, verbatim, EXCEPT:
#     --arch v16 --routed 1 --aux-coeff 1.0
# Same PPO, same GAE, same terminal-only reward, same opponent pool and weights,
# same entropy, same lr, same eval cadence — and literally the same code path
# (train_v15_rich.py + V15RichTrainer; V16RichTrainer only adds two hooks).
# Paths below are the LOCAL equivalents of the VM paths in the original.
#
# DELIBERATE DEVIATIONS from the champion's launcher, none of them semantic:
#   --device mps          (was cuda)  — hardware.
#   --opp-device cpu      (was cuda)  — opponents do batch-1 inference per
#                                       game-turn; v15's own loader comments say
#                                       GPU buys nothing there, and batch-1 on
#                                       MPS is actively pathological.
#   --save-interval-sec 600 (was 120) — owner's call; affects crash recovery
#                                       only, never the science.
# Everything that touches the gradient is unchanged.
#
#
# STAGE 2 OPPONENT POOL — the champion's, verbatim (switched 2026-08-05
# after stage 1 plateaued at ~79% across 5 evals / 40k games).
# STAGE 1 was (owner's call): Expert 25 / Heuristic 20 /
# self 25 — i.e. train_v15_rich.py's OWN default pool minus the neural
# opponent. The strong pool (v13.2 30 / v13.6 35 / Depth2Expectimax 15) is the
# champion's STAGE-2 recipe and is held back until this stage plateaus; their
# weights are left at 0 here so stage 2 is a one-number change, and at 0 the
# checkpoints are not even loaded.
# Ghosts (frozen past selves) are wired but OFF. They exist to stop naive
# self-play drifting into a mutual equilibrium — the exact failure the
# twosignal run hit at 100% self-play. Here self-play is only ~36% of the mix
# with two scripted bots anchoring it, so they are not needed yet. Turn on with
# --opp-weight-ghost if the eval climbs while H2H versus a fixed reference does
# not: that divergence IS the drift signature.
#
# INIT: the champion's own stage-1 RL checkpoint (eval 0.796). v16 loads its
# bulk weights into `a` with `b`=0, so w = a*(1+tanh(0)) = a exactly and the
# network starts NUMERICALLY IDENTICAL to it. train_v15_rich.py asserts this at
# startup and refuses to run if the mapping is wrong — otherwise an init bug
# would later be misread as an effect of routing.
set -uo pipefail
cd /Users/sumit/Github/AlphaLudo/td_ludo_v15
export PYTHONPATH=.:/Users/sumit/Github/AlphaLudo/td_ludo
PYBIN=/Users/sumit/Github/AlphaLudo/td_ludo/td_env/bin/python

RUN=v16_routed
export TD_LUDO_RUN_NAME=$RUN
RUN_DIR=checkpoints/$RUN
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/console.log
BK=/Users/sumit/Github/AlphaLudo/checkpoint_backups
V136=$BK/v136_gaeterminal_2026-06-17/v136_gaeterminal_BEST_eval85.5pct_G200k_2026-06-16.pt
V132=$BK/v132_20260505_230124/model_latest.pt
INIT=checkpoints/h2h_compare/v152_rl_stage1_G261K/model_best.pt

if pgrep -f "run-name $RUN" > /dev/null; then echo "$RUN already running"; exit 0; fi

# Clear a STALE lock before launching. acquire_train_lock() stores a pid and
# tests liveness with kill(0) — but a reboot recycles pids, so after a power cut
# the lock's pid often belongs to some unrelated process and the lock reads as
# LIVE forever. That silently refused two restarts on 2026-08-05 and training
# sat dead for ~40 minutes. Only remove the lock when no matching trainer is
# actually running.
LOCKF="$HOME/.alphaludo_locks/$RUN.lock"
if [[ -f "$LOCKF" ]] && ! pgrep -f "run-name $RUN" > /dev/null; then
  echo "clearing stale lock $LOCKF"
  rm -f "$LOCKF"
fi

RESUME=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then RESUME="--resume"; fi

nohup $PYBIN -u train_v15_rich.py \
  --arch v16 --routed 1 --aux-coeff 1.0 \
  --init "$INIT" \
  --d-model 128 --n-layers 4 --n-heads 4 --ffn-dim 256 --history-len 1 \
  --use-gae --gae-lambda 0.95 --terminal-only --use-shaped-reward 0 \
  --opp-weight-self 20 \
  --opp-weight-expert 0 --opp-weight-heuristic 0 \
  --opp-v132    "$V132" --opp-weight-v132 30 \
  --opp-v135-rl "$V136" --opp-weight-v135-rl 35 \
  --opp-weight-depth2-expectimax 15 \
  --opp-weight-ghost 0 --ghost-interval 5000 --ghost-pool-size 5 \
  --entropy-coeff 0.03 --lr 1e-5 \
  --device mps --opp-device cpu \
  --parallel-games 128 --rollout-workers 6 \
  --eval-interval 10000 --eval-games 2000 \
  --save-interval-sec 600 \
  --port 8801 --run-name $RUN \
  $RESUME >> "$LOG" 2>&1 &
disown
echo "launched $RUN (pid $!)  log: $LOG"
