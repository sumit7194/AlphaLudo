#!/bin/bash
# V17 2P+4P chain (runs detached; started by start_v17_4p.sh). Same network as V17,
# warm-started from the frozen 2P model:
#   SL  stage A: 85% 4P teacher positions + 15% 2P (MCTS-3000) anchor, until the 4P
#       validation metrics plateau; stage B: 50/50 with LR drops and auto-stop.
#   RL  50/50 2P/4P training decisions from the start (2026-10-08: after SL the model already
#       plays 4P at teacher level, and 15% 2P let 2P strength slip), 4P ladder starting at depth2;
#       infinite. Per-run overrides: <run>/rl_overrides.json.
# Resumable at every point; `touch <run>/stop` pauses.
cd "$(dirname "${BASH_SOURCE[0]}")"
RUN="${RUN:-checkpoints/v17_4p}"
P=./td_ludo/td_env/bin/python
mkdir -p "$RUN"
log() { echo "[$(date '+%F %T')] $*" >> "$RUN/console.log"; }
paused() { [ -f "$RUN/stop" ] || [ -f stop ]; }

if [ ! -f "$RUN/init_2p.pt" ]; then
  log "missing $RUN/init_2p.pt (start with ./start_v17_4p.sh)"; exit 1
fi

if [ ! -f "$RUN/sl_final.pt" ]; then
  log "stage 1: SL 2P+4P (4P share ${SL_STAGES:-0.85,0.5}; 2P teacher MCTS-${SIMS:-3000}; ${SL_THREADS:-10} threads)"
  $P -u -m v17.sl_train --run-dir "$RUN" --threads "${SL_THREADS:-10}" --sims "${SIMS:-3000}" \
      --device "${DEVICE:-mps}" --init "$RUN/init_2p.pt" --frac4-stages "${SL_STAGES:-0.85,0.5}" \
      --lr "${SL_LR:-1e-4}" --warmup-steps 1000 --min-games 0 \
      --min-positions "${SL_MIN_POSITIONS:-6000000}" --max-positions "${SL_MAX_POSITIONS:-40000000}" \
      --min-stage-positions "${SL_MIN_STAGE_POSITIONS:-4000000}" ${SL_ARGS:-} >> "$RUN/console.log" 2>&1
  paused && { log "paused during SL"; exit 0; }
fi
[ -f "$RUN/sl_final.pt" ] || { log "SL did not finish (see log); not starting RL"; exit 1; }
[ -n "${SL_ONLY:-}" ] && { log "SL finished; SL_ONLY set, not starting RL"; exit 0; }

log "stage 2: RL 2P+4P (infinite; 4P share ${RL_STAGES:-0.5})"
$P -u -m v17.rl_train --run-dir "$RUN" --concurrency "${CONCURRENCY:-256}" --device "${DEVICE:-mps}" \
    --frac4-stages "${RL_STAGES:-0.5}" --start-rung 3 --start-rung4 2 ${RL_ARGS:-} >> "$RUN/console.log" 2>&1
paused && log "paused during RL" || log "RL exited (see log)"
