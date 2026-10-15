#!/bin/bash
# V17 chain (runs detached; started by start_v17.sh): SL stage until its auto-stop,
# then the infinite RL stage. Resumable at every point; `touch <run>/stop` pauses.
cd "$(dirname "${BASH_SOURCE[0]}")"
RUN="${RUN:-checkpoints/v17}"
P=./td_ludo/td_env/bin/python
mkdir -p "$RUN"
log() { echo "[$(date '+%F %T')] $*" >> "$RUN/console.log"; }
paused() { [ -f "$RUN/stop" ] || [ -f stop ]; }

if [ ! -f "$RUN/sl_final.pt" ]; then
  log "stage 1: SL (teacher MCTS-${SIMS:-3000} self-play, ${SL_THREADS:-10} threads)"
  $P -u -m v17.sl_train --run-dir "$RUN" --threads "${SL_THREADS:-10}" --sims "${SIMS:-3000}" \
      --device "${DEVICE:-mps}" ${SL_ARGS:-} >> "$RUN/console.log" 2>&1
  paused && { log "paused during SL"; exit 0; }
fi
[ -f "$RUN/sl_final.pt" ] || { log "SL did not finish (see log); not starting RL"; exit 1; }
[ -n "${SL_ONLY:-}" ] && { log "SL finished; SL_ONLY set, not starting RL"; exit 0; }

log "stage 2: RL (infinite)"
$P -u -m v17.rl_train --run-dir "$RUN" --concurrency "${CONCURRENCY:-256}" --device "${DEVICE:-mps}" \
    ${RL_ARGS:-} >> "$RUN/console.log" 2>&1
paused && log "paused during RL" || log "RL exited (see log)"
