#!/bin/bash
# Resume V11.1 RL training detached (survives Claude Code closing + power blips)
#
# Usage from td_ludo/ directory:
#   ./resume_v11_1.sh
#
# After running, monitor with:
#   tail -f logs/v11_1_rl_latest.log         # live log
#   ps -p $(cat logs/v11_1_rl_latest.pid)    # is process alive?
#   open http://localhost:8789/v11_dashboard.html  # web dashboard

set -e

# Make sure we're in the td_ludo directory
cd "$(dirname "$0")"

# Clean any stale lock file
rm -f checkpoints/ac_v11_1/train.pid

# Pick a unique log filename based on timestamp
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/v11_1_rl_${TS}.log"
PIDFILE="logs/v11_1_rl_${TS}.pid"
mkdir -p logs

# Launch detached via Python (start_new_session=True) so it survives
# Claude Code closing, terminal closing, etc. PPID becomes 1 (launchd).
./td_env/bin/python <<PYEOF
import subprocess, os
env = os.environ.copy()
env['TD_LUDO_RUN_NAME'] = 'ac_v11_1'
proc = subprocess.Popen(
    ['./td_env/bin/python', '-u', 'train_v11.py',
     '--resume',
     '--port', '8789',
     '--num-res-blocks', '4',
     '--num-channels', '96',
     '--num-attn-layers', '1',
     '--num-heads', '2',
     '--ffn-ratio', '4',
     '--attn-dim', '64',
     '--ppo-minibatch-size', '128',
    ],
    stdin=subprocess.DEVNULL,
    stdout=open('${LOG}', 'w'),
    stderr=subprocess.STDOUT,
    start_new_session=True,
    env=env,
)
print(f"V11.1 RL resumed, PID: {proc.pid}")
with open('${PIDFILE}', 'w') as f:
    f.write(str(proc.pid))
PYEOF

# Update the "latest" symlinks for easy monitoring
ln -sf "v11_1_rl_${TS}.log" logs/v11_1_rl_latest.log
ln -sf "v11_1_rl_${TS}.pid" logs/v11_1_rl_latest.pid

# Wait briefly and verify it started cleanly
sleep 8
PID=$(cat "${PIDFILE}")
echo
echo "=== Status check ==="
ps -p $PID -o pid,ppid,etime,stat 2>/dev/null && \
  echo "✓ Process alive, fully detached (PPID=1)" || \
  echo "✗ Process not running — check ${LOG}"
echo
echo "=== Resume info ==="
grep -E "Resumed|Safe-load|PPO minibatch|Eval scheduler" "${LOG}" 2>/dev/null

echo
echo "=== Dashboard ==="
echo "  http://localhost:8789/v11_dashboard.html"
echo
echo "=== To stop gracefully ==="
echo "  kill -INT \$(cat logs/v11_1_rl_latest.pid)"
