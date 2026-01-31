#!/bin/bash

# Cleanup Function
cleanup() {
    echo -e "\n\n🔴 Stopping Remote Session..."
    kill $PID_TRAIN $PID_PROXY $PID_NGROK 2>/dev/null
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

echo "🚀 Starting AlphaLudo Remote Session..."
echo "----------------------------------------"
export PYTHONUNBUFFERED=1

# Environment Setup
export PATH="/opt/homebrew/bin:$PATH"
source .venv/bin/activate


# 1. Start Training (Background)
echo "1️⃣  Starting Training Loop..."
# We run run_training.sh in background, piping output to log to keep terminal clean
# Or keeping it visible? User wants to see logs probably.
# But we have multiple processes.
# Let's pipe training log to file or let it print?
# If we let it print, it might mess up our URL display.
# Let's pipe to training.log and tail it?
./run_training.sh > training_remote.log 2>&1 &
PID_TRAIN=$!
echo "   Training PID: $PID_TRAIN (Logs: tail -f training_remote.log)"

# 2. Start Proxy (Background)
echo "2️⃣  Starting Proxy Server (Port 8090)..."
python remote_proxy.py > proxy.log 2>&1 &
PID_PROXY=$!
echo "   Proxy PID: $PID_PROXY"

# 3. Start Ngrok
echo "3️⃣  Starting Ngrok Tunnel..."
ngrok http 8090 --log=stdout > ngrok.log 2>&1 &
PID_NGROK=$!
echo "   Ngrok PID: $PID_NGROK"

# 4. Fetch URL
echo "⏳ Waiting for public URL..."
sleep 5
# Query Ngrok API
# Sometimes needs curl retry
URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])" 2>/dev/null)

echo "----------------------------------------"
if [ -z "$URL" ]; then
    echo "❌ Failed to fetch URL. Check ngrok.log"
    echo "   Ensure 'ngrok' is installed and authenticated."
else
    echo "✅ REMOTE ACCESS LIVE!"
    echo "   📱 Open this URL on your mobile:"
    echo "   👉 $URL"
    echo ""
    echo "   Dashboard: $URL/index.html"
    echo "   Game Stream: $URL/game.html?id=0"
fi
echo "----------------------------------------"
echo "Press Ctrl+C to Stop All."

# Wait forever (or until child dies)
wait $PID_TRAIN
