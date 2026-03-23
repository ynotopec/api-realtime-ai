#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
source .env 2>/dev/null || true

# Activate venv
source ~/venv/external-test/bin/activate

echo "🚀 Starting external-test services..."

# Kill existing processes
if [ -f .run/proxy.pid ]; then
    kill $(cat .run/proxy.pid) 2>/dev/null || true
    sleep 1
fi
if [ -f .run/static.pid ]; then
    kill $(cat .run/static.pid) 2>/dev/null || true
    sleep 1
fi
pkill -f "python proxy.py" 2>/dev/null || true
pkill -f "http.server 5173" 2>/dev/null || true

# Start proxy server
nohup python proxy.py > .run/proxy.log 2>&1 &
echo $! > .run/proxy.pid
sleep 2

# Start static file server
nohup python -m http.server 5173 --bind 127.0.0.1 > .run/static.log 2>&1 &
echo $! > .run/static.pid
sleep 1

echo "✅ Services started:"
echo "   - Proxy:    http://127.0.0.1:8000 (/docs)"
echo "   - Static:   http://127.0.0.1:5173"
echo ""
echo "📊 Monitor logs: tail -f .run/*.log"
echo "🔋 To stop:     ./stop.sh"
