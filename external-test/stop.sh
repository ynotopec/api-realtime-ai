#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🛑 Stopping external-test services..."

# Kill proxy
if [ -f .run/proxy.pid ]; then
    kill $(cat .run/proxy.pid) 2>/dev/null || true
    rm .run/proxy.pid
fi

# Kill static server
if [ -f .run/static.pid ]; then
    kill $(cat .run/static.pid) 2>/dev/null || true
    rm .run/static.pid
fi

# Force cleanup
pkill -f "python proxy.py" 2>/dev/null || true
pkill -f "http.server 5173" 2>/dev/null || true

echo "✅ All services stopped."
