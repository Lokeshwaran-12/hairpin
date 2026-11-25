#!/usr/bin/env bash
# ==============================================================================
# Hairpin Guard AI - Linux & macOS Launch Script
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "          HAIRPIN GUARD AI - TRAFFIC MONITORING SYSTEM                "
echo "======================================================================"
echo ""

# Find Python executable
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[Error] Python not found. Please install Python 3.10+."
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Check Node.js
if ! command -v node &>/dev/null; then
    echo "[Error] Node.js not found. Please install Node.js (v18+)."
    exit 1
fi
echo "Using Node.js: $(node --version)"
echo ""

# Launch Node gateway (it will automatically launch Python backend on port 8000)
export PYTHON="$PYTHON_CMD"
echo "Starting Full-Stack Application..."
echo "- Frontend Dashboard: http://localhost:3000"
echo "- Python AI Backend:  http://127.0.0.1:8000 (Swagger: /docs)"
echo "Press Ctrl+C to stop all services."
echo "======================================================================"
exec node server.js
