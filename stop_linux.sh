#!/usr/bin/env bash
# Dung cac service da khoi dong boi start_linux.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

stop_pid_file() {
    local name="$1"
    local pid_file="$PID_DIR/${name}.pid"
    if [ -f "$pid_file" ]; then
        local pid
        pid="$(cat "$pid_file")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "Dung $name (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

stop_pid_file "ui"
stop_pid_file "omnivoice"

# Don them process con python neu con treo (tuy chon, chi trong thu muc project)
pkill -f "python app.py" 2>/dev/null || true
pkill -f "python audio.py" 2>/dev/null || true

echo "Da dung cac service."
