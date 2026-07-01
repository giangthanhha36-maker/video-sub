#!/usr/bin/env bash
# ===========================================================================
# Linux headless: khoi dong CA HAI service (OmniVoice + UI) trong background.
#
#   ./start_linux.sh          # khoi dong, ghi log vao thu muc logs/
#   ./start_linux.sh --share  # UI + link public gradio.live (GRADIO_SHARE=1)
#   ./stop_linux.sh           # dung tat ca
#
# CONG CAN BIET:
#   7860  -> GIAO DIEN CHINH (PUBLIC) - ban truy cap URL nay
#   7861  -> OmniVoice API (NOI BO) - KHONG mo firewall cho cong nay
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs
PID_DIR="$SCRIPT_DIR/.pids"
mkdir -p "$PID_DIR"

SHARE_FLAG="${1:-}"

if [ -f "$PID_DIR/omnivoice.pid" ] && kill -0 "$(cat "$PID_DIR/omnivoice.pid")" 2>/dev/null; then
    echo "[CANH BAO] OmniVoice da chay (PID $(cat "$PID_DIR/omnivoice.pid")). Bo qua khoi dong lai."
else
    echo "[1/2] Khoi dong OmniVoice (cong 7861, noi bo)..."
    nohup env PYTHONUTF8=1 OMNIVOICE_PORT=7861 bash "$SCRIPT_DIR/run_omnivoice.sh" \
        > "$SCRIPT_DIR/logs/omnivoice.log" 2>&1 &
    echo $! > "$PID_DIR/omnivoice.pid"
    echo "      PID: $(cat "$PID_DIR/omnivoice.pid") | log: logs/omnivoice.log"
    echo "      Doi model nap (~1-3 phut). Theo doi: tail -f logs/omnivoice.log"
    echo "      Tim dong 'Model loaded.' de biet san sang."
fi

sleep 2

if [ -f "$PID_DIR/ui.pid" ] && kill -0 "$(cat "$PID_DIR/ui.pid")" 2>/dev/null; then
    echo "[CANH BAO] UI da chay (PID $(cat "$PID_DIR/ui.pid")). Bo qua khoi dong lai."
else
    echo "[2/2] Khoi dong giao dien chinh (cong 7860, PUBLIC)..."
    UI_ENV="PYTHONUTF8=1 GRADIO_PORT=7860"
    if [ "$SHARE_FLAG" = "--share" ]; then
        UI_ENV="$UI_ENV GRADIO_SHARE=1"
        echo "      GRADIO_SHARE=1 -> se co link *.gradio.live trong log"
    fi
    # shellcheck disable=SC2086
    nohup env $UI_ENV bash "$SCRIPT_DIR/run_ui.sh" \
        > "$SCRIPT_DIR/logs/ui.log" 2>&1 &
    echo $! > "$PID_DIR/ui.pid"
    echo "      PID: $(cat "$PID_DIR/ui.pid") | log: logs/ui.log"
fi

# Lay IP noi bo de hien thi
SERVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || echo '<IP-server>')"

echo ""
echo "=========================================="
echo " DA KHOI DONG (background)"
echo "=========================================="
echo " Giao dien chinh (PUBLIC):  http://${SERVER_IP}:7860"
echo " OmniVoice API (noi bo):    http://127.0.0.1:7861"
echo ""
echo " Neu chua mo cong 7860 tren firewall:"
echo "   sudo ufw allow 7860/tcp"
echo ""
echo " Hoac dung SSH tunnel tu may ca nhan:"
echo "   ssh -L 7860:localhost:7860 user@${SERVER_IP}"
echo "   roi mo: http://localhost:7860"
echo ""
echo " Link gradio.live (neu dung --share): xem trong logs/ui.log"
echo " Theo doi log:"
echo "   tail -f logs/ui.log"
echo "   tail -f logs/omnivoice.log"
echo " Dung dich vu: ./stop_linux.sh"
echo "=========================================="
