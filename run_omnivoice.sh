#!/usr/bin/env bash
# ===========================================================================
# Linux: Khoi dong SERVICE OmniVoice (long tieng) - moi truong conda RIENG.
#
# Cong NOI BO (KHONG mo ra internet):
#   - Mac dinh: 7861  ->  chi goi tu localhost (app.py goi qua gradio_client)
#
# Service nap model 1 lan, giu trong VRAM -> tiet kiem thoi gian/chi phi GPU.
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUTF8="${PYTHONUTF8:-1}"

OMNIVOICE_PORT="${OMNIVOICE_PORT:-7861}"
OMNIVOICE_MODEL="${OMNIVOICE_MODEL:-k2-fsa/OmniVoice}"
# Dat OMNIVOICE_NO_ASR=1 neu LUON nhap san Reference Text -> nhe VRAM, khoi dong nhanh
OMNIVOICE_NO_ASR="${OMNIVOICE_NO_ASR:-0}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "[LOI] Khong tim thay conda.sh. Hay 'conda activate omnivoice' thu cong."
    exit 1
fi

conda activate omnivoice

EXTRA_ARGS=()
if [ "$OMNIVOICE_NO_ASR" = "1" ]; then
    EXTRA_ARGS+=(--no-asr)
fi

echo "=========================================="
echo " OmniVoice service (audio.py)"
echo " Cong NOI BO: ${OMNIVOICE_PORT} (chi localhost)"
echo " Doi den khi thay: Model loaded."
echo "=========================================="

exec python audio.py \
    --model "$OMNIVOICE_MODEL" \
    --device cuda \
    --ip 0.0.0.0 \
    --port "$OMNIVOICE_PORT" \
    "${EXTRA_ARGS[@]}"
