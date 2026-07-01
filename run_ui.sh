#!/usr/bin/env bash
# ===========================================================================
# Linux: Khoi dong GIAO DIEN WEB chinh (app.py) tren server KHONG co man hinh.
#
# Cong PUBLIC (mo ra internet hoac SSH tunnel):
#   - Mac dinh: 7860  ->  http://<IP-server>:7860
#
# Bien moi truong huu ich:
#   GRADIO_PORT=7860     # doi cong (mac dinh 7860)
#   GRADIO_SHARE=1         # tao link public *.gradio.live (khong can mo firewall)
#   PYTHONUTF8=1           # doc file tieng Trung/Viet on dinh
#
# Truoc khi chay: service OmniVoice phai dang chay (./run_omnivoice.sh hoac start_linux.sh)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUTF8="${PYTHONUTF8:-1}"
export GRADIO_PORT="${GRADIO_PORT:-7860}"
# Khong tu mo trinh duyet (server headless khong co GUI)
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"

# Kich hoat conda (tu dong tim duong dan pho bien)
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
    echo "[LOI] Khong tim thay conda.sh. Hay 'conda activate ste' thu cong roi chay: python app.py"
    exit 1
fi

conda activate ste

# Pipeline chinh (Paddle OCR + STTN) uu tien GPU 0 khi server co nhieu GPU.
if [ -z "${STE_CUDA_DEVICE:-}" ] && command -v nvidia-smi &>/dev/null; then
    STE_CUDA_DEVICE=0
fi
if [ -n "${STE_CUDA_DEVICE:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${STE_CUDA_DEVICE}"
fi

# Paddle GPU (OCR): link libcudnn + LD_LIBRARY_PATH (Docker/Linux headless).
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/ste_gpu_env.sh"
ste_gpu_prepare_env

echo "=========================================="
echo " SubErase UI (app.py)"
echo " Cong: ${GRADIO_PORT}"
echo " Truy cap: http://<IP-server>:${GRADIO_PORT}"
if [ "${GRADIO_SHARE:-0}" = "1" ]; then
    echo " GRADIO_SHARE=1 -> se co them link *.gradio.live"
fi
echo "=========================================="

exec python app.py
