#!/usr/bin/env bash
# ===========================================================================
# Cai dat lan dau tren Linux GPU server (headless, chi terminal).
#
# Cach dung:
#   chmod +x setup_linux_gpu.sh
#   ./setup_linux_gpu.sh
#
# Script tao 2 moi truong conda:
#   ste        - pipeline chinh (OCR, xoa phu de, dich, UI)
#   omnivoice  - long tieng OmniVoice (service rieng)
#
# Sau khi chay xong, ban van can:
#   1) Tai models/ (xem LINUX_SERVER.md muc 4)
#   2) copy config-template.yaml -> config.yaml va dien API key
#   3) ./start_linux.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " Setup SubErase-Translate-Embed (Linux GPU)"
echo "=========================================="

# --- Kiem tra GPU ---
if ! command -v nvidia-smi &>/dev/null; then
    echo "[LOI] Khong tim thay nvidia-smi. Can driver NVIDIA + GPU."
    exit 1
fi
echo "[OK] GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Kiem tra conda ---
if ! command -v conda &>/dev/null; then
    echo "[LOI] Chua cai conda. Cai Miniconda truoc:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# --- Kiem tra ffmpeg ---
if ! command -v ffmpeg &>/dev/null; then
    echo "[CANH BAO] Chua co ffmpeg. Cai: sudo apt install -y ffmpeg imagemagick"
fi

# --- Phat hien doi card (sm_120 = 50-series) ---
ARCH_LIST=""
if python3 -c "import torch; print(torch.cuda.get_arch_list())" 2>/dev/null; then
    ARCH_LIST="$(python3 -c "import torch; print(torch.cuda.get_arch_list())" 2>/dev/null || true)"
fi
USE_CASE_B=0
if echo "$ARCH_LIST" | grep -q "sm_120"; then
    USE_CASE_B=1
    echo "[INFO] Phat hien sm_120 (card 50-series) -> CASE B"
else
    echo "[INFO] Dung CASE A (card 30/40-series hoac chua co torch)"
fi

# ===========================================================================
# Moi truong 1: ste (pipeline chinh)
# ===========================================================================
echo ""
echo ">>> [1/2] Tao moi truong conda 'ste' (Python 3.10)..."
if conda env list | grep -qE '^ste\s'; then
    echo "      Moi truong 'ste' da ton tai, bo qua conda create."
else
    conda create -n ste python=3.10 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ste

echo ">>> Cai PyTorch + PaddlePaddle (GPU)..."
if [ "$USE_CASE_B" -eq 1 ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ \
        || echo "[CANH BAO] Paddle cu129 that bai - xem LINUX_SERVER.md CASE B"
else
    pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install paddlepaddle-gpu==2.6.1.post120 \
        -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
fi

echo ">>> Cai requirements chinh + UI..."
pip install -r requirements.txt
pip install -r requirements-ui.txt

conda deactivate

# ===========================================================================
# Moi truong 2: omnivoice (long tieng)
# ===========================================================================
echo ""
echo ">>> [2/2] Tao moi truong conda 'omnivoice' (Python 3.12)..."
if conda env list | grep -qE '^omnivoice\s'; then
    echo "      Moi truong 'omnivoice' da ton tai, bo qua conda create."
else
    conda create -n omnivoice python=3.12 -y
fi

conda activate omnivoice

echo ">>> Cai torch cu128 + OmniVoice..."
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-omnivoice.txt

conda deactivate

# ===========================================================================
# Config + quyen thuc thi script
# ===========================================================================
if [ ! -f config.yaml ]; then
    cp config-template.yaml config.yaml
    echo "[OK] Da tao config.yaml tu template. Hay dien API key dich."
fi

chmod +x run_ui.sh run_omnivoice.sh start_linux.sh stop_linux.sh setup_linux_gpu.sh 2>/dev/null || true

echo ""
echo "=========================================="
echo " SETUP XONG"
echo "=========================================="
echo " Buoc tiep theo:"
echo "   1. Tai models/ (OCR + sttn.pth) - xem LINUX_SERVER.md muc 4"
echo "   2. Sua config.yaml (API key dich)"
echo "   3. chmod +x *.sh && ./start_linux.sh"
echo ""
echo " CONG PUBLIC (giao dien web): 7860"
echo " CONG NOI BO (OmniVoice):      7861 (khong mo firewall)"
echo " Doc chi tiet: LINUX_SERVER.md"
echo "=========================================="
