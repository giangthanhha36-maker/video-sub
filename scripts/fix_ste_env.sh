#!/usr/bin/env bash
# ===========================================================================
# Sua moi truong ste da cai san (khong tao lai env).
# Dung khi gap loi: cuDNN, Gradio, huggingface_hub, pydantic, Cython...
#
#   chmod +x scripts/fix_ste_env.sh
#   ./scripts/fix_ste_env.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ste

echo ">>> [1/5] Sua setuptools + Cython (neu bi hong)..."
pip cache purge 2>/dev/null || true
pip install --no-cache-dir "setuptools>=69,<76" "Cython>=3.0,<4"
# Paddle 2.6 + run_check() chua tuong thich numpy 2.x.
pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

echo ">>> [2/5] Cai NVIDIA cuDNN/cuBLAS (torch 2.5 cu121 can cudnn == 9.1.0.70)..."
pip install --no-cache-dir "nvidia-cudnn-cu12==9.1.0.70" nvidia-cublas-cu12

echo ">>> [3/5] Sua phien ban Gradio/UI dependencies..."
pip install --no-cache-dir -r requirements-ui.txt
pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/ste_gpu_env.sh"
echo ">>> [4/5] Link cuDNN + kiem tra torch + paddle GPU..."
ste_gpu_prepare_env
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ! ste_gpu_verify_paddle; then
    echo "[LOI] Kiem tra Paddle GPU that bai."
    python -c "import numpy; print('numpy', numpy.__version__)"
    exit 1
fi
echo ">>> [5/5] Kiem tra Gradio UI..."
python -c "import gradio; print('gradio', gradio.__version__, 'OK')"

conda deactivate

echo ""
echo "=========================================="
echo " FIX XONG — khoi dong lai dich vu:"
echo "   ./stop_linux.sh && ./start_linux.sh --share"
echo "=========================================="
