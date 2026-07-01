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

echo ">>> [1/4] Sua setuptools + Cython (neu bi hong)..."
pip cache purge 2>/dev/null || true
pip install --no-cache-dir "setuptools>=69,<76" "Cython>=3.0,<4"

echo ">>> [2/4] Sua NVIDIA cuDNN/cuBLAS cho Paddle..."
PADDLE_VER="$(python -c "import paddle; print(paddle.__version__)" 2>/dev/null || echo "0")"
if echo "$PADDLE_VER" | grep -qE '^3\.'; then
    echo "      Paddle ${PADDLE_VER} (CASE B) -> cuDNN 9.x"
    pip install --no-cache-dir nvidia-cudnn-cu12 nvidia-cublas-cu12
else
    echo "      Paddle ${PADDLE_VER} (CASE A) -> cuDNN 8.9"
    pip install --no-cache-dir "nvidia-cudnn-cu12==8.9.7.29" nvidia-cublas-cu12 \
        || pip install --no-cache-dir nvidia-cudnn-cu12 nvidia-cublas-cu12
fi

echo ">>> [3/4] Sua phien ban Gradio/UI dependencies..."
pip install --no-cache-dir -r requirements-ui.txt

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/ste_gpu_env.sh"
echo ">>> [4/4] Link cuDNN + kiem tra Paddle GPU..."
ste_gpu_prepare_env
python -c "import paddle; paddle.utils.run_check()"
python -c "import gradio; print('gradio', gradio.__version__, 'OK')"

conda deactivate

echo ""
echo "=========================================="
echo " FIX XONG — khoi dong lai dich vu:"
echo "   ./stop_linux.sh && ./start_linux.sh --share"
echo "=========================================="
