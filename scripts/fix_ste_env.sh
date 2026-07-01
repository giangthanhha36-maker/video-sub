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

echo ">>> [2/4] Cai NVIDIA cuDNN/cuBLAS (torch 2.5 cu121 can cudnn == 9.1.0.70)..."
pip install --no-cache-dir "nvidia-cudnn-cu12==9.1.0.70" nvidia-cublas-cu12

echo ">>> [3/4] Sua phien ban Gradio/UI dependencies..."
pip install --no-cache-dir -r requirements-ui.txt

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/scripts/ste_gpu_env.sh"
echo ">>> [4/4] Link cuDNN + kiem tra torch + paddle GPU..."
ste_gpu_prepare_env
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import paddle; paddle.utils.run_check()"
python -c "import gradio; print('gradio', gradio.__version__, 'OK')"

conda deactivate

echo ""
echo "=========================================="
echo " FIX XONG — khoi dong lai dich vu:"
echo "   ./stop_linux.sh && ./start_linux.sh --share"
echo "=========================================="
