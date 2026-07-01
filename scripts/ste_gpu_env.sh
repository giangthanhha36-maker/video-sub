#!/usr/bin/env bash
# ===========================================================================
# Chuan bi moi truong GPU cho env conda "ste" (Paddle OCR + PyTorch STTN).
# Duoc goi tu setup_linux_gpu.sh va run_ui.sh.
#
# Van de xu ly:
#   - Paddle tim libcudnn tai /usr/local/cuda/lib64/libcudnn.so (Docker thuong thieu)
#   - pip chi cai libcudnn.so.8/.9, can symlink thanh libcudnn.so
#   - LD_LIBRARY_PATH can tro toi nvidia/cudnn/lib va torch/lib
# ===========================================================================

# Kiem tra da conda activate env ste chua (CONDA_PREFIX phai co gia tri).
ste_gpu_require_ste_env() {
    local prefix="${1:-${CONDA_PREFIX:-}}"
    if [ -z "$prefix" ]; then
        echo "[LOI] Chua kich hoat moi truong conda 'ste' (CONDA_PREFIX trong)." >&2
        echo "      Hay chay day du:" >&2
        echo "        source \"\$(conda info --base)/etc/profile.d/conda.sh\"" >&2
        echo "        conda activate ste" >&2
        echo "        source scripts/ste_gpu_env.sh" >&2
        echo "        ste_gpu_prepare_env && ste_gpu_verify_paddle" >&2
        return 1
    fi
    if [ "${CONDA_DEFAULT_ENV:-}" != "ste" ] && [ "$(basename "$prefix")" != "ste" ]; then
        echo "[CANH BAO] Dang o moi truong '${CONDA_DEFAULT_ENV:-$(basename "$prefix")}', khong phai 'ste'." >&2
    fi
    return 0
}

ste_gpu_site_packages() {
    local prefix="${1:-${CONDA_PREFIX:-}}"
    if [ -z "$prefix" ]; then
        echo "[LOI] ste_gpu_site_packages: CONDA_PREFIX trong — chua 'conda activate ste'." >&2
        return 1
    fi
    local pyver
    pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    echo "${prefix}/lib/python${pyver}/site-packages"
}

# Tao symlink libcudnn.so -> /usr/local/cuda/lib64 (Paddle 2.6 hay tim path co dinh nay).
ste_gpu_link_cudnn() {
    local site
    site="$(ste_gpu_site_packages "$1")" || return 1
    local cudnn_lib="${site}/nvidia/cudnn/lib"
    local cublas_lib="${site}/nvidia/cublas/lib"

    if [ ! -d "$cudnn_lib" ]; then
        echo "[CANH BAO] Khong tim thay ${cudnn_lib} — bo qua link cuDNN."
        return 0
    fi

    local cudnn_so=""
    # uu tien .so.9 vi torch cu121 can libcudnn.so.9
    for candidate in \
        "${cudnn_lib}/libcudnn.so.9" \
        "${cudnn_lib}/libcudnn.so.8" \
        "${cudnn_lib}/libcudnn.so"
    do
        if [ -e "$candidate" ]; then
            cudnn_so="$candidate"
            break
        fi
    done
    if [ -z "$cudnn_so" ]; then
        cudnn_so="$(find "$cudnn_lib" -maxdepth 1 -name 'libcudnn.so.*' -type f 2>/dev/null | sort -V | tail -1)"
    fi
    if [ -z "$cudnn_so" ]; then
        echo "[CANH BAO] Khong tim thay libcudnn.so* trong ${cudnn_lib}"
        return 1
    fi

    ln -sf "$cudnn_so" "${cudnn_lib}/libcudnn.so"

    local torch_lib="${site}/torch/lib"
    if [ -d "$torch_lib" ]; then
        for f in "${cudnn_lib}"/libcudnn*.so*; do
            [ -e "$f" ] || continue
            ln -sf "$f" "${torch_lib}/$(basename "$f")"
        done
    fi

    mkdir -p /usr/local/cuda/lib64
    ln -sf "$cudnn_so" /usr/local/cuda/lib64/libcudnn.so
    for f in "${cudnn_lib}"/libcudnn*.so*; do
        [ -e "$f" ] || continue
        ln -sf "$f" "/usr/local/cuda/lib64/$(basename "$f")"
    done

    if [ -d "$cublas_lib" ]; then
        for f in "${cublas_lib}"/libcublas*.so*; do
            [ -e "$f" ] || continue
            ln -sf "$f" "/usr/local/cuda/lib64/$(basename "$f")"
        done
    fi

    return 0
}

# Gan LD_LIBRARY_PATH cho Paddle/PyTorch trong env ste.
# KHONG them ${prefix}/lib — se lam ffprobe/ffmpeg nap nham libncurses cua conda.
ste_gpu_export_ld() {
    local site
    site="$(ste_gpu_site_packages "$1")" || return 1
    local extra=""

    for libdir in \
        "${site}/nvidia/cudnn/lib" \
        "${site}/nvidia/cublas/lib" \
        "${site}/torch/lib"
    do
        if [ -d "$libdir" ]; then
            extra="${extra:+$extra:}${libdir}"
        fi
    done

    if [ -n "$extra" ]; then
        export LD_LIBRARY_PATH="${extra}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

# Goi day du: link + export LD. Tham so 1: CONDA_PREFIX (mac dinh $CONDA_PREFIX).
ste_gpu_prepare_env() {
    local prefix="${1:-${CONDA_PREFIX:-}}"
    ste_gpu_require_ste_env "$prefix" || return 1
    ste_gpu_link_cudnn "$prefix"
    ste_gpu_export_ld "$prefix"
}

# Kiem tra Paddle GPU (tra ve 0 neu thanh cong).
ste_gpu_verify_paddle() {
    local prefix="${1:-${CONDA_PREFIX:-}}"
    ste_gpu_require_ste_env "$prefix" || return 1
    ste_gpu_prepare_env "$prefix"
    # python -c "import paddle; paddle.utils.run_check()"
    local _verify_script
    _verify_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/verify_paddle_gpu.py"
    python "$_verify_script"
    unset _verify_script
}
