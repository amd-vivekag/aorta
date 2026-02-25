#!/bin/bash
# =============================================================================
# Build PyTorch wheel from source with custom hipBLASLt and latest CK
#
# Target build configuration:
#   PyTorch:      latest main
#   CK:           latest develop branch
#   hipBLASLt:    pre-built from ROCm/rocm-libraries therock-7.11 release
#   GCN arch:     gfx950:sramecc+:xnack-
#   ROCm:         7.0 (from container base image)
#
# Usage (run INSIDE the build container):
#   bash /workspace/aorta/docker/build_pytorch_wheel.sh
#
# Optional env vars:
#   SKIP_ROCM_UPDATE=1   — skip ROCm repo/package update (step 2)
#   SKIP_HIPBLASLT=1     — skip hipBLASLt download (reuse previous install)
#   SKIP_PYTORCH_UPDATE=1— skip PyTorch git pull & submodule sync (steps 4-5)
#   SKIP_CK_UPDATE=1     — skip CK develop update (step 6)
#   SKIP_CLEAN=1         — incremental PyTorch build (skip setup.py clean)
#   FORCE_HIPIFY=1       — force re-run HIPify
#   NJOBS=64             — override parallelism (default: min(nproc, 256))
#   BUILD_ROOT=/path     — temp build area (default: /workspace/build)
#
# For a rebuild-only run (ROCm/hipBLASLt/CK already prepared):
#   SKIP_ROCM_UPDATE=1 SKIP_PYTORCH_UPDATE=1 SKIP_CK_UPDATE=1 SKIP_CLEAN=1 \
#     bash /workspace/aorta/docker/build_pytorch_wheel.sh
# =============================================================================
set -euo pipefail

PYTORCH_SRC="/workspace/pytorch"
WHEEL_OUT="${WHEEL_OUT_DIR:-/workspace/aorta/docker/wheels}"
BUILD_ROOT="${BUILD_ROOT:-/workspace/build}"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_TARGET="gfx950"
PYTORCH_ARCH="gfx950:sramecc+:xnack-"

NCPU=$(nproc)
NJOBS="${NJOBS:-$(( NCPU > 256 ? 256 : NCPU ))}"

HIPBLASLT_INSTALL="/workspace/aorta/docker/hipblaslt_install"
HIPBLASLT_RELEASE_URL="${HIPBLASLT_RELEASE_URL:-https://github.com/ROCm/rocm-libraries/releases/download/therock-7.11/hipblaslt.tar.gz}"

export ROCM_PATH
export HIP_PATH="${ROCM_PATH}"

echo "============================================="
echo " Build Configuration"
echo "  PyTorch source      : $PYTORCH_SRC"
echo "  Build root          : $BUILD_ROOT"
echo "  ROCm                : $ROCM_PATH"
echo "  GPU target          : $GPU_TARGET"
echo "  Jobs                : $NJOBS"
echo "  SKIP_ROCM_UPDATE    : ${SKIP_ROCM_UPDATE:-0}"
echo "  SKIP_HIPBLASLT      : ${SKIP_HIPBLASLT:-0}"
echo "  SKIP_PYTORCH_UPDATE : ${SKIP_PYTORCH_UPDATE:-0}"
echo "  SKIP_CK_UPDATE      : ${SKIP_CK_UPDATE:-0}"
echo "  SKIP_CLEAN          : ${SKIP_CLEAN:-0}"
echo "============================================="

mkdir -p "$BUILD_ROOT"

# =============================================================================
# [1/9] Pre-flight checks
# =============================================================================
echo ""
echo "============================================="
echo " [1/9] Pre-flight checks"
echo "============================================="

if [ ! -d "$PYTORCH_SRC" ]; then
    echo "ERROR: PyTorch source not found at $PYTORCH_SRC"
    echo "Make sure docker-compose mounts /apps/oyazdanb/pytorch:/workspace/pytorch"
    exit 1
fi

git config --global --add safe.directory '*'

if command -v rocminfo &>/dev/null; then
    ROCM_VER=$(rocminfo 2>/dev/null | grep -oP 'HSA Runtime Version:\s+\K[\d.]+' || echo "unknown")
    echo "[INFO] ROCm runtime version: $ROCM_VER"
fi

echo ""
echo "--- ROCm Library Versions (before update) ---"
ROCBLAS_VER=$(rpm -q rocblas 2>/dev/null || echo "not installed")
echo "[INFO] rocblas:             $ROCBLAS_VER"
HIPBLASLT_SYS_VER=$(rpm -q hipblaslt 2>/dev/null || echo "not installed")
echo "[INFO] hipblaslt (system):  $HIPBLASLT_SYS_VER"
echo "-----------------------------------------------"

# =============================================================================
# [2/9] Update ROCm repo and packages
# =============================================================================
echo ""
echo "============================================="
echo " [2/9] Updating ROCm repo to compute-rocm-rel-7.0-meta/19"
echo "============================================="

if [ "${SKIP_ROCM_UPDATE:-0}" = "1" ]; then
    echo "  SKIP_ROCM_UPDATE=1 — skipping ROCm package update"
else
    AMDGPU_BUILD="${AMDGPU_BUILD:-2281818}"
    ROCM_BUILD="${ROCM_BUILD:-compute-rocm-rel-7.0-meta/19}"

    echo "  amdgpu-build: $AMDGPU_BUILD"
    echo "  rocm-build:   $ROCM_BUILD"

    amdgpu-repo --amdgpu-build="$AMDGPU_BUILD" --rocm-build="$ROCM_BUILD"

    echo "  Updating ROCm packages..."
    yum update -y --skip-broken \
        rocm-hip \
        rocm-libs \
        rocm-hip-libraries \
        rocm-hip-runtime-devel \
        hip-base \
        hip-dev \
        hip-runtime-amd \
        rocm-core \
        rocblas \
        rocm-llvm-dev 2>/dev/null || echo "  Updated available ROCm packages"

    echo ""
    echo "--- ROCm Library Versions (after update) ---"
    ROCBLAS_VER=$(rpm -q rocblas 2>/dev/null || echo "not installed")
    echo "[INFO] rocblas:             $ROCBLAS_VER"
    HIPBLASLT_SYS_VER=$(rpm -q hipblaslt 2>/dev/null || echo "not installed")
    echo "[INFO] hipblaslt (system):  $HIPBLASLT_SYS_VER"
    echo "----------------------------------------------"
fi

# =============================================================================
# [3/9] Download pre-built hipBLASLt from therock release
# =============================================================================
echo ""
echo "============================================="
echo " [3/9] Downloading pre-built hipBLASLt (therock-7.11)"
echo "============================================="

if [ "${SKIP_HIPBLASLT:-0}" = "1" ]; then
    echo "  SKIP_HIPBLASLT=1 — skipping hipBLASLt download"
    if [ -f "$HIPBLASLT_INSTALL/lib/libhipblaslt.so" ]; then
        echo "  [INFO] Using existing hipBLASLt at $HIPBLASLT_INSTALL"
    else
        echo "  [WARN] hipBLASLt not found at $HIPBLASLT_INSTALL — build may fail"
    fi
elif [ -f "$HIPBLASLT_INSTALL/lib/libhipblaslt.so" ]; then
    echo "  hipBLASLt already installed at $HIPBLASLT_INSTALL — skipping"
    ls -l "$HIPBLASLT_INSTALL/lib/libhipblaslt"* 2>/dev/null
else
    rm -rf "$HIPBLASLT_INSTALL"
    mkdir -p "$HIPBLASLT_INSTALL"

    HIPBLASLT_TARBALL="$BUILD_ROOT/hipblaslt.tar.gz"
    if [ ! -f "$HIPBLASLT_TARBALL" ]; then
        echo "  Downloading from: $HIPBLASLT_RELEASE_URL"
        wget -q --show-progress -O "$HIPBLASLT_TARBALL" "$HIPBLASLT_RELEASE_URL"
    else
        echo "  Reusing cached tarball: $HIPBLASLT_TARBALL"
    fi

    echo "  Extracting to: $HIPBLASLT_INSTALL"
    tar xzf "$HIPBLASLT_TARBALL" -C "$HIPBLASLT_INSTALL"

    # The tarball may have a top-level directory; flatten if needed
    if [ ! -d "$HIPBLASLT_INSTALL/lib" ]; then
        NESTED=$(find "$HIPBLASLT_INSTALL" -maxdepth 2 -name "lib" -type d | head -1)
        if [ -n "$NESTED" ]; then
            NESTED_ROOT=$(dirname "$NESTED")
            echo "  Flattening nested directory: $NESTED_ROOT"
            mv "$NESTED_ROOT"/* "$HIPBLASLT_INSTALL/" 2>/dev/null || true
            rmdir "$NESTED_ROOT" 2>/dev/null || true
        fi
    fi

    echo "  [OK] hipBLASLt installed to: $HIPBLASLT_INSTALL"
    ls -l "$HIPBLASLT_INSTALL/lib/libhipblaslt"* 2>/dev/null || echo "  [WARN] libhipblaslt.so not found — check tarball layout"
fi

# =============================================================================
# [4/9] Update PyTorch to latest main
# =============================================================================
echo ""
echo "============================================="
echo " [4/9] Updating PyTorch to latest main"
echo "============================================="

cd "$PYTORCH_SRC"

if [ "${SKIP_PYTORCH_UPDATE:-0}" = "1" ]; then
    echo "  SKIP_PYTORCH_UPDATE=1 — skipping PyTorch git pull & submodule sync"
    PYTORCH_VERSION=$(cat version.txt | tr -d '[:space:]')
    echo "  [INFO] PyTorch $PYTORCH_VERSION at: $(git log -1 --format='%H %s (%ci)')"

    echo ""
    echo "============================================="
    echo " [5/9] Syncing submodules — SKIPPED (SKIP_PYTORCH_UPDATE=1)"
    echo "============================================="
else
    # Force HTTPS for all submodule URLs
    sed -i 's|git@github\.com:|https://github.com/|g' .gitmodules
    sed -i 's|ssh://git@github\.com/|https://github.com/|g' .gitmodules
    git config --global url."https://github.com/".insteadOf "git@github.com:"
    git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"

    git fetch origin main
    git checkout main
    git reset --hard origin/main

    PYTORCH_VERSION=$(cat version.txt | tr -d '[:space:]')
    echo "  [OK] PyTorch $PYTORCH_VERSION at: $(git log -1 --format='%H %s (%ci)')"

    # =============================================================================
    # [5/9] Sync submodules
    # =============================================================================
    echo ""
    echo "============================================="
    echo " [5/9] Syncing submodules"
    echo "============================================="

    git config --global http.postBuffer 524288000
    git config --global http.lowSpeedLimit 1000
    git config --global http.lowSpeedTime 300
    git config --global protocol.version 2

    git submodule sync
    if [ -d ".git/modules" ]; then
        find .git/modules -name config -exec \
            sed -i 's|git@github\.com:|https://github.com/|g' {} +
        find .git/modules -name config -exec \
            sed -i 's|ssh://git@github\.com/|https://github.com/|g' {} +
    fi

    MAX_RETRIES=3
    for attempt in $(seq 1 $MAX_RETRIES); do
        echo "  Submodule sync attempt $attempt/$MAX_RETRIES ..."
        git submodule sync
        if git submodule update --init --force --depth 1 --jobs 4; then
            echo "  [OK] Top-level submodules synced"
            break
        fi
        if [ "$attempt" -eq "$MAX_RETRIES" ]; then
            echo "ERROR: Failed to sync submodules after $MAX_RETRIES attempts"
            exit 1
        fi
        echo "  Retrying in 10 seconds..."
        sleep 10
    done

    # Verify critical submodules
    echo ""
    echo "  Verifying critical submodule working trees..."
    for sm_dir in psimd FP16 FXdiv NNPACK cpuinfo flatbuffers fmt gloo XNNPACK; do
        sm_path="third_party/$sm_dir"
        if [ ! -d "$sm_path" ] || [ -z "$(ls -A "$sm_path" 2>/dev/null | grep -v '^\.\(git\|gitignore\)$' | head -1)" ]; then
            echo "  [WARN] $sm_path appears empty — re-initializing..."
            git submodule update --init --force --depth 1 "$sm_path" || true
        fi
    done
    if [ ! -f "third_party/psimd/CMakeLists.txt" ]; then
        echo "  [ERROR] psimd still missing, attempting deep clone..."
        git submodule update --init --force third_party/psimd || exit 1
    fi
fi

# =============================================================================
# [6/9] Update CK to latest develop
# =============================================================================
echo ""
echo "============================================="
echo " [6/9] Updating composable_kernel to latest develop"
echo "============================================="

if [ "${SKIP_CK_UPDATE:-0}" = "1" ]; then
    echo "  SKIP_CK_UPDATE=1 — skipping CK update"
    echo "  [INFO] composable_kernel at: $(cd "$PYTORCH_SRC/third_party/composable_kernel" && git log -1 --format='%H %s (%ci)' 2>/dev/null || echo 'not checked out')"
    echo "  [INFO] AITer at: $(cd "$PYTORCH_SRC/third_party/aiter" && git log -1 --format='%H %s (%ci)' 2>/dev/null || echo 'not checked out')"
else
    cd "$PYTORCH_SRC/third_party/composable_kernel"
    git fetch --unshallow origin 2>/dev/null || git fetch origin
    git checkout develop
    git pull origin develop
    echo "  [OK] composable_kernel at: $(git log -1 --format='%H %s (%ci)')"

    # AITer submodules
    cd "$PYTORCH_SRC/third_party/aiter"
    echo "  [OK] AITer at: $(git log -1 --format='%H %s (%ci)')"
    git submodule update --init --depth 1 3rdparty/composable_kernel 2>/dev/null || true
    echo "  [OK] AITer CK sub: $(cd 3rdparty/composable_kernel && git log -1 --format='%H (%ci)' 2>/dev/null || echo 'not initialized')"
fi

cd "$PYTORCH_SRC"

# =============================================================================
# [7/9] Build configuration
# =============================================================================
echo ""
echo "============================================="
echo " [7/9] Build flags"
echo "============================================="

export PYTORCH_ROCM_ARCH="$PYTORCH_ARCH"
export USE_ROCM=1
export USE_CUDA=0
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1
export USE_ROCM_CK_SDPA=1
export DISABLE_AOTRITON=1
export MAX_JOBS="$NJOBS"
export BUILD_TEST=0
export BUILD_CAFFE2_OPS=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0

# Point to custom hipBLASLt
export CMAKE_PREFIX_PATH="$HIPBLASLT_INSTALL:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$HIPBLASLT_INSTALL/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export HIPBLASLT_DIR="$HIPBLASLT_INSTALL"
export hipblaslt_DIR="$HIPBLASLT_INSTALL/lib/cmake/hipblaslt"

export CMAKE_ARGS="-DCK_TILE_FMHA_FWD_FAST_EXP2=1 \
-DFLASH_NAMESPACE=pytorch_flash \
-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

echo "  PYTORCH_ROCM_ARCH     = $PYTORCH_ROCM_ARCH"
echo "  USE_ROCM              = $USE_ROCM"
echo "  USE_FLASH_ATTENTION   = $USE_FLASH_ATTENTION"
echo "  USE_MEM_EFF_ATTENTION = $USE_MEM_EFF_ATTENTION"
echo "  USE_ROCM_CK_SDPA     = $USE_ROCM_CK_SDPA"
echo "  DISABLE_AOTRITON     = $DISABLE_AOTRITON"
echo "  MAX_JOBS              = $MAX_JOBS"
echo "  hipBLASLt install     = $HIPBLASLT_INSTALL"
echo "  CMAKE_ARGS            = $CMAKE_ARGS"
echo ""
echo "  Submodule versions:"
echo "    AITer:             $(cd third_party/aiter && git log -1 --format='%H (%ci)')"
echo "    composable_kernel: $(cd third_party/composable_kernel && git log -1 --format='%H (%ci)')"

# =============================================================================
# [8/9] Build PyTorch wheel
# =============================================================================
echo ""
echo "============================================="
echo " [8/9] Building PyTorch wheel"
echo "============================================="

pip install -r requirements.txt 2>/dev/null || true
pip install ninja cmake wheel setuptools 2>/dev/null || true

# ccache
if command -v ccache &>/dev/null; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_HIP_COMPILER_LAUNCHER=ccache
    ccache --max-size=50G 2>/dev/null || true
    echo "  [OK] ccache enabled ($(ccache --version | head -1))"
fi

# Clean
if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    echo "  Cleaning previous build..."
    python3 setup.py clean 2>/dev/null || true
else
    echo "  SKIP_CLEAN=1 — incremental build"
fi

# HIPify
if [ ! -f "c10/hip/impl/hip_cmake_macros.h.in" ] || [ "${FORCE_HIPIFY:-0}" = "1" ]; then
    echo "  Running HIPify (CUDA -> HIP conversion)..."
    python3 tools/amd_build/build_amd.py
else
    echo "  HIPify already done — skipping (set FORCE_HIPIFY=1 to redo)"
fi

echo ""
echo "  Starting PyTorch build (this will take a while)..."
python3 setup.py bdist_wheel

# =============================================================================
# [9/9] Copy outputs
# =============================================================================
echo ""
echo "============================================="
echo " [9/9] Copying outputs"
echo "============================================="

mkdir -p "$WHEEL_OUT"
rm -f "$WHEEL_OUT"/torch-*.whl

WHEEL=$(ls -t dist/torch-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: No wheel found in dist/ after build!"
    exit 1
fi

cp "$WHEEL" "$WHEEL_OUT/"
WHEEL_NAME=$(basename "$WHEEL")

# Post-build verification
echo ""
pip install "$WHEEL_OUT/$WHEEL_NAME" --force-reinstall --no-deps 2>/dev/null || true
python3 -c "
import torch
print(f'  PyTorch version:  {torch.__version__}')
print(f'  ROCm available:   {torch.cuda.is_available()}')
print(f'  GPU count:        {torch.cuda.device_count()}')
print(f'  HIP version:      {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')
" 2>/dev/null || echo "  (verification skipped — no GPU access during build)"

echo ""
echo "============================================="
echo " SUCCESS!"
echo "  Wheel:      $WHEEL_NAME"
echo "  Saved to:   $WHEEL_OUT/$WHEEL_NAME"
echo "  hipBLASLt:  $HIPBLASLT_INSTALL"
echo "============================================="
echo ""
echo " Next steps:"
echo "   1. Exit the container"
echo "   2. Rebuild the Docker image:"
echo "      cd /apps/oyazdanb/aorta/docker"
echo "      docker compose -f docker-compose.rocm70_9-1-shampoo.yaml build"
echo ""
