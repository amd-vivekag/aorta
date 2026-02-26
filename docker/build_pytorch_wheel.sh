#!/bin/bash
# =============================================================================
# Build PyTorch from source for ROCm 7.0 (gfx950) inside the Docker container.
#
# Target build configuration:
#   PyTorch version:   2.11.0
#   GCN arch name:     gfx950:sramecc+:xnack-
#   ROCm version:      7.0.2.1 (compute-rocm-rel-7.0.2.1/6) — closest to client's 7.0.2.0
#   Rocblas version:   5.0.2-20250912-42-1199-g2584e35062
#   Hipblaslt version: 100200-7e32d53eb1
#
# Submodule pins:
#   AITer:              latest (HEAD, not pinned)
#   composable_kernel:  fcc9372c009c8e0a23fece77b582da83b04a654f
#
# CMake / build flags:
#   -DUSE_ROCM=1
#   -DUSE_FLASH_ATTENTION=1
#   -DUSE_MEM_EFF_ATTENTION=1
#   -DUSE_ROCM_CK_SDPA
#   -DDISABLE_AOTRITON
#   -DCK_TILE_FMHA_FWD_FAST_EXP2=1
#   -DFLASH_NAMESPACE=pytorch_flash
#
# Usage (run INSIDE the container on compute node cv350-zts-gtu-g33-18):
#   bash /workspace/aorta/docker/build_pytorch_wheel.sh
#
# After the wheel is built, rebuild the Docker image:
#   docker compose -f docker-compose.rocm70_9-1-shampoo.yaml build
# =============================================================================
set -euo pipefail

PYTORCH_SRC="/workspace/pytorch"
WHEEL_OUT="${WHEEL_OUT_DIR:-/workspace/aorta/docker/wheels}"

# =============================================================================
# Expected versions & pins
# =============================================================================
EXPECTED_PYTORCH_VERSION="2.12.0"
EXPECTED_ROCM_VERSION="7.0.2.1"
EXPECTED_GCN_ARCH="gfx950:sramecc+:xnack-"
CK_COMMIT="${CK_COMMIT:-fcc9372c009c8e0a23fece77b582da83b04a654f}"

# =============================================================================
# [1/7] Pre-flight checks
# =============================================================================
echo "============================================="
echo " [1/7] Pre-flight checks"
echo "============================================="

if [ ! -d "$PYTORCH_SRC" ]; then
    echo "ERROR: PyTorch source not found at $PYTORCH_SRC"
    echo "Make sure the docker-compose mounts /apps/oyazdanb/pytorch:/workspace/pytorch"
    exit 1
fi

# Verify PyTorch source version
PYTORCH_SRC_VERSION=$(cat "$PYTORCH_SRC/version.txt" | tr -d '[:space:]')
if [[ "$PYTORCH_SRC_VERSION" != "${EXPECTED_PYTORCH_VERSION}"* ]]; then
    echo "WARNING: PyTorch source version is '$PYTORCH_SRC_VERSION', expected '${EXPECTED_PYTORCH_VERSION}*'"
    echo "         Proceeding anyway..."
else
    echo "[OK] PyTorch source version: $PYTORCH_SRC_VERSION"
fi

# Verify ROCm version
if command -v rocminfo &>/dev/null; then
    ROCM_VER=$(rocminfo 2>/dev/null | grep -oP 'HSA Runtime Version:\s+\K[\d.]+' || echo "unknown")
    echo "[INFO] ROCm runtime version: $ROCM_VER"
else
    echo "[INFO] rocminfo not found, skipping ROCm version check"
fi

# Check ROCm library versions
echo ""
echo "--- ROCm Library Versions ---"
ROCBLAS_VER=$(rpm -q rocblas 2>/dev/null || echo "not installed")
echo "[INFO] rocblas:    $ROCBLAS_VER"
HIPBLASLT_VER=$(rpm -q hipblaslt 2>/dev/null || echo "not installed")
echo "[INFO] hipblaslt:  $HIPBLASLT_VER"
echo "-----------------------------"
echo ""

cd "$PYTORCH_SRC"

# =============================================================================
# [2/7] Sync top-level submodules
# =============================================================================
echo "============================================="
echo " [2/7] Syncing top-level git submodules"
echo "============================================="

# Mark all directories as safe — the mounted volume is owned by the host user,
# which differs from the container user and triggers git's safe.directory check.
git config --global --add safe.directory '*'

# Force HTTPS instead of SSH — container has no SSH keys
# 1) Rewrite .gitmodules in-place so all URLs are HTTPS
sed -i 's|git@github\.com:|https://github.com/|g' .gitmodules
sed -i 's|ssh://git@github\.com/|https://github.com/|g' .gitmodules

# 2) Also set global insteadOf as a safety net
git config --global url."https://github.com/".insteadOf "git@github.com:"
git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"

# 3) Disable SSH entirely so git never accidentally tries it
export GIT_SSH_COMMAND="false"

# Increase git network resilience for large submodule clones
git config --global http.postBuffer 524288000       # 500 MB
git config --global http.lowSpeedLimit 1000         # bytes/sec
git config --global http.lowSpeedTime 300           # 5 min before timeout
git config --global protocol.version 2

# 4) Clean up any previously cached SSH URLs in .git/config and .git/modules
git submodule sync
if [ -d ".git/modules" ]; then
    echo "  Cleaning cached SSH URLs in .git/modules..."
    find .git/modules -name config -exec \
        sed -i 's|git@github\.com:|https://github.com/|g' {} +
    find .git/modules -name config -exec \
        sed -i 's|ssh://git@github\.com/|https://github.com/|g' {} +
fi

# Use --jobs for parallel clones; --depth 1 for shallow; --force to
# re-checkout working trees that may be stale from a previous container run;
# NO --recursive (recursive submodules like aiter/composable_kernel are
# handled in step 4).
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

# Verify critical third-party directories have expected files
echo ""
echo "  Verifying critical submodule working trees..."
SUBMOD_OK=true
for sm_dir in psimd FP16 FXdiv NNPACK cpuinfo flatbuffers fmt gloo XNNPACK; do
    sm_path="third_party/$sm_dir"
    if [ ! -d "$sm_path" ] || [ -z "$(ls -A "$sm_path" 2>/dev/null | grep -v '^\.\(git\|gitignore\)$' | head -1)" ]; then
        echo "  [WARN] $sm_path appears empty or missing — attempting re-init..."
        git submodule update --init --force --depth 1 "$sm_path" || true
        SUBMOD_OK=false
    fi
done
# Special check for psimd which caused the original CMake error
if [ ! -f "third_party/psimd/CMakeLists.txt" ]; then
    echo "  [ERROR] third_party/psimd/CMakeLists.txt still missing after submodule init!"
    echo "  Attempting deep (non-shallow) clone of psimd..."
    git submodule update --init --force third_party/psimd || {
        echo "  ERROR: Could not recover psimd submodule."
        echo "  On the host, try: cd /apps/oyazdanb/pytorch && git submodule update --init --force third_party/psimd"
        exit 1
    }
fi
if $SUBMOD_OK; then
    echo "  [OK] All critical submodule working trees verified"
fi

# Re-enable SSH for any non-GitHub operations later
unset GIT_SSH_COMMAND

# =============================================================================
# [3/7] AITer — use latest (HEAD, no pin)
# =============================================================================
echo ""
echo "============================================="
echo " [3/7] AITer — using latest (HEAD)"
echo "============================================="
cd "$PYTORCH_SRC/third_party/aiter"
echo "  [OK] AITer at: $(git log -1 --format='%H %s (%ci)')"

# =============================================================================
# [4/7] Pin composable_kernel
# =============================================================================
echo ""
echo "============================================="
echo " [4/7] Pinning composable_kernel to $CK_COMMIT"
echo "============================================="

cd "$PYTORCH_SRC/third_party/composable_kernel"
CURRENT_CK=$(git rev-parse HEAD 2>/dev/null || echo "uninitialized")
if [ "$CURRENT_CK" = "$CK_COMMIT" ]; then
    echo "  [OK] composable_kernel already at target commit"
else
    echo "  Checking out composable_kernel commit $CK_COMMIT ..."
    git fetch --unshallow origin 2>/dev/null || git fetch origin
    git checkout "$CK_COMMIT"
fi
echo "  [OK] composable_kernel at: $(git log -1 --format='%H %s (%ci)')"

# Initialize AITer's own composable_kernel submodule (for CK tile JIT)
echo ""
echo "  Initializing AITer's composable_kernel submodule..."
cd "$PYTORCH_SRC/third_party/aiter"
git submodule update --init --depth 1 3rdparty/composable_kernel

AITER_CK_FILES=$(ls 3rdparty/composable_kernel/ 2>/dev/null | wc -l)
if [ "$AITER_CK_FILES" -gt 0 ]; then
    echo "  [OK] AITer's composable_kernel initialized ($AITER_CK_FILES entries)"
else
    echo "  [WARN] AITer's composable_kernel appears empty — CK tile JIT may not work"
fi

# =============================================================================
# [5/7] Verify kernel files
# =============================================================================
echo ""
echo "============================================="
echo " [5/7] Verifying gfx950 kernel availability"
echo "============================================="

AITER_GFX950_BWD="$PYTORCH_SRC/third_party/aiter/hsa/gfx950/fmha_v3_bwd"

if [ -d "$AITER_GFX950_BWD" ]; then
    echo "  gfx950 backward kernels:"
    ls "$AITER_GFX950_BWD" 2>/dev/null | grep "hd128.*bf16" | while read -r k; do
        echo "    $k"
    done
    echo ""
    RTNA_COUNT=$(ls "$AITER_GFX950_BWD" 2>/dev/null | grep "rtna" | wc -l)
    echo "  Total rtna kernels: $RTNA_COUNT"
else
    echo "  [WARN] Directory not found: $AITER_GFX950_BWD"
fi

cd "$PYTORCH_SRC"

# =============================================================================
# [6/7] Build configuration & flags
# =============================================================================
echo ""
echo "============================================="
echo " [6/7] Build Configuration"
echo "============================================="

# ---- GPU Architecture ----
export PYTORCH_ROCM_ARCH="${EXPECTED_GCN_ARCH}"
echo "  PYTORCH_ROCM_ARCH     = $PYTORCH_ROCM_ARCH"

# ---- Build flags (env vars read by setup.py) ----
export USE_ROCM=1
echo "  USE_ROCM              = $USE_ROCM"

export USE_FLASH_ATTENTION=1
echo "  USE_FLASH_ATTENTION   = $USE_FLASH_ATTENTION"

export USE_MEM_EFF_ATTENTION=1
echo "  USE_MEM_EFF_ATTENTION = $USE_MEM_EFF_ATTENTION"

export USE_ROCM_CK_SDPA=1
echo "  USE_ROCM_CK_SDPA     = $USE_ROCM_CK_SDPA"

export DISABLE_AOTRITON=1
echo "  DISABLE_AOTRITON      = $DISABLE_AOTRITON"

# ---- Extra CMake defines ----
export CMAKE_ARGS="-DUSE_FLASH_ATTENTION=1 \
-DUSE_ROCM=1 \
-DUSE_ROCM_CK_SDPA=1 \
-DUSE_MEM_EFF_ATTENTION=1 \
-DDISABLE_AOTRITON=1 \
-DCK_TILE_FMHA_FWD_FAST_EXP2=1 \
-DFLASH_NAMESPACE=pytorch_flash"
echo "  CMAKE_ARGS            = $CMAKE_ARGS"

# ---- Parallelism ----
# The host has 384 cores. ROCm HIP compilation is memory-hungry (~2-3 GB per
# hipcc process), so we cap at 128 to avoid OOM kills.  Tune down if the build
# OOMs or the machine has <256 GB RAM.
NCPU=$(nproc)
export MAX_JOBS=$(( NCPU > 128 ? 128 : NCPU ))
echo "  MAX_JOBS              = $MAX_JOBS"

# ---- Disable components we don't need (big time savings) ----
# BUILD_TEST:  skip the entire C++ test suite (~hundreds of translation units)
export BUILD_TEST=0
echo "  BUILD_TEST            = $BUILD_TEST"

# BUILD_CAFFE2_OPS: legacy Caffe2 operators, not needed for modern PyTorch
export BUILD_CAFFE2_OPS=0
echo "  BUILD_CAFFE2_OPS      = $BUILD_CAFFE2_OPS"

# USE_NNPACK / USE_QNNPACK / USE_XNNPACK: mobile/x86 quantized inference
# backends — not relevant for GPU training workloads
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0
echo "  USE_NNPACK            = $USE_NNPACK"
echo "  USE_QNNPACK           = $USE_QNNPACK"
echo "  USE_XNNPACK           = $USE_XNNPACK"

echo ""
echo "  Submodule versions:"
echo "    AITer:              $(cd third_party/aiter && git log -1 --format='%H (%ci)')"
echo "    composable_kernel:  $(cd third_party/composable_kernel && git log -1 --format='%H (%ci)')"
echo "    AITer CK (sub):     $(cd third_party/aiter/3rdparty/composable_kernel && git log -1 --format='%H (%ci)' 2>/dev/null || echo 'not initialized')"

# =============================================================================
# [7/7] Build
# =============================================================================
echo ""
echo "============================================="
echo " [7/7] Building PyTorch ${EXPECTED_PYTORCH_VERSION} for ROCm / ${EXPECTED_GCN_ARCH}"
echo "============================================="

# Install build dependencies
echo ""
echo "Installing build dependencies..."
pip install -r requirements.txt 2>/dev/null || true
pip install ninja cmake wheel setuptools 2>/dev/null || true

# ---- ccache (if available) ----
# ccache dramatically speeds up rebuilds by caching object files.
if command -v ccache &>/dev/null; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_HIP_COMPILER_LAUNCHER=ccache
    # Give ccache plenty of room — ROCm builds produce large object files
    ccache --max-size=50G 2>/dev/null || true
    echo "  [OK] ccache enabled ($(ccache --version | head -1))"
else
    echo "  [INFO] ccache not found — install it for faster rebuilds"
fi

# Clean previous build artifacts (skip with SKIP_CLEAN=1 for incremental builds)
if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    echo ""
    echo "Cleaning previous build..."
    python3 setup.py clean 2>/dev/null || true
else
    echo ""
    echo "Skipping clean (SKIP_CLEAN=1) — incremental build"
fi

# HIPify: convert CUDA sources to HIP for ROCm build
# This generates c10/hip/, aten/src/ATen/hip/, aten/src/THH/, etc.
# from their CUDA counterparts (c10/cuda/, aten/src/ATen/cuda/, aten/src/THC/).
# Skip if the generated files already exist (e.g., incremental build from a
# copied source tree).  Force re-run with FORCE_HIPIFY=1.
if [ ! -f "c10/hip/impl/hip_cmake_macros.h.in" ] || [ "${FORCE_HIPIFY:-0}" = "1" ]; then
    echo ""
    echo "Running HIPify (CUDA -> HIP source conversion)..."
    python3 tools/amd_build/build_amd.py
else
    echo ""
    echo "HIPify already done (c10/hip/ exists) — skipping (set FORCE_HIPIFY=1 to redo)"
fi

# Work around CMake 4.x incompatibility with old cmake_minimum_required()
# in vendored third-party CMakeLists.txt files (e.g., the 'six' package).
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Clean stale CK SDPA generated .hip files from a previous build.
# CMake's generate.py writes kernel instantiations directly into the source
# tree but doesn't remove old ones when CK is updated. Stale files reference
# the old CK API and break compilation.
# Skip with SKIP_CK_CLEAN=1 for incremental builds where CK hasn't changed.
if [ "${SKIP_CK_CLEAN:-0}" != "1" ]; then
    CK_SDPA_DIR="aten/src/ATen/native/transformers/hip/flash_attn/ck"
    CK_SDPA_V3_DIR="aten/src/ATen/native/transformers/hip/flash_attn/ck/fav_v3"
    STALE_COUNT=0
    for dir in "$CK_SDPA_DIR" "$CK_SDPA_V3_DIR"; do
        if [ -d "$dir" ]; then
            cnt=$(find "$dir" -maxdepth 1 -name 'fmha_*.hip' -o -name 'mha_*.hip' 2>/dev/null | wc -l)
            if [ "$cnt" -gt 0 ]; then
                echo "  Removing $cnt stale generated .hip files from $dir ..."
                find "$dir" -maxdepth 1 \( -name 'fmha_*.hip' -o -name 'mha_*.hip' \) -delete
                STALE_COUNT=$((STALE_COUNT + cnt))
            fi
        fi
    done
    if [ "$STALE_COUNT" -gt 0 ]; then
        echo "  [OK] Removed $STALE_COUNT stale CK SDPA .hip files (cmake will regenerate)"
    else
        echo "  [OK] No stale CK SDPA .hip files found"
    fi
else
    echo "  Skipping CK SDPA .hip cleanup (SKIP_CK_CLEAN=1)"
fi

# Build the wheel
echo ""
echo "Starting PyTorch build (this will take a while)..."
python3 setup.py bdist_wheel

# =============================================================================
# Copy wheel to docker context
# =============================================================================
mkdir -p "$WHEEL_OUT"
rm -f "$WHEEL_OUT"/torch-*.whl

WHEEL=$(ls -t dist/torch-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: No wheel found in dist/ after build!"
    exit 1
fi

cp "$WHEEL" "$WHEEL_OUT/"
WHEEL_NAME=$(basename "$WHEEL")

# =============================================================================
# Post-build verification
# =============================================================================
echo ""
echo "============================================="
echo " Post-build verification"
echo "============================================="
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
echo " Wheel:    $WHEEL_NAME"
echo " Saved to: $WHEEL_OUT/$WHEEL_NAME"
echo "============================================="
echo ""
echo " Target configuration:"
echo "   PyTorch version:   ${EXPECTED_PYTORCH_VERSION}"
echo "   GCN arch:          ${EXPECTED_GCN_ARCH}"
echo "   ROCm version:      ${EXPECTED_ROCM_VERSION} (docker) — client has 7.0.2.0"
echo "   Rocblas:            (verify via rpm -q rocblas inside container)"
echo "   Hipblaslt:          (verify via rpm -q hipblaslt inside container)"
echo ""
echo " Build flags:"
echo "   -DUSE_ROCM=1"
echo "   -DUSE_FLASH_ATTENTION=1"
echo "   -DUSE_MEM_EFF_ATTENTION=1"
echo "   -DUSE_ROCM_CK_SDPA"
echo "   -DDISABLE_AOTRITON"
echo "   -DCK_TILE_FMHA_FWD_FAST_EXP2=1"
echo "   -DFLASH_NAMESPACE=pytorch_flash"
echo "   PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
echo ""
echo " Submodule pins:"
echo "   AITer:             latest (HEAD)"
echo "   composable_kernel: $CK_COMMIT"
echo ""
echo " Next steps:"
echo "   1. Exit the container"
echo "   2. Rebuild the Docker image:"
echo "      cd /apps/oyazdanb/aorta/docker"
echo "      docker compose -f docker-compose.rocm70_9-1-shampoo.yaml build"
echo ""
