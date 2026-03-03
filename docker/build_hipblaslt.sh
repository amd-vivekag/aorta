#!/bin/bash
# =============================================================================
# Build hipBLASLt from the latest therock release and package install artifacts
# into a tarball that can be overlaid onto /opt/rocm in a Docker image.
#
# Usage (run INSIDE a ROCm container, e.g. rocm/pytorch base):
#   bash /workspace/aorta/docker/build_hipblaslt.sh
#
# Output:
#   docker/hipblaslt_install/hipblaslt-install.tar.gz
#   (extract at / to overlay onto /opt/rocm)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/hipblaslt_install"
OUTPUT_TAR="${OUTPUT_DIR}/hipblaslt-install.tar.gz"
GPU_TARGETS="${GPU_TARGETS:-gfx942;gfx950}"
STAGED="/tmp/hipblaslt-staged"

# =============================================================================
# [1/5] Install build dependencies
# =============================================================================
echo "============================================="
echo " [1/5] Installing build dependencies"
echo "============================================="
apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libmsgpack-dev libnuma1 wget curl libssl-dev \
    && rm -rf /var/lib/apt/lists/*

echo "  Installing CMake 3.25.2..."
cd /tmp
wget -q https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.tar.gz
tar xzf cmake-3.25.2-linux-x86_64.tar.gz
cp -a cmake-3.25.2-linux-x86_64/bin/* /usr/local/bin/
cp -a cmake-3.25.2-linux-x86_64/share/* /usr/local/share/
rm -rf cmake-3.25.2-linux-x86_64 cmake-3.25.2-linux-x86_64.tar.gz
echo "  [OK] $(cmake --version | head -1)"

# =============================================================================
# [2/5] Discover latest therock release
# =============================================================================
echo ""
echo "============================================="
echo " [2/5] Discovering latest therock release"
echo "============================================="

if [ -n "${THEROCK_TAG:-}" ]; then
    LATEST_TAG="$THEROCK_TAG"
    echo "  Using user-specified tag: ${LATEST_TAG}"
else
    LATEST_TAG=$(curl -s https://api.github.com/repos/ROCm/rocm-libraries/releases \
        | python3 -c "import json,sys; \
           tags=[r['tag_name'] for r in json.load(sys.stdin) if r['tag_name'].startswith('therock-')]; \
           print(tags[0])")
    echo "  Latest therock release: ${LATEST_TAG}"
fi

# =============================================================================
# [3/5] Download source
# =============================================================================
echo ""
echo "============================================="
echo " [3/5] Downloading hipBLASLt source"
echo "============================================="

cd /tmp
rm -rf hipblaslt
wget -q --show-progress \
    "https://github.com/ROCm/rocm-libraries/releases/download/${LATEST_TAG}/hipblaslt.tar.gz"
tar xzf hipblaslt.tar.gz
rm hipblaslt.tar.gz
echo "  [OK] Source extracted to /tmp/hipblaslt"

# =============================================================================
# [4/5] Build and install to staging prefix
# =============================================================================
echo ""
echo "============================================="
echo " [4/5] Building hipBLASLt (GPU_TARGETS=${GPU_TARGETS})"
echo "============================================="

rm -rf "${STAGED}"
mkdir -p "${STAGED}/opt/rocm"

cd /tmp/hipblaslt
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="${STAGED}/opt/rocm" \
    -DGPU_TARGETS="${GPU_TARGETS}" \
    -DAMDGPU_TARGETS="${GPU_TARGETS}" \
    -DHIPBLASLT_ENABLE_CLIENT=OFF \
    -DROCM_LIBS_SUPERBUILD=ON

NCPU=$(nproc)
BUILD_JOBS=$(( NCPU > 64 ? 64 : NCPU ))
echo "  Building with ${BUILD_JOBS} parallel jobs..."
make -j${BUILD_JOBS}
make install

echo ""
echo "  Installed files:"
find "${STAGED}" -type f | head -30
TOTAL_FILES=$(find "${STAGED}" -type f | wc -l)
echo "  ... (${TOTAL_FILES} files total)"

# =============================================================================
# [5/5] Package and verify
# =============================================================================
echo ""
echo "============================================="
echo " [5/5] Packaging and verification"
echo "============================================="

mkdir -p "${OUTPUT_DIR}"
cd "${STAGED}"
tar czf "${OUTPUT_TAR}" opt/
echo "  [OK] Packaged to: ${OUTPUT_TAR}"
echo "  Size: $(du -h "${OUTPUT_TAR}" | cut -f1)"

if [ -f "${STAGED}/opt/rocm/lib/cmake/hipblaslt/hipblaslt-config-version.cmake" ]; then
    VERSION=$(grep PACKAGE_VERSION "${STAGED}/opt/rocm/lib/cmake/hipblaslt/hipblaslt-config-version.cmake" | head -1)
    echo "  ${VERSION}"
fi

if [ -f "${STAGED}/opt/rocm/lib/libhipblaslt.so" ]; then
    XF32=$(strings "${STAGED}/opt/rocm/lib/libhipblaslt.so" | grep HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32 || echo "NOT FOUND")
    echo "  HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32: ${XF32}"
fi

# Cleanup
rm -rf /tmp/hipblaslt "${STAGED}"

echo ""
echo "============================================="
echo " SUCCESS!"
echo " therock release: ${LATEST_TAG}"
echo " GPU targets:     ${GPU_TARGETS}"
echo " Output:          ${OUTPUT_TAR}"
echo "============================================="
echo ""
echo " Next steps:"
echo "   1. Exit the container"
echo "   2. Rebuild the Docker image:"
echo "      cd ${SCRIPT_DIR}"
echo "      docker compose -f docker-compose.public-rocm72-torch2.12-shampoo.yaml build"
echo ""
