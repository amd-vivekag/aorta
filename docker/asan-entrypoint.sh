#!/bin/bash
# =============================================================================
# ASAN overlay entrypoint for TheRock HOST_ASAN Docker
# =============================================================================
#
# Activates ASAN-instrumented HIP runtime libraries at container startup.
#
#   Normal ROCm libs:  /opt/rocm/lib
#   ASAN overlay libs: /opt/rocm-asan/lib  (libamdhip64.so, libhsa-runtime64.so,
#                                           libamd_comgr.so)
#
# The overlay is activated by:
#   1. Prepending /opt/rocm-asan/lib to LD_LIBRARY_PATH (shadows normal libs)
#   2. Adding the ASAN runtime's directory to LD_LIBRARY_PATH so it is found
#      as a NEEDED dependency of the ASAN-built libraries
#   3. Setting verify_asan_link_order=0 in ASAN_OPTIONS so the runtime works
#      without LD_PRELOAD (LD_PRELOAD breaks HIP's internal dlopen/dlsym)
#
# Environment variables:
#   ASAN_DISABLE=1     — skip ASAN activation (run as normal image)
#   ASAN_OPTIONS=...   — override ASAN runtime options
# =============================================================================

ASAN_OVERLAY_DIR="/opt/rocm-asan/lib"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"

if [ "${ASAN_DISABLE:-0}" = "1" ]; then
    echo "[asan-entrypoint] ASAN_DISABLE=1 — running without ASAN overlay"
    exec "$@"
fi

# --- Locate ASAN runtime from the ROCm LLVM toolchain ---

ASAN_LIB=$(find "$ROCM_HOME/llvm/lib/clang" -name "libclang_rt.asan-x86_64.so" 2>/dev/null | head -1)

if [ -z "$ASAN_LIB" ]; then
    echo "[asan-entrypoint] WARNING: ASAN runtime (libclang_rt.asan-x86_64.so) not found"
    echo "  Searched: $ROCM_HOME/llvm/lib/clang/"
    echo "  Running without ASAN overlay"
    exec "$@"
fi

ASAN_RT_DIR=$(dirname "$ASAN_LIB")

# --- Validate overlay directory ---

if [ ! -d "$ASAN_OVERLAY_DIR" ]; then
    echo "[asan-entrypoint] WARNING: overlay directory $ASAN_OVERLAY_DIR not found"
    echo "  Running without ASAN overlay"
    exec "$@"
fi

if ! ls "$ASAN_OVERLAY_DIR"/libamdhip64.so* >/dev/null 2>&1; then
    echo "[asan-entrypoint] WARNING: libamdhip64.so not found in $ASAN_OVERLAY_DIR"
    echo "  Running without ASAN overlay"
    exec "$@"
fi

# --- Activate ASAN overlay ---
#
# IMPORTANT: We do NOT use LD_PRELOAD for the ASAN runtime.
# LD_PRELOAD causes ASAN to globally intercept dlopen/dlsym, which breaks
# HIP's internal dynamic loading of libhsa-runtime64.so (SEGV in
# amd::Device::init at a null function pointer).
#
# Instead, the ASAN runtime is loaded as a NEEDED dependency of the
# ASAN-built libraries.  verify_asan_link_order=0 suppresses the ASAN
# check that it must be the first loaded library.

export LD_LIBRARY_PATH="${ASAN_OVERLAY_DIR}:${ASAN_RT_DIR}:${LD_LIBRARY_PATH:-}"
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:halt_on_error=0:symbolize=1:verify_asan_link_order=0}"

SYMBOLIZER="$ROCM_HOME/llvm/bin/llvm-symbolizer"
if [ -x "$SYMBOLIZER" ]; then
    export ASAN_SYMBOLIZER_PATH="$SYMBOLIZER"
fi

export ASAN_OVERLAY_ACTIVE=1

echo "[asan-entrypoint] ASAN overlay active"
echo "  LD_LIBRARY_PATH prefix: $ASAN_OVERLAY_DIR"
echo "  ASAN runtime (NEEDED):  $ASAN_LIB"
echo "  ASAN_OPTIONS:           $ASAN_OPTIONS"
echo "  ASAN_SYMBOLIZER_PATH:   ${ASAN_SYMBOLIZER_PATH:-not set}"
echo ""
echo "  Disable:  ASAN_DISABLE=1"

exec "$@"
