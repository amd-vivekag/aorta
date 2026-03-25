#!/usr/bin/env bash
# verify_asan_therock.sh — Verify TheRock's HOST_ASAN build is working.
#
# Run inside the ASAN Docker container:
#   docker run --device=/dev/kfd --device=/dev/dri \
#     --ipc=host --group-add=video --group-add=render \
#     therock-host-asan-pytorch  /workspace/asan_tests/verify_asan_therock.sh
#
# Exit codes:
#   0 — all checks passed
#   1 — a check failed

set -euo pipefail

ROCM=${ROCM_HOME:-/opt/rocm}
CLANG="$ROCM/llvm/bin/clang++"
ASAN_LIB=$(find "$ROCM/llvm/lib/clang" -name "libclang_rt.asan-x86_64.so" 2>/dev/null | head -1)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMPDIR=$(mktemp -d /tmp/asan_test.XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass=0
fail=0

check() {
    local name="$1"; shift
    if "$@"; then
        echo -e "  ${GREEN}[PASS]${NC} $name"
        ((pass++))
    else
        echo -e "  ${RED}[FAIL]${NC} $name"
        ((fail++))
    fi
}

# ===== Section 1: Environment checks =====
echo ""
echo "=========================================="
echo " 1. Environment checks"
echo "=========================================="

check "ROCm install exists" test -d "$ROCM"
check "hipcc available" test -x "$ROCM/bin/hipcc"
check "clang++ available" test -x "$CLANG"
check "ASAN runtime found" test -n "$ASAN_LIB" -a -f "$ASAN_LIB"

# Check that libamdhip64.so is ASAN-instrumented
check "libamdhip64.so has ASAN symbols" \
    bash -c "nm -D $ROCM/lib/libamdhip64.so 2>/dev/null | grep -q __asan_report"

# ===== Section 2: ASAN catches host memory errors =====
echo ""
echo "=========================================="
echo " 2. ASAN detects host memory errors"
echo "=========================================="

# Compile the test binary
if [ -f "$SCRIPT_DIR/test_hip_asan.cpp" ]; then
    TEST_SRC="$SCRIPT_DIR/test_hip_asan.cpp"
else
    echo -e "  ${YELLOW}[SKIP]${NC} test_hip_asan.cpp not found at $SCRIPT_DIR"
    TEST_SRC=""
fi

if [ -n "$TEST_SRC" ]; then
    echo "  Compiling test_hip_asan.cpp with ASAN..."
    ARCH=${PYTORCH_ROCM_ARCH:-gfx950}
    "$CLANG" -fsanitize=address -shared-libasan \
        --offload-arch="$ARCH" -x hip \
        -I"$ROCM/include" -L"$ROCM/lib" -lamdhip64 \
        -o "$TMPDIR/test_hip_asan" "$TEST_SRC" 2>&1

    if [ $? -ne 0 ]; then
        echo -e "  ${RED}[FAIL]${NC} Compilation failed"
        ((fail++))
    else
        echo -e "  ${GREEN}[PASS]${NC} Compilation succeeded"
        ((pass++))

        # Test: heap_overflow should produce ASAN report
        echo "  Running heap_overflow test..."
        ASAN_OUTPUT=$(LD_PRELOAD="$ASAN_LIB" "$TMPDIR/test_hip_asan" heap_overflow 2>&1 || true)
        check "ASAN catches heap-buffer-overflow" \
            bash -c "echo '$ASAN_OUTPUT' | grep -q 'heap-buffer-overflow'"

        # Test: use_after_free should produce ASAN report
        echo "  Running use_after_free test..."
        ASAN_OUTPUT=$(LD_PRELOAD="$ASAN_LIB" "$TMPDIR/test_hip_asan" use_after_free 2>&1 || true)
        check "ASAN catches use-after-free" \
            bash -c "echo '$ASAN_OUTPUT' | grep -q 'use-after-free\|heap-use-after-free'"
    fi
fi

# ===== Section 3: GPU functionality (requires GPU) =====
echo ""
echo "=========================================="
echo " 3. GPU functionality under ASAN"
echo "=========================================="

DEVICE_COUNT=$(hipinfo 2>/dev/null | grep -c "Device Type.*GPU" || echo "0")
if [ "$DEVICE_COUNT" -eq 0 ]; then
    # Try alternative detection
    DEVICE_COUNT=$(LD_PRELOAD="$ASAN_LIB" "$TMPDIR/test_hip_asan" clean 2>&1 | grep -oP 'HIP devices: \K\d+' || echo "0")
fi

if [ "$DEVICE_COUNT" -gt 0 ] && [ -f "$TMPDIR/test_hip_asan" ]; then
    echo "  Found $DEVICE_COUNT GPU(s)"

    echo "  Running clean test (no errors expected)..."
    CLEAN_OUTPUT=$(LD_PRELOAD="$ASAN_LIB" "$TMPDIR/test_hip_asan" clean 2>&1)
    CLEAN_RC=$?
    check "Clean HIP program runs correctly" test "$CLEAN_RC" -eq 0
    check "No ASAN errors in clean run" \
        bash -c "! echo '$CLEAN_OUTPUT' | grep -q 'ERROR: AddressSanitizer'"

    echo "  Running event_query stress test..."
    EVENT_OUTPUT=$(LD_PRELOAD="$ASAN_LIB" ASAN_OPTIONS="detect_leaks=0:halt_on_error=0" \
        "$TMPDIR/test_hip_asan" event_query 2>&1)
    EVENT_RC=$?
    check "hipEventQuery stress test completes" test "$EVENT_RC" -eq 0

    # Report any ASAN findings from event_query (informational)
    if echo "$EVENT_OUTPUT" | grep -q "ERROR: AddressSanitizer"; then
        echo -e "  ${YELLOW}[INFO]${NC} ASAN found issues in hipEventQuery path:"
        echo "$EVENT_OUTPUT" | grep "ERROR: AddressSanitizer" | head -5
    else
        echo -e "  ${GREEN}[INFO]${NC} No ASAN errors in hipEventQuery path"
    fi

    echo "  Running multi_stream test..."
    STREAM_OUTPUT=$(LD_PRELOAD="$ASAN_LIB" ASAN_OPTIONS="detect_leaks=0:halt_on_error=0" \
        "$TMPDIR/test_hip_asan" multi_stream 2>&1)
    STREAM_RC=$?
    check "Multi-stream event polling completes" test "$STREAM_RC" -eq 0
else
    echo -e "  ${YELLOW}[SKIP]${NC} No GPU available — skipping GPU tests"
    echo "         Run with --device=/dev/kfd --device=/dev/dri for GPU tests"
fi

# ===== Summary =====
echo ""
echo "=========================================="
echo " Summary"
echo "=========================================="
echo -e "  ${GREEN}Passed: $pass${NC}"
echo -e "  ${RED}Failed: $fail${NC}"
echo ""

if [ "$fail" -gt 0 ]; then
    echo -e "${RED}SOME CHECKS FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}ALL CHECKS PASSED${NC}"
    exit 0
fi
