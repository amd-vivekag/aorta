"""Verify TF32x1 and TF32x3 use different hipblaslt compute passes.

PyTorch sends HIPBLAS_COMPUTE_32F_FAST_TF32 (xf32) to hipBLASLt when
allow_tf32=True.  HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32 overrides the
compute type inside hipBLASLt:

  Override=1 (default) → HIPBLAS_COMPUTE_32F_FAST_TF32  (triple BF16 acc, x3)
  Override=2           → HIPBLAS_COMPUTE_32F_FAST_16BF  (single BF16 acc, x1)
  Override=0           → HIPBLAS_COMPUTE_32F            (pure FP32 fallback)
"""

import os
import torch


def run_matmul(A, B, device, label, override_val=None):
    allow = override_val is not None
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.set_float32_matmul_precision("high" if allow else "highest")

    if override_val is not None:
        os.environ["HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32"] = str(override_val)
    else:
        os.environ.pop("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32", None)

    torch.cuda.synchronize(device)
    result = torch.mm(A, B)
    torch.cuda.synchronize(device)

    print(f"  {label:20s} | OVERRIDE_XF32={str(override_val):>5s} | "
          f"allow_tf32={str(allow):>5s} | precision={torch.get_float32_matmul_precision()}")
    return result


def main():
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    print(f"Platform: {'ROCm ' + str(torch.version.hip) if is_rocm else 'CUDA'}")
    print()

    torch.cuda.manual_seed(98765)
    M, K, N = 4096, 4096, 4096
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    ref_f64 = torch.mm(A.double(), B.double())

    print("Running matmuls...")
    out_fp32       = run_matmul(A, B, device, "FP32 (disabled)")
    out_x3         = run_matmul(A, B, device, "TF32x3 (Override=1)", override_val=1)
    out_x1         = run_matmul(A, B, device, "TF32x1 (Override=2)", override_val=2)
    out_fp32_force = run_matmul(A, B, device, "FP32   (Override=0)", override_val=0)

    fp32_err = (out_fp32.double() - ref_f64).abs()
    x3_err   = (out_x3.double()   - ref_f64).abs()
    x1_err   = (out_x1.double()   - ref_f64).abs()

    x1_x3_diff       = (out_x1 - out_x3).abs()
    fp32_x3_diff     = (out_fp32 - out_x3).abs()
    fp32_x1_diff     = (out_fp32 - out_x1).abs()
    fp32_force_diff  = (out_fp32 - out_fp32_force).abs()

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    print()
    print("Accuracy vs fp64 ground truth (max abs error):")
    print(f"  FP32   : {fp32_err.max().item():.4e}  (mean {fp32_err.mean().item():.4e})")
    print(f"  TF32x3 : {x3_err.max().item():.4e}  (mean {x3_err.mean().item():.4e})")
    print(f"  TF32x1 : {x1_err.max().item():.4e}  (mean {x1_err.mean().item():.4e})")

    print()
    print("Pairwise deltas (max abs diff):")
    print(f"  FP32  vs TF32x3 (Override=1): {fp32_x3_diff.max().item():.4e}  (mean {fp32_x3_diff.mean().item():.4e})")
    print(f"  FP32  vs TF32x1 (Override=2): {fp32_x1_diff.max().item():.4e}  (mean {fp32_x1_diff.mean().item():.4e})")
    print(f"  TF32x3 vs TF32x1            : {x1_x3_diff.max().item():.4e}  (mean {x1_x3_diff.mean().item():.4e})")
    print(f"  FP32  vs FP32   (Override=0): {fp32_force_diff.max().item():.4e}  (sanity check, should be 0)")

    print()
    print("Verification:")
    x3_active       = fp32_x3_diff.max().item() > 0
    x1_active       = fp32_x1_diff.max().item() > 0
    x1_x3_different = x1_x3_diff.max().item() > 0
    fp32_sane       = fp32_force_diff.max().item() == 0.0

    print(f"  TF32x3 active (Override=1 differs from FP32) : {'YES' if x3_active else 'NO'}")
    print(f"  TF32x1 active (Override=2 differs from FP32) : {'YES' if x1_active else 'NO'}")
    print(f"  x1 != x3 (different hipblaslt compute passes): {'YES' if x1_x3_different else 'NO'}")
    print(f"  Override=0 == FP32 (sanity check)            : {'YES' if fp32_sane else 'NO'}")

    print()
    if x3_active and x1_active and x1_x3_different and fp32_sane:
        print("PASS: TF32x1 and TF32x3 use two different hipblaslt compute passes")
        print("  TF32x3 (Override=1) → HIPBLAS_COMPUTE_32F_FAST_TF32  (triple BF16 accumulation)")
        print("  TF32x1 (Override=2) → HIPBLAS_COMPUTE_32F_FAST_16BF  (single BF16 accumulation)")
    elif not x3_active and not x1_active:
        print("FAIL: Neither TF32 mode is active — hipblaslt may not support TF32 on this HW")
    elif not x1_x3_different:
        print("FAIL: TF32x1 and TF32x3 produce identical results — same pass is being used")
    else:
        print("PARTIAL: Only one TF32 mode is active")
        if not x3_active:
            print("  TF32x3 (FAST_TF32) is NOT active")
        if not x1_active:
            print("  TF32x1 (FAST_16BF) is NOT active")


if __name__ == "__main__":
    main()
