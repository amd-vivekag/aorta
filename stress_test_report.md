# NaN Stress Test Report

**Date:** 2026-02-26
**Host:** cv350-zts-gtu-h30-08
**Command:**
```
GPU_MAX_HW_QUEUES=4 PYTHONPATH=src torchrun --nproc_per_node=8 \
    scripts/nan_stress_test.py --config config/nan_stress_test.yaml \
    --max-steps 500 --hw-queues 4
```

---

## Summary

| # | Docker Image / Compose | Result | Failure Mode |
|---|------------------------|--------|--------------|
| 1 | `docker-torchenv-rocm70-shampoo` (9-1-shampoo) | **FAIL** | SIGSEGV on all 8 ranks |
| 2 | `docker-torchenv-rocm70-shampoo:latest` (working tarball) | **PASS** | 500/500 steps, 0 NaN |
| 3 | `docker-torchenv-rocm70-shampoo-vivekag` | **N/A** | No `nan_stress_test.py` in mounted aorta |
| 4 | `docker-torchenv` (vivekag-20260210) | **N/A** | No `nan_stress_test.py` in mounted aorta |
| 5 | `docker-torchenv-rocm70-2-1-shampoo` (newly built) | **FAIL** | SIGSEGV on all 8 ranks |
| 6 | `docker-torchenv-rocm70` (rocm70_9-1 base) | **FAIL** | `ModuleNotFoundError: distributed_shampoo` |
| 7 | `rocm/pytorch-private:20251030...nightly` (base) | **FAIL** | `ModuleNotFoundError: distributed_shampoo` |

---

## Detailed Results

### 1. docker-torchenv-rocm70-shampoo (compose: rocm70_9-1-shampoo)

| Field | Value |
|-------|-------|
| Container | `training-overlap-bugs-rocm70_9-1-shampoo` |
| Dockerfile | `Dockerfile.rocm70_9-1-shampoo` |
| ROCm build | `compute-rocm-rel-7.0-meta/19` |
| Python | 3.10.18 |
| PyTorch | 2.12.0a0+git580a6e2 |
| ROCm | 7.0.2 |
| HIP | 7.0.51831 |
| Shampoo | Installed |
| **Result** | **FAIL — SIGSEGV (signal 11) on all 8 ranks within ~3s of training start** |

### 2. docker-torchenv-rocm70-shampoo:latest (working tarball)

| Field | Value |
|-------|-------|
| Container | `training-overlap-bugs-rocm70-working` (temp) |
| Source | `/apps/oyazdanb/docker-torchenv-rocm70-shampoo-h30-working.tar.gz` |
| Python | 3.10.18 |
| PyTorch | 2.9.0+git25a49ce |
| ROCm | 7.0.2 |
| HIP | 7.0.51831-7c9236b16 |
| Shampoo | Installed |
| **Result** | **PASS — 500/500 steps, 0 NaN/Inf, loss 0.1271 → 0.0419, ~444ms/step** |

### 3. docker-torchenv-rocm70-shampoo-vivekag

| Field | Value |
|-------|-------|
| Container | `training-overlap-bugs-rocm70_9-1-shampoo-vivekag` |
| Python | 3.10.18 |
| PyTorch | 2.9.0+git25a49ce |
| ROCm | 7.0.2 |
| HIP | 7.0.51831-7c9236b16 |
| Shampoo | Installed |
| **Result** | **N/A — mounts `/apps/vivekag/aorta_work/aorta_1` which lacks `nan_stress_test.py`** |

### 4. docker-torchenv (vivekag-rocm70_9-1-shampoo-20260210)

| Field | Value |
|-------|-------|
| Container | `vivekag-rocm70_9-1-shampoo-20260210` |
| Python | 3.10.18 |
| PyTorch | 2.9.0+git25a49ce |
| ROCm | 7.0.2 |
| HIP | 7.0.51831-7c9236b16 |
| Shampoo | Installed |
| **Result** | **N/A — mounts `/apps/vivekag/aorta_work/aorta_1` which lacks `nan_stress_test.py`** |

### 5. docker-torchenv-rocm70-2-1-shampoo (compose: rocm70_2-1-shampoo)

| Field | Value |
|-------|-------|
| Container | `stress-test-rocm70-2-1-shampoo` (temp) |
| Dockerfile | `Dockerfile.rocm70_2-1-shampoo` |
| ROCm build | `compute-rocm-rel-7.0.2.1/6` |
| Python | 3.10.18 |
| PyTorch | 2.11.0a0+git51cc634 |
| ROCm | 7.0.2 |
| HIP | 7.0.51831 |
| Shampoo | Installed |
| **Result** | **FAIL — SIGSEGV (signal 11) on all 8 ranks within ~3s of training start** |

### 6. docker-torchenv-rocm70 (compose: rocm70_9-1, base image)

| Field | Value |
|-------|-------|
| Container | `stress-test-rocm70-base` (temp) |
| Dockerfile | `Dockerfile.rocm70_9-1` |
| Base image | `rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1` |
| Python | 3.10.18 |
| PyTorch | 2.9.0+git25a49ce |
| ROCm | 7.0.2 |
| HIP | 7.0.51831-7c9236b16 |
| Shampoo | **Not installed** |
| **Result** | **FAIL — `ModuleNotFoundError: No module named 'distributed_shampoo'`** |

### 7. rocm/pytorch-private:20251030...nightly (compose: docker-compose.yaml)

| Field | Value |
|-------|-------|
| Container | `stress-test-nightly-base` (temp) |
| Image | `rocm/pytorch-private:20251030_rocm_e2e_phantom_mi350_genai_nightly` |
| Python | 3.10.18 |
| PyTorch | 2.9.0+git25a49ce |
| ROCm | 7.0.2 |
| HIP | 7.0.51831-7c9236b16 |
| Shampoo | **Not installed** |
| **Result** | **FAIL — `ModuleNotFoundError: No module named 'distributed_shampoo'`** |

---

## Key Observations

### The only PASSING image uses PyTorch 2.9.0 (git 25a49ce)

The working tarball (`docker-torchenv-rocm70-shampoo-h30-working.tar.gz`) is the **only** configuration that passes. It uses the original base image PyTorch (`2.9.0+git25a49ce`) without replacing it with a custom-built wheel.

### All SIGSEGV failures use custom PyTorch wheels

| Image | PyTorch | Wheel | SIGSEGV? |
|-------|---------|-------|----------|
| working tarball | 2.9.0+git25a49ce | Base image default | No |
| 9-1-shampoo | 2.12.0a0+git580a6e2 | `torch-*.whl` (custom) | **Yes** |
| 2-1-shampoo | 2.11.0a0+git51cc634 | `torch-*.whl` (custom) | **Yes** |

Both crashing images install a custom PyTorch wheel (`COPY wheels/torch-*.whl`) built from newer PyTorch source, while the working image uses the base image's PyTorch 2.9.0. The ROCm version (7.0.2) and HIP runtime are the same across all three.

### HIP version difference

- SIGSEGV images: HIP `7.0.51831` (no git suffix — updated ROCm packages)
- Working image: HIP `7.0.51831-7c9236b16` (with git suffix — original base image)

This suggests the ROCm package update (`yum update rocm-hip ...`) in the Dockerfiles may also contribute to the regression.

### Root cause candidates

1. **Custom PyTorch wheel** (2.11/2.12) introduces a regression causing SIGSEGV during distributed Shampoo + multi-stream training
2. **ROCm HIP runtime update** via `yum update` strips the HIP git-versioned build, possibly introducing incompatibility
3. **Custom hipBLASLt** (`/opt/hipblaslt/lib`) in the SIGSEGV images may conflict with the runtime
