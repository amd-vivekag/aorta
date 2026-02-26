# Precision Testing: TF32x1 vs TF32x3

Verify that hipBLASLt dispatches TF32x1 and TF32x3 through different compute passes on AMD GPUs.

## Background

On AMD/ROCm, PyTorch sends `HIPBLAS_COMPUTE_32F_FAST_TF32` (xf32) to hipBLASLt whenever `allow_tf32=True`. The env var `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32` overrides the compute type **inside** hipBLASLt to select the actual accumulation strategy:

| tf32_mode | Override value | hipBLAS compute type | Accumulation |
|-----------|---------------|----------------------|--------------|
| disabled  | (unset)       | `HIPBLAS_COMPUTE_32F` | Full FP32 |
| x3        | 1 (default)   | `HIPBLAS_COMPUTE_32F_FAST_TF32` | Triple BF16 (BF16x3 on gfx950) |
| x1        | 2             | `HIPBLAS_COMPUTE_32F_FAST_16BF` | Single BF16 (BF16x1 on gfx950) |
| (fp32)    | 0             | `HIPBLAS_COMPUTE_32F` | Forces FP32 even when TF32 requested |

On **gfx942** (MI300X/MI308), x1 uses native TF32 (10-bit mantissa). On **gfx950** (MI350X/MI355), there is no native TF32 — x1 uses a single BF16 matmul and x3 uses three BF16 matmuls for higher accuracy.

## Prerequisites

- Docker with GPU access (`--device=/dev/kfd --device=/dev/dri`)
- AMD Instinct GPU (MI300X, MI350X, or similar)
- The `private-base-shampoo-only` Docker image (or any image with PyTorch + ROCm)

## Step 1: Build the Docker Image

```bash
cd docker/
docker build --network=host \
  -f Dockerfile.private-base-shampoo-only \
  -t private-base-shampoo-only:latest .
```

> **Note**: `--network=host` is required on systems where the Docker bridge network is not available.

The Dockerfile uses `rocm/pytorch-private:20251030_rocm_e2e_phantom_mi350_genai_nightly` as the base image, which includes hipBLASLt with `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32` support. The build step verifies this at image creation time.

## Step 2: Start a Container

```bash
docker run -d --name precision-tf32-test \
  --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host --shm-size=64g \
  -v /path/to/aorta:/workspace/aorta \
  -w /workspace/aorta \
  private-base-shampoo-only:latest \
  sleep infinity
```

Install the aorta package inside the container:

```bash
docker exec precision-tf32-test pip install -e /workspace/aorta
```

## Step 3: Run the Standalone TF32 Probe

```bash
docker exec precision-tf32-test python3.10 scripts/tf32_probe.py
```

### Expected Output (PASS)

```
Device: AMD Instinct MI350X
PyTorch: 2.9.0+git25a49ce
Platform: ROCm 7.0.51831-7c9236b16

Running matmuls...
  FP32 (disabled)      | OVERRIDE_XF32= None | allow_tf32=False | precision=highest
  TF32x3 (Override=1)  | OVERRIDE_XF32=    1 | allow_tf32= True | precision=high
  TF32x1 (Override=2)  | OVERRIDE_XF32=    2 | allow_tf32= True | precision=high
  FP32   (Override=0)  | OVERRIDE_XF32=    0 | allow_tf32= True | precision=high

========================================================================
RESULTS
========================================================================

Accuracy vs fp64 ground truth (max abs error):
  FP32   : 1.0638e-03  (mean 5.1402e-05)
  TF32x3 : 1.5786e-03  (mean 2.2891e-04)
  TF32x1 : 8.2363e-01  (mean 1.1993e-01)

Pairwise deltas (max abs diff):
  FP32  vs TF32x3 (Override=1): 1.7700e-03  (mean 2.3603e-04)
  FP32  vs TF32x1 (Override=2): 8.2367e-01  (mean 1.1993e-01)
  TF32x3 vs TF32x1            : 8.2305e-01  (mean 1.1993e-01)
  FP32  vs FP32   (Override=0): 0.0000e+00  (sanity check, should be 0)

Verification:
  TF32x3 active (Override=1 differs from FP32) : YES
  TF32x1 active (Override=2 differs from FP32) : YES
  x1 != x3 (different hipblaslt compute passes): YES
  Override=0 == FP32 (sanity check)            : YES

PASS: TF32x1 and TF32x3 use two different hipblaslt compute passes
  TF32x3 (Override=1) → HIPBLAS_COMPUTE_32F_FAST_TF32  (triple BF16 accumulation)
  TF32x1 (Override=2) → HIPBLAS_COMPUTE_32F_FAST_16BF  (single BF16 accumulation)
```

The probe runs a 4096x4096 fp32 matmul under each mode and checks:
1. Both x1 and x3 differ from pure FP32 (TF32 is active)
2. x1 and x3 differ from each other (different hipBLASLt kernel paths)
3. Override=0 matches FP32 exactly (sanity check)

### If the Probe FAILs

| Failure | Meaning | Fix |
|---------|---------|-----|
| Neither mode active | hipBLASLt does not support TF32 on this hardware | Update hipBLASLt (see below) |
| x1 == x3 identical | Both modes route to the same kernel | hipBLASLt is missing the override support — update it |
| Override=0 != FP32 | hipBLASLt ignoring the override env var entirely | Check `strings libhipblaslt.so \| grep OVERRIDE` |

## Updating hipBLASLt

The base Docker image ships hipBLASLt v1.2.0 which already supports `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32`. If you need a newer version:

### Check Current Version

```bash
docker exec precision-tf32-test bash -c '
  cat /opt/rocm/lib/cmake/hipblaslt/hipblaslt-config-version.cmake | grep PACKAGE_VERSION
  strings /opt/rocm/lib/libhipblaslt.so | grep HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32
'
```

### Install from Source (therock releases)

The [ROCm rocm-libraries releases](https://github.com/ROCm/rocm-libraries/releases) publish hipBLASLt **source** tarballs (not pre-built binaries). To update:

```bash
# Download source (example: therock-7.11)
cd /tmp
wget https://github.com/ROCm/rocm-libraries/releases/download/therock-7.11/hipblaslt.tar.gz
tar xzf hipblaslt.tar.gz

# Build from source (requires ROCm dev tools)
cd hipblaslt
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/rocm -DAMDGPU_TARGETS="gfx942;gfx950"
make -j$(nproc)
make install
```

### Verify the Override is Supported

After installing, confirm the env var exists in the binary:

```bash
strings /opt/rocm/lib/libhipblaslt.so | grep HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32
```

Then re-run the probe to verify.

## Using TF32 Modes in Training

### Config File

Set `tf32_mode` in the precision section of your YAML config:

```yaml
# TF32x3 — triple BF16 accumulation (most accurate)
precision:
  param_dtype: bf16
  reduce_dtype: fp32
  buffer_dtype: fp32
  tf32_mode: x3
```

```yaml
# TF32x1 — single BF16 accumulation (fastest)
precision:
  param_dtype: bf16
  reduce_dtype: fp32
  buffer_dtype: fp32
  tf32_mode: x1
```

Pre-built configs are available:
- `config/multi_node/shampoo_opt_multi_node_seed42_tf32x1.yaml`
- `config/multi_node/shampoo_opt_multi_node_seed42_tf32x3.yaml`

### How It Works

When `fsdp_trainer.py` starts:

1. `configure_tf32_precision()` reads `tf32_mode` from the config
2. Sets `torch.set_float32_matmul_precision("high")` (enables TF32 in PyTorch)
3. Sets `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32` to select the hipBLASLt pass:
   - `x1` → override=2 → `HIPBLAS_COMPUTE_32F_FAST_16BF`
   - `x3` → override=1 → `HIPBLAS_COMPUTE_32F_FAST_TF32`
4. `verify_tf32_active()` runs a matmul probe to confirm the mode is active

### Comparing Precision Runs

After running training with both modes, compare loss curves:

```bash
python scripts/compare_precision_runs.py \
  --baseline-dir experiments/multinode_*_precision_tf32x3 \
  --compare-dir  experiments/multinode_*_precision_tf32x1
```

## Debugging

### Enable hipBLASLt Logging

To see which compute type hipBLASLt receives for each matmul:

```bash
HIPBLASLT_LOG_LEVEL=5 HIPBLASLT_LOG_MASK=32 python3.10 scripts/tf32_probe.py 2>&1 | grep computeType
```

Expected output:
- `computeType=COMPUTE_32F` for pure FP32
- `computeType=COMPUTE_32XF` for TF32 (before override is applied)

### Key Env Vars

| Variable | Read by | Purpose |
|----------|---------|---------|
| `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32` | hipBLASLt | Override xf32 compute type (0=fp32, 1=keep xf32, 2=bf16) |
| `HIPBLASLT_LOG_LEVEL` | hipBLASLt | Logging verbosity (0-5) |
| `HIPBLASLT_LOG_MASK` | hipBLASLt | Log category bitmask (32 = API + trace) |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | hipBLASLt | Custom kernel selection overrides |

### Common Pitfall: `HIPBLASLT_ALLOW_TF32`

This env var **does not exist** in hipBLASLt or PyTorch. It was a hypothetical flag. The correct env var is `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32`. If you see code or docs referencing `HIPBLASLT_ALLOW_TF32`, it needs to be updated.

## Cleanup

```bash
docker stop precision-tf32-test && docker rm precision-tf32-test
```
