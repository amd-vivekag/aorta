# Docker Setup for Aorta

This directory contains Docker configurations for building and running Aorta training workloads.

## Overview

We provide a unified Docker Compose configuration that supports multiple Dockerfile variants through environment variables. Each user can maintain their own `.env` file (git-ignored) with personalized settings.

## Quick Start

### Method 1: Interactive Setup (Recommended)

Run the setup script to create your `.env` file interactively:

```bash
bash setup-env.sh
docker compose -f docker-compose.build.yaml up -d
```

The script will guide you through:
- Selecting a Dockerfile
- Naming your container
- Configuring volume mounts
- Setting environment variables

### Method 2: Manual Configuration

Copy the example and edit manually:

```bash
cp .env.example .env
# Edit .env with your settings
docker compose -f docker-compose.build.yaml up -d
```

## Available Dockerfiles

| Dockerfile | Base Image | PyTorch | ROCm Build | Shampoo | Stress Test |
|------------|-----------|---------|------------|---------|-------------|
| `Dockerfile.private-base-shampoo-only` | `rocm/pytorch-private:20251030...nightly` | 2.9.0 (base image) | None (base image) | Pinned (840de49) | **PASS** |
| `Dockerfile.private-meta19-torch2.12-shampoo` | `rocm/pytorch-private:20251030...nightly` | 2.12.0a0 (custom wheel) | `compute-rocm-rel-7.0-meta/19` | Latest | **FAIL (SIGSEGV)** |
| `Dockerfile.private-rocm7021-torch2.11-shampoo` | `rocm/pytorch-private:20251030...nightly` | 2.11.0a0 (custom wheel) | `compute-rocm-rel-7.0.2.1/6` | Latest | **FAIL (SIGSEGV)** |
| `Dockerfile.public-rocm72-base` | `rocm/pytorch:rocm7.2_ubuntu22.04` | 2.9.1 (base image) | None (public ROCm 7.2) | Not installed | N/A |
| `Dockerfile.ubuntu-meta19-nightly-pip` | `ubuntu:22.04` | 2.11.0 nightly (pip) | `compute-rocm-rel-7.0-meta/19` | Not installed | N/A |
| `Dockerfile.ubuntu-meta19-source-build` | `ubuntu:22.04` | Built from source | `compute-rocm-rel-7.0-meta/19` | Not installed | N/A |

### Stress Test Notes

- The **only passing** Dockerfile is `Dockerfile.private-base-shampoo-only`, which uses the unmodified base image PyTorch 2.9.0 with Shampoo pinned to commit `840de49`.
- Both custom-wheel Dockerfiles (torch 2.11 and 2.12) crash with SIGSEGV on all 8 GPUs within seconds of starting training.
- All images use the same hipBLASLt (`1.2.0-54d5b15ade`) from the base image; the custom hipBLASLt in `hipblaslt_install/` is source only (not compiled).
- The SIGSEGV is caused by the custom PyTorch wheels, not ROCm or hipBLASLt.

## Configuration Variables

### Required

- **`DOCKERFILE`**: Which Dockerfile to build from
- **`CONTAINER_NAME`**: Unique name for your container (avoid conflicts with other users)

### Volume Mounts

- **`AORTA_WORKSPACE`**: Path to aorta workspace (default: `..`)
- **`RCCL_PATH`**: Optional. Leave unset to use the image's RCCL (no YAML edit needed). To use a custom RCCL build, set this and run with `-f docker-compose.rccl.yaml` (see [Using custom RCCL](#using-custom-rccl)).

### Optional

- **`AMDGPU_DRIVER_VARIANT`**: Driver variant for environment_info.json
- **`EXTRA_MOUNT_SRC_*`** / **`EXTRA_MOUNT_DST_*`**: Additional volume mounts

## Example Configurations

### Example 1: Working Shampoo Setup (Recommended)

```bash
# .env
DOCKERFILE=Dockerfile.private-base-shampoo-only
CONTAINER_NAME=myuser-shampoo-working
AORTA_WORKSPACE=..
```

Run: `docker compose -f docker-compose.build.yaml up -d`

### Example 2: Standard Development (image RCCL)

```bash
# .env
DOCKERFILE=Dockerfile.public-rocm72-base
CONTAINER_NAME=myuser-dev-20260205
AORTA_WORKSPACE=..
# RCCL_PATH unset = use image RCCL
```

Run: `docker compose -f docker-compose.build.yaml up -d`

### Example 3: Shampoo with Custom RCCL

```bash
# .env
DOCKERFILE=Dockerfile.private-base-shampoo-only
CONTAINER_NAME=shampoo-experiment-1
AORTA_WORKSPACE=/apps/username/aorta_work/aorta_1
RCCL_PATH=/apps/username/rccl
```

Run: `docker compose -f docker-compose.build.yaml -f docker-compose.rccl.yaml up -d`

## Using custom RCCL

By default, the container uses the RCCL bundled in the image. You do not need to set or remove any RCCL path in the YAML.

To use a custom RCCL build:

1. Set `RCCL_PATH` in your `.env` to your RCCL build directory.
2. Run with the RCCL override file:

   ```bash
   docker compose -f docker-compose.build.yaml -f docker-compose.rccl.yaml up -d
   ```

The override file adds the RCCL volume and RCCL-related environment variables only when you use it.

## File Structure

```
docker/
├── docker-compose.build.yaml                          # Unified compose file (use this!)
├── docker-compose.rccl.yaml                           # Optional: use with -f when RCCL_PATH is set
├── docker-compose.private-base-nightly.yaml           # Image-based compose (alternative)
├── docker-compose.private-meta19-torch2.12-shampoo.yaml   # Meta/19 + torch 2.12 (SIGSEGV)
├── docker-compose.private-rocm7021-torch2.11-shampoo.yaml # ROCm 7.0.2.1 + torch 2.11 (SIGSEGV)
├── docker-compose.public-rocm72-base.yaml             # Public ROCm 7.2 base
├── .env.example                                       # Template for your .env
├── .env                                               # Your personal config (git-ignored)
├── setup-env.sh                                       # Interactive setup script
├── Dockerfile.private-base-shampoo-only               # Base image + pinned Shampoo (WORKING)
├── Dockerfile.private-meta19-torch2.12-shampoo        # Meta/19 + custom torch 2.12 (SIGSEGV)
├── Dockerfile.private-rocm7021-torch2.11-shampoo      # ROCm 7.0.2.1 + custom torch 2.11 (SIGSEGV)
├── Dockerfile.public-rocm72-base                      # Public ROCm 7.2 Ubuntu base
├── Dockerfile.ubuntu-meta19-nightly-pip               # Ubuntu + meta/19 + torch nightly pip
├── Dockerfile.ubuntu-meta19-source-build              # Ubuntu + meta/19 + torch from source
├── hipblaslt_install/                                 # hipBLASLt source (not compiled)
├── wheels/                                            # Custom PyTorch wheels (if any)
└── rccl_test/                                         # Separate RCCL testing setup
```

## Common Commands

### Start Container

```bash
docker compose -f docker-compose.build.yaml up -d
```

### Stop Container

```bash
docker compose -f docker-compose.build.yaml down
```

### View Logs

```bash
docker compose -f docker-compose.build.yaml logs -f
```

### Connect to Container

```bash
docker exec -it <your-container-name> bash
```

### Rebuild After Dockerfile Changes

```bash
docker compose -f docker-compose.build.yaml build
docker compose -f docker-compose.build.yaml up -d
```

### View Resolved Configuration

See what environment variables are being used:

```bash
docker compose -f docker-compose.build.yaml config
```

## Tips

1. **Unique Container Names**: Use descriptive, unique names to avoid conflicts with other users on shared systems
   - Good: `username-shampoo-2026-02-05`
   - Bad: `training` (too generic)

2. **Git Ignore**: Your `.env` file is git-ignored, so your personal configuration won't be committed

3. **Environment Override**: You can override any variable at runtime:
   ```bash
   CONTAINER_NAME=test-run docker compose -f docker-compose.build.yaml up
   ```

4. **VSCode Integration**: Use VSCode's "Attach to Running Container" feature for an IDE experience

5. **Multiple Variants**: You can run multiple containers with different Dockerfiles simultaneously by using different container names

## Troubleshooting

### "container name already in use"

Another user or previous run is using that name. Choose a different `CONTAINER_NAME`.

### "No such file or directory" for volumes

Check that paths in your `.env` exist and are accessible:
```bash
ls -la $AORTA_WORKSPACE
ls -la $RCCL_PATH
```

### Changes to .env not taking effect

Stop and restart the container:
```bash
docker compose -f docker-compose.build.yaml down
docker compose -f docker-compose.build.yaml up -d
```

### Need to add more volume mounts

Edit your `.env` and add:
```bash
EXTRA_MOUNT_SRC_1=/path/on/host
EXTRA_MOUNT_DST_1=/path/in/container
```

Then update `docker-compose.build.yaml` to reference them in the volumes section.

## Migration from Old Compose Files

If you were using:
- `docker-compose.rocm70_9-1.yaml` → Use `docker-compose.build.yaml` with `DOCKERFILE=Dockerfile.public-rocm72-base`
- `docker-compose.rocm70_9-1-shampoo.yaml` → Use `docker-compose.build.yaml` with `DOCKERFILE=Dockerfile.private-base-shampoo-only`

These old files are deprecated and will be removed in a future update.

## Related Documentation

- [Getting Started Guide](../docs/getting-started.md)
- [Running Benchmarks](../docs/running-benchmark.md)
- [Profiling Guide](../docs/profiling.md)
