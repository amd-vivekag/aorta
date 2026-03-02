# SDPA Multi-Node Testing

This guide explains how to run SDPA (Scaled Dot Product Attention) backward testing on multi-node distributed setups using AORTA's infrastructure.

## Overview

The SDPA testing infrastructure allows you to:
- Run SDPA backward operations in a distributed FSDP context
- Test for NaN/Inf issues across multiple nodes and GPUs
- Reproduce race conditions and numerical instabilities
- Iterate until NaN is detected or max iterations reached

## Quick Start

### Single Node Test

```bash
# Using torchrun directly
torchrun --nproc_per_node=8 scripts/run_sdpa_distributed_test.py \
  --config config/multi_node/sdpa_test_local_inputs.yaml

# Using local_launch.sh (if in Docker)
./scripts/multi_node/local_launch.sh \
  --nproc 8 \
  --config config/multi_node/sdpa_test_local_inputs.yaml \
  --entry-script scripts/run_sdpa_distributed_test.py
```

### Multi-Node Test

```bash
# From master node
./scripts/multi_node/master_launch.sh \
  --config config/multi_node/sdpa_test_multi_node.yaml \
  --nproc 8 \
  --label sdpa_test
```

## Configuration

### Configuration Files

Two configuration files are provided:

1. **`config/multi_node/sdpa_test_multi_node.yaml`**
   - Uses original input location: `/home/vivekag/scratch/apps/aorta_work/nan_issue/sdpa/input`
   - Suitable when the source directory is accessible across nodes

2. **`config/multi_node/sdpa_test_local_inputs.yaml`**
   - Uses local copy: `data/sdpa_inputs/`
   - Recommended for containerized environments

### Key Configuration Options

#### SDPA Test Settings (`sdpa_test` section)

```yaml
sdpa_test:
  # Path to SDPA input files
  input_dir: data/sdpa_inputs
  
  # Maximum iterations before stopping
  max_iterations: 1000
  
  # Device to use (cuda or cpu)
  device: cuda
  
  # Enable verbose logging
  verbose: false
  
  # Broadcast inputs from rank 0 (recommended for multi-node)
  broadcast_inputs: true
  
  # Output directory for results
  output_dir: artifacts/sdpa_test
```

#### Distributed Settings

```yaml
distributed:
  mode: fsdp           # Use FSDP for multi-node
  backend: nccl        # NCCL for GPU communication

fsdp:
  sharding_strategy: hybrid_shard  # Shard across nodes, replicate within nodes
  sync_module_states: false        # No model to sync
```

#### Race Condition Testing

```yaml
race_experiment:
  gpu_max_hw_queues: 4          # Enable hardware queue parallelism
  skip_rccl_warmup: false       # Enable RCCL warmup (recommended)
  rccl_warmup_iterations: 5     # Warmup iterations
```

## Input Files

### Required Input Files

The SDPA test requires the following input files:

- `metadata.json` - Metadata about the saved operation
- `input_grad_out.pt` - Gradient output tensor
- `input_query.pt` - Query tensor
- `input_key.pt` - Key tensor
- `input_value.pt` - Value tensor
- `input_out.pt` - Output from forward pass
- `input_logsumexp.pt` - LogSumExp from forward pass

### Optional Input Files

- `input_philox_seed.pt` - Philox seed for dropout
- `input_philox_offset.pt` - Philox offset for dropout
- `input_cum_seq_q.pt` - Cumulative sequence lengths (query)
- `input_cum_seq_k.pt` - Cumulative sequence lengths (key)

### Preparing Input Files

If you need to create new input files:

1. Run a training job with NaN detection enabled
2. Use the original `replay_sdpa_backward.py` script to save inputs
3. Copy files to `data/sdpa_inputs/` or update config to point to your location

## Architecture

### Execution Flow

```
┌─────────────────────────────────────────────┐
│  Master Launch (master_launch.sh)           │
│  - Coordinates all nodes                    │
│  - Starts torchrun on each node            │
└────────────────┬────────────────────────────┘
                 │
                 ├──> Node 0 (Master)
                 ├──> Node 1
                 └──> Node N
                      │
                      ├──> Rank 0
                      ├──> Rank 1
                      └──> Rank N
                           │
                           ├──> Initialize distributed
                           ├──> Load/broadcast SDPA inputs
                           ├──> Run iteration loop
                           ├──> Execute SDPA backward
                           ├──> Check for NaN/Inf
                           └──> Report results
```

### Input Broadcasting Strategy

For efficient multi-node execution:

1. **Rank 0** loads SDPA inputs from disk
2. **Rank 0** broadcasts tensor metadata (shapes, dtypes) to all ranks
3. **All ranks** allocate tensors based on metadata
4. **Rank 0** broadcasts actual tensor data
5. **All ranks** execute SDPA backward with identical inputs

This ensures consistency while minimizing I/O operations.

### NaN Detection

NaN detection is synchronized across all ranks:

1. Each rank checks its output tensors for NaN/Inf
2. Results are aggregated using `torch.distributed.all_reduce()`
3. If any rank detects NaN, all ranks stop
4. Rank 0 aggregates and reports final results

## Running Tests

### Example: Single Node, 8 GPUs

```bash
cd /home/vivekag/scratch/apps/aorta_work/aorta

# Activate environment (if needed)
# source venv/bin/activate

# Run test
torchrun --nproc_per_node=8 \
  --master_port=29500 \
  scripts/run_sdpa_distributed_test.py \
  --config config/multi_node/sdpa_test_local_inputs.yaml
```

### Example: Multi-Node (2 nodes, 8 GPUs each)

```bash
# Ensure node_ip_list.txt is configured
cd /home/vivekag/scratch/apps/aorta_work/aorta/scripts/multi_node

# Start Docker containers on all nodes
./start_docker_all_nodes.sh

# Launch test from master node
./master_launch.sh \
  --config ../../config/multi_node/sdpa_test_multi_node.yaml \
  --nproc 8 \
  --channels 28 \
  --threads 256 \
  --label sdpa_nan_test
```

### Example: Custom Iteration Count

Create a custom config or override:

```yaml
sdpa_test:
  max_iterations: 5000  # Run up to 5000 iterations
  verbose: true         # Enable detailed logging
```

## Interpreting Results

### Successful Run (No NaN)

```
[Rank 0] Starting SDPA test loop (max_iterations=1000)...
[Rank 0] Iteration 10/1000
[Rank 0] Iteration 20/1000
...
[Rank 0] Completed 1000 iterations without NaN/Inf
======================================================================
SDPA DISTRIBUTED TEST RESULTS
======================================================================
PASS - Completed 1000 iterations without NaN/Inf on all ranks
======================================================================
```

Exit code: 0

### NaN Detected

```
[Rank 0] Iteration 347/1000
[Rank 0] Iteration 347: NaN/Inf DETECTED!
[Rank 0]   grad_query: NaN=True (count=1247), Inf=False (count=0)
======================================================================
SDPA DISTRIBUTED TEST RESULTS
======================================================================
NaN/Inf DETECTED across ranks!
  Rank 0: NaN detected at iteration 347
  Rank 1: NaN detected at iteration 347
  Rank 2: No NaN detected
  ...
======================================================================
```

Exit code: 1

### Execution Failure

```
[Rank 0] Iteration 123: ERROR running SDPA backward: ...
======================================================================
FAILURES detected across ranks!
  Rank 0: Failed at iteration 123
======================================================================
```

Exit code: 2

## Logs and Artifacts

### Log Files

Per-rank log files are created in the output directory:

```
artifacts/sdpa_test/
├── rank0.log
├── rank1.log
├── rank2.log
└── ...
```

### Viewing Logs

```bash
# View rank 0 log
tail -f artifacts/sdpa_test/rank0.log

# Search for NaN across all ranks
grep -r "NaN DETECTED" artifacts/sdpa_test/

# View summary from all ranks
for f in artifacts/sdpa_test/rank*.log; do
  echo "=== $f ==="
  tail -20 $f
done
```

## Troubleshooting

### Issue: "metadata.json not found"

**Cause**: Input directory is incorrect or files are missing

**Solution**:
```bash
# Verify input directory exists and contains files
ls -la data/sdpa_inputs/
ls -la /home/vivekag/scratch/apps/aorta_work/nan_issue/sdpa/input/

# Update config to point to correct location
vim config/multi_node/sdpa_test_multi_node.yaml
```

### Issue: "Missing required inputs"

**Cause**: Required tensor files are missing

**Solution**:
Ensure all required files exist:
- `input_grad_out.pt`
- `input_query.pt`
- `input_key.pt`
- `input_value.pt`
- `input_out.pt`
- `input_logsumexp.pt`

### Issue: Distributed initialization timeout

**Cause**: Network connectivity issues or NCCL configuration problems

**Solution**:
```bash
# Check NCCL environment variables
echo $NCCL_SOCKET_IFNAME
echo $NCCL_IB_HCA

# Verify network connectivity between nodes
ping <other_node_ip>

# Check if RCCL warmup is enabled
grep "skip_rccl_warmup" config/multi_node/sdpa_test_multi_node.yaml

# Increase timeout
export TORCH_DIST_INIT_TIMEOUT=300
```

### Issue: Ranks hang or deadlock

**Cause**: Mismatched collective operations or broadcast issues

**Solution**:
```bash
# Enable debug logging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Check if broadcast_inputs is enabled
grep "broadcast_inputs" config/multi_node/sdpa_test_multi_node.yaml

# Verify all ranks are starting
grep "Initialized distributed" artifacts/sdpa_test/rank*.log
```

### Issue: Out of memory

**Cause**: SDPA inputs are too large or too many GPUs per node

**Solution**:
```bash
# Run with fewer GPUs per node
./scripts/multi_node/master_launch.sh --nproc 4 ...

# Check tensor sizes
grep "shape=" artifacts/sdpa_test/rank0.log
```

### Issue: CUDA errors or invalid device

**Cause**: GPU not available or incorrect device configuration

**Solution**:
```bash
# Verify GPU availability
nvidia-smi
# or
rocm-smi

# Check device configuration
grep "device:" config/multi_node/sdpa_test_multi_node.yaml

# Verify LOCAL_RANK environment variable
echo $LOCAL_RANK
```

## Advanced Usage

### Custom SDPA Inputs

To test with custom SDPA inputs:

1. Create input directory:
   ```bash
   mkdir -p custom_inputs
   ```

2. Save tensors in the required format:
   ```python
   import torch
   
   # Save tensors
   torch.save(grad_out, "custom_inputs/input_grad_out.pt")
   torch.save(query, "custom_inputs/input_query.pt")
   # ... etc
   
   # Save metadata
   metadata = {
       "func_name": "_scaled_dot_product_flash_attention_backward.default",
       "saved_inputs": {
           "grad_out": "input_grad_out.pt",
           "query": "input_query.pt",
           # ... etc
       }
   }
   import json
   with open("custom_inputs/metadata.json", "w") as f:
       json.dump(metadata, f)
   ```

3. Update config:
   ```yaml
   sdpa_test:
     input_dir: custom_inputs
   ```

### Integration with Existing Training

To capture SDPA inputs from actual training:

1. Enable NaN detection in training config
2. Training will save inputs when NaN is first detected
3. Copy saved inputs to SDPA test directory
4. Run distributed test to reproduce

### Profiling SDPA Operations

To profile SDPA execution:

```yaml
profiling:
  enabled: true
  active: 10           # Profile 10 iterations
  chrome_trace: true
  trace_filename: sdpa_profile.json
```

View traces:
```bash
google-chrome artifacts/sdpa_test/sdpa_profile.json
```

## Reference

### Environment Variables

- `TORCH_DIST_INIT_TIMEOUT` - Distributed initialization timeout (seconds)
- `NCCL_DEBUG` - NCCL debug level (INFO, WARN, ERROR)
- `NCCL_SOCKET_IFNAME` - Network interface for NCCL
- `NCCL_IB_HCA` - InfiniBand HCAs to use
- `GPU_MAX_HW_QUEUES` - Hardware queue parallelism (set via config)
- `LOCAL_RANK` - Local rank (set by torchrun)
- `RANK` - Global rank (set by torchrun)
- `WORLD_SIZE` - Total number of ranks (set by torchrun)

### Script Parameters

```bash
scripts/run_sdpa_distributed_test.py --help

Arguments:
  --config CONFIG       Path to YAML configuration file (required)
  --local-rank RANK     Local rank (set by torchrun, optional)
```

### Multi-Node Launch Parameters

```bash
scripts/multi_node/master_launch.sh --help

Options:
  -c, --channels        NCCL_MAX_NCHANNELS (default: 28)
  -t, --threads         RCCL_THREADS_PER_BLOCK (default: 256)
  -p, --nproc           GPUs per node (default: 8)
  -f, --config          Config file path (required)
  -d, --docker          Docker container name
  -l, --label           Experiment label suffix
  --master-port         Master port (auto-selected if not specified)
```

## Related Documentation

- [Running Multi-Node Training](running-benchmark.md)
- [Profiling Guide](profiling.md)
- [Troubleshooting](troubleshooting.md)
- [Multi-Node Setup](../scripts/multi_node/README.md)

## Contact

For issues or questions, refer to the project README or contact the development team.
