# ROCm / AMD GPU Internals — Deep-Dive Knowledge Base

**Date:** March 2026
**Target Hardware:** AMD MI350X (CDNA architecture)
**Purpose:** Low-level reference for debugging GPU race conditions, NaN corruption,
and stream synchronization issues on the ROCm stack.

---

## Table of Contents

1. [GPU Memory Hierarchy](#1-gpu-memory-hierarchy)
2. [AQL — Architected Queuing Language](#2-aql--architected-queuing-language)
3. [Streams vs Hardware Queues](#3-streams-vs-hardware-queues)
4. [Hardware Queue Virtualization](#4-hardware-queue-virtualization)
5. [MES — Micro Engine Scheduler](#5-mes--micro-engine-scheduler)
6. [Kernel Dispatch Pipeline](#6-kernel-dispatch-pipeline)
7. [Cross-Stream Synchronization](#7-cross-stream-synchronization)
8. [SDMA Engines](#8-sdma-engines)
9. [RCCL Internals](#9-rccl-internals)
10. [Stream Priorities](#10-stream-priorities)
11. [ROCm Software Stack](#11-rocm-software-stack)
12. [Debugging & Observability](#12-debugging--observability)
13. [CSAN Race Analysis — Meta NaN Case Study](#13-csan-race-analysis--meta-nan-case-study)

---

## 1. GPU Memory Hierarchy

### HBM vs VRAM vs L2 Cache

These are three distinct layers, often conflated but fundamentally different:

| Property | HBM | VRAM | L2 Cache |
|----------|-----|------|----------|
| **What it is** | Physical DRAM technology (HBM3e on MI350X) | Software memory domain | On-die SRAM cache |
| **Size** | ~288 GB | Same footprint as HBM | Tens of MB |
| **Technology** | Stacked DRAM on silicon interposer | Logical concept (address space) | On-die SRAM |
| **Speed** | High bandwidth (~TB/s), higher latency | N/A (it *is* HBM) | Lower bandwidth, much lower latency |
| **Role** | Bulk storage | How software classifies GPU-local memory | Reduces traffic to HBM |
| **Coherence** | Not a coherence point | Not a coherence point | **The** coherence point |

- **HBM** is the physical medium — stacked DRAM dies connected via silicon interposer.
- **VRAM** is the software abstraction. `hipMalloc` returns VRAM allocations; the amdgpu driver's
  TTM subsystem categorizes them as `AMDGPU_GEM_DOMAIN_VRAM` vs `AMDGPU_GEM_DOMAIN_GTT`.
- **L2 Cache** is SRAM sitting between CUs and HBM. It's the point of coherence — all CUs
  across all Shader Engines see a consistent view of memory at the L2 level.

### Memory Access Path

```
CU (Compute Unit)
  → L0 vector cache (per-SIMD, ~16KB)
    → L1 cache (per-CU, ~32KB, read-only for global memory on CDNA)
      → L2 cache (shared across all CUs, SRAM, multi-MB)
        → HBM (physical DRAM)
```

### GPU Virtual Memory (GPUVM)

AMD GPUs have a full virtual memory system with multi-level page tables
(PD0 → PD1 → PD2 → PD3 → PTE), analogous to x86:

- **MMHUB** (Memory Management Hub) handles VRAM-side translations.
- **ATHUB** (Address Translation Hub) handles system memory (GTT) access via PCIe/xGMI.

### Memory Types

- **VRAM (Local)** — HBM on the GPU die, managed by the memory controller.
- **GTT (Graphics Translation Table)** — CPU-visible memory accessible by the GPU over PCIe.
  This is where pinned host memory lives.
- **GART (Graphics Aperture Remapping Table)** — maps system memory into the GPU's address space.

### HIP Allocator Chain

```
hipMalloc → hsa_amd_memory_allocate → amdgpu ioctl → GEM BO in VRAM
hipHostMalloc (mapped) → GTT BO with GPU mapping
PyTorch CachingAllocator → recycles freed BOs (critical for NaN investigation)
```

### L2 Coherence and Cross-Queue Correctness

When two kernels on different streams (queues) access the same memory:

1. Kernel A on stream 1 writes → data flows through L2 → eventually to HBM.
2. Kernel B on stream 2 reads the same address → checks L2.

If the synchronization barrier is correct, the CP inserts:
- `buffer_wbl2` (write-back L2) after kernel A
- `buffer_invl2` (invalidate L2) before kernel B

Without correct barrier packets, kernel B may read stale L2 data — not NaN per se,
but stale embeddings or partial writes that downstream attention/softmax turns into NaN
via `log(0)` or `exp(huge)`.

---

## 2. AQL — Architected Queuing Language

AQL is the fundamental CPU→GPU command submission interface on AMD hardware,
defined by the HSA (Heterogeneous System Architecture) specification.

### Ring Buffer Location

AQL packets live in a **ring buffer in host DRAM (CPU memory)**, not in VRAM:

```
Host DRAM                              GPU Die
┌──────────────────┐                   ┌─────────────────┐
│                  │                   │                  │
│  AQL Ring Buffer │◄── PCIe read ────│  Command         │
│  (pinned pages)  │                   │  Processor (CP)  │
│                  │                   │                  │
│  Doorbell page   │◄── MMIO write ──│  Doorbell         │
│  (uncacheable)   │    from CPU      │  Controller       │
│                  │                   │                  │
└──────────────────┘                   └─────────────────┘
     CPU writes here                     GPU reads from here
     (normal stores +                    (over PCIe/xGMI)
      doorbell MMIO)
```

The ring buffer is:
- **Physically in host DRAM** (CPU memory)
- **Pinned** (non-pageable) so the GPU always has a valid physical address
- **Mapped into GPUVM page tables** via ATHUB, domain = GTT
- **Read by the GPU** over the PCIe (or xGMI) bus

### Queue Creation

When ROCr creates a queue via `hsa_queue_create()`:

1. Allocates contiguous host memory for the ring buffer (e.g., 16,384 × 64 bytes = 1 MB).
2. KFD ioctl `AMDKFD_IOC_CREATE_QUEUE`:
   - Allocates ring buffer pages in GTT (system memory)
   - Pins pages (GPU needs reliable access)
   - Maps pages into GPU's GPUVM page tables
   - Allocates a doorbell page and returns its address to userspace
3. Returns `hsa_queue_t` with `base_address`, `doorbell_signal`, `size`, `write_index/read_index`.

### AQL Packet Structure

Each AQL packet is **64 bytes** containing:
- Header (packet type, barrier bit, acquire/release fences)
- Kernel object pointer (ISA code in VRAM)
- Kernarg address (kernel arguments, usually in system memory)
- Grid dimensions, workgroup size
- Completion signal (HSA signal pointer)

### AQL Packet Types

| Type | Purpose |
|------|---------|
| `HSA_PACKET_TYPE_KERNEL_DISPATCH` | Launch a compute kernel |
| `HSA_PACKET_TYPE_BARRIER_AND` | Wait until ALL referenced signals are satisfied |
| `HSA_PACKET_TYPE_BARRIER_OR` | Wait until ANY referenced signal is satisfied |
| `HSA_PACKET_TYPE_AGENT_DISPATCH` | Host-side callbacks |

### CPU Write Sequence

```
1. Atomically increment write_index (CAS loop) → reserves a slot
2. Write 64-byte AQL packet fields to ring_buffer[slot]
3. Set header's "valid" bit LAST (release-store semantics)
   → all other fields must be visible before header becomes valid
4. Write to doorbell register (MMIO store) → notifies GPU CP
   → doorbell value = new write_index
```

The header-last ordering is mandated by the AQL spec: the GPU reads the header first
with acquire semantics.

### GPU Read Sequence

The CP (Command Processor) is a **fixed-function microcontroller**, not a shader:

```
1. CP watches doorbell register → wakes on CPU write
2. Compares doorbell value (new write_index) with its read_index
3. Reads AQL packets from ring buffer over PCIe (via ATHUB)
4. Decodes packet type → KERNEL_DISPATCH, BARRIER, etc.
5. For kernel dispatch: fetches ISA code (from VRAM) + kernarg block
6. Programs SPI → CUs begin executing wavefronts
7. On completion: decrements completion signal, advances read_index
```

### Doorbell Mechanism

The doorbell is **memory-mapped IO (MMIO)**, not regular memory:
- Kernel driver maps a doorbell page into userspace via `mmap()` on `/dev/kfd`
- Each queue gets a unique doorbell slot (4 or 8 bytes)
- CPU write → PCIe BAR → GPU's doorbell controller (bypasses CPU caches)
- Fast: single PCIe posted write, no round-trip needed

### Queue Depth and CPU-GPU Asymmetry

AMD MI-series GPUs support a **16,384-entry AQL queue** by default
(vs. NVIDIA's ~1K entries). This is the root cause of Issue A:

- CPU writes 64 bytes to pinned memory + 8 bytes MMIO per dispatch → microseconds
- GPU reads over PCIe, decodes, schedules, executes → milliseconds per kernel
- CPU can enqueue **thousands of kernels** before the ring buffer blocks
- In a triple-buffered pipeline, the CPU may recycle a buffer slot while
  the GPU hasn't even started the kernels referencing its old data

Controllable via: `ROCM_AQL_QUEUE_SIZE=1024` (reduce from 16K to 1K).

---

## 3. Streams vs Hardware Queues

### The Fundamental Difference

- A **stream** is a software abstraction: an ordered sequence of operations.
- A **hardware queue (HWQ)** is the physical AQL ring buffer read by the CP.

### AMD vs NVIDIA: Architectural Divergence

**NVIDIA:** Many streams multiplexed onto a small number of hardware channels (~28-32).
The CUDA driver maintains ordering via dependencies.

**AMD/ROCm:** **One stream = one AQL hardware queue** (one-to-one mapping).
Every HIP stream gets its own dedicated ring buffer, doorbell, and read/write indices.

```
HIP Stream 0  ──→  Hardware Queue 0 (own AQL ring buffer, 16K slots)
HIP Stream 1  ──→  Hardware Queue 1 (own AQL ring buffer, 16K slots)
HIP Stream 2  ──→  Hardware Queue 2 (own AQL ring buffer, 16K slots)
```

This stems from HSA's design philosophy: queues are cheap, so give every producer
its own queue rather than multiplexing.

### Dispatch Is Asynchronous

```python
with torch.cuda.stream(stream_a):
    kernel_1()   # CPU: writes AQL packet → returns IMMEDIATELY
    kernel_2()   # CPU: writes another AQL packet → returns IMMEDIATELY
```

Both calls return to the CPU in microseconds. The GPU may not have started
`kernel_1` yet. The in-order guarantee is on the **GPU side** (CP processes
the single HW queue in FIFO order), not the CPU side.

### Cross-Stream Ordering

Python executes sequentially, so dispatch order follows Python's line order.
But **GPU execution order across different HW queues is NOT guaranteed**.
The hardware provides no cross-queue ordering promise.

The practical concern isn't reordering — it's **overlap**. Kernels on different
streams may overlap on the GPU, accessing the same memory concurrently.

---

## 4. Hardware Queue Virtualization

The GPU has a fixed number of physical queue slots, but supports many more
software queues through virtualization.

### The Queue Hierarchy

```
Software Streams (unlimited)
    │
    ▼
AQL Queues / User Queues (many — managed by KFD)
    │
    ▼
HQD Slots / Hardware Queue Descriptors (limited — ~4-8 per pipe)
    │
    ▼
ACE Pipes (fixed — ~4 compute pipes)
    │
    ▼
CUs (execution units)
```

### ACE Pipes and HQD Slots

- **ACE (Asynchronous Compute Engine)** pipes: typically 4 on CDNA parts.
- **HQD (Hardware Queue Descriptor)** slots: typically 4-8 per ACE pipe.
- Total: ~16-32 HQD slots (actual registers on the GPU die).

An HQD slot holds: ring buffer base address, read/write pointers,
doorbell offset, priority, VMID (page table binding).

### When Queues Exceed HQD Slots

The KFD scheduler virtualizes hardware queue slots, like OS process scheduling:

```
Time ──────────────────────────────────────────►

HQD Slot 0: [Queue 0][Queue 0][Queue 4][Queue 4][Queue 0]
HQD Slot 1: [Queue 1][Queue 5][Queue 5][Queue 1][Queue 1]
                           ▲           ▲
                     preempt/swap  preempt/swap
```

For the typical 3-stream pipeline (memcpy + datadist + default), all 3 queues
fit within HQD slots — fully concurrent, no preemption needed.

Controllable via: `GPU_MAX_HW_QUEUES=1` (force single HW queue for debugging).

---

## 5. MES — Micro Engine Scheduler

MES is a **dedicated RISC microcontroller** embedded on the GPU die that manages
queue scheduling in hardware. It replaced the older software-based KFD scheduling.

### What It Is Physically

- Firmware-driven RISC microcontroller with its own instruction memory, data SRAM,
  register access, and interrupt interface.
- Firmware loaded at driver init (`/lib/firmware/amdgpu/gc_*_mes*.bin`).
- Runs independently of both the CPU and shader CUs.

### Responsibilities

1. **Runlist Management** — KFD sends MES an ordered list of active AQL queues.
   MES maps queues to HQD slots based on priority and availability.

2. **Queue Preemption** — When a higher-priority queue arrives or a time quantum expires:
   - Sends PREEMPT to the target HQD slot
   - CP saves in-flight wavefront state to VRAM (register file, PC, LDS)
   - Unmaps old queue, maps new queue
   - Entirely in hardware/firmware — no CPU involvement

3. **Gang Scheduling** — Maps groups of queues atomically for cooperative kernels.

4. **Hang Detection** — Detects stuck queues and resets without full GPU reset.

### Performance vs Old Software Path

| Property | MES | Old KFD Software Scheduler |
|----------|-----|---------------------------|
| Where it runs | On-die microcontroller | CPU (kernel driver) |
| Queue switch latency | ~100s of nanoseconds | ~10s of microseconds |
| Preemption | Hardware-initiated | CPU interrupt driven |
| CPU involvement | None (after runlist setup) | Every queue switch |

### Inspecting MES

```bash
dmesg | grep -i mes
# amdgpu: MES firmware version: X.Y.Z
ls /lib/firmware/amdgpu/gc_*_mes*.bin
```

---

## 6. Kernel Dispatch Pipeline

Full path from HIP API call to GPU execution:

```
HIP API (hipLaunchKernel)
  → ROCr runtime (libhsa-runtime64.so)
    → Write AQL packet to queue ring buffer
      → Doorbell write (MMIO)
        → Command Processor (CP) reads packet
          → CP programs SPI (Shader Processor Input)
            → SPI dispatches wavefronts to CUs
              → CUs execute CDNA ISA instructions
```

### Wavefront Scheduling on CDNA

- Each CU has 4 SIMD units, each 16-wide.
- A wavefront = 64 work-items (4 cycles on a 16-wide SIMD).
- Each CU can hold up to 16 wavefronts (occupancy).
- The SPI distributes workgroups across CUs within a Shader Engine.

### Kernarg Block

The kernarg block (kernel arguments — pointers to tensors, grid dims, etc.)
is typically in system memory. For each dispatch, the CP does two PCIe reads:
1. AQL packet (from ring buffer)
2. Kernarg data (from kernarg region)

The actual kernel ISA and tensor data live in VRAM — heavy memory traffic during
execution stays on-die (CUs → L2 → HBM).

---

## 7. Cross-Stream Synchronization

### Barrier Packets and Signals

When `stream_b.wait_event(event_recorded_on_stream_a)`:

```
HW Queue 0 (stream_a):              HW Queue 1 (stream_b):
┌─────────────────────┐             ┌─────────────────────┐
│ kernel_1            │             │                     │
├─────────────────────┤             │ BARRIER_AND         │
│ kernel_2            │             │ dep_signal = sig_X  │
├─────────────────────┤             ├─────────────────────┤
│ Signal: write sig_X │── sig_X ──→│ kernel_3            │
└─────────────────────┘             └─────────────────────┘
```

1. `event.record(stream_a)` → appends signal-completion AQL packet to queue 0.
2. `stream_b.wait_event(event)` → appends BARRIER_AND packet to queue 1 referencing `sig_X`.
3. CP for queue 1 checks `sig_X`: if not satisfied, **stalls queue 1 only** (other queues continue).
4. When queue 0's signal fires, queue 1 resumes.

A BARRIER_AND packet can reference up to **5 dependency signals**.

### HSA Signals

Signals are 64-bit integers in **system memory** (host DRAM, GPU-accessible).
`hipStreamSynchronize` ultimately calls `hsa_signal_wait_scacquire()` which
spins or blocks on this memory location.

### Memory Ordering

AMD GPUs have a relaxed memory model:
- `buffer_wbl2` / `buffer_invl2` instructions enforce L2 visibility.
- `s_waitcnt` waits for outstanding memory ops (vmcnt, lgkmcnt, expcnt).
- Within a single queue: CP inserts implicit cache flushes between kernels.
- Across queues: **explicit** barriers or signals required.

---

## 8. SDMA Engines

### Separate Hardware for DMA

SDMA engines are **completely independent hardware** from the compute CUs:

```
Compute HW Queue 0  ──→  ACE 0  ──→  CUs (shader execution)
Compute HW Queue 1  ──→  ACE 1  ──→  CUs (shader execution)
SDMA HW Queue 0     ──→  SDMA Engine 0  ──→  Memory controller (DMA)
SDMA HW Queue 1     ──→  SDMA Engine 1  ──→  Memory controller (DMA)
```

- MI350X has multiple SDMA engines (typically 2-4).
- SDMA has its own queue and packet format (not AQL).
- `hipMemcpyAsync` on a separate stream can truly overlap with compute.

### Copy Path Selection

When `hipMemcpyAsync` is called, the runtime decides:
- **SDMA path:** uses a dedicated SDMA HW queue (default for H2D/D2H).
- **Blit (shader) path:** launches a copy kernel on the compute HW queue.

`HSA_ENABLE_SDMA=0` forces all copies through the shader blit path.

---

## 9. RCCL Internals

RCCL (ROCm Communication Collectives Library) uses multiple communication paths
depending on the operation and topology.

### Three Communication Paths

#### Path 1: Kernel-Based P2P (Intra-Node via xGMI)

```
GPU 0: RCCL kernel ── xGMI direct load/store ──→ GPU 1: RCCL kernel
       (on user's HW queue)                      (on user's HW queue)
```

- Launched on the **user's compute HW queue** — no separate queue.
- Uses global memory operations to peer GPU's VRAM through cross-GPU GPUVM mappings.
- Fastest path; most common for MI-series intra-node communication.

#### Path 2: SDMA-Based Copy (Intra-Node, Sometimes)

- Uses a separate SDMA HW queue.
- RCCL coordinates compute and SDMA queues using signals/barriers.

#### Path 3: Network Proxy Thread (Inter-Node)

```
GPU kernel → staging buffer → CPU proxy thread → network (RoCE/IB) → remote proxy → remote GPU
```

- RCCL spawns background CPU threads (`ncclProxyService`) that poll GPU staging buffers
  and call network APIs (ibv_post_send, etc.).
- Proxy threads use HSA signals to synchronize with GPU, not HW queues.
- GDR (GPU Direct RDMA) can bypass the CPU for some transfers.

### RCCL Channels

RCCL parallelizes collectives using **channels** — multiple kernel instances handling
different data chunks. All channel kernels are launched on the **same user stream**.
Parallelism comes from workgroups within the kernel, not multiple HW queues.

### Key Point for Pipeline Debugging

When PyTorch calls RCCL:

```python
dist.all_to_all(output, input)
# Under the hood: ncclAllToAll(..., comm, current_stream)
```

The RCCL kernel runs on **your stream's HW queue**, serialized with your other work
in that queue. RCCL does not create hidden HW queues for the collective itself.

### RCCL Internal Streams

RCCL creates **internal HIP streams** with HIGH priority for auxiliary GPU operations
(buffer zeroing, scratch setup, signaling). These are separate HW queues used for
small fast kernels that keep the communication pipeline flowing — not for the
collective data movement itself.

---

## 10. Stream Priorities

### API

```python
high_stream = torch.cuda.Stream(priority=-1)  # high priority
low_stream  = torch.cuda.Stream(priority=0)   # normal priority
```

AMD GPUs typically support only **2 priority levels** (HIGH and NORMAL).

### How Priority Maps to Hardware

```
hipStreamCreateWithPriority(priority=HIGH)
  → hsa_queue_create(HSA_QUEUE_PRIORITY_HIGH)
    → AMDKFD_IOC_CREATE_QUEUE { priority = HIGH }
      → MES assigns higher scheduling weight
```

Priority affects two things:
1. **HQD slot allocation** — HIGH queues get mapped first; MES preempts NORMAL
   queues to make room for HIGH queues if needed.
2. **CU dispatch priority** — SPI dispatches workgroups from higher-priority queues
   first when CUs become available.

Priority does **NOT** preempt running wavefronts mid-execution.

### RCCL and Priority

| Scenario | Stream used | Priority |
|----------|-------------|----------|
| `dist.all_to_all()` | Your stream | Your stream's priority |
| RCCL internal async ops | RCCL's internal streams | HIGH |

### CU Sharing

AMD GPUs have **no hardware CU partitioning**. All queues compete for all CUs.
The SPI assigns workgroups based on availability + priority.

---

## 11. ROCm Software Stack

```
Application (PyTorch / TorchRec)
  ↓
torch.compile → Triton → AMDGPU LLVM backend → ISA
  ↓
HIP Runtime (libamdhip64.so)
  ↓
ROCr Runtime (libhsa-runtime64.so) — HSA core runtime
  ↓
KFD (Kernel Fusion Driver) — /dev/kfd ioctls
  ↓
amdgpu kernel driver — GEM/TTM, VM, scheduler, SDMA
  ↓
Hardware: CP, CUs, SDMA, MMHUB, xGMI
```

### KFD Key IOCTLs

| IOCTL | Purpose |
|-------|---------|
| `AMDKFD_IOC_CREATE_QUEUE` | Create an AQL hardware queue |
| `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` | Allocate GPU memory |
| `AMDKFD_IOC_MAP_MEMORY_TO_GPU` | Map memory into GPU page tables |

---

## 12. Debugging & Observability

### Environment Variables

| Variable | Effect |
|----------|--------|
| `ROCM_AQL_QUEUE_SIZE=1024` | Reduce AQL queue depth from 16K to 1K |
| `GPU_MAX_HW_QUEUES=1` | Force all streams through a single HW queue |
| `HSA_ENABLE_SDMA=0` | Force shader blit copies instead of SDMA |
| `AMD_LOG_LEVEL=4` | Full HIP runtime logging (AQL packet submissions) |
| `HSA_ENABLE_INTERRUPT=0` | Force polling instead of interrupt-based signal wait |
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | Disable CCA (control experiment for race bugs) |
| `TORCH_CUDA_SANITIZER=1` | Enable CUDA Stream Sanitizer (CSAN) |

### Tools

| Tool | Purpose |
|------|---------|
| `rocprof` / `rocprofv2` | Kernel timings, HW counters, AQL packet traces |
| `rocr_debug_agent` | GPU exception handler (memory faults, illegal instructions) |
| `umr` | Read GPU registers, dump page tables, inspect VRAM |
| `amdgpu_top` | Real-time GPU utilization |
| CSAN (PyTorch) | Detect unsynchronized cross-stream tensor accesses |

---

## 13. CSAN Race Analysis — Meta NaN Case Study

### Sanitizer Configuration

```
ROCM_AQL_QUEUE_SIZE=1024   (reduced from 16K)
GPU_MAX_HW_QUEUES=1         (single HW queue)
TORCH_CUDA_SANITIZER=1
```

### What CSAN Detected

A data race on tensor with data pointer `139991783932688` between two streams:

**Access 1 — `data_dist_stream`** (stream `140011564741504`):
- Operation: `aten::empty.memory_format` → allocating a new tensor
- Location: `wait_sparse_data_dist` → `KJTAllToAllTensorsAwaitable.__init__` → `torch.empty()`
- Role: Data distribution stage requesting a buffer for all-to-all output

**Access 2 — `default_stream`** (stream `0`):
- Operation: `c10d::alltoall_base_` → reading the `input` argument
- Location: `model_fwd` → RCCL all-to-all collective
- Role: Forward pass stage still reading from a tensor used as all-to-all input

**The race:** `data_dist_stream` writes to memory that `default_stream` is still reading,
with no synchronization between them.

### The Mechanism

```
Iteration N:
  default_stream (0):    [... all_to_all(input=buf_X) ...]
                              still reading buf_X...

Iteration N+1 or N+2:
  data_dist_stream:      [torch.empty() → CCA returns buf_X!]
                              WRITES to buf_X
                              *** RACE ***
```

The Caching Allocator (CCA) recycled the same memory block for `torch.empty()`
while the `default_stream` still had in-flight RCCL kernels reading from it.

### Three Hypotheses

#### Hypothesis 1: Missing `record_stream` (Most Likely)

When a tensor crosses streams, `tensor.record_stream(other_stream)` must be called
so the CCA knows not to recycle the memory until both streams are done:

```python
# Without record_stream: CCA only tracks data_dist_stream
# CCA recycles block as soon as data_dist_stream moves past it
# default_stream still reading → RACE

# With record_stream: CCA tracks both streams
buf.record_stream(torch.cuda.default_stream())
```

**Test:** Audit `record_stream` usage in TorchRec's `dist_data.py` around
`KJTAllToAllTensorsAwaitable`. Add explicit `record_stream` calls and check
if CSAN race disappears.

#### Hypothesis 2: HIP Event Premature Completion

Even with correct `record_stream`, the CCA relies on HIP events to know when
a stream is done with a block:

```
1. Tensor freed (refcount → 0)
2. CCA inserts hipEvent on each stream that used the tensor
3. CCA polls: are all events complete?
4. If yes → return block to free pool
5. If no  → keep in "pending free" list
```

If HIP events report completion prematurely (before GPU actually finishes),
the CCA recycles too early. Possible causes:
- ROCm HIP event implementation bug
- HSA signal vs CUDA event semantic differences
- Edge case with 16K AQL queue depth (event "completed" but RCCL kernel
  data movement not finished)

**Test:** Poll HIP events immediately after recording and verify they
correctly report "not done" while kernels are still running.

#### Hypothesis 3: RCCL Internal Stream Lifetime Mismatch

RCCL's all-to-all may internally reference memory on RCCL's internal
HIGH-priority streams that the CCA doesn't know about. The CCA tracks
the tensor as used on stream 0, but RCCL internally accesses it on
a different stream. When stream 0's event completes, RCCL's internal
stream may still be accessing the data.

**Test:** Set `RCCL_FORCE_SYNC=1` or add `torch.cuda.synchronize()` after
every collective. If the race disappears, RCCL's internal async behavior
is the culprit.

### Recommended Next Steps

1. **Audit `record_stream`** in TorchRec `dist_data.py` for the all-to-all
   input/output tensor flow between `data_dist_stream` and `default_stream`.
2. **HIP event correctness test** — verify events don't report premature completion.
3. **`PYTORCH_NO_CUDA_MEMORY_CACHING=1`** — if race disappears, confirms CCA
   is recycling blocks prematurely.
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`** — force simpler
   allocation strategy for easier debugging.

---

## Appendix: Quick Reference Diagram

```
                     CPU Side                    │            GPU Side
                                                 │
  Python / PyTorch                               │
    │                                            │
    ▼                                            │
  HIP Runtime (libamdhip64.so)                   │
    │                                            │
    ▼                                            │
  ROCr Runtime (libhsa-runtime64.so)             │
    │                                            │
    ▼                                            │
  AQL Ring Buffer ◄── pinned host DRAM           │
    │                                            │
    ├──── PCIe / xGMI ──────────────────────────→│ ATHUB
    │                                            │   │
    │  Doorbell ── MMIO ────────────────────────→│ Doorbell Controller
    │                                            │   │
    │                                            │   ▼
    │                                            │ Command Processor (CP)
    │                                            │   │
    │                                            │   ▼
    │                                            │ MES (Micro Engine Scheduler)
    │                                            │   │
    │                                            │   ▼
    │                                            │ SPI → CUs → L2 → HBM
    │                                            │
  SDMA Queue ── PCIe ───────────────────────────→│ SDMA Engines
                                                 │
  RCCL Proxy Thread ── network ─────────────────→│ NIC (RoCE/IB)
```
