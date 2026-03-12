"""
Pipeline CachingAllocator race reproducer.

Tests the exact CSAN-flagged pattern from Meta's workload:
  1. H2D copy on memcpy_stream (pipeline prefetch)
  2. NCCL all_to_all on data_dist_stream (embedding redistribution)
  3. Forward/backward/optimizer on default_stream
  4. CCA recycles batch tensors from old iterations without record_stream

The race: a tensor from iteration N-2 is H2D-copied on memcpy_stream,
passed to data_dist_stream for NCCL all_to_all, then freed. The CCA
recycles the memory while NCCL is still reading it, because no
record_stream was called for the NCCL stream.

Usage:
    # Test without record_stream (should show corruption)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/pipeline_cca_race.py

    # Test with record_stream fix (should show no corruption)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/pipeline_cca_race.py --record-stream

    # Test with CCA disabled (should show no corruption -- confirms CCA is the cause)
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/pipeline_cca_race.py
"""

import argparse
import logging
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] R%(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    def __init__(self, d_model=256, num_layers=4, vocab_size=50000, num_tasks=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, num_tasks)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x.mean(1))


class PipelineWithRace:
    """3-stage pipeline that exercises the CCA + NCCL race pattern.

    The key race: NCCL all_to_all reads send_buf on an RCCL-internal stream.
    If the send_buf is freed (Python reference dropped) without record_stream
    being called for RCCL's internal stream, the CCA will recycle the memory
    while RCCL is still reading. The next allocation may overwrite the memory,
    causing RCCL to read garbage → corrupted gradients → NaN.

    Pipeline stages (matching Meta's TrainPipelineSparseDist):
      - Stage 1 (memcpy_stream): H2D copy for iteration N+2
      - Stage 2 (data_dist_stream): NCCL all_to_all for iteration N+1
      - Stage 3 (default_stream): forward/backward/optimizer for iteration N
    """

    def __init__(self, device, batch_size, seq_len, vocab_size, world_size, rank,
                 nccl_payload_mb=16.0, use_record_stream=False):
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.use_record_stream = use_record_stream

        self.memcpy_stream = torch.cuda.Stream()
        self.data_dist_stream = torch.cuda.Stream()

        self.batch_queue = []
        self.pending_nccl_works = []
        self.iteration = 0
        self.nccl_payload_nelems = int(nccl_payload_mb * 1024 * 1024 / 2)

    def _generate_batch_cpu(self):
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        targets = torch.randint(0, 2, (self.batch_size, 8)).float()
        emb_send = torch.randn(self.nccl_payload_nelems, dtype=torch.bfloat16)
        return {
            "input_ids": input_ids.pin_memory(),
            "targets": targets.pin_memory(),
            "emb_send": emb_send.pin_memory(),
        }

    def _h2d_and_distribute(self, cpu_batch):
        """H2D copy + NCCL all_to_all in a pipelined fashion."""
        with torch.cuda.stream(self.memcpy_stream):
            gpu_batch = {
                "input_ids": cpu_batch["input_ids"].to(self.device, non_blocking=True),
                "targets": cpu_batch["targets"].to(self.device, non_blocking=True),
            }
            send_buf = cpu_batch["emb_send"].to(self.device, non_blocking=True)

        with torch.cuda.stream(self.data_dist_stream):
            self.data_dist_stream.wait_stream(self.memcpy_stream)

            recv_buf = torch.empty_like(send_buf)

            if self.use_record_stream:
                send_buf.record_stream(self.data_dist_stream)

            work = dist.all_to_all_single(recv_buf, send_buf, async_op=True)

            gpu_batch["_send"] = send_buf
            gpu_batch["_recv"] = recv_buf
            gpu_batch["_work"] = work

        return gpu_batch

    def fill(self):
        for _ in range(3):
            cpu_batch = self._generate_batch_cpu()
            gpu_batch = self._h2d_and_distribute(cpu_batch)
            self.batch_queue.append(gpu_batch)
            self.iteration += 1

    def get_batch(self):
        """Get batch for forward pass.

        The race: we pop a batch and free its NCCL send buffer.
        The all_to_all that used this send buffer was launched 2-3 iterations ago.
        We do NOT call record_stream, so the CCA doesn't know RCCL's
        internal stream is still reading from it. The CCA recycles the
        memory, and the next _h2d_and_distribute may overwrite it.
        """
        torch.cuda.current_stream().wait_stream(self.data_dist_stream)

        batch = self.batch_queue.pop(0)

        work = batch.pop("_work", None)
        if work is not None:
            work.wait()

        send_buf = batch.pop("_send", None)
        recv_buf = batch.pop("_recv", None)
        del send_buf
        del recv_buf

        for _ in range(4):
            p = torch.empty(self.nccl_payload_nelems, device=self.device,
                            dtype=torch.float32)
            del p

        cpu_batch = self._generate_batch_cpu()
        gpu_batch = self._h2d_and_distribute(cpu_batch)
        self.batch_queue.append(gpu_batch)
        self.iteration += 1

        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nccl-payload-mb", type=float, default=16.0)
    parser.add_argument("--record-stream", action="store_true",
                        help="Add record_stream calls (the fix)")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--optimizer", choices=["shampoo", "adam"], default="shampoo")
    parser.add_argument("--precondition-frequency", type=int, default=10)
    parser.add_argument("--start-preconditioning-step", type=int, default=10)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    os.environ.setdefault("GPU_MAX_HW_QUEUES", "4")

    model = SimpleModel(
        d_model=args.d_model, num_layers=args.num_layers,
        vocab_size=args.vocab_size,
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    emb_params = []
    dense_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embedding" in name:
            emb_params.append(p)
        else:
            dense_params.append(p)

    if args.optimizer == "shampoo":
        from distributed_shampoo import DDPDistributedConfig, DistributedShampoo
        opt = DistributedShampoo(
            dense_params,
            lr=args.lr, betas=(0.9, 0.985), epsilon=1e-8, weight_decay=0.01,
            max_preconditioner_dim=8192,
            precondition_frequency=args.precondition_frequency,
            start_preconditioning_step=args.start_preconditioning_step,
            distributed_config=DDPDistributedConfig(
                communication_dtype=torch.float32,
                num_trainers_per_group=-1,
                communicate_params=False,
            ),
        )
        emb_opt = torch.optim.Adagrad(emb_params, lr=args.lr)
    else:
        opt = torch.optim.AdamW(dense_params, lr=args.lr)
        emb_opt = torch.optim.Adagrad(emb_params, lr=args.lr)

    pipeline = PipelineWithRace(
        device=device, batch_size=args.batch_size, seq_len=args.seq_len,
        vocab_size=args.vocab_size, world_size=world_size, rank=rank,
        nccl_payload_mb=args.nccl_payload_mb,
        use_record_stream=args.record_stream,
    )
    pipeline.fill()

    dense_count = sum(p.numel() for p in dense_params) / 1e6
    if rank == 0:
        log.info("=" * 60)
        log.info("PIPELINE CCA RACE REPRODUCER")
        log.info("=" * 60)
        log.info(f"World: {world_size} GPUs")
        log.info(f"Dense: {dense_count:.1f}M params, Optimizer: {args.optimizer}")
        log.info(f"record_stream: {args.record_stream}")
        log.info(f"NCCL payload: {args.nccl_payload_mb}MB")
        log.info(f"PYTORCH_NO_CUDA_MEMORY_CACHING={os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING', 'not set')}")
        log.info(f"GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES')}")
        log.info("=" * 60)

    total_nans = 0
    first_nan_step = None
    t0 = time.time()

    for step in range(args.max_steps):
        opt.zero_grad()
        emb_opt.zero_grad()

        batch = pipeline.get_batch()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["input_ids"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["targets"].to(device))

        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            total_nans += 1
            first_nan_step = first_nan_step or step
            log.error(f"NaN in LOSS at step {step}")
            break

        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                total_nans += 1
                first_nan_step = first_nan_step or step
                log.error(f"NaN in grad/{name} step {step}")

        opt.step()
        emb_opt.step()

        for name, p in model.named_parameters():
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                total_nans += 1
                first_nan_step = first_nan_step or step
                log.error(f"NaN in param/{name} step {step}")
                break

        if total_nans > 0:
            break

        if rank == 0 and (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            ms = 1000 * elapsed / (step + 1)
            log.info(f"Step {step+1}/{args.max_steps} | loss={loss_val:.4f} | {ms:.0f} ms/step")

    elapsed = time.time() - t0
    try:
        dist.barrier()
    except Exception:
        pass

    if rank == 0:
        log.info("")
        log.info("=" * 60)
        if total_nans > 0:
            log.info(f"NaN at step {first_nan_step}")
        else:
            log.info(f"No NaN in {step+1} steps ({elapsed:.0f}s)")
        log.info(f"record_stream={args.record_stream}, "
                 f"CCA={'OFF' if os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') else 'ON'}")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
