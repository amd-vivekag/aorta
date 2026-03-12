"""
RCCL CCA race → NaN reproducer.

Uses the PROVEN corruption mechanism from meta_nan_hip_event_stress.py
(rccl_event_race mode: corruption on every iteration) and injects it
into a Shampoo training loop to produce NaN.

The proven race:
  1. RCCL all_to_all reads send_buf on an RCCL-internal stream
  2. User frees send_buf without record_stream for RCCL's stream
  3. CCA recycles the block → new allocation overwrites while RCCL reads
  4. RCCL reads garbage → recv_buf contains corrupted data

Integration into training:
  We simulate TorchRec's pipeline by running RCCL all_to_all with
  embedding-like data, then intentionally freeing the send buffer
  without record_stream. The freed memory gets recycled for gradient
  buffers or optimizer state. This corrupts the training.

Usage:
    # Reproduce NaN via CCA race
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/rccl_nan_inject.py

    # With record_stream fix (should NOT produce NaN)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/rccl_nan_inject.py --record-stream

    # With CCA disabled (should NOT produce NaN)
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/rccl_nan_inject.py
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


class TrainingModel(nn.Module):
    def __init__(self, d_model=256, num_layers=6, vocab_size=50000, num_tasks=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            ))
        self.head = nn.Linear(d_model, num_tasks)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x.mean(1))


def rccl_race_cycle(send_buf, recv_buf, data_dist_stream, use_record_stream):
    """Execute one RCCL all_to_all race cycle.

    This is the proven corruption pattern from meta_nan_hip_event_stress.py.
    Returns the work handle.
    """
    with torch.cuda.stream(data_dist_stream):
        if use_record_stream:
            send_buf.record_stream(data_dist_stream)

        work = dist.all_to_all_single(recv_buf, send_buf, async_op=True)

    return work


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-payload-mb", type=float, default=32.0,
                        help="Simulated embedding all_to_all payload (MB)")
    parser.add_argument("--record-stream", action="store_true",
                        help="Add record_stream (the fix)")
    parser.add_argument("--optimizer", choices=["shampoo", "adam"], default="shampoo")
    parser.add_argument("--precondition-frequency", type=int, default=10)
    parser.add_argument("--start-preconditioning-step", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    os.environ.setdefault("GPU_MAX_HW_QUEUES", "4")

    model = TrainingModel(
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

    if args.optimizer == "shampoo" and dense_params:
        from distributed_shampoo import DDPDistributedConfig, DistributedShampoo
        shampoo = DistributedShampoo(
            dense_params, lr=args.lr, betas=(0.9, 0.985), epsilon=1e-8,
            weight_decay=0.01, max_preconditioner_dim=8192,
            precondition_frequency=args.precondition_frequency,
            start_preconditioning_step=args.start_preconditioning_step,
            distributed_config=DDPDistributedConfig(
                communication_dtype=torch.float32,
                num_trainers_per_group=-1,
                communicate_params=False,
            ),
        )
    else:
        shampoo = torch.optim.AdamW(dense_params, lr=args.lr)
    emb_opt = torch.optim.Adagrad(emb_params, lr=args.lr) if emb_params else None

    data_dist_stream = torch.cuda.Stream()

    payload_nelems = int(args.emb_payload_mb * 1024 * 1024 / 2)

    dense_count = sum(p.numel() for p in dense_params) / 1e6
    emb_count = sum(p.numel() for p in emb_params) / 1e6

    if rank == 0:
        log.info("=" * 60)
        log.info("RCCL CCA RACE → NaN REPRODUCER")
        log.info("=" * 60)
        log.info(f"World: {world_size} GPUs")
        log.info(f"Dense: {dense_count:.1f}M (Shampoo), Emb: {emb_count:.1f}M (AdaGrad)")
        log.info(f"Payload: {args.emb_payload_mb}MB per all_to_all")
        log.info(f"record_stream: {args.record_stream}")
        log.info(f"PYTORCH_NO_CUDA_MEMORY_CACHING="
                 f"{os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING', 'not set')}")
        log.info(f"GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES')}")
        log.info("=" * 60)

    total_nans = 0
    first_nan_step = None
    first_nan_loc = None
    t0 = time.time()

    prev_send = None
    prev_work = None

    for step in range(args.max_steps):
        shampoo.zero_grad()
        if emb_opt:
            emb_opt.zero_grad()

        send_buf = torch.randn(payload_nelems, device=device, dtype=torch.bfloat16)
        recv_buf = torch.empty_like(send_buf)

        work = rccl_race_cycle(send_buf, recv_buf, data_dist_stream, args.record_stream)

        if prev_send is not None:
            del prev_send
        if prev_work is not None:
            prev_work.wait()
            prev_work = None

        for _ in range(4):
            p = torch.empty(payload_nelems // 2, device=device, dtype=torch.float32)
            p.fill_(float('nan'))
            del p

        prev_send = send_buf
        prev_work = work

        input_ids = torch.randint(0, args.vocab_size,
                                  (args.batch_size, args.seq_len), device=device)
        targets = torch.randint(0, 2, (args.batch_size, 8), device=device).float()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            total_nans += 1
            first_nan_step = first_nan_step or step
            first_nan_loc = first_nan_loc or "loss"
            log.error(f"NaN in LOSS at step {step}: {loss_val}")
            break

        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                total_nans += 1
                first_nan_step = first_nan_step or step
                first_nan_loc = first_nan_loc or f"grad/{name}"
                nan_ct = torch.isnan(p.grad).sum().item()
                log.error(f"NaN grad/{name} step {step} NaN={nan_ct}")

        shampoo.step()
        if emb_opt:
            emb_opt.step()

        for name, p in model.named_parameters():
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                total_nans += 1
                first_nan_step = first_nan_step or step
                first_nan_loc = first_nan_loc or f"param/{name}"
                log.error(f"NaN param/{name} step {step}")
                break

        if total_nans > 0:
            break

        if rank == 0 and (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            ms = 1000 * elapsed / (step + 1)
            log.info(f"Step {step+1}/{args.max_steps} | loss={loss_val:.4f} | {ms:.0f} ms/step")

    if prev_work:
        prev_work.wait()

    elapsed = time.time() - t0
    try:
        dist.barrier()
    except Exception:
        pass

    if rank == 0:
        log.info("")
        log.info("=" * 60)
        if total_nans > 0:
            log.info(f"NaN REPRODUCED at step {first_nan_step} in {first_nan_loc}")
            log.info(f"Total NaN events: {total_nans}")
            if args.record_stream:
                log.info("record_stream was ON → race is NOT in send_buf lifecycle")
            else:
                log.info("Try --record-stream to test the hypothesis")
        else:
            log.info(f"No NaN in {step+1} steps ({elapsed:.0f}s)")
        log.info(f"record_stream={args.record_stream}, "
                 f"CCA={'OFF' if os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') else 'ON'}")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
