"""
Minimal Shampoo DDPDistributedConfig race condition reproducer.

Targets the exact race pattern from Meta's trace:
  1. DDP backward fires all-reduce on NCCL's internal stream
  2. CachingAllocator may reuse memory freed by the optimizer
  3. Shampoo's all_gather_into_tensor on stream 0 may collide

Key strategies:
  - Many small parameter groups → many DDP buckets → many NCCL ops
  - Very low precondition_frequency (every step) → constant preconditioner activity
  - Large intermediate tensors → CachingAllocator churn
  - bf16 autocast → dtype conversion creates extra allocations
  - No grad clipping → any corruption propagates fully
  - GPU_MAX_HW_QUEUES=4 → maximum kernel concurrency on AMD

Usage:
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/shampoo_race_minimal.py
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/shampoo_race_minimal.py --sync-fix
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


class WideResBlock(nn.Module):
    """Wide residual block creating many independent parameter groups."""

    def __init__(self, dim: int, width_mult: int = 4):
        super().__init__()
        wide = dim * width_mult
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, wide)
        self.fc2 = nn.Linear(wide, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return residual + g * h


class MultiStreamModel(nn.Module):
    """Model designed to maximize DDP communication overhead and CachingAllocator pressure.

    Creates many independent parameter groups (each becomes a DDP bucket),
    forcing many NCCL all-reduce ops during backward. The output branches
    create allocation pressure from intermediate tensors.
    """

    def __init__(self, d_model: int = 256, num_blocks: int = 12,
                 width_mult: int = 8, num_tasks: int = 8,
                 seq_len: int = 128, vocab_size: int = 50000):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.blocks = nn.ModuleList([
            WideResBlock(d_model, width_mult) for _ in range(num_blocks)
        ])

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            for _ in range(num_blocks)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_blocks)
        ])

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.attn_layers):
                residual = x
                x = self.attn_norms[i](x)
                x, _ = self.attn_layers[i](x, x, x)
                x = residual + x

        pooled = x.mean(dim=1)

        logits_list = []
        for head in self.task_heads:
            logits_list.append(head(pooled).squeeze(-1))

        return torch.stack(logits_list, dim=-1)


def split_params_for_shampoo(model):
    """Split into embedding (AdaGrad) and dense (Shampoo) params."""
    emb_params = []
    dense_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embedding" in name:
            emb_params.append(p)
        else:
            dense_params.append(p)
    return dense_params, emb_params


def create_shampoo_optimizer(dense_params, args):
    from distributed_shampoo import DDPDistributedConfig, DistributedShampoo

    kwargs = dict(
        lr=args.lr,
        betas=(0.9, 0.985),
        epsilon=1e-8,
        weight_decay=0.01,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
    )

    if not args.no_ddp_config:
        kwargs["distributed_config"] = DDPDistributedConfig(
            communication_dtype=torch.float32,
            num_trainers_per_group=-1,
            communicate_params=False,
        )

    return DistributedShampoo(dense_params, **kwargs)


def generate_batch(batch_size, seq_len, vocab_size, device):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 2, (batch_size, 8), device=device).float()
    return input_ids, targets


def alloc_churn(device, n=20, mb=8.0):
    """Pressure the CachingAllocator with allocate/free cycles."""
    nelems = int(mb * 1024 * 1024 / 4)
    bufs = [torch.empty(nelems, device=device, dtype=torch.float32) for _ in range(n)]
    del bufs


def run_nccl_side_traffic(send_buf, recv_buf, stream):
    """Launch all_to_all on a side stream to overlap with optimizer."""
    with torch.cuda.stream(stream):
        dist.all_to_all_single(recv_buf, send_buf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=12)
    parser.add_argument("--width-mult", type=int, default=8)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--precondition-frequency", type=int, default=1)
    parser.add_argument("--start-preconditioning-step", type=int, default=5)
    parser.add_argument("--max-preconditioner-dim", type=int, default=8192)
    parser.add_argument("--no-ddp-config", action="store_true")
    parser.add_argument("--sync-fix", action="store_true",
                        help="Sync before optimizer.step() (should fix the race)")
    parser.add_argument("--alloc-stress", action="store_true", default=True)
    parser.add_argument("--nccl-side-traffic", action="store_true", default=True)
    parser.add_argument("--nccl-payload-mb", type=float, default=32.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--nan-check-interval", type=int, default=1)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    os.environ.setdefault("GPU_MAX_HW_QUEUES", "4")

    model = MultiStreamModel(
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        width_mult=args.width_mult,
        num_tasks=args.num_tasks,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    dense_params, emb_params = split_params_for_shampoo(model)

    shampoo_opt = create_shampoo_optimizer(dense_params, args)
    emb_opt = torch.optim.Adagrad(emb_params, lr=args.lr) if emb_params else None

    dense_count = sum(p.numel() for p in dense_params) / 1e6
    emb_count = sum(p.numel() for p in emb_params) / 1e6

    nccl_stream = torch.cuda.Stream()
    nccl_nelems = int(args.nccl_payload_mb * 1024 * 1024 / 2)
    nccl_send = torch.randn(nccl_nelems, device=device, dtype=torch.bfloat16)
    nccl_recv = torch.empty_like(nccl_send)

    if rank == 0:
        log.info("=" * 60)
        log.info("SHAMPOO RACE CONDITION REPRODUCER (minimal)")
        log.info("=" * 60)
        log.info(f"World: {world_size} GPUs")
        log.info(f"Model: d={args.d_model}, blocks={args.num_blocks}, "
                 f"width_mult={args.width_mult}")
        log.info(f"Dense: {dense_count:.1f}M params (Shampoo), "
                 f"Emb: {emb_count:.1f}M params (AdaGrad)")
        log.info(f"Batch: {args.batch_size}x{args.seq_len}")
        log.info(f"Precondition every {args.precondition_frequency} steps "
                 f"(start at {args.start_preconditioning_step})")
        log.info(f"DDPDistributedConfig: {'OFF' if args.no_ddp_config else 'ON'}")
        log.info(f"Sync fix: {args.sync_fix}")
        log.info(f"Alloc stress: {args.alloc_stress}")
        log.info(f"NCCL side traffic: {args.nccl_side_traffic} ({args.nccl_payload_mb}MB)")
        log.info(f"GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES')}")
        log.info("=" * 60)

    total_nans = 0
    first_nan_step = None
    first_nan_loc = None
    t0 = time.time()

    for step in range(args.max_steps):
        shampoo_opt.zero_grad()
        if emb_opt:
            emb_opt.zero_grad()

        input_ids, targets = generate_batch(
            args.batch_size, args.seq_len, args.vocab_size, device,
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            total_nans += 1
            if first_nan_step is None:
                first_nan_step = step
                first_nan_loc = "loss"
            log.error(f"NaN in LOSS at step {step}: {loss_val}")
            break

        loss.backward()

        if args.nccl_side_traffic:
            run_nccl_side_traffic(nccl_send, nccl_recv, nccl_stream)

        if args.alloc_stress:
            alloc_churn(device)

        if step % args.nan_check_interval == 0:
            for name, p in model.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    total_nans += 1
                    if first_nan_step is None:
                        first_nan_step = step
                        first_nan_loc = f"grad/{name}"
                    nan_ct = torch.isnan(p.grad).sum().item()
                    inf_ct = torch.isinf(p.grad).sum().item()
                    log.error(f"NaN/Inf in grad/{name} step {step}: "
                              f"NaN={nan_ct} Inf={inf_ct} shape={list(p.grad.shape)}")

        if args.sync_fix:
            torch.cuda.synchronize()

        shampoo_opt.step()
        if emb_opt:
            emb_opt.step()

        if step % args.nan_check_interval == 0:
            for name, p in model.named_parameters():
                if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                    total_nans += 1
                    if first_nan_step is None:
                        first_nan_step = step
                        first_nan_loc = f"param/{name}"
                    nan_ct = torch.isnan(p.data).sum().item()
                    inf_ct = torch.isinf(p.data).sum().item()
                    log.error(f"NaN/Inf in param/{name} step {step}: "
                              f"NaN={nan_ct} Inf={inf_ct}")
                    break

        if total_nans > 0:
            log.error(f"Stopping due to NaN at step {step}")
            break

        if rank == 0 and (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            ms_per_step = 1000 * elapsed / (step + 1)
            is_precond = (
                step >= args.start_preconditioning_step
                and step % args.precondition_frequency == 0
            )
            log.info(f"Step {step+1}/{args.max_steps} | loss={loss_val:.4f} | "
                     f"{ms_per_step:.0f} ms/step"
                     f"{' [PRECOND]' if is_precond else ''}")

    elapsed = time.time() - t0
    dist.barrier()

    if rank == 0:
        log.info("")
        log.info("=" * 60)
        if total_nans > 0:
            log.info(f"NaN DETECTED at step {first_nan_step} in {first_nan_loc}")
            log.info(f"Total NaN events: {total_nans}")
        else:
            log.info(f"No NaN in {step+1} steps ({elapsed:.0f}s)")
        log.info(f"Config: ddp_config={'OFF' if args.no_ddp_config else 'ON'}, "
                 f"sync_fix={args.sync_fix}")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
