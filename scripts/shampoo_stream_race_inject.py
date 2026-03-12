"""
Shampoo stream race condition reproducer via monkey-patching.

This script reproduces the exact race condition observed in Meta's trace:
  Streams 16-30 each run bf16-to-fp32 gradient copy kernels during backward.
  These are NOT synchronized with stream 0 before Shampoo reads the gradients.

We reproduce this by monkey-patching Shampoo's merge_and_block_gradients()
to copy gradients on separate CUDA streams (without synchronization to the
default stream), exactly as observed in the trace.

The race:
  1. Backward produces bf16 gradients on stream 0
  2. We copy gradients bf16→fp32 on separate streams (16..N) [NO SYNC]
  3. Shampoo reads these fp32 copies on stream 0 for preconditioner computation
  4. If the copy on stream N isn't finished, Shampoo reads partial/garbage data
  5. Corrupted preconditioner → corrupted search direction → NaN

Usage:
    # Reproduce the race (should eventually produce NaN)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/shampoo_stream_race_inject.py

    # Verify the fix (should NOT produce NaN)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/shampoo_stream_race_inject.py --sync-streams

    # Disable the injection (baseline)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/shampoo_stream_race_inject.py --no-inject
"""

import argparse
import logging
import math
import os
import time
from functools import wraps

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


class HSTUBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.gate = nn.Linear(d_model, d_model)
        ffn_dim = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = self.norm(x)
        a, _ = self.attn(x, x, x)
        x = res + torch.sigmoid(self.gate(x)) * a
        res = x
        x = res + self.ffn(self.ffn_norm(x))
        return x


class TraceModel(nn.Module):
    """Model with many dense params to stress Shampoo."""
    def __init__(self, d_model=256, num_blocks=8, num_heads=8,
                 ffn_mult=8, vocab_size=50000, seq_len=128, num_tasks=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.blocks = nn.ModuleList([
            HSTUBlock(d_model, num_heads, ffn_mult) for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, num_tasks),
        )

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(1))


def inject_stream_race(shampoo_optimizer, num_copy_streams=15, sync_streams=False):
    """Monkey-patch Shampoo to copy gradients on separate CUDA streams.

    This reproduces the exact pattern from Meta's trace:
    Streams 16-30 each run bf16→fp32 gradient conversion, WITHOUT
    synchronization to the default stream.
    """
    copy_streams = [torch.cuda.Stream() for _ in range(num_copy_streams)]
    fp32_grad_cache = {}

    for state_lists in shampoo_optimizer._per_group_state_lists:
        distributor = state_lists.get("distributor")
        if distributor is None:
            continue

        original_merge = distributor._merge_and_block_gradients

        def patched_merge(orig=original_merge, streams=copy_streams,
                          cache=fp32_grad_cache, do_sync=sync_streams):
            """Intercept gradient merge to inject stream race."""
            result = orig()

            if not result:
                return result

            patched_result = []
            for i, grad_block in enumerate(result):
                stream = streams[i % len(streams)]

                if grad_block.dtype == torch.bfloat16:
                    with torch.cuda.stream(stream):
                        fp32_copy = grad_block.to(torch.float32)

                    if do_sync:
                        torch.cuda.current_stream().wait_stream(stream)

                    patched_result.append(fp32_copy)
                else:
                    patched_result.append(grad_block)

            return tuple(patched_result)

        distributor._merge_and_block_gradients = patched_merge

    return copy_streams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--precondition-frequency", type=int, default=1)
    parser.add_argument("--start-preconditioning-step", type=int, default=5)
    parser.add_argument("--max-preconditioner-dim", type=int, default=8192)
    parser.add_argument("--num-copy-streams", type=int, default=15)
    parser.add_argument("--sync-streams", action="store_true",
                        help="Add synchronization after stream copies (the fix)")
    parser.add_argument("--no-inject", action="store_true",
                        help="Don't inject stream race (baseline)")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    os.environ.setdefault("GPU_MAX_HW_QUEUES", "4")

    model = TraceModel(
        d_model=args.d_model, num_blocks=args.num_blocks,
        num_heads=args.num_heads, ffn_mult=args.ffn_mult,
        vocab_size=args.vocab_size, seq_len=args.seq_len,
        num_tasks=args.num_tasks,
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

    from distributed_shampoo import DDPDistributedConfig, DistributedShampoo
    shampoo = DistributedShampoo(
        dense_params,
        lr=args.lr,
        betas=(0.9, 0.985),
        epsilon=1e-8,
        weight_decay=0.01,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        distributed_config=DDPDistributedConfig(
            communication_dtype=torch.float32,
            num_trainers_per_group=-1,
            communicate_params=False,
        ),
    )
    emb_opt = torch.optim.Adagrad(emb_params, lr=args.lr) if emb_params else None

    if not args.no_inject:
        copy_streams = inject_stream_race(
            shampoo,
            num_copy_streams=args.num_copy_streams,
            sync_streams=args.sync_streams,
        )
        if rank == 0:
            inject_mode = "SYNCED" if args.sync_streams else "UNSYNCED (race!)"
            log.info(f"Injected {args.num_copy_streams} copy streams [{inject_mode}]")

    dense_count = sum(p.numel() for p in dense_params) / 1e6
    emb_count = sum(p.numel() for p in emb_params) / 1e6

    if rank == 0:
        log.info("=" * 60)
        log.info("SHAMPOO STREAM RACE INJECTOR")
        log.info("=" * 60)
        log.info(f"World: {world_size} GPUs, device: {device}")
        log.info(f"Dense: {dense_count:.1f}M params (Shampoo), "
                 f"Emb: {emb_count:.1f}M params (AdaGrad)")
        log.info(f"Inject: {'OFF' if args.no_inject else 'ON'}, "
                 f"sync_streams: {args.sync_streams}")
        log.info(f"Precondition freq: {args.precondition_frequency}")
        log.info(f"GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES')}")
        log.info("=" * 60)

    total_nans = 0
    first_nan_step = None
    first_nan_loc = None
    t0 = time.time()

    for step in range(args.max_steps):
        shampoo.zero_grad()
        if emb_opt:
            emb_opt.zero_grad()

        input_ids = torch.randint(0, args.vocab_size,
                                  (args.batch_size, args.seq_len), device=device)
        targets = torch.randint(0, 2, (args.batch_size, args.num_tasks),
                                device=device).float()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            total_nans += 1
            first_nan_step = first_nan_step or step
            first_nan_loc = first_nan_loc or "loss"
            log.error(f"NaN in LOSS at step {step}")
            break

        loss.backward()

        shampoo.step()
        if emb_opt:
            emb_opt.step()

        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                total_nans += 1
                first_nan_step = first_nan_step or step
                first_nan_loc = first_nan_loc or f"grad/{name}"
                log.error(f"NaN in grad/{name} step {step}")

            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                total_nans += 1
                first_nan_step = first_nan_step or step
                first_nan_loc = first_nan_loc or f"param/{name}"
                log.error(f"NaN in param/{name} step {step}")

        if total_nans > 0:
            log.error(f"Stopping at step {step}")
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
            log.info(f"NaN REPRODUCED at step {first_nan_step} in {first_nan_loc}")
        else:
            log.info(f"No NaN in {step+1} steps ({elapsed:.0f}s)")
        log.info(f"Mode: inject={'OFF' if args.no_inject else 'ON'}, "
                 f"sync={args.sync_streams}")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
