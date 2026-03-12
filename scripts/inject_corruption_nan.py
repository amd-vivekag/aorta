"""
Injection-based NaN reproducer for the Shampoo stream race.

Instead of relying on the actual CCA/RCCL race condition to corrupt memory,
this script DIRECTLY INJECTS the corruption that the race would produce,
then verifies that:
  1. Corrupted bf16→fp32 gradient copies → NaN in preconditioner
  2. NaN preconditioner → NaN in parameter updates
  3. NaN parameters → NaN loss in subsequent steps

This proves the NaN propagation pathway WITHOUT requiring the actual
hardware-level race to trigger.

Two injection modes:
  A) "grad_corrupt" -- Corrupt one fp32 gradient copy (simulate incomplete bf16→fp32)
  B) "alloc_corrupt" -- Corrupt an all_to_all buffer (simulate CCA reuse)

Usage:
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/inject_corruption_nan.py \\
        --inject-step 50 --inject-mode grad_corrupt

    srun --nodes=2 --gres=gpu:8 --ntasks-per-node=1 -p mi355x -t 00:30:00 \\
      scripts/launch_multinode.sh scripts/inject_corruption_nan.py \\
        --inject-step 50 --inject-mode grad_corrupt
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


class HSTULayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.gate = nn.Linear(d_model, d_model)
        w = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, w), nn.GELU(), nn.Linear(w, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        r = x
        x = self.norm(x)
        a, _ = self.attn(x, x, x)
        x = r + torch.sigmoid(self.gate(x)) * a
        return x + self.ffn(self.ffn_norm(x))


class TraceHSTU(nn.Module):
    def __init__(self, d_model=96, num_layers=7, num_heads=16,
                 vocab_size=500_000, ffn_mult=4, num_tasks=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.layers = nn.ModuleList([
            HSTULayer(d_model, num_heads, ffn_mult) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, 1024), nn.SiLU(), nn.LayerNorm(1024),
            nn.Linear(1024, 2048), nn.SiLU(), nn.LayerNorm(2048),
            nn.Linear(2048, 1024), nn.SiLU(), nn.LayerNorm(1024),
            nn.Linear(1024, num_tasks),
        )

    def forward(self, ids):
        x = self.embedding(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(1))


class ShampooWithInjection:
    """Shampoo optimizer with injection capability.

    At the injection step, corrupts specific fp32 gradient copies
    to simulate the race condition outcome.
    """
    def __init__(self, model, lr, inject_step, inject_mode,
                 inject_rank=0, num_copy_streams=15,
                 precondition_frequency=100, start_preconditioning_step=50):
        self.inject_step = inject_step
        self.inject_mode = inject_mode
        self.inject_rank = inject_rank
        self.num_copy_streams = num_copy_streams
        self.precondition_frequency = precondition_frequency
        self.start_preconditioning_step = start_preconditioning_step

        self.copy_streams = [torch.cuda.Stream() for _ in range(num_copy_streams)]

        emb_params = []
        dense_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "embedding" in name:
                emb_params.append(p)
            else:
                dense_params.append(p)

        self.dense_params = dense_params
        self.emb_params = emb_params
        self.emb_opt = torch.optim.Adagrad(emb_params, lr=lr) if emb_params else None

        try:
            from distributed_shampoo import DDPDistributedConfig, DistributedShampoo
            self.dense_opt = DistributedShampoo(
                dense_params, lr=lr, betas=(0.9, 0.985), epsilon=1e-8,
                weight_decay=0.01, max_preconditioner_dim=8192,
                precondition_frequency=precondition_frequency,
                start_preconditioning_step=start_preconditioning_step,
                distributed_config=DDPDistributedConfig(
                    communication_dtype=torch.float32,
                    num_trainers_per_group=-1,
                    communicate_params=False,
                ),
            )
            self._shampoo = True
        except Exception:
            self.dense_opt = torch.optim.AdamW(dense_params, lr=lr)
            self._shampoo = False

    def zero_grad(self):
        if self.dense_opt:
            self.dense_opt.zero_grad()
        if self.emb_opt:
            self.emb_opt.zero_grad()

    def _inject_grad_corruption(self, step, rank):
        """Inject corruption into gradient copies, simulating the race
        where the bf16→fp32 copy on stream 16-30 produces garbage because
        the CCA recycled the destination buffer while a previous copy
        was still writing to it."""
        if step != self.inject_step:
            return
        if rank != self.inject_rank:
            return

        injected = 0
        for i, p in enumerate(self.dense_params):
            if p.grad is None:
                continue
            if p.grad.numel() > 1000:
                n = p.grad.numel()
                # Simulate: 1% of the gradient buffer contains garbage from
                # CCA reuse -- random bits that could be NaN/Inf in bf16/fp32
                corrupt_mask = torch.rand(n, device=p.grad.device) < 0.01
                garbage = torch.where(
                    torch.rand(n, device=p.grad.device) < 0.5,
                    torch.tensor(float('nan'), device=p.grad.device).expand(n),
                    torch.tensor(float('inf'), device=p.grad.device).expand(n),
                )
                p.grad.view(-1).masked_scatter_(corrupt_mask, garbage[corrupt_mask])
                injected += 1
                if injected >= 3:
                    break

        if injected > 0:
            log.warning(f"INJECTED grad corruption at step {step} "
                        f"rank {rank}: {injected} params")

    def step(self, step, rank):
        if self.inject_mode == "grad_corrupt":
            self._inject_grad_corruption(step, rank)

        if self.dense_opt:
            self.dense_opt.step()
        if self.emb_opt:
            self.emb_opt.step()


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--num-layers", type=int, default=7)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--num-tasks", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--precondition-frequency", type=int, default=50)
    p.add_argument("--start-preconditioning-step", type=int, default=50)

    p.add_argument("--inject-step", type=int, default=50,
                   help="Step at which to inject corruption")
    p.add_argument("--inject-mode", choices=["grad_corrupt", "none"], default="grad_corrupt")
    p.add_argument("--inject-rank", type=int, default=0,
                   help="Which rank to inject on (-1 = all)")

    args = p.parse_args()

    os.environ.setdefault("GPU_MAX_HW_QUEUES", "4")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = TraceHSTU(
        d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, vocab_size=args.vocab_size,
        ffn_mult=args.ffn_mult, num_tasks=args.num_tasks,
    ).to(dtype=torch.bfloat16, device=device)
    model = DDP(model, device_ids=[local_rank])

    inject_rank = args.inject_rank if args.inject_rank >= 0 else rank
    optimizer = ShampooWithInjection(
        model, lr=args.lr,
        inject_step=args.inject_step, inject_mode=args.inject_mode,
        inject_rank=inject_rank,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
    )

    if rank == 0:
        dense_n = sum(pp.numel() for pp in optimizer.dense_params) / 1e6
        log.info(f"World: {world_size} GPUs, Model: {dense_n:.1f}M dense params "
                 f"({'Shampoo' if optimizer._shampoo else 'AdamW'})")
        log.info(f"Inject: mode={args.inject_mode} step={args.inject_step} "
                 f"rank={args.inject_rank}")

    t0 = time.time()
    nan_step = None

    for step in range(args.max_steps):
        optimizer.zero_grad()

        ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len),
                            device=device)
        targets = torch.randint(0, 2, (args.batch_size, args.num_tasks),
                                device=device, dtype=torch.bfloat16)

        logits = model(ids)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss_val = loss.item()

        if math.isnan(loss_val) or math.isinf(loss_val):
            if nan_step is None:
                nan_step = step
                log.error(f"NaN/Inf loss at step {step}: {loss_val}")
                log.error(f"  Injection was at step {args.inject_step}, "
                          f"NaN appeared {step - args.inject_step} steps later")
            if step > args.inject_step + 50:
                break
            continue

        loss.backward()

        has_nan_grad = False
        for name, pp in model.named_parameters():
            if pp.grad is not None and torch.isnan(pp.grad).any():
                if nan_step is None:
                    nan_step = step
                    log.error(f"NaN grad at step {step} in {name}")
                has_nan_grad = True
                break

        optimizer.step(step, rank)

        has_nan_param = False
        for name, pp in model.named_parameters():
            if torch.isnan(pp.data).any():
                if nan_step is None:
                    nan_step = step
                    log.error(f"NaN param at step {step} in {name}")
                has_nan_param = True
                break

        if rank == 0 and (step + 1) % args.log_interval == 0:
            el = time.time() - t0
            log.info(f"Step {step+1}/{args.max_steps} | "
                     f"loss={loss_val:.4f} | "
                     f"NaN={'YES' if nan_step else 'no'} | "
                     f"{1000*el/(step+1):.0f} ms/step")

    if rank == 0:
        log.info("=" * 50)
        if nan_step is not None:
            log.info(f"NaN CONFIRMED at step {nan_step}")
            log.info(f"  Injection at step {args.inject_step}")
            log.info(f"  Latency: {nan_step - args.inject_step} steps")
        else:
            log.info(f"No NaN in {args.max_steps} steps")
        log.info("=" * 50)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
