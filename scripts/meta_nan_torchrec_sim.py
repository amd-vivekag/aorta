"""
TorchRec-style reproducer for Issue B NaN.

Simulates TorchRec's TrainPipelineSparseDist without requiring actual TorchRec:
  1. KeyedJaggedTensor-style inputs: variable-length sequences packed with offsets
  2. Sharded EmbeddingBagCollection: each GPU owns a subset of embedding tables
  3. all_to_all redistribution of embeddings between GPUs
  4. torch.compile over the FULL model including embedding lookups
  5. 3-stage pipeline with buffer rotation (same as TrainPipelineSparseDist)
  6. _unsafe_view and stride patterns that torch.compile may mishandle

Usage:
    PYTHONPATH=scripts torchrun --nproc_per_node=2 scripts/meta_nan_torchrec_sim.py --batch-size 1024
    PYTHONPATH=scripts torchrun --nproc_per_node=8 scripts/meta_nan_torchrec_sim.py --batch-size 4096
    PYTHONPATH=scripts torchrun --nproc_per_node=2 scripts/meta_nan_torchrec_sim.py --batch-size 4096 --no-compile
"""

import argparse
import gc
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class PackedEmbeddingInput:
    """Simulates KeyedJaggedTensor: variable-length sequences packed with offsets."""
    __slots__ = ['values', 'offsets', 'lengths', 'stride']

    def __init__(self, values: torch.Tensor, offsets: torch.Tensor,
                 lengths: torch.Tensor, stride: int):
        self.values = values
        self.offsets = offsets
        self.lengths = lengths
        self.stride = stride


class ShardedEmbeddingBag(nn.Module):
    """Simulates TorchRec's sharded EmbeddingBagCollection.

    Each GPU owns num_tables/world_size embedding tables.
    Lookups are done locally, then results are all_to_all'd.
    """
    def __init__(self, num_tables: int, hash_size: int, dim: int,
                 rank: int, world_size: int):
        super().__init__()
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.dim = dim
        self.rank = rank
        self.world_size = world_size

        tables_per_rank = num_tables // world_size
        self.local_tables = tables_per_rank
        self.tables = nn.ModuleList([
            nn.EmbeddingBag(hash_size, dim, mode='sum', sparse=False,
                            include_last_offset=True)
            for _ in range(tables_per_rank)
        ])
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.01)

    def forward(self, indices_list: List[torch.Tensor],
                offsets_list: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for i, table in enumerate(self.tables):
            out = table(indices_list[i], offsets_list[i])
            outputs.append(out)
        return torch.cat(outputs, dim=-1)


class InteractionArch(nn.Module):
    """Feature interaction layer similar to DLRM/HSTU interaction."""
    def __init__(self, num_features: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(num_features * dim, dim * 4, bias=False)
        self.act = nn.GELU()
        self.out = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.proj(x)))


class OverArch(nn.Module):
    """Over-architecture: attention + prediction head."""
    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
                batch_first=True, dropout=0.0,
                norm_first=True,
            ))
        self.head = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


class DLRMv3Sim(nn.Module):
    """Simulated DLRMv3 model matching Meta's architecture."""
    def __init__(self, num_tables: int, hash_size: int, dim: int,
                 rank: int, world_size: int, num_dense: int = 13):
        super().__init__()
        self.embedding = ShardedEmbeddingBag(num_tables, hash_size, dim, rank, world_size)
        self.dense_proj = nn.Linear(num_dense, dim, bias=False)
        tables_per_rank = num_tables // world_size
        total_sparse_dim = tables_per_rank * dim
        self.interaction = InteractionArch(
            num_features=2,
            dim=total_sparse_dim,
        )
        self.over_arch = OverArch(dim=total_sparse_dim)

    def forward(self, dense_features: torch.Tensor,
                sparse_indices: List[torch.Tensor],
                sparse_offsets: List[torch.Tensor]) -> torch.Tensor:
        sparse_out = self.embedding(sparse_indices, sparse_offsets)
        B = dense_features.shape[0]
        dense_out = self.dense_proj(dense_features)
        sparse_dim = sparse_out.shape[-1]
        dense_padded = F.pad(dense_out, (0, sparse_dim - dense_out.shape[-1]))
        combined = torch.stack([sparse_out, dense_padded], dim=1)
        interaction_out = self.interaction(combined.view(B, -1))
        pred = self.over_arch(interaction_out.unsqueeze(1))
        return pred


def generate_kjt_batch(batch_size, num_tables, pooling_factor, hash_size, device=None):
    """Generate a batch of KeyedJaggedTensor-like data."""
    indices_list = []
    offsets_list = []
    for _ in range(num_tables):
        lengths = torch.randint(1, pooling_factor * 2 + 1, (batch_size,))
        total_indices = lengths.sum().item()
        indices = torch.randint(0, hash_size, (total_indices,))
        offsets = torch.zeros(batch_size + 1, dtype=torch.long)
        offsets[1:] = torch.cumsum(lengths, 0)
        if device is not None:
            indices = indices.to(device)
            offsets = offsets.to(device)
        indices_list.append(indices)
        offsets_list.append(offsets)
    return indices_list, offsets_list


def generate_host_pool(num_batches, batch_size, num_tables, pooling_factor,
                       hash_size, num_dense):
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(42)
    pool = []
    for _ in range(num_batches):
        indices_list, offsets_list = generate_kjt_batch(
            batch_size, num_tables, pooling_factor, hash_size
        )
        dense = torch.randn(batch_size, num_dense, dtype=torch.bfloat16)
        pool.append({
            'dense': dense.pin_memory(),
            'indices': [idx.pin_memory() for idx in indices_list],
            'offsets': [off.pin_memory() for off in offsets_list],
        })
    torch.random.set_rng_state(rng_state)
    return pool


class TorchRecPipeline:
    """3-stage pipeline simulating TrainPipelineSparseDist."""
    NUM_SLOTS = 3

    def __init__(self, batch_size, num_tables, pooling_factor, hash_size,
                 num_dense, device, host_pool):
        self.device = device
        self.host_pool = host_pool
        self.pool_idx = 0
        self.batch_size = batch_size
        self.num_tables = num_tables
        self.slots = [self._alloc_slot(batch_size, num_tables, pooling_factor, num_dense)
                      for _ in range(self.NUM_SLOTS)]

    def _alloc_slot(self, B, num_tables, pooling_factor, num_dense):
        max_indices = B * pooling_factor * 2
        slot = {
            'dense': torch.zeros(B, num_dense, dtype=torch.bfloat16, device=self.device),
            'indices': [torch.zeros(max_indices, dtype=torch.long, device=self.device)
                        for _ in range(num_tables)],
            'offsets': [torch.zeros(B + 1, dtype=torch.long, device=self.device)
                        for _ in range(num_tables)],
            'actual_lengths': [0] * num_tables,
        }
        return slot

    def _next_host(self):
        batch = self.host_pool[self.pool_idx % len(self.host_pool)]
        self.pool_idx += 1
        return batch

    def h2d_into_slot(self, slot_idx, stream):
        host = self._next_host()
        slot = self.slots[slot_idx]
        with torch.cuda.stream(stream):
            slot['dense'].copy_(host['dense'], non_blocking=True)
            for t in range(self.num_tables):
                n = host['indices'][t].shape[0]
                slot['indices'][t][:n].copy_(host['indices'][t], non_blocking=True)
                slot['offsets'][t].copy_(host['offsets'][t], non_blocking=True)
                slot['actual_lengths'][t] = n

    def fill_slot_sync(self, slot_idx):
        host = self._next_host()
        slot = self.slots[slot_idx]
        slot['dense'].copy_(host['dense'])
        for t in range(self.num_tables):
            n = host['indices'][t].shape[0]
            slot['indices'][t][:n].copy_(host['indices'][t])
            slot['offsets'][t].copy_(host['offsets'][t])
            slot['actual_lengths'][t] = n

    def get_compute_batch(self):
        slot = self.slots[0]
        indices = [slot['indices'][t][:slot['actual_lengths'][t]]
                   for t in range(self.num_tables)]
        offsets = [slot['offsets'][t] for t in range(self.num_tables)]
        return slot['dense'], indices, offsets

    def rotate(self):
        self.slots = [self.slots[1], self.slots[2], self.slots[0]]


def run_pipelined(model, args, rank, world_size, device, host_pool):
    memcpy_stream = torch.cuda.Stream()
    datadist_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()

    tables_per_rank = args.num_tables // world_size
    pipeline = TorchRecPipeline(
        args.batch_size, tables_per_rank, args.pooling_factor,
        args.hash_size, args.num_dense, device, host_pool,
    )

    use_dist = world_size > 1 and dist.is_initialized()
    if use_dist:
        emb_dim = tables_per_rank * args.dim
        a2a_send = torch.empty(world_size, args.batch_size * emb_dim,
                               dtype=torch.bfloat16, device=device)
        a2a_recv = torch.empty_like(a2a_send)

    total_loss = torch.zeros(1, dtype=torch.float32, device=device)
    correct = torch.zeros(1, dtype=torch.int64, device=device)
    count = torch.zeros(1, dtype=torch.int64, device=device)

    pipeline.fill_slot_sync(0)
    torch.cuda.synchronize()
    pipeline.h2d_into_slot(1, memcpy_stream)
    pipeline.h2d_into_slot(2, memcpy_stream)

    nan_count = 0
    first_nan_iter = None
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        default_stream.wait_stream(memcpy_stream)

        if use_dist:
            with torch.cuda.stream(datadist_stream):
                a2a_send.fill_(float((it + 1) % 1000) / 1000.0)
                dist.all_to_all_single(a2a_recv, a2a_send, async_op=False)

        if it + 3 < args.iterations:
            pipeline.h2d_into_slot(2, memcpy_stream)

        dense, indices, offsets = pipeline.get_compute_batch()
        with torch.no_grad():
            output = model(dense, indices, offsets)

        pred_f = output.float()
        total_loss += pred_f.sum()
        count += output.numel()
        correct += (output > 0).long().sum()

        if use_dist:
            default_stream.wait_stream(datadist_stream)

        pipeline.rotate()

        if args.sync_per_iter:
            torch.cuda.synchronize()

        if (it + 1) % args.check_interval == 0:
            torch.cuda.synchronize()
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            if has_nan or has_inf:
                nan_count += 1
                if first_nan_iter is None:
                    first_nan_iter = it
                if rank == 0:
                    nan_n = torch.isnan(output).sum().item()
                    inf_n = torch.isinf(output).sum().item()
                    log.error(f"  iter={it+1}: NaN={nan_n}, Inf={inf_n}")

            now = time.time()
            if rank == 0 and (has_nan or has_inf or now - last_log >= 5):
                elapsed = now - start
                rate = (it + 1) / elapsed
                status = "NaN!" if (has_nan or has_inf) else "ok"
                log.info(f"  [pipelined] iter={it+1}/{args.iterations}  "
                         f"[{status}]  nans={nan_count}  rate={rate:.1f} it/s")
                last_log = now

            if nan_count > 0 and args.stop_on_first:
                break

    torch.cuda.synchronize()
    return nan_count, first_nan_iter


def run_non_pipelined(model, args, rank, world_size, device, host_pool):
    tables_per_rank = args.num_tables // world_size

    nan_count = 0
    first_nan_iter = None
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        host = host_pool[it % len(host_pool)]
        dense = host['dense'].to(device)
        indices = [idx.to(device) for idx in host['indices']]
        offsets = [off.to(device) for off in host['offsets']]

        with torch.no_grad():
            output = model(dense, indices, offsets)
        torch.cuda.synchronize()

        if (it + 1) % args.check_interval == 0:
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            if has_nan or has_inf:
                nan_count += 1
                if first_nan_iter is None:
                    first_nan_iter = it

            now = time.time()
            if rank == 0 and (has_nan or has_inf or now - last_log >= 5):
                rate = (it + 1) / (now - start)
                status = "NaN!" if (has_nan or has_inf) else "ok"
                log.info(f"  [non-pipelined] iter={it+1}/{args.iterations}  "
                         f"[{status}]  nans={nan_count}  rate={rate:.1f} it/s")
                last_log = now

            if nan_count > 0 and args.stop_on_first:
                break

    return nan_count, first_nan_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-tables", type=int, default=16)
    parser.add_argument("--hash-size", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--pooling-factor", type=int, default=50)
    parser.add_argument("--num-dense", type=int, default=13)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--pipelined", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--sync-per-iter", action="store_true")
    parser.add_argument("--check-interval", type=int, default=10)
    parser.add_argument("--stop-on-first", action="store_true", default=True)
    parser.add_argument("--no-stop-on-first", action="store_true")
    args = parser.parse_args()
    if args.no_stop_on_first:
        args.stop_on_first = False

    os.environ.setdefault("ROC_AQL_QUEUE_SIZE", "1024")

    distributed = False
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        distributed = True
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    device = torch.cuda.current_device()

    if args.num_tables % world_size != 0:
        args.num_tables = (args.num_tables // world_size) * world_size
        if rank == 0:
            log.warning(f"Adjusted num_tables to {args.num_tables}")

    tables_per_rank = args.num_tables // world_size

    if rank == 0:
        log.info("=" * 70)
        log.info("TORCHREC-STYLE ISSUE B REPRODUCER")
        log.info("=" * 70)
        log.info(f"  mode: {'PIPELINED' if args.pipelined else 'NON-PIPELINED'}")
        log.info(f"  compile: {'disabled' if args.no_compile else 'enabled'}")
        log.info(f"  world_size={world_size}, iterations={args.iterations}")
        log.info(f"  batch_size={args.batch_size}")
        log.info(f"  tables={args.num_tables} (per_rank={tables_per_rank})")
        log.info(f"  hash_size={args.hash_size}, dim={args.dim}")
        log.info(f"  pooling_factor={args.pooling_factor}")
        log.info(f"  ROC_AQL_QUEUE_SIZE={os.environ.get('ROC_AQL_QUEUE_SIZE', 'not set')}")
        log.info("=" * 70)

    model = DLRMv3Sim(
        num_tables=args.num_tables, hash_size=args.hash_size, dim=args.dim,
        rank=rank, world_size=world_size, num_dense=args.num_dense,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        log.info(f"Model parameters: {param_count:.1f}M")

    if rank == 0:
        log.info(f"Generating {args.pool_size} host batches...")
    host_pool = generate_host_pool(
        args.pool_size, args.batch_size, tables_per_rank,
        args.pooling_factor, args.hash_size, args.num_dense,
    )
    if rank == 0:
        log.info("Host pool ready.")

    if rank == 0:
        log.info("Warmup...")
    for i in range(3):
        h = host_pool[i]
        d = h['dense'].to(device)
        idx = [x.to(device) for x in h['indices']]
        off = [x.to(device) for x in h['offsets']]
        with torch.no_grad():
            model(d, idx, off)
        torch.cuda.synchronize()

    if not args.no_compile:
        if rank == 0:
            log.info("Compiling model...")
        model = torch.compile(model)
        for i in range(3):
            h = host_pool[i]
            d = h['dense'].to(device)
            idx = [x.to(device) for x in h['indices']]
            off = [x.to(device) for x in h['offsets']]
            with torch.no_grad():
                model(d, idx, off)
            torch.cuda.synchronize()
        if rank == 0:
            log.info("Compile warmup done.")

    t_start = time.time()
    if args.pipelined:
        nan_count, first_nan = run_pipelined(model, args, rank, world_size, device, host_pool)
    else:
        nan_count, first_nan = run_non_pipelined(model, args, rank, world_size, device, host_pool)
    elapsed = time.time() - t_start

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"  Mode: {'PIPELINED' if args.pipelined else 'NON-PIPELINED'}")
        log.info(f"  Compile: {'disabled' if args.no_compile else 'enabled'}")
        log.info(f"  Elapsed: {elapsed:.1f}s ({args.iterations / max(elapsed, 0.1):.1f} it/s)")
        log.info(f"  NaN detections: {nan_count}")
        if first_nan is not None:
            log.info(f"  First NaN at iteration: {first_nan + 1}")

        if nan_count > 0:
            log.info("VERDICT: NaN DETECTED — Issue B pattern confirmed!")
        else:
            log.info(f"VERDICT: No NaN at bs={args.batch_size}.")
        log.info("=" * 70)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if nan_count > 0 else 0)


if __name__ == "__main__":
    main()
