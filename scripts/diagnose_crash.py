"""
Diagnostic script to isolate the HSA_STATUS_ERROR_EXCEPTION crash.

Tests combinations of:
  A) EmbeddingStressSimulator (large GPU allocations + multi-stream + all_to_all)
  B) H2D double-buffered batch generation (pinned memory + memcpy_stream)
  C) DDP + Shampoo optimizer
  D) GPU_MAX_HW_QUEUES setting

Usage:
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/diagnose_crash.py --mode <MODE>

Modes:
    gpu_only     - GPU-generated data + model + DDP + Shampoo + emb_stress (no H2D)
    h2d_only     - H2D double-buffer + model + DDP + Shampoo (no emb_stress)
    emb_only     - emb_stress + trivial model (no DDP, no Shampoo, no H2D)
    full         - everything combined (reproduces crash)
    full_no_a2a  - everything but skip all_to_all in emb_stress
"""
import argparse
import logging
import os
import signal
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from aorta.models import ModelConfig, RankingTransformerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

VOCAB = 350_000
EMB_DIM = 256
DENSE_F = 32
DENSE_D = 256
MODEL_DIM = 1024
LAYERS = 18
B, T, S = 64, 64, 64

EMB_TABLES = 8
EMB_ROWS = 2_000_000
EMB_TABLE_DIM = 128
EMB_LOOKUPS = 8192
EMB_STREAMS = 4


class EmbStress:
    """Minimal embedding stress simulator for diagnostics."""
    def __init__(self, world_size, dtype, skip_a2a=False):
        self.world_size = world_size
        self.skip_a2a = skip_a2a
        self.emb_streams = [torch.cuda.Stream() for _ in range(EMB_STREAMS)]
        self.datadist_stream = torch.cuda.Stream()
        self.default_stream = torch.cuda.current_stream()

        self.tables = []
        for _ in range(EMB_TABLES):
            t = nn.Embedding(EMB_ROWS, EMB_TABLE_DIM).to("cuda", dtype=dtype)
            self.tables.append(t)

        total = EMB_LOOKUPS * EMB_TABLE_DIM
        self.send_buf = torch.empty(world_size, total, dtype=dtype, device="cuda")
        self.recv_buf = torch.empty_like(self.send_buf)
        self.indices = torch.randint(0, EMB_ROWS, (EMB_LOOKUPS,), device="cuda")

    def run(self):
        self.indices.random_(0, EMB_ROWS)
        idx_event = torch.cuda.current_stream().record_event()
        for s in self.emb_streams:
            s.wait_event(idx_event)

        chunk = self.send_buf.shape[1] // EMB_TABLES
        for t_idx, table in enumerate(self.tables):
            s = self.emb_streams[t_idx % EMB_STREAMS]
            with torch.cuda.stream(s):
                out = table(self.indices).reshape(-1)
                cs = t_idx * chunk
                sz = min(out.numel(), chunk)
                self.send_buf[:, cs:cs+sz] = out[:sz]

        for s in self.emb_streams:
            self.datadist_stream.wait_stream(s)

        work = None
        if not self.skip_a2a:
            with torch.cuda.stream(self.datadist_stream):
                work = dist.all_to_all_single(
                    self.recv_buf, self.send_buf, async_op=True)
        return work

    def wait(self, work):
        self.default_stream.wait_stream(self.datadist_stream)
        if work is not None:
            work.wait()


class H2DBatchGen:
    """Minimal H2D batch generator for diagnostics."""
    def __init__(self, device, dtype, sync_before_fill=False, verify=False,
                 blocking_h2d=False, no_pin=False):
        self.device = device
        self.dtype = dtype
        self.sync_before_fill = sync_before_fill
        self.verify = verify
        self.blocking_h2d = blocking_h2d
        self.no_pin = no_pin
        self.memcpy_stream = torch.cuda.Stream()
        self.cpu_bufs = [self._alloc(), self._alloc()]
        self.gpu_bufs: List[Optional[Dict]] = [None, None]
        self.cur = 0
        self.prefetched = False

    def _alloc(self):
        d = torch.empty(B, T, DENSE_F, DENSE_D, dtype=torch.float32)
        c = torch.empty(B, T, S, dtype=torch.long)
        t = torch.empty(B, T, dtype=torch.float32)
        if not self.no_pin:
            d = d.pin_memory()
            c = c.pin_memory()
            t = t.pin_memory()
        return {"dense": d, "categorical": c, "target": t}

    def _fill(self, buf):
        if self.sync_before_fill:
            self.memcpy_stream.synchronize()
        buf["dense"].normal_()
        buf["categorical"].random_(0, VOCAB)
        buf["target"].uniform_()
        if self.verify:
            cmin = buf["categorical"].min().item()
            cmax = buf["categorical"].max().item()
            if cmin < 0 or cmax >= VOCAB:
                raise RuntimeError(
                    f"CPU fill produced OOB: [{cmin}, {cmax}]")

    def _h2d(self, cpu_buf):
        gpu = {}
        if self.blocking_h2d:
            for k, v in cpu_buf.items():
                dt = self.dtype if v.is_floating_point() else v.dtype
                gpu[k] = v.to(self.device, dtype=dt)
        else:
            with torch.cuda.stream(self.memcpy_stream):
                for k, v in cpu_buf.items():
                    dt = self.dtype if v.is_floating_point() else v.dtype
                    gpu[k] = v.to(self.device, non_blocking=True, dtype=dt)
        if self.verify:
            if not self.blocking_h2d:
                self.memcpy_stream.synchronize()
            gmin = gpu["categorical"].min().item()
            gmax = gpu["categorical"].max().item()
            if gmin < 0 or gmax >= VOCAB:
                raise RuntimeError(
                    f"GPU H2D produced OOB: [{gmin}, {gmax}]")
        return gpu

    def get(self):
        if self.gpu_bufs[self.cur] is None:
            self._fill(self.cpu_bufs[self.cur])
            self.gpu_bufs[self.cur] = self._h2d(self.cpu_bufs[self.cur])
        return self.gpu_bufs[self.cur]

    def wait(self, use_full_sync=False):
        if use_full_sync:
            self.memcpy_stream.synchronize()
        else:
            torch.cuda.current_stream().wait_stream(self.memcpy_stream)

    def prefetch(self):
        nxt = 1 - self.cur
        self._fill(self.cpu_bufs[nxt])
        self.gpu_bufs[nxt] = self._h2d(self.cpu_bufs[nxt])
        self.prefetched = True

    def swap(self):
        old = self.cur
        if self.prefetched:
            self.cur = 1 - self.cur
            self.prefetched = False
        self.gpu_bufs[old] = None


def make_gpu_batch(device, dtype):
    return {
        "dense": torch.randn(B, T, DENSE_F, DENSE_D, device=device, dtype=dtype),
        "categorical": torch.randint(0, VOCAB, (B, T, S), device=device),
        "target": torch.rand(B, T, device=device, dtype=dtype),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["gpu_only", "h2d_only", "emb_only",
                                 "full", "full_no_a2a"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--full-sync", action="store_true",
                        help="Use memcpy_stream.synchronize() instead of wait_stream")
    parser.add_argument("--sync-before-fill", action="store_true",
                        help="Sync memcpy_stream before filling CPU buffers")
    parser.add_argument("--verify", action="store_true",
                        help="Verify CPU fill and GPU H2D produce valid values")
    parser.add_argument("--blocking-h2d", action="store_true",
                        help="Use blocking H2D (no non_blocking, no memcpy_stream)")
    parser.add_argument("--no-pin", action="store_true",
                        help="Don't use pinned memory (regular CPU memory)")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    dtype = torch.bfloat16

    signal.signal(signal.SIGABRT, lambda s, f: (
        log.error(f"SIGABRT at step {step_holder[0]} on rank {rank}"),
        sys.exit(134)))
    step_holder = [0]

    use_emb = args.mode in ("gpu_only", "full", "full_no_a2a", "emb_only")
    use_h2d = args.mode in ("h2d_only", "full", "full_no_a2a")
    use_ddp = args.mode != "emb_only"
    use_shampoo = args.mode != "emb_only"
    skip_a2a = args.mode == "full_no_a2a"

    cfg = ModelConfig(vocab_size=VOCAB, embedding_dim=EMB_DIM,
                      num_dense_features=DENSE_F, dense_dim=DENSE_D,
                      model_dim=MODEL_DIM, num_heads=16, num_layers=LAYERS,
                      dropout=0.1, mlp_hidden_dim=4096)
    model = RankingTransformerModel(cfg).to(dev)
    if use_ddp:
        model = DDP(model, device_ids=[lr])

    if use_shampoo:
        from distributed_shampoo import DistributedShampoo, DDPDistributedConfig
        optimizer = DistributedShampoo(
            model.parameters(), lr=2e-4, betas=(0.9, 0.985), epsilon=1e-8,
            max_preconditioner_dim=8192, precondition_frequency=50,
            start_preconditioning_step=50, weight_decay=0.01,
            distributed_config=DDPDistributedConfig(
                communication_dtype=torch.float32,
                num_trainers_per_group=-1, communicate_params=False))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    emb = EmbStress(ws, dtype, skip_a2a) if use_emb else None
    h2d = H2DBatchGen(dev, dtype, sync_before_fill=args.sync_before_fill,
                       verify=args.verify, blocking_h2d=args.blocking_h2d,
                       no_pin=args.no_pin) if use_h2d else None
    loss_fn = nn.MSELoss()

    if rank == 0:
        log.info(f"MODE={args.mode} steps={args.steps} grad_accum={args.grad_accum}")
        log.info(f"  emb={use_emb} h2d={use_h2d} ddp={use_ddp} "
                 f"shampoo={use_shampoo} skip_a2a={skip_a2a} "
                 f"full_sync={args.full_sync}")
        log.info(f"  HWQ={os.environ.get('GPU_MAX_HW_QUEUES', 'unset')}")

    t0 = time.time()
    for step in range(args.steps):
        step_holder[0] = step
        optimizer.zero_grad()

        for micro in range(args.grad_accum):
            do_sync = (micro == args.grad_accum - 1)
            ctx = nullcontext() if do_sync else model.no_sync() if use_ddp else nullcontext()

            if h2d is not None:
                batch = h2d.get()
                emb_work = emb.run() if emb else None
                h2d.wait(use_full_sync=args.full_sync)
            else:
                batch = make_gpu_batch(dev, dtype)
                emb_work = emb.run() if emb else None

            if emb is not None:
                emb.wait(emb_work)

            # Bounds check (no sync to not mask races - just record for post-mortem)
            cat = batch["categorical"]
            cat_max = cat.max().item()
            cat_min = cat.min().item()
            if cat_min < 0 or cat_max >= VOCAB:
                log.error(f"[rank {rank}] OOB at step {step} micro {micro}: "
                          f"[{cat_min}, {cat_max}]")
                dist.barrier()
                sys.exit(99)

            with ctx:
                with torch.amp.autocast("cuda", dtype=dtype):
                    scores = model(batch)
                    loss = loss_fn(scores, batch["target"])
                    if args.grad_accum > 1:
                        loss = loss / args.grad_accum

                if h2d is not None:
                    h2d.prefetch()
                loss.backward()

            if h2d is not None:
                h2d.swap()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if rank == 0 and (step + 1) % 20 == 0:
            ms = (time.time() - t0) / (step + 1) * 1000
            log.info(f"Step {step+1}/{args.steps} | loss={loss.item():.4f} | "
                     f"{ms:.0f} ms/step")

    dist.barrier()
    elapsed = time.time() - t0
    if rank == 0:
        log.info(f"PASSED: {args.steps} steps in {elapsed:.1f}s "
                 f"({elapsed/args.steps*1000:.0f} ms/step)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
