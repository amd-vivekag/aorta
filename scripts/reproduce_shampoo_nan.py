"""
DLRMv3/HSTU Shampoo NaN Reproducer.

Reproduces the nondeterministic NaN in Meta's DLRMv3/HSTU workload on AMD GPUs.
The NaN is caused by a race between:
  1. Shampoo's DDPDistributedConfig all_gather on the default CUDA stream
  2. NCCL collectives (all_to_all for embedding redistribution) on a side stream
  3. PyTorch CachingAllocator reusing memory that NCCL is still reading

Key reproduction strategies:
  - Overlap NCCL all_to_all on a side stream with Shampoo optimizer on stream 0
  - Heavy CachingAllocator churn: allocate/free large tensors between backward and
    optimizer step to maximize chance of NCCL reading freed+reused memory
  - Low precondition_frequency to hit preconditioner computation often
  - No grad clipping (so corrupted values aren't clamped away)
  - GPU_MAX_HW_QUEUES=4 to maximize concurrent kernel execution

Usage:
    # Aggressive NaN reproduction (8 GPUs)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 \
        scripts/reproduce_shampoo_nan.py \
        --batch-size 1024 --seq-len 200 --max-steps 10000 \
        --precondition-frequency 10 --start-preconditioning-step 50 \
        --scale quarter --grad-clip-norm 0 --alloc-stress

    # With sync fix (should eliminate NaN if hypothesis is correct)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 \
        scripts/reproduce_shampoo_nan.py \
        --batch-size 1024 --seq-len 200 --max-steps 10000 \
        --precondition-frequency 10 --start-preconditioning-step 50 \
        --scale quarter --sync-before-precondition

    # Quick 2-GPU smoke test
    torchrun --nproc_per_node=2 \
        scripts/reproduce_shampoo_nan.py \
        --batch-size 128 --seq-len 100 --max-steps 200 \
        --precondition-frequency 10 --start-preconditioning-step 10 \
        --scale tiny --force-fallback
"""

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | R%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================================================================
# Scale configs for embedding table sizes
# =========================================================================

SCALE_CONFIGS = {
    "full": {
        "item_hash_size": 1_000_000_000,
        "user_hash_size": 10_000_000,
        "category_hash_size": 128,
        "small_table_sizes": [1_100_000, 16_000_000, 32_500_000, 3_400, 838_000, 33_500_000],
        "large_table_rows": 32_400_000,
        "large_table_count": 14,
        "weighted_table_rows": 6_700_000,
    },
    "half": {
        "item_hash_size": 500_000_000,
        "user_hash_size": 5_000_000,
        "category_hash_size": 128,
        "small_table_sizes": [550_000, 8_000_000, 16_250_000, 1_700, 419_000, 16_750_000],
        "large_table_rows": 16_200_000,
        "large_table_count": 14,
        "weighted_table_rows": 3_350_000,
    },
    "quarter": {
        "item_hash_size": 250_000_000,
        "user_hash_size": 2_500_000,
        "category_hash_size": 128,
        "small_table_sizes": [275_000, 4_000_000, 8_125_000, 850, 210_000, 8_375_000],
        "large_table_rows": 8_100_000,
        "large_table_count": 14,
        "weighted_table_rows": 1_675_000,
    },
    "tiny": {
        "item_hash_size": 10_000_000,
        "user_hash_size": 100_000,
        "category_hash_size": 128,
        "small_table_sizes": [11_000, 160_000, 325_000, 128, 8_380, 335_000],
        "large_table_rows": 324_000,
        "large_table_count": 14,
        "weighted_table_rows": 67_000,
    },
}


# =========================================================================
# Model setup using generative_recommenders
# =========================================================================


def try_import_generative_recommenders():
    """Try importing generative_recommenders, return None components on failure."""
    try:
        from generative_recommenders.modules.dlrm_hstu import (
            DlrmHSTU,
            DlrmHSTUConfig,
            SequenceEmbedding,
        )
        from generative_recommenders.modules.multitask_module import (
            MultitaskTaskType,
            TaskConfig,
        )
        return DlrmHSTU, DlrmHSTUConfig, SequenceEmbedding, MultitaskTaskType, TaskConfig
    except ImportError:
        return None


def create_hstu_config(args):
    """Create DlrmHSTUConfig with trace-matched dimensions."""
    gr = try_import_generative_recommenders()
    if gr is None:
        raise ImportError(
            "generative_recommenders not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/generative-recommenders.git"
        )
    _, DlrmHSTUConfig, _, MultitaskTaskType, TaskConfig = gr

    multitask_configs = []
    for i in range(args.num_binary_tasks):
        multitask_configs.append(
            TaskConfig(
                task_name=f"binary_{i}",
                task_weight=1 << i,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            )
        )

    config = DlrmHSTUConfig(
        max_seq_len=args.seq_len,
        max_num_candidates=args.num_candidates,
        hstu_num_heads=args.num_heads,
        hstu_attn_linear_dim=args.attn_linear_dim,
        hstu_attn_qk_dim=args.attn_qk_dim,
        hstu_attn_num_layers=args.num_layers,
        hstu_embedding_table_dim=args.embedding_table_dim,
        hstu_preprocessor_hidden_dim=256,
        hstu_transducer_embedding_dim=args.d_model,
        hstu_group_norm=False,
        hstu_input_dropout_ratio=0.2,
        hstu_linear_dropout_rate=0.1,
        causal_multitask_weights=0.2,
        multitask_configs=multitask_configs,
        user_embedding_feature_names=["item_id", "user_id", "item_category_id"],
        item_embedding_feature_names=["item_candidate_id", "item_candidate_category_id"],
        uih_post_id_feature_name="item_id",
        uih_action_time_feature_name="action_timestamp",
        candidates_querytime_feature_name="item_query_time",
        candidates_weight_feature_name="item_action_weights",
        uih_weight_feature_name="item_weights",
        candidates_watchtime_feature_name="item_rating",
        action_weights=[1, 2, 4, 8, 16],
        action_embedding_init_std=5.0,
        contextual_feature_to_max_length={"user_id": 1},
        contextual_feature_to_min_uih_length={"user_id": 20},
        merge_uih_candidate_feature_mapping=[
            ("item_id", "item_candidate_id"),
            ("item_rating", "item_candidate_rating"),
            ("action_timestamp", "item_query_time"),
            ("item_weights", "item_action_weights"),
            ("dummy_watch_time", "item_dummy_watchtime"),
            ("item_category_id", "item_candidate_category_id"),
        ],
        hstu_uih_feature_names=[
            "user_id", "item_id", "item_rating",
            "action_timestamp", "item_weights",
            "dummy_watch_time", "item_category_id",
        ],
        hstu_candidate_feature_names=[
            "item_candidate_id", "item_candidate_rating",
            "item_query_time", "item_action_weights",
            "item_dummy_watchtime", "item_candidate_category_id",
        ],
    )
    return config


def create_embedding_table_config(args):
    """Create embedding table configs matching trace groups."""
    from torchrec.modules.embedding_configs import DataType, EmbeddingConfig

    scale = SCALE_CONFIGS[args.scale]
    return {
        "item_id": EmbeddingConfig(
            num_embeddings=scale["item_hash_size"],
            embedding_dim=args.embedding_table_dim,
            name="item_id",
            data_type=DataType.FP16,
            feature_names=["item_id", "item_candidate_id"],
        ),
        "item_category_id": EmbeddingConfig(
            num_embeddings=scale["category_hash_size"],
            embedding_dim=args.embedding_table_dim,
            name="item_category_id",
            data_type=DataType.FP16,
            weight_init_max=1.0,
            weight_init_min=-1.0,
            feature_names=["item_category_id", "item_candidate_category_id"],
        ),
        "user_id": EmbeddingConfig(
            num_embeddings=scale["user_hash_size"],
            embedding_dim=args.embedding_table_dim,
            name="user_id",
            data_type=DataType.FP16,
            feature_names=["user_id"],
        ),
    }


def create_model(args, device):
    """Create DlrmHSTU model using generative_recommenders."""
    gr = try_import_generative_recommenders()
    DlrmHSTU = gr[0]

    config = create_hstu_config(args)
    table_config = create_embedding_table_config(args)

    model = DlrmHSTU(
        hstu_configs=config,
        embedding_tables=table_config,
        is_inference=False,
        is_dense=False,
        bf16_training=True,
    )
    model = model.to(device)
    model.set_training_dtype(torch.bfloat16)
    model.train()
    return model, config


# =========================================================================
# Fallback model when generative_recommenders is not available
# =========================================================================


class FallbackHSTULayer(nn.Module):
    """HSTU attention layer matching trace: self-attention + gating + wide FFN."""

    def __init__(self, d_model: int, num_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.gate = nn.Linear(d_model, d_model)
        ffn_dim = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        gate = torch.sigmoid(self.gate(x))
        x = residual + gate * attn_out
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        return x


class FallbackHSTUModel(nn.Module):
    """HSTU model with trace-matched dense parameter count.

    The model is intentionally designed with many dense parameters
    so that Shampoo's DDPDistributedConfig has a large all_gather
    payload, maximizing the window for NCCL/allocator races.
    """

    def __init__(
        self,
        d_model: int = 96,
        num_layers: int = 7,
        num_heads: int = 16,
        embedding_dim: int = 128,
        num_embeddings: int = 1_000_000,
        num_binary_tasks: int = 8,
        seq_len: int = 200,
        num_candidates: int = 10,
        ffn_mult: int = 4,
        output_mlp_width: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_candidates = num_candidates

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.embed_proj = nn.Linear(embedding_dim, d_model)
        self.pos_embed = nn.Embedding(seq_len + num_candidates, d_model)

        self.layers = nn.ModuleList([
            FallbackHSTULayer(d_model, num_heads, ffn_mult=ffn_mult)
            for _ in range(num_layers)
        ])

        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1)

        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, output_mlp_width),
            nn.SiLU(),
            nn.LayerNorm(output_mlp_width),
            nn.Linear(output_mlp_width, output_mlp_width),
            nn.SiLU(),
            nn.LayerNorm(output_mlp_width),
            nn.Linear(output_mlp_width, d_model),
            nn.LayerNorm(d_model),
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, output_mlp_width),
            nn.SiLU(),
            nn.LayerNorm(output_mlp_width),
            nn.Linear(output_mlp_width, output_mlp_width),
            nn.SiLU(),
            nn.LayerNorm(output_mlp_width),
            nn.Linear(output_mlp_width, num_binary_tasks),
        )

    def forward(self, item_ids, candidate_ids, seq_lengths):
        B = item_ids.shape[0]
        S = min(item_ids.shape[1], self.seq_len)
        C = min(candidate_ids.shape[1], self.num_candidates)

        uih_emb = self.embedding(item_ids[:, :S])
        cand_emb = self.embedding(candidate_ids[:, :C])

        uih_proj = self.embed_proj(uih_emb)
        combined = torch.cat([uih_proj, self.embed_proj(cand_emb)], dim=1)
        total_len = combined.shape[1]
        positions = torch.arange(total_len, device=combined.device).unsqueeze(0)
        combined = combined + self.pos_embed(positions)

        combined = combined + self.conv1d(combined.transpose(1, 2)).transpose(1, 2)

        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=combined.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            combined = layer(combined, attn_mask=causal_mask)

        candidate_out = combined[:, S:S+C, :]

        item_features = self.item_mlp(cand_emb)
        interaction = candidate_out * item_features
        logits = self.output_head(interaction)
        return logits

    def compute_loss(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)


def create_fallback_model(args, device):
    """Create fallback HSTU when generative_recommenders is unavailable."""
    scale = SCALE_CONFIGS[args.scale]
    emb_size_limit = {
        "tiny": 1_000_000,
        "quarter": 5_000_000,
        "half": 10_000_000,
        "full": 50_000_000,
    }
    max_emb = emb_size_limit.get(args.scale, 1_000_000)
    num_embeddings = min(scale["item_hash_size"], max_emb)

    model = FallbackHSTUModel(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        embedding_dim=args.embedding_table_dim,
        num_embeddings=num_embeddings,
        num_binary_tasks=args.num_binary_tasks,
        seq_len=args.seq_len,
        num_candidates=args.num_candidates,
        ffn_mult=args.ffn_mult,
        output_mlp_width=args.output_mlp_width,
    )
    model = model.to(device)
    model.train()
    return model


# =========================================================================
# Data generator: DLRMv3-style synthetic batches
# =========================================================================

@dataclass
class DataConfig:
    num_categories: int = 128
    categories_per_user: int = 4
    num_timestamps: int = 100
    alpha_range: Tuple[int, int] = (1, 500)
    rating_values: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0])
    rating_probs: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.25, 0.25, 0.2])


class DLRMv3BatchGenerator:
    """Generates DLRMv3-realistic batches packed as KJT-compatible dicts."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_candidates: int,
        item_hash_size: int,
        user_hash_size: int,
        category_hash_size: int,
        data_config: Optional[DataConfig] = None,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_candidates = num_candidates
        self.item_hash_size = item_hash_size
        self.user_hash_size = user_hash_size
        self.category_hash_size = category_hash_size
        self.cfg = data_config or DataConfig()

        self.rng = np.random.RandomState(42)
        self.items_per_category = item_hash_size // self.cfg.num_categories

        self.user_categories = []
        for _ in range(batch_size):
            cats = self.rng.choice(
                self.cfg.num_categories, size=self.cfg.categories_per_user, replace=False,
            ).tolist()
            alpha = self.rng.randint(*self.cfg.alpha_range)
            self.user_categories.append((cats, alpha))

    def _sample_seq_len(self) -> int:
        length = round(self.rng.normal(self.seq_len, self.seq_len // 4))
        return max(self.num_candidates + 1, min(length, self.seq_len))

    def _sample_items(self, user_idx: int, n: int, timestamp: int) -> Tuple[np.ndarray, np.ndarray]:
        cats, alpha = self.user_categories[user_idx % len(self.user_categories)]
        p = np.ones(len(cats)) / len(cats)
        chosen_cats = self.rng.choice(cats, size=n, p=p)

        sample_end = max(int(self.items_per_category * (timestamp + 1) / self.cfg.num_timestamps), 1)
        within_cat = self.rng.randint(0, sample_end, size=n)
        offsets = np.array([c * self.items_per_category for c in chosen_cats])
        item_ids = (within_cat + offsets) % self.item_hash_size
        return item_ids, chosen_cats

    def generate_batch(self, iteration: int) -> Dict[str, torch.Tensor]:
        B = self.batch_size
        S = self.seq_len
        C = self.num_candidates
        timestamp = iteration % self.cfg.num_timestamps

        item_ids = np.zeros((B, S), dtype=np.int64)
        user_ids = np.zeros((B,), dtype=np.int64)
        category_ids = np.zeros((B, S), dtype=np.int64)
        candidate_ids = np.zeros((B, C), dtype=np.int64)
        candidate_cats = np.zeros((B, C), dtype=np.int64)
        seq_lengths = np.zeros(B, dtype=np.int64)
        action_timestamps = np.zeros((B, S), dtype=np.int64)
        item_weights = np.ones((B, S), dtype=np.int64)
        item_ratings = np.zeros((B, S), dtype=np.float32)
        candidate_ratings = np.zeros((B, C), dtype=np.float32)
        candidate_query_times = np.full((B, C), timestamp, dtype=np.int64)
        candidate_action_weights = np.ones((B, C), dtype=np.int64)
        dummy_watch_times = np.zeros((B, S), dtype=np.float32)
        candidate_dummy_watchtimes = np.zeros((B, C), dtype=np.float32)

        for b in range(B):
            sl = self._sample_seq_len()
            seq_lengths[b] = sl
            items, cats = self._sample_items(b, sl, timestamp)
            item_ids[b, :sl] = items
            category_ids[b, :sl] = cats % self.category_hash_size
            user_ids[b] = (b + iteration * B) % self.user_hash_size
            action_timestamps[b, :sl] = np.arange(sl)
            item_ratings[b, :sl] = self.rng.choice(
                self.cfg.rating_values, size=sl, p=self.cfg.rating_probs,
            )
            cand_items, cand_cats_arr = self._sample_items(b, C, timestamp)
            candidate_ids[b] = cand_items
            candidate_cats[b] = cand_cats_arr % self.category_hash_size
            candidate_ratings[b] = self.rng.choice(
                self.cfg.rating_values, size=C, p=self.cfg.rating_probs,
            )

        return {
            "item_ids": torch.from_numpy(item_ids),
            "user_ids": torch.from_numpy(user_ids),
            "category_ids": torch.from_numpy(category_ids),
            "candidate_ids": torch.from_numpy(candidate_ids),
            "candidate_cats": torch.from_numpy(candidate_cats),
            "seq_lengths": torch.from_numpy(seq_lengths),
            "action_timestamps": torch.from_numpy(action_timestamps),
            "item_weights": torch.from_numpy(item_weights),
            "item_ratings": torch.from_numpy(item_ratings.astype(np.float32)),
            "candidate_ratings": torch.from_numpy(candidate_ratings.astype(np.float32)),
            "candidate_query_times": torch.from_numpy(candidate_query_times),
            "candidate_action_weights": torch.from_numpy(candidate_action_weights),
            "dummy_watch_times": torch.from_numpy(dummy_watch_times.astype(np.float32)),
            "candidate_dummy_watchtimes": torch.from_numpy(candidate_dummy_watchtimes.astype(np.float32)),
        }

    def generate_pinned_batch(self, iteration: int) -> Dict[str, torch.Tensor]:
        batch = self.generate_batch(iteration)
        return {k: v.pin_memory() for k, v in batch.items()}


# =========================================================================
# TorchRec KJT packing
# =========================================================================


def pack_batch_to_kjt(batch: Dict[str, torch.Tensor], device: torch.device, config):
    """Pack batch tensors into TorchRec KeyedJaggedTensor format."""
    try:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
        return _pack_to_kjt_real(batch, device, config)
    except ImportError:
        return _pack_to_kjt_fallback(batch, device, config)


def _pack_to_kjt_real(batch, device, config):
    """Pack using real TorchRec KJT."""
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    B = batch["item_ids"].shape[0]
    seq_lengths = batch["seq_lengths"]

    uih_keys = config.hstu_uih_feature_names
    candidate_keys = config.hstu_candidate_feature_names

    uih_values_list = []
    uih_lengths_list = []
    for key in uih_keys:
        if key == "item_id":
            vals = batch["item_ids"]
        elif key == "user_id":
            vals = batch["user_ids"].unsqueeze(1).expand(-1, 1)
            uih_lengths_list.append(torch.ones(B, dtype=torch.long, device=device))
            uih_values_list.append(vals.reshape(-1).to(device))
            continue
        elif key == "item_category_id":
            vals = batch["category_ids"]
        elif key == "action_timestamp":
            vals = batch["action_timestamps"]
        elif key == "item_weights":
            vals = batch["item_weights"]
        elif key == "item_rating":
            vals = batch["item_ratings"].long()
        elif key == "dummy_watch_time":
            vals = batch["dummy_watch_times"].long()
        else:
            vals = torch.zeros(B, batch["item_ids"].shape[1], dtype=torch.long)

        flat_vals = []
        lengths = []
        for b in range(B):
            sl = seq_lengths[b].item()
            flat_vals.append(vals[b, :sl])
            lengths.append(sl)
        uih_values_list.append(torch.cat(flat_vals).to(device))
        uih_lengths_list.append(torch.tensor(lengths, dtype=torch.long, device=device))

    uih_features = KeyedJaggedTensor.from_lengths_sync(
        keys=uih_keys,
        values=torch.cat(uih_values_list),
        lengths=torch.cat(uih_lengths_list),
    )

    C = batch["candidate_ids"].shape[1]
    cand_values_list = []
    cand_lengths_list = []
    for key in candidate_keys:
        if key == "item_candidate_id":
            vals = batch["candidate_ids"]
        elif key == "item_candidate_category_id":
            vals = batch["candidate_cats"]
        elif key == "item_query_time":
            vals = batch["candidate_query_times"]
        elif key == "item_action_weights":
            vals = batch["candidate_action_weights"]
        elif key == "item_candidate_rating":
            vals = batch["candidate_ratings"].long()
        elif key == "item_dummy_watchtime":
            vals = batch["candidate_dummy_watchtimes"].long()
        else:
            vals = torch.zeros(B, C, dtype=torch.long)

        cand_values_list.append(vals.reshape(-1).to(device))
        cand_lengths_list.append(torch.full((B,), C, dtype=torch.long, device=device))

    candidates_features = KeyedJaggedTensor.from_lengths_sync(
        keys=candidate_keys,
        values=torch.cat(cand_values_list),
        lengths=torch.cat(cand_lengths_list),
    )

    return uih_features, candidates_features


def _pack_to_kjt_fallback(batch, device, config):
    """Fallback when TorchRec is not available -- return raw tensors."""
    return {k: v.to(device) for k, v in batch.items()}, None


# =========================================================================
# TorchRec 3-stage pipeline (simplified)
# =========================================================================


class SimplePipeline:
    """3-stage pipeline matching TorchRec TrainPipelineSparseDist.

    Stage layout (matching Meta's trace):
      - memcpy_stream (Stream 8): H2D copy for iteration N+2
      - datadist_stream (Stream 9): all_to_all redistribution for iteration N+1
      - default_stream (Stream 0): forward+backward+optimizer for iteration N
    """

    def __init__(self, device: torch.device, batch_gen, config, use_pipeline: bool = True):
        self.device = device
        self.batch_gen = batch_gen
        self.config = config
        self.use_pipeline = use_pipeline
        self.batches: deque = deque()
        self.iteration = 0

        if use_pipeline:
            self.memcpy_stream = torch.cuda.Stream()
            self.datadist_stream = torch.cuda.Stream()
        else:
            self.memcpy_stream = None
            self.datadist_stream = None

    def _generate_and_transfer(self):
        raw_batch = self.batch_gen.generate_batch(self.iteration)
        self.iteration += 1

        if self.use_pipeline and self.memcpy_stream is not None:
            with torch.cuda.stream(self.memcpy_stream):
                device_batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in raw_batch.items()
                }
        else:
            device_batch = {k: v.to(self.device) for k, v in raw_batch.items()}

        return device_batch

    def fill(self, depth: int = 3):
        while len(self.batches) < depth:
            self.batches.append(self._generate_and_transfer())

    def get_batch(self) -> Dict[str, torch.Tensor]:
        if self.use_pipeline:
            if self.memcpy_stream is not None:
                torch.cuda.current_stream().wait_stream(self.memcpy_stream)

        if len(self.batches) == 0:
            self.fill(1)

        batch = self.batches.popleft()

        self.batches.append(self._generate_and_transfer())

        return batch


class NCCLTrafficGenerator:
    """Generates concurrent NCCL traffic on a side stream to simulate
    TorchRec's all_to_all embedding redistribution.

    In Meta's trace, NCCL all_to_all kernels (10-44ms each) on stream 4
    overlap heavily with compute on stream 0. This creates the window
    for the CachingAllocator race: stream 0 frees a tensor, allocator
    hands it out for Shampoo's all_gather buffer, but NCCL on stream 4
    is still reading from it.
    """

    def __init__(self, device: torch.device, world_size: int, rank: int,
                 payload_mb: float = 16.0):
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.nccl_stream = torch.cuda.Stream()
        nelems = int(payload_mb * 1024 * 1024 / 2)  # bf16 = 2 bytes
        self.send_buf = torch.randn(nelems, device=device, dtype=torch.bfloat16)
        self.recv_buf = torch.empty(nelems, device=device, dtype=torch.bfloat16)
        self._work = None

    def launch_async_alltoall(self):
        """Launch an all_to_all on the NCCL side stream (non-blocking)."""
        with torch.cuda.stream(self.nccl_stream):
            self._work = dist.all_to_all_single(
                self.recv_buf, self.send_buf, async_op=True,
            )

    def wait(self):
        """Wait for the all_to_all to complete."""
        if self._work is not None:
            self._work.wait()
            self._work = None


class AllocationStressor:
    """Creates CachingAllocator churn to maximize the chance that
    freed memory blocks are handed out while NCCL is still reading them.

    This mimics the aten::empty_strided burst seen in Finding 6:
    20-26 allocations right before Shampoo's all_gather_base.
    """

    def __init__(self, device: torch.device, alloc_mb: float = 8.0, num_allocs: int = 24):
        self.device = device
        self.alloc_mb = alloc_mb
        self.num_allocs = num_allocs

    def churn(self):
        """Allocate and immediately free tensors to pressure the CachingAllocator."""
        nelems = int(self.alloc_mb * 1024 * 1024 / 4)  # fp32 = 4 bytes
        tensors = []
        for _ in range(self.num_allocs):
            t = torch.empty(nelems, device=self.device, dtype=torch.float32)
            tensors.append(t)
        del tensors


# =========================================================================
# NaN checker
# =========================================================================


class NaNChecker:
    """Tracks NaN/Inf occurrences across training."""

    def __init__(self):
        self.nan_steps: List[int] = []
        self.total_nans = 0
        self.first_nan_location: Optional[str] = None

    def check_tensor(self, tensor: torch.Tensor, name: str, step: int) -> bool:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            nan_ct = torch.isnan(tensor).sum().item()
            inf_ct = torch.isinf(tensor).sum().item()
            self.nan_steps.append(step)
            self.total_nans += 1
            if self.first_nan_location is None:
                self.first_nan_location = f"{name} at step {step}"
            log.error(
                f"NaN/Inf in {name} at step {step}: "
                f"NaN={nan_ct}, Inf={inf_ct}, shape={list(tensor.shape)}"
            )
            return False
        return True

    def check_loss(self, loss: torch.Tensor, step: int) -> bool:
        val = loss.item()
        if math.isnan(val) or math.isinf(val):
            self.nan_steps.append(step)
            self.total_nans += 1
            if self.first_nan_location is None:
                self.first_nan_location = f"loss at step {step}"
            log.error(f"NaN/Inf in LOSS at step {step}: {val}")
            return False
        return True

    def check_gradients(self, model: nn.Module, step: int) -> bool:
        ok = True
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not self.check_tensor(param.grad, f"grad/{name}", step):
                    ok = False
        return ok

    def check_parameters(self, model: nn.Module, step: int) -> bool:
        ok = True
        for name, param in model.named_parameters():
            if not self.check_tensor(param.data, f"param/{name}", step):
                ok = False
        return ok

    def summary(self) -> str:
        if self.total_nans == 0:
            return "No NaN/Inf detected."
        return (
            f"TOTAL NaN/Inf events: {self.total_nans}, "
            f"first: {self.first_nan_location}, "
            f"affected steps: {sorted(set(self.nan_steps))[:20]}"
        )


# =========================================================================
# Signal handler
# =========================================================================

_CRASH_STATE: Dict = {"step": -1, "rank": -1, "nan_checker": None}


def _crash_handler(signum, frame):
    step = _CRASH_STATE["step"]
    rank = _CRASH_STATE["rank"]
    nan_checker = _CRASH_STATE["nan_checker"]
    try:
        sig_name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        sig_name = str(signum)

    log.error(f"SIGNAL {sig_name} at step {step} on rank {rank}")

    crash_info = {"signal": sig_name, "step": step, "rank": rank}
    if nan_checker:
        crash_info["nan_summary"] = nan_checker.summary()

    try:
        path = f"reproduce_nan_crash_rank{rank}.json"
        with open(path, "w") as f:
            json.dump(crash_info, f, indent=2)
        log.error(f"Crash state saved to {path}")
    except Exception:
        pass

    sys.exit(128 + signum)


# =========================================================================
# Distributed setup
# =========================================================================


def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# =========================================================================
# Optimizer setup
# =========================================================================


def _split_params(model):
    """Split model params into dense (for Shampoo) and embedding (for AdaGrad).

    In Meta's trace, Shampoo only handles dense params (attention, MLP, norms).
    Embeddings use rowwise AdaGrad via FBGEMM TBE.
    """
    embedding_params = []
    dense_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_embedding = any(k in name for k in ["embedding", "embed"])
        if is_embedding:
            embedding_params.append(param)
        else:
            dense_params.append(param)
    return dense_params, embedding_params


def create_optimizer(model, args, disable_ddp_config=False):
    """Create optimizer: Shampoo on dense params, AdaGrad on embedding params.

    This matches Meta's setup where FBGEMM TBE uses rowwise AdaGrad for
    embeddings, and Distributed Shampoo handles all dense parameters.
    """
    dense_params, embedding_params = _split_params(model)

    optimizers = []

    if embedding_params:
        emb_opt = torch.optim.Adagrad(embedding_params, lr=args.lr)
        optimizers.append(("adagrad_emb", emb_opt))

    if args.optimizer == "shampoo" and dense_params:
        from distributed_shampoo import DistributedShampoo

        shampoo_kwargs = dict(
            lr=args.lr,
            betas=(0.9, 0.985),
            epsilon=1e-8,
            weight_decay=args.weight_decay,
            max_preconditioner_dim=args.max_preconditioner_dim,
            precondition_frequency=args.precondition_frequency,
            start_preconditioning_step=args.start_preconditioning_step,
        )

        if not disable_ddp_config:
            from distributed_shampoo import DDPDistributedConfig
            shampoo_kwargs["distributed_config"] = DDPDistributedConfig(
                communication_dtype=torch.float32,
                num_trainers_per_group=-1,
                communicate_params=False,
            )

        dense_opt = DistributedShampoo(dense_params, **shampoo_kwargs)
        optimizers.append(("shampoo_dense", dense_opt))
    elif dense_params:
        dense_opt = torch.optim.AdamW(
            dense_params,
            lr=args.lr,
            betas=(0.9, 0.985),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        optimizers.append(("adam_dense", dense_opt))

    return MultiOptimizer(optimizers)


class MultiOptimizer:
    """Wraps multiple optimizers to present a single optimizer interface."""

    def __init__(self, optimizer_list):
        self.optimizers = {name: opt for name, opt in optimizer_list}

    def zero_grad(self):
        for opt in self.optimizers.values():
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers.values():
            opt.step()

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers.values():
            groups.extend(opt.param_groups)
        return groups


# =========================================================================
# Training loop
# =========================================================================


def train_with_generative_recommenders(model, config, optimizer, pipeline, args,
                                        rank, nan_checker, device):
    """Training loop using generative_recommenders DlrmHSTU model."""
    dtype = torch.bfloat16

    for step in range(args.max_steps):
        _CRASH_STATE["step"] = step

        optimizer.zero_grad()

        batch = pipeline.get_batch()

        if args.sync_before_precondition and step > 0 and step % args.precondition_frequency == 0:
            torch.cuda.synchronize()
            if rank == 0 and step % args.log_interval == 0:
                log.info(f"Step {step}: sync before precondition step")

        try:
            uih_features, candidates_features = pack_batch_to_kjt(batch, device, config)
        except Exception as e:
            if rank == 0:
                log.warning(f"KJT packing failed at step {step}: {e}, using fallback forward")
            loss = _fallback_forward(model, batch, dtype)
            if loss is None:
                continue
            if not nan_checker.check_loss(loss, step):
                if args.stop_on_nan:
                    break
                continue
            loss.backward()
            optimizer.step()
            _log_step(step, loss, rank, args, nan_checker, time.time())
            continue

        with torch.amp.autocast("cuda", dtype=dtype):
            if isinstance(uih_features, dict):
                loss = _fallback_forward_from_dict(model, uih_features, dtype)
            else:
                try:
                    outputs = model(
                        uih_features=uih_features,
                        candidates_features=candidates_features,
                    )
                    _, _, aux_losses, mt_preds, mt_labels, mt_weights = outputs

                    if aux_losses:
                        loss = sum(aux_losses.values())
                    elif mt_preds is not None:
                        loss = mt_preds.sum() * 0.0
                        if mt_labels is not None:
                            loss = F.binary_cross_entropy_with_logits(
                                mt_preds, mt_labels.float(),
                            )
                    else:
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                except Exception as e:
                    if rank == 0 and step % args.log_interval == 0:
                        log.warning(f"Forward failed at step {step}: {e}")
                    loss = _fallback_forward(model, batch, dtype)
                    if loss is None:
                        continue

        if not nan_checker.check_loss(loss, step):
            if args.stop_on_nan:
                log.error(f"Stopping at step {step} due to NaN in loss")
                break
            continue

        loss.backward()

        if step % args.nan_check_interval == 0:
            if not nan_checker.check_gradients(model, step):
                if args.stop_on_nan:
                    log.error(f"Stopping at step {step} due to NaN in gradients")
                    break

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer.step()

        if step % args.nan_check_interval == 0:
            if not nan_checker.check_parameters(model, step):
                if args.stop_on_nan:
                    log.error(f"Stopping at step {step} due to NaN in parameters")
                    break

        _log_step(step, loss, rank, args, nan_checker, time.time())

    return step


def train_with_fallback(model, optimizer, pipeline, args, rank, nan_checker, device,
                        nccl_traffic=None, alloc_stressor=None):
    """Training loop using fallback HSTU model.

    The loop is structured to maximize the race window:
      1. Forward + backward on stream 0
      2. Launch async NCCL all_to_all on side stream (simulates TorchRec data_dist)
      3. Allocation churn (simulates aten::empty_strided burst before all_gather)
      4. optimizer.step() (Shampoo does all_gather on stream 0 while NCCL is active)
    """
    dtype = torch.bfloat16

    for step in range(args.max_steps):
        _CRASH_STATE["step"] = step

        optimizer.zero_grad()

        batch = pipeline.get_batch()

        if args.sync_before_precondition and step > 0 and step % args.precondition_frequency == 0:
            torch.cuda.synchronize()
            if rank == 0 and step % args.log_interval == 0:
                log.info(f"Step {step}: sync before precondition")

        with torch.amp.autocast("cuda", dtype=dtype):
            inner_model = model.module if hasattr(model, "module") else model
            logits = model(
                batch["item_ids"],
                batch["candidate_ids"],
                batch["seq_lengths"],
            )
            B, C, T = logits.shape
            targets = torch.zeros(B, C, T, device=device, dtype=logits.dtype)
            labels = (batch["candidate_ratings"][:, :C] > 3.0).float()
            for t_idx in range(T):
                targets[:, :, t_idx] = labels

            loss = inner_model.compute_loss(logits, targets)

        if not nan_checker.check_loss(loss, step):
            if args.stop_on_nan:
                log.error(f"Stopping at step {step} due to NaN in loss")
                break
            continue

        loss.backward()

        if nccl_traffic is not None:
            nccl_traffic.launch_async_alltoall()

        if alloc_stressor is not None:
            alloc_stressor.churn()

        if step % args.nan_check_interval == 0:
            if not nan_checker.check_gradients(model, step):
                if args.stop_on_nan:
                    break

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer.step()

        if nccl_traffic is not None:
            nccl_traffic.wait()

        if step % args.nan_check_interval == 0:
            if not nan_checker.check_parameters(model, step):
                if args.stop_on_nan:
                    break

        _log_step(step, loss, rank, args, nan_checker, time.time())

    return step


def _fallback_forward(model, batch, dtype):
    """Attempt a simple forward when the full pipeline fails."""
    try:
        if hasattr(model, 'module'):
            m = model.module
        else:
            m = model
        if isinstance(m, FallbackHSTUModel):
            logits = m(batch["item_ids"], batch["candidate_ids"], batch["seq_lengths"])
            B, C, T = logits.shape
            targets = torch.zeros_like(logits)
            return m.compute_loss(logits, targets)
        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
    except Exception:
        return None


def _fallback_forward_from_dict(model, batch_dict, dtype):
    """Forward when we only have dict tensors (no KJT)."""
    try:
        if hasattr(model, 'module'):
            m = model.module
        else:
            m = model
        if isinstance(m, FallbackHSTUModel):
            logits = m(batch_dict["item_ids"], batch_dict["candidate_ids"], batch_dict["seq_lengths"])
            targets = torch.zeros_like(logits)
            return m.compute_loss(logits, targets)
        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
    except Exception:
        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)


_train_start_time = None


def _log_step(step, loss, rank, args, nan_checker, current_time):
    global _train_start_time
    if _train_start_time is None:
        _train_start_time = current_time

    if rank == 0 and (step + 1) % args.log_interval == 0:
        elapsed = current_time - _train_start_time
        steps_per_sec = (step + 1) / max(elapsed, 0.001)
        ms_per_step = 1000.0 / max(steps_per_sec, 0.001)
        is_precond = (
            step >= args.start_preconditioning_step
            and step % args.precondition_frequency == 0
        )
        precond_marker = " [PRECOND]" if is_precond else ""
        log.info(
            f"Step {step + 1}/{args.max_steps} | "
            f"loss={loss.item():.4f} | "
            f"{ms_per_step:.1f} ms/step | "
            f"NaN={nan_checker.total_nans}"
            f"{precond_marker}"
        )


# =========================================================================
# CLI
# =========================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="DLRMv3/HSTU Shampoo NaN Reproducer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--d-model", type=int, default=96,
                             help="HSTU transducer embedding dim (d_model)")
    model_group.add_argument("--num-layers", type=int, default=7,
                             help="Number of HSTU attention layers")
    model_group.add_argument("--num-heads", type=int, default=16,
                             help="Number of attention heads")
    model_group.add_argument("--attn-linear-dim", type=int, default=128,
                             help="HSTU attention linear (hidden) dim")
    model_group.add_argument("--attn-qk-dim", type=int, default=96,
                             help="HSTU attention Q/K dim")
    model_group.add_argument("--embedding-table-dim", type=int, default=128,
                             help="Embedding table dimension")
    model_group.add_argument("--num-binary-tasks", type=int, default=8,
                             help="Number of binary classification tasks")
    model_group.add_argument("--num-candidates", type=int, default=10,
                             help="Number of candidates per sample")

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--batch-size", type=int, default=1024,
                            help="Batch size per GPU")
    data_group.add_argument("--seq-len", type=int, default=200,
                            help="Sequence length")
    data_group.add_argument("--scale", type=str, default="tiny",
                            choices=list(SCALE_CONFIGS.keys()),
                            help="Embedding table scale")

    optim_group = parser.add_argument_group("Optimizer")
    optim_group.add_argument("--optimizer", type=str, default="shampoo",
                             choices=["shampoo", "adam"],
                             help="Optimizer type")
    optim_group.add_argument("--lr", type=float, default=2e-4)
    optim_group.add_argument("--weight-decay", type=float, default=0.01)
    optim_group.add_argument("--grad-clip-norm", type=float, default=1.0)
    optim_group.add_argument("--precondition-frequency", type=int, default=4500,
                             help="Shampoo precondition frequency")
    optim_group.add_argument("--start-preconditioning-step", type=int, default=4500,
                             help="Step to start preconditioning")
    optim_group.add_argument("--max-preconditioner-dim", type=int, default=8192,
                             help="Max preconditioner dimension")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--max-steps", type=int, default=25000)
    train_group.add_argument("--log-interval", type=int, default=20)
    train_group.add_argument("--nan-check-interval", type=int, default=1)
    train_group.add_argument("--stop-on-nan", action="store_true", default=True)
    train_group.add_argument("--no-stop-on-nan", dest="stop_on_nan",
                             action="store_false")

    ablation_group = parser.add_argument_group("Ablation Flags")
    ablation_group.add_argument("--sync-before-precondition", action="store_true",
                                help="Add torch.cuda.synchronize() before preconditioner steps")
    ablation_group.add_argument("--disable-ddp-config", action="store_true",
                                help="Use Shampoo WITHOUT DDPDistributedConfig")
    ablation_group.add_argument("--single-stream-pipeline", action="store_true",
                                help="Disable TorchRec pipeline (no side streams)")
    ablation_group.add_argument("--gpu-max-hw-queues", type=int, default=None,
                                help="Set GPU_MAX_HW_QUEUES env var")
    ablation_group.add_argument("--disable-compile", action="store_true",
                                help="Disable torch.compile")
    ablation_group.add_argument("--force-fallback", action="store_true",
                                help="Force use of fallback model (skip generative_recommenders)")

    stress_group = parser.add_argument_group("Race Condition Stress")
    stress_group.add_argument("--nccl-traffic", action="store_true",
                              help="Generate concurrent NCCL all_to_all on side stream")
    stress_group.add_argument("--nccl-payload-mb", type=float, default=16.0,
                              help="NCCL all_to_all payload size in MB")
    stress_group.add_argument("--alloc-stress", action="store_true",
                              help="CachingAllocator churn between backward and optimizer step")
    stress_group.add_argument("--alloc-stress-mb", type=float, default=8.0,
                              help="Size of each stress allocation in MB")
    stress_group.add_argument("--alloc-stress-count", type=int, default=24,
                              help="Number of alloc/free cycles per step")
    stress_group.add_argument("--ffn-mult", type=int, default=4,
                              help="FFN width multiplier (increase for more dense params)")
    stress_group.add_argument("--output-mlp-width", type=int, default=512,
                              help="Output MLP hidden width")

    profile_group = parser.add_argument_group("Profiling")
    profile_group.add_argument("--profile", action="store_true",
                               help="Enable PyTorch profiler")
    profile_group.add_argument("--profile-start-step", type=int, default=100)
    profile_group.add_argument("--profile-steps", type=int, default=5)
    profile_group.add_argument("--profile-dir", type=str, default="./traces")

    return parser.parse_args()


# =========================================================================
# Main
# =========================================================================


def main():
    args = parse_args()

    if args.gpu_max_hw_queues is not None:
        os.environ["GPU_MAX_HW_QUEUES"] = str(args.gpu_max_hw_queues)

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    nan_checker = NaNChecker()
    _CRASH_STATE.update(rank=rank, nan_checker=nan_checker)
    signal.signal(signal.SIGABRT, _crash_handler)

    if rank == 0:
        log.info("=" * 70)
        log.info("DLRMv3/HSTU SHAMPOO NaN REPRODUCER")
        log.info("=" * 70)
        log.info(f"World size: {world_size}")
        log.info(f"Model: d_model={args.d_model}, layers={args.num_layers}, "
                 f"heads={args.num_heads}, scale={args.scale}")
        log.info(f"Training: batch_size={args.batch_size}, seq_len={args.seq_len}, "
                 f"max_steps={args.max_steps}")
        log.info(f"Optimizer: {args.optimizer}")
        if args.optimizer == "shampoo":
            log.info(f"  precondition_frequency={args.precondition_frequency}")
            log.info(f"  start_preconditioning_step={args.start_preconditioning_step}")
            log.info(f"  max_preconditioner_dim={args.max_preconditioner_dim}")
        log.info(f"Ablation flags:")
        log.info(f"  sync_before_precondition={args.sync_before_precondition}")
        log.info(f"  disable_ddp_config={args.disable_ddp_config}")
        log.info(f"  single_stream_pipeline={args.single_stream_pipeline}")

        env_vars = [
            "GPU_MAX_HW_QUEUES", "ROC_AQL_QUEUE_SIZE",
            "HSA_FORCE_FINE_GRAIN_PCIE", "PYTORCH_CUDA_ALLOC_CONF",
            "NCCL_MAX_NCHANNELS",
        ]
        log.info("Environment:")
        for var in env_vars:
            val = os.environ.get(var, "(not set)")
            log.info(f"  {var}={val}")
        log.info("=" * 70)

    use_gr = not args.force_fallback and try_import_generative_recommenders() is not None
    if rank == 0:
        log.info(f"Using {'generative_recommenders DlrmHSTU' if use_gr else 'fallback HSTU model'}")

    if use_gr:
        model, config = create_model(args, device)
    else:
        model = create_fallback_model(args, device)
        config = None

    model = DDP(model, device_ids=[local_rank])

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        log.info(f"Model parameters: {param_count:.1f}M")

    optimizer = create_optimizer(model, args, disable_ddp_config=args.disable_ddp_config)
    if rank == 0:
        dense_params, emb_params = _split_params(model)
        dense_count = sum(p.numel() for p in dense_params) / 1e6
        emb_count = sum(p.numel() for p in emb_params) / 1e6
        log.info(f"Optimizer split: dense={dense_count:.1f}M params (Shampoo), "
                 f"embedding={emb_count:.1f}M params (AdaGrad)")
        if args.disable_ddp_config:
            log.info("Shampoo: DDPDistributedConfig DISABLED (ablation)")

    scale = SCALE_CONFIGS[args.scale]
    emb_size_limit = {
        "tiny": 1_000_000, "quarter": 5_000_000,
        "half": 10_000_000, "full": 50_000_000,
    }
    max_emb = emb_size_limit.get(args.scale, 1_000_000)
    effective_item_hash = min(scale["item_hash_size"], max_emb)
    effective_user_hash = min(scale["user_hash_size"], max_emb)

    batch_gen = DLRMv3BatchGenerator(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_candidates=args.num_candidates,
        item_hash_size=effective_item_hash,
        user_hash_size=min(effective_user_hash, 1_000_000),
        category_hash_size=scale["category_hash_size"],
    )

    use_pipeline = not args.single_stream_pipeline
    hstu_config = config if use_gr else None

    pipeline = SimplePipeline(
        device=device,
        batch_gen=batch_gen,
        config=hstu_config,
        use_pipeline=use_pipeline,
    )
    pipeline.fill(3 if use_pipeline else 1)

    nccl_traffic = None
    if args.nccl_traffic and world_size > 1:
        nccl_traffic = NCCLTrafficGenerator(
            device=device, world_size=world_size, rank=rank,
            payload_mb=args.nccl_payload_mb,
        )
        if rank == 0:
            log.info(f"NCCL traffic: all_to_all on side stream, {args.nccl_payload_mb}MB payload")

    alloc_stressor = None
    if args.alloc_stress:
        alloc_stressor = AllocationStressor(
            device=device,
            alloc_mb=args.alloc_stress_mb,
            num_allocs=args.alloc_stress_count,
        )
        if rank == 0:
            log.info(f"Alloc stress: {args.alloc_stress_count}x {args.alloc_stress_mb}MB per step")

    if rank == 0:
        log.info(f"Pipeline: {'3-stage' if use_pipeline else 'single-stream'}")
        log.info("Starting training...")

    global _train_start_time
    _train_start_time = time.time()

    if use_gr and config is not None:
        final_step = train_with_generative_recommenders(
            model, config, optimizer, pipeline, args, rank, nan_checker, device,
        )
    else:
        final_step = train_with_fallback(
            model, optimizer, pipeline, args, rank, nan_checker, device,
            nccl_traffic=nccl_traffic, alloc_stressor=alloc_stressor,
        )

    elapsed = time.time() - _train_start_time

    try:
        dist.barrier()
    except Exception:
        pass

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"Total steps: {final_step + 1}")
        log.info(f"Elapsed: {elapsed:.1f}s ({elapsed / 3600:.2f}h)")
        log.info(f"Model: {'generative_recommenders DlrmHSTU' if use_gr else 'fallback HSTU'}")
        log.info(f"Optimizer: {args.optimizer}")
        log.info(f"Scale: {args.scale}")
        log.info(f"Ablations: sync_precond={args.sync_before_precondition}, "
                 f"no_ddp={args.disable_ddp_config}, "
                 f"single_stream={args.single_stream_pipeline}")
        log.info(f"NaN summary: {nan_checker.summary()}")

        if nan_checker.total_nans > 0:
            log.info("")
            log.info("NaN REPRODUCED!")
            if args.sync_before_precondition:
                log.info("  sync_before_precondition was ON but NaN still occurred")
                log.info("  -> stream sync hypothesis may be INSUFFICIENT")
            else:
                log.info("  Try --sync-before-precondition to test the hypothesis")
        else:
            log.info("")
            log.info("No NaN detected in this run.")
            if not args.sync_before_precondition:
                log.info("  The race may need more steps or higher GPU_MAX_HW_QUEUES")
            else:
                log.info("  sync_before_precondition was ON -- hypothesis may be correct")

        log.info("=" * 70)

    sys.exit(1 if nan_checker.total_nans > 0 else 0)


if __name__ == "__main__":
    main()
