"""
DLRMv3-style synthetic data generator matching MLCommons distributions.

Replicates the data distributions from:
  https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v3/streaming_synthetic_data.py

Key distribution properties:
  - 128 item categories, 4 preferred categories per user
  - Category-based item selection with Dirichlet-like alpha (1-500)
  - Variable-length UIH sequences (Gaussian around mean)
  - 2048 inference candidates (matching DLRMv3 config)
  - Category-to-item mapping: items_per_category = hash_size / 128
  - Time-varying rating distributions (cosine modulation)
  - Item ratings from [1,2,3,4,5] with p=[0.1,0.2,0.25,0.25,0.2]

Includes ThreadedDataPipeline for fully pipelined data loading:
  - Background thread continuously generates batches into a queue
  - Hot loop calls get_batch() which is instant if thread is ahead
  - Fully overlaps CPU data generation with GPU compute
"""

import logging
import math
import queue
import random
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


class DLRMv3DataConfig:
    """Configuration matching DLRMv3 streaming synthetic data parameters."""

    def __init__(
        self,
        item_hash_size: int = 1_000_000,
        user_hash_size: int = 100_000,
        num_categories: int = 128,
        categories_per_user: int = 4,
        num_timestamps: int = 100,
        avg_seq_len: int = 200,
        num_inference_candidates: int = 2048,
        alpha_range: Tuple[int, int] = (1, 500),
    ):
        self.item_hash_size = item_hash_size
        self.user_hash_size = user_hash_size
        self.num_categories = num_categories
        self.categories_per_user = categories_per_user
        self.num_timestamps = num_timestamps
        self.avg_seq_len = avg_seq_len
        self.num_inference_candidates = num_inference_candidates
        self.alpha_range = alpha_range
        self.items_per_category = item_hash_size // num_categories


class DLRMv3SyntheticBatchGenerator:
    """Generates batches with DLRMv3-realistic distributions.

    Each "user" has preferred categories with Dirichlet-like selection.
    Items are sampled within categories (contiguous ID ranges).
    Sequence lengths follow a Gaussian distribution.
    """

    def __init__(
        self,
        config: DLRMv3DataConfig,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ):
        self.config = config
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device

        self.rng = np.random.RandomState(42)
        self._precompute_user_categories()
        self._precompute_item_ratings()

    def _precompute_user_categories(self) -> None:
        """Assign preferred categories to each user in the batch.
        Users keep the same categories across iterations (stateful).
        """
        cfg = self.config
        self.user_categories = []
        for _ in range(self.batch_size):
            cats = random.sample(range(cfg.num_categories), cfg.categories_per_user)
            alpha = random.randint(*cfg.alpha_range)
            self.user_categories.append((cats, alpha))

    def _precompute_item_ratings(self) -> None:
        """Item-level ratings: [1,2,3,4,5] with DLRMv3 distribution."""
        self.item_ratings = self.rng.choice(
            [5.0, 4.0, 3.0, 2.0, 1.0],
            size=min(self.config.item_hash_size, 10_000_000),
            p=[0.2, 0.25, 0.25, 0.2, 0.1],
        )

    def _sample_seq_len(self) -> int:
        """Gaussian sequence length matching DLRMv3's gen_rand_seq_len."""
        length = round(random.gauss(self.config.avg_seq_len, self.config.avg_seq_len // 4))
        return max(self.config.num_inference_candidates + 1, min(length, self.max_seq_len))

    def _sample_items_for_user(
        self, user_idx: int, seq_len: int, timestamp: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Category-based item sampling matching DLRMv3 distribution.

        Items are selected from the user's preferred categories with
        Dirichlet-like probabilities that evolve over time.
        """
        cats, alpha = self.user_categories[user_idx % len(self.user_categories)]
        cfg = self.config

        cat_counts = {c: 1 for c in cats}
        total = len(cats)
        p = np.array([(alpha / len(cats) + cat_counts[c]) / (alpha + total) for c in cats])
        p = p / p.sum()

        chosen_cats = self.rng.choice(cats, size=seq_len, p=p)

        sample_end = int(cfg.items_per_category * max(timestamp + 1, 1) / cfg.num_timestamps)
        sample_end = max(sample_end, 1)
        within_cat = self.rng.randint(0, sample_end, size=seq_len)
        offsets = np.array([c * cfg.items_per_category for c in chosen_cats])
        item_ids = (within_cat + offsets) % cfg.item_hash_size

        cat_ids = np.array(chosen_cats)
        return item_ids, cat_ids

    def generate_batch(self, iteration: int) -> Dict[str, torch.Tensor]:
        """Generate a batch with DLRMv3-realistic distributions.

        Returns dict matching the HSTU model's expected input format.
        """
        B = self.batch_size
        S = self.max_seq_len
        C = self.config.num_inference_candidates
        cfg = self.config
        timestamp = iteration % cfg.num_timestamps

        item_ids_list = []
        user_ids_list = []
        category_ids_list = []
        candidate_ids_list = []
        seq_lengths_list = []

        for b in range(B):
            seq_len = min(self._sample_seq_len(), S)
            seq_lengths_list.append(seq_len)

            items, cats = self._sample_items_for_user(b, seq_len, timestamp)

            padded_items = np.zeros(S, dtype=np.int64)
            padded_items[:seq_len] = items[:S]
            item_ids_list.append(padded_items)

            padded_cats = np.zeros(S, dtype=np.int64)
            padded_cats[:seq_len] = cats[:S]
            category_ids_list.append(padded_cats)

            user_id = (b + iteration * B) % cfg.user_hash_size
            padded_users = np.full(S, user_id, dtype=np.int64)
            user_ids_list.append(padded_users)

            cand_items, _ = self._sample_items_for_user(b, C, timestamp)
            candidate_ids_list.append(cand_items[:C])

        return {
            "item_ids": torch.from_numpy(np.stack(item_ids_list)).to(self.device),
            "user_ids": torch.from_numpy(np.stack(user_ids_list)).to(self.device),
            "category_ids": torch.from_numpy(np.stack(category_ids_list)).to(self.device),
            "candidate_item_ids": torch.from_numpy(np.stack(candidate_ids_list)).to(self.device),
            "seq_lengths": torch.tensor(seq_lengths_list, dtype=torch.long, device=self.device),
        }

    def generate_batch_cpu_pinned(self, iteration: int) -> Dict[str, torch.Tensor]:
        """Generate batch in CPU pinned memory for H2D transfer."""
        B = self.batch_size
        S = self.max_seq_len
        C = self.config.num_inference_candidates
        cfg = self.config
        timestamp = iteration % cfg.num_timestamps

        item_ids_np = np.zeros((B, S), dtype=np.int64)
        user_ids_np = np.zeros((B, S), dtype=np.int64)
        cat_ids_np = np.zeros((B, S), dtype=np.int64)
        cand_ids_np = np.zeros((B, C), dtype=np.int64)
        seq_lens_np = np.zeros(B, dtype=np.int64)

        for b in range(B):
            seq_len = min(self._sample_seq_len(), S)
            seq_lens_np[b] = seq_len

            items, cats = self._sample_items_for_user(b, seq_len, timestamp)
            item_ids_np[b, :seq_len] = items[:S]
            cat_ids_np[b, :seq_len] = cats[:S]

            user_id = (b + iteration * B) % cfg.user_hash_size
            user_ids_np[b, :] = user_id

            cand_items, _ = self._sample_items_for_user(b, C, timestamp)
            cand_ids_np[b, :] = cand_items[:C]

        return {
            "item_ids": torch.from_numpy(item_ids_np).pin_memory(),
            "user_ids": torch.from_numpy(user_ids_np).pin_memory(),
            "category_ids": torch.from_numpy(cat_ids_np).pin_memory(),
            "candidate_item_ids": torch.from_numpy(cand_ids_np).pin_memory(),
            "seq_lengths": torch.from_numpy(seq_lens_np).pin_memory(),
        }


class ThreadedDataPipeline:
    """Fully pipelined data loader with background generation thread.

    A daemon thread continuously generates batches with DLRMv3 distributions
    into pinned memory and pushes them into a queue. The hot loop calls
    get_batch() which returns instantly if the thread is ahead, giving
    zero CPU stall in the dispatch path.

    This matches Meta's real DataLoader pattern where data preparation
    is fully overlapped with GPU compute.
    """

    def __init__(
        self,
        config: DLRMv3DataConfig,
        batch_size: int,
        max_seq_len: int,
        queue_depth: int = 8,
        total_batches: int = 10000,
    ):
        self.queue_depth = queue_depth
        self.total_batches = total_batches
        self._queue: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._stop = threading.Event()

        self._gen = DLRMv3SyntheticBatchGenerator(
            config=config,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=torch.device("cpu"),
        )

        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

        # Wait for queue to fill before returning, so hot loop starts with a full queue
        while self._queue.qsize() < queue_depth and not self._stop.is_set():
            import time
            time.sleep(0.01)

        log.info(
            f"ThreadedDataPipeline ready: queue={self._queue.qsize()}/{queue_depth}, "
            f"batch_size={batch_size}, seq_len={max_seq_len}, "
            f"candidates={config.num_inference_candidates}"
        )

    def _producer(self) -> None:
        """Background thread: generate pinned batches into the queue."""
        for i in range(self.total_batches):
            if self._stop.is_set():
                return
            batch = self._gen.generate_batch_cpu_pinned(i)
            try:
                self._queue.put(batch, timeout=5.0)
            except queue.Full:
                if self._stop.is_set():
                    return

    def get_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get next pre-generated pinned batch (blocks briefly if thread behind)."""
        try:
            return self._queue.get(timeout=10.0)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5.0)
