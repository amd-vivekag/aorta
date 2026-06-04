"""Single transformer block repeated N times — comm/compute stage probe.

The war-room execution model is a series of repeating
``comm_kernel_i -> compute_kernel_i`` stages. Each block here IS one such
stage: under FSDP2 ``fully_shard`` the block's parameters are all-gathered
on entry and grads reduce-scattered on exit, so the activation entering
and leaving each block is the tensor that crosses GPU boundaries. The
determinism workload hooks the per-block boundary to checksum exactly
those tensors at every repetition.

Optional MoE: when ``num_experts > 1`` the FFN is a top-1 token-router
over ``num_experts`` GLU experts. The router selection is deterministic
under a fixed seed and exercises the same all-to-all-shaped traffic an
MoE workload would generate (without depending on a real MoE framework).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BlockConfig:
    """Knobs for one repeated transformer block + how many times to repeat it.

    ``num_layers`` is the repetition count — the only knob that scales the
    amount of cross-GPU traffic when wrapped in FSDP2. Everything else
    sizes the per-block compute.
    """

    vocab_size: int = 32_000
    hidden_size: int = 2048
    ffn_size: int = 5632
    num_heads: int = 16
    num_layers: int = 24
    seq_len: int = 512
    num_experts: int = 1  # 1 == dense FFN; >1 == top-1 MoE router.
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.num_experts < 1:
            raise ValueError("num_experts must be >= 1")


class _GluFFN(nn.Module):
    def __init__(self, hidden: int, ffn: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, ffn, bias=False)
        self.up = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class _Top1MoE(nn.Module):
    """Top-1 token routing across ``num_experts`` GLU experts.

    Token dispatch uses ``argmax`` + indexed gather — deterministic under a
    fixed seed. No load balancing loss; this is a structural stand-in for
    MoE traffic patterns, not an MoE quality model.
    """

    def __init__(self, hidden: int, ffn: int, num_experts: int) -> None:
        super().__init__()
        self.router = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList([_GluFFN(hidden, ffn) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        flat = x.reshape(b * t, h)
        choice = self.router(flat).argmax(dim=-1)
        out = torch.zeros_like(flat)
        for idx, expert in enumerate(self.experts):
            mask = choice == idx
            if mask.any():
                out[mask] = expert(flat[mask])
        return out.reshape(b, t, h)


class RepeatedTransformerBlock(nn.Module):
    """One pre-norm transformer block: LayerNorm + MHA + FFN (dense or MoE).

    Manual scaled-dot-product attention so the kernel sequence is the same
    on every call — ``torch.nn.functional.scaled_dot_product_attention``
    picks a backend whose determinism varies by build.
    """

    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.attn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.attn_out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.ffn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)
        if cfg.num_experts == 1:
            self.ffn: nn.Module = _GluFFN(cfg.hidden_size, cfg.ffn_size)
        else:
            self.ffn = _Top1MoE(cfg.hidden_size, cfg.ffn_size, cfg.num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        head_dim = h // self.cfg.num_heads
        qkv = self.qkv(self.attn_norm(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.cfg.num_heads, head_dim).transpose(1, 2)
        k = k.view(b, t, self.cfg.num_heads, head_dim).transpose(1, 2)
        v = v.view(b, t, self.cfg.num_heads, head_dim).transpose(1, 2)
        scale = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device, dtype=scores.dtype), diagonal=1)
        attn = torch.softmax((scores + mask).float(), dim=-1).to(x.dtype)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, h)
        x = x + self.attn_out(ctx)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RepeatedBlockModel(nn.Module):
    """``cfg.num_layers`` repetitions of one ``RepeatedTransformerBlock``.

    Each repetition is the war-room "comm+compute" stage. Hooks attached
    by the workload run before and after each block's ``forward`` to
    checksum the boundary tensors.
    """

    def __init__(self, cfg: BlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([RepeatedTransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return torch.nn.functional.linear(x, self.embed.weight)


__all__ = ["BlockConfig", "RepeatedBlockModel", "RepeatedTransformerBlock"]
