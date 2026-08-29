"""Paged attention, written in pure PyTorch.

The KV pool stores K/V in fixed-size physical blocks. A sequence's K/V is
scattered across those blocks; attention:

  1. writes the new tokens' K/V into their pool slots (paged write),
  2. gathers the full context K/V back through the block table
     (block i -> physical block), producing a contiguous [ctx_len, ...] view,
  3. runs standard scaled-dot-product attention with a causal mask over the
     query window, using GQA expansion.

Production vLLM swaps steps (1)(2)(3) for a single fused CUDA kernel that
walks the block table; the memory layout and semantics here are the same.
"""
from typing import List

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------- RoPE
def rope_inv_freq(head_dim: int, theta: float, device) -> torch.Tensor:
    """HF convention: inv_freq = 1 / theta ** (arange(0, dim, 2) / dim)."""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                         device=device) / head_dim))


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor,
               inv_freq: torch.Tensor) -> tuple:
    """q: [T, n_heads, D], k: [T, n_kv_heads, D], positions: [T] (absolute)."""
    freqs = positions.float()[:, None] * inv_freq[None, :]        # [T, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)                       # [T, D]
    cos = emb.cos().to(q.dtype)[:, None, :]                       # [T, 1, D]
    sin = emb.sin().to(q.dtype)[:, None, :]

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ------------------------------------------------------------- paged KV I/O
def write_kv_to_pool(k_view: torch.Tensor, v_view: torch.Tensor,
                     k: torch.Tensor, v: torch.Tensor,
                     positions: torch.Tensor, block_table: torch.Tensor,
                     block_size: int):
    """Write new tokens' K/V into their pool slots.

    k/v: [n, kv_heads, head_dim]; positions: [n] absolute token positions;
    block_table: [nb] physical block ids of the sequence.
    """
    phys = block_table[positions // block_size]     # [n]
    slots = positions % block_size                  # [n]
    # k_view: [num_blocks, kv_heads, block_size, head_dim]
    k_view[phys, :, slots, :] = k
    v_view[phys, :, slots, :] = v


def gather_kv(k_view: torch.Tensor, v_view: torch.Tensor,
              block_table: torch.Tensor, ctx_len: int,
              block_size: int) -> tuple:
    """Gather a sequence's context K/V from physical blocks into
    [kv_heads, ctx_len, head_dim] contiguous tensors."""
    nb = (ctx_len + block_size - 1) // block_size
    idx = block_table[:nb]
    k = k_view[idx]                                  # [nb, kvh, bs, D]
    v = v_view[idx]
    kvh = k.shape[1]
    k = k.permute(1, 0, 2, 3).reshape(kvh, nb * block_size, -1)[:, :ctx_len]
    v = v.permute(1, 0, 2, 3).reshape(kvh, nb * block_size, -1)[:, :ctx_len]
    return k, v


# ----------------------------------------------------------------- attention
def _probe_sdpa_gqa() -> bool:
    """torch >= 2.5 supports GQA inside SDPA without materializing repeats."""
    try:
        q = torch.zeros(1, 2, 1, 2)
        k = torch.zeros(1, 1, 1, 2)
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
        return True
    except Exception:
        return False


_SDPA_ENABLE_GQA = _probe_sdpa_gqa()


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask, scale: float,
          num_heads: int, num_kv_heads: int) -> torch.Tensor:
    if num_heads != num_kv_heads and _SDPA_ENABLE_GQA:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                              scale=scale, enable_gqa=True)
    if num_heads != num_kv_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=-3)
        v = v.repeat_interleave(rep, dim=-3)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def paged_attention(query: torch.Tensor, k_ctx: torch.Tensor, v_ctx: torch.Tensor,
                    q_start: int, num_heads: int) -> torch.Tensor:
    """One sequence's attention.

    query:  [q_len, n_heads, head_dim]
    k_ctx / v_ctx: [kv_heads, ctx_len, head_dim] (gathered, includes the new
            tokens themselves)
    q_start: absolute position of the first query token (= ctx_len - q_len)

    Returns [q_len, n_heads, head_dim].
    """
    q_len, _, d = query.shape
    ctx_len = k_ctx.shape[1]
    scale = 1.0 / d ** 0.5

    q = query.permute(1, 0, 2)[None]                 # [1, H, q_len, D]
    k = k_ctx[None]                                  # [1, kvh, ctx, D]
    v = v_ctx[None]

    if q_len > 1:                                    # causal within query window
        i = torch.arange(q_len, device=query.device)[:, None]    # [q_len, 1]
        j = torch.arange(ctx_len, device=query.device)[None, :]  # [1, ctx]
        allow = j <= (q_start + i)                        # [q_len, ctx]
        mask = torch.where(allow, 0.0, float("-inf")).to(query.dtype)
        out = _sdpa(q, k, v, mask[None, None], scale, num_heads, k.shape[1])
    else:
        out = _sdpa(q, k, v, None, scale, num_heads, k.shape[1])

    return out[0].permute(1, 0, 2)                   # [q_len, H, D]


class SeqInput:
    """Per-sequence metadata consumed by the model forward.

    q_start: absolute token position of the sequence's first new token
             (used for RoPE and the causal mask).
    t0:      index of that token inside the flat batch tensor
             (used to slice the batched Q/K/V stream).
    """

    __slots__ = ("q_start", "q_len", "block_table", "ctx_len", "t0")

    def __init__(self, q_start: int, q_len: int, block_table: torch.Tensor,
                 t0: int = 0):
        self.q_start = q_start
        self.q_len = q_len
        self.block_table = block_table
        self.ctx_len = q_start + q_len
        self.t0 = t0
