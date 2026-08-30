"""Triton PagedAttention decode kernel (q_len=1) + torch reference.

    Q [B, H, D]
        |
    block_table[B, max_blocks]      (logical block -> physical block)
        |
    Triton kernel: program (b, h) walks the sequence's physical KV blocks
    directly in the paged pool -- NO contiguous K/V is ever materialized
        |
    block-wise online softmax (running max m, exp-sum l, weighted-V acc)
        |
    out [B, H, D]

Scope (deliberately narrow, mirroring production first steps):
  * decode only (q_len == 1) -- no causal mask needed;
  * MHA and GQA (kv_head = query_head // group_size, group_size constexpr,
    requires num_q_heads % num_kv_heads == 0);
  * fp16 / bf16 storage with fp32 accumulation.

`paged_attention_decode_torch` is the PyTorch reference: batched gather +
SDPA with the same math (and the engine's fallback when Triton is
unavailable).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:                                    # Triton is optional (Linux/Windows pkgs)
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_attention_decode_kernel(
        q_ptr,                          # [B, H, D]
        k_ptr,                          # [nblk, kvh, bs, D]
        v_ptr,                          # [nblk, kvh, bs, D]
        out_ptr,                        # [B, H, D]
        block_tables_ptr,               # [B, max_nb] int32/64
        context_lens_ptr,               # [B]
        scale,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_od,
        stride_tb, stride_tn,
        GROUP_SIZE: tl.constexpr,       # H // kvh
        BLOCK_SIZE: tl.constexpr,       # tokens per physical KV block
        HEAD_DIM: tl.constexpr,
    ):
        seq = tl.program_id(0)
        head = tl.program_id(1)
        kv_head = head // GROUP_SIZE

        d_offs = tl.arange(0, HEAD_DIM)
        s_offs = tl.arange(0, BLOCK_SIZE)

        q = tl.load(q_ptr + seq * stride_qb + head * stride_qh
                    + d_offs * stride_qd).to(tl.float32)

        ctx_len = tl.load(context_lens_ptr + seq)
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        nb = tl.cdiv(ctx_len, BLOCK_SIZE)
        for blk in range(0, nb):
            phys = tl.load(block_tables_ptr + seq * stride_tb
                           + blk * stride_tn)
            pos = blk * BLOCK_SIZE + s_offs             # token positions
            mask = pos < ctx_len
            k = tl.load(k_ptr + phys * stride_kb + kv_head * stride_kh
                        + s_offs * stride_ks + d_offs[:, None] * stride_kd,
                        mask=mask[None, :], other=0.0).to(tl.float32)
            # scores[HEAD_DIM, BLOCK_SIZE] = q^T K
            scores = tl.sum(q[:, None] * k, axis=0) * scale
            scores = tl.where(mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_i = alpha * l_i + tl.sum(p, axis=0)

            v = tl.load(v_ptr + phys * stride_vb + kv_head * stride_vh
                        + s_offs * stride_vs + d_offs[:, None] * stride_vd,
                        mask=mask[None, :], other=0.0).to(tl.float32)
            acc = alpha * acc + tl.sum(p[None, :] * v, axis=1)
            m_i = m_new

        out = acc / l_i
        tl.store(out_ptr + seq * stride_ob + head * stride_oh
                 + d_offs * stride_od,
                 out.to(out_ptr.dtype.element_ty))

    _TRITON_AVAILABLE = True
except ImportError:                         # pragma: no cover - CI has no triton
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    return _TRITON_AVAILABLE


def paged_attention_decode_triton(
    q: torch.Tensor,                        # [B, H, D]
    k_cache: torch.Tensor,                  # [nblk, kvh, bs, D]
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,             # [B, max_nb]
    context_lens: torch.Tensor,             # [B]
    scale: float,
) -> torch.Tensor:
    """Decode attention straight from the paged KV pool. Returns
    [B, H, D]. Requires CUDA + Triton; callers fall back to
    `paged_attention_decode_torch` otherwise."""
    B, H, D = q.shape
    _, kvh, bs, _ = k_cache.shape
    assert H % kvh == 0, "query heads must be divisible by kv heads"
    assert q.is_contiguous()
    block_tables = block_tables.to(device="cuda", dtype=torch.int32)         .contiguous()
    context_lens = context_lens.to(device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    _paged_attention_decode_kernel[(B, H)](
        q, k_cache, v_cache, out,
        block_tables, context_lens,
        scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        GROUP_SIZE=H // kvh, BLOCK_SIZE=bs, HEAD_DIM=D,
    )
    return out


def paged_attention_decode_torch(
    q: torch.Tensor,                        # [B, H, D]
    k_cache: torch.Tensor,                  # [nblk, kvh, bs, D]
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,             # [B, max_nb]
    context_lens: torch.Tensor,             # [B]
    scale: float,
) -> torch.Tensor:
    """Reference: gather each sequence's K/V through its block table and run
    SDPA (no mask needed for q_len == 1). Same math as the Triton kernel."""
    B, H, D = q.shape
    kvh, bs = k_cache.shape[1], k_cache.shape[2]
    ctx = int(context_lens.max())
    nb = (ctx + bs - 1) // bs
    tables = block_tables[:, :nb]                        # [B, nb]
    k = k_cache[tables]                                  # [B, nb, kvh, bs, D]
    v = v_cache[tables]
    kk = k.permute(0, 2, 1, 3, 4).reshape(B, kvh, nb * bs, D)[:, :, :ctx]
    vv = v.permute(0, 2, 1, 3, 4).reshape(B, kvh, nb * bs, D)[:, :, :ctx]
    mask = (torch.arange(ctx, device=q.device)[None, None, None, :]
            < context_lens[:, None, None, None])         # [B,1,1,ctx]
    if kvh != H and hasattr(F, "scaled_dot_product_attention"):
        try:
            out = F.scaled_dot_product_attention(
                q[:, :, None], kk, vv, attn_mask=mask, scale=scale,
                enable_gqa=True)
            return out[:, :, 0, :]
        except TypeError:
            pass
    rep = H // kvh
    kk = kk.repeat_interleave(rep, dim=1)
    vv = vv.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q[:, :, None], kk, vv,
                                         attn_mask=mask, scale=scale)
    return out[:, :, 0, :]
