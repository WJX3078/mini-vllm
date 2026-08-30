"""Triton PagedAttention decode kernel: numerical correctness matrix.

The kernel must match the PyTorch reference (gather + SDPA) across batch /
context length / block size / head layout / head_dim / dtype combinations,
including the ragged cases (context just above/below a block boundary and
context == 1). GPU + Triton only -- skipped elsewhere; CPU CI never needs
Triton.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from minivllm.kernels.paged_attention import (
    paged_attention_decode_torch,
    paged_attention_decode_triton,
    triton_available,
)

pytestmark = [pytest.mark.gpu, pytest.mark.triton]

if not (torch.cuda.is_available() and triton_available()):
    pytest.skip("requires CUDA + Triton", allow_module_level=True)

DEV = "cuda"


def _make_inputs(batch, ctx, block_size, num_q_heads, num_kv_heads, head_dim,
                 dtype):
    torch.manual_seed(0)
    max_nb = (ctx + block_size - 1) // block_size
    # random distinct physical blocks per (seq, logical block)
    nblk = batch * max_nb + 4
    q = torch.randn(batch, num_q_heads, head_dim, device=DEV, dtype=dtype)
    k_cache = torch.randn(nblk, num_kv_heads, block_size, head_dim,
                          device=DEV, dtype=dtype)
    v_cache = torch.randn(nblk, num_kv_heads, block_size, head_dim,
                          device=DEV, dtype=dtype)
    perm = torch.randperm(nblk)
    tables = torch.stack([perm[i * max_nb:(i + 1) * max_nb]
                          for i in range(batch)]).to(DEV)
    lens = torch.randint(max(1, ctx - block_size // 2), ctx + 1,
                         (batch,), device=DEV)
    lens = lens.clamp(min=1, max=ctx)
    return q, k_cache, v_cache, tables, lens


HEAD_LAYOUTS = [(4, 4), (8, 2), (16, 4)]          # (num_q_heads, num_kv_heads)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("heads", HEAD_LAYOUTS)
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("ctx", [1, 15, 16, 17, 128, 511, 1024])
@pytest.mark.parametrize("batch", [1, 4, 16])
def test_triton_matches_torch_reference(dtype, head_dim, heads, block_size,
                                        ctx, batch):
    num_q, num_kv = heads
    q, k, v, tables, lens = _make_inputs(batch, ctx, block_size, num_q,
                                         num_kv, head_dim, dtype)
    scale = head_dim ** -0.5
    ref = paged_attention_decode_torch(q, k, v, tables, lens, scale)
    got = paged_attention_decode_triton(q, k, v, tables, lens, scale)
    assert got.shape == (batch, num_q, head_dim)
    diff = (got.float() - ref.float()).abs()
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    assert diff.max().item() < tol, \
        f"max diff {diff.max().item():.4f} at ctx={ctx} bs={block_size}" \
        f" heads={heads} dim={head_dim} {dtype}"


def test_gqa_head_mapping_exact():
    """GQA 4Q/2KV: query heads 0,1 must read KV head 0 and heads 2,3 must
    read KV head 1 (kv = q // group_size). Every slot of a KV head carries
    the same key, so the softmax over that head's slots is uniform and the
    output equals that KV head's constant value exactly."""
    bs, D = 4, 8
    q = torch.zeros(1, 4, D, device=DEV, dtype=torch.float16)
    k = torch.zeros(1, 2, bs, D, device=DEV, dtype=torch.float16)   # 1 block
    v = torch.zeros(1, 2, bs, D, device=DEV, dtype=torch.float16)
    for h in range(4):
        q[0, h, h] = 1.0
    k[0, 0] = 1.0                            # KV head 0: all keys = e0
    v[0, 0] = 10.0                           # KV head 0: value 10 everywhere
    k[0, 1] = 1.0                            # KV head 1: all keys = e0
    v[0, 1] = 20.0                           # KV head 1: value 20 everywhere
    tables = torch.tensor([[0]], device=DEV)
    lens = torch.tensor([4], device=DEV)
    scale = 1.0
    out = paged_attention_decode_triton(q, k, v, tables, lens, scale=scale)
    ref = paged_attention_decode_torch(q, k, v, tables, lens, scale=scale)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2)
    for h in (0, 1):                         # -> KV head 0
        assert abs(out[0, h, 0].item() - 10.0) < 1e-2
    for h in (2, 3):                         # -> KV head 1
        assert abs(out[0, h, 0].item() - 20.0) < 1e-2


def test_block_table_reordering_is_honored():
    """Logical block i maps to ANY physical block: shuffling the table
    must move the data accordingly."""
    bs, D = 8, 32
    torch.manual_seed(1)
    nblk = 4
    k = torch.randn(nblk, 1, bs, D, device=DEV, dtype=torch.float16)
    v = torch.randn(nblk, 1, bs, D, device=DEV, dtype=torch.float16)
    q = torch.randn(2, 1, D, device=DEV, dtype=torch.float16)
    identity = torch.arange(4, device=DEV).repeat(2, 1)
    swapped = torch.tensor([[2, 3, 0, 1], [2, 3, 0, 1]], device=DEV)
    lens = torch.tensor([16, 16], device=DEV)
    ref_id = paged_attention_decode_torch(q, k, v, identity, lens, D ** -0.5)
    got_sw = paged_attention_decode_triton(q, k, v, swapped, lens, D ** -0.5)
    ref_sw = paged_attention_decode_torch(q, k, v, swapped, lens, D ** -0.5)
    assert torch.allclose(got_sw.float(), ref_sw.float(), atol=1e-2)
    assert not torch.allclose(ref_id.float(), ref_sw.float(), atol=1e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
