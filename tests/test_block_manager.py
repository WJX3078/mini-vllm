"""Unit tests for the block-level KV cache manager."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from minivllm.block_manager import BlockSpaceManager
from minivllm.sequence import SamplingParams, Sequence


def make_bm(num_blocks=16, block_size=4, caching=True):
    return BlockSpaceManager(num_blocks=num_blocks, block_size=block_size,
                             num_layers=2, num_kv_heads=2, head_dim=8,
                             dtype=torch.float32, device="cpu",
                             enable_prefix_caching=caching)


def make_seq(tokens):
    return Sequence(tokens, SamplingParams())


def test_basic_alloc_layout():
    bm = make_bm()
    s = make_seq(list(range(10)))              # 10 tokens, bs=4 -> 2 full + 1 tail
    cached = bm.allocate_sequence(s)
    assert cached == 0                          # nothing cached yet
    assert len(s.block_table) == 3
    assert len(set(s.block_table)) == 3         # distinct physical blocks
    assert bm.blocks[s.block_table[0]].ref_count == 1

    bm.free_sequence(s)
    assert bm.num_free_blocks == 16


def test_prefix_cache_hit_and_partial_hit():
    bm = make_bm()
    a = make_seq(list(range(10)))
    assert bm.allocate_sequence(a) == 0
    bm.register_filled_blocks(a, upto_tokens=10)   # blocks 0,1 become shareable

    b = make_seq(list(range(10)))                  # identical prompt
    cached = bm.allocate_sequence(b)
    assert cached == 2 * 4                          # 8 tokens served from cache
    assert a.block_table[0] == b.block_table[0]     # same physical block
    assert bm.blocks[b.block_table[0]].ref_count == 2

    c = make_seq([0, 1, 2, 3, 99, 98])              # shares only block 0
    cached = bm.allocate_sequence(c)
    assert cached == 4

    d = make_seq([7, 7, 7, 7, 7, 7])                # different first block
    assert bm.allocate_sequence(d) == 0


def test_prefix_cache_disabled():
    bm = make_bm(caching=False)
    a = make_seq(list(range(8)))
    bm.allocate_sequence(a)
    bm.register_filled_blocks(a, 8)
    b = make_seq(list(range(8)))
    assert bm.allocate_sequence(b) == 0             # no reuse without caching
    assert b.block_table[0] != a.block_table[0]


def test_copy_on_write_on_fork():
    bm = make_bm()
    parent = make_seq(list(range(10)))              # tail block (idx 2) partial
    bm.allocate_sequence(parent)
    child = make_seq(list(range(10)))
    bm.fork_sequence(parent, child)
    tail = parent.block_table[2]
    assert bm.blocks[tail].ref_count == 2

    # parent appends one token -> must COW the shared partial tail
    parent.output_token_ids.append(100)
    parent.num_computed_tokens = 10                 # prompt KV already computed
    assert bm.prepare_slots(parent)
    assert parent.block_table[2] != tail
    assert bm.blocks[parent.block_table[2]].ref_count == 1
    assert bm.blocks[tail].ref_count == 1           # child still holds it
    # copied content must equal the original
    assert torch.equal(bm.pool.data[parent.block_table[2]],
                       bm.pool.data[tail])

    # child appends into its (now private) tail in place -- no COW
    child.output_token_ids.append(200)
    child.num_computed_tokens = 10
    assert bm.prepare_slots(child)
    assert child.block_table[2] == tail


def test_fork_full_tail_needs_no_cow():
    bm = make_bm()
    parent = make_seq(list(range(8)))               # exactly 2 full blocks
    bm.allocate_sequence(parent)
    child = make_seq(list(range(8)))
    bm.fork_sequence(parent, child)
    parent.output_token_ids.append(1)
    parent.num_computed_tokens = 8
    assert bm.prepare_slots(parent)                 # new block, not COW
    assert parent.block_table[2] not in parent.block_table[:2]
    assert len(set(parent.block_table)) == 3
    assert parent.block_table[:2] == child.block_table[:2]


def test_lru_eviction_of_cached_blocks():
    bm = make_bm(num_blocks=4)
    a = make_seq([1] * 8)                           # blocks 0,1
    bm.allocate_sequence(a)
    bm.register_filled_blocks(a, 8)
    bm.free_sequence(a)                             # cached, ref 0

    b = make_seq([2] * 12)                          # needs 3 blocks -> must evict
    bm.allocate_sequence(b)
    assert 0 in b.block_table                       # LRU victim (block 0) reused
    old_key = bm._block_key(None, (1, 1, 1, 1))
    assert old_key not in bm.cached_blocks
    assert bm.blocks[0].ref_count == 1


def test_eviction_skips_blocks_in_use():
    bm = make_bm(num_blocks=4)
    live = make_seq([5] * 8)
    bm.allocate_sequence(live)
    bm.register_filled_blocks(live, 8)
    bm.free_sequence(live)
    # re-allocate same prompt: cache hit, ref back to 1
    again = make_seq([5] * 8)
    assert bm.allocate_sequence(again) == 8
    # a bigger seq now needs 3 blocks: the only cached ones are in use
    # (ref>1) and the free list is empty -> allocation fails cleanly
    other = make_seq([6] * 12)
    assert bm.allocate_sequence(other) is None      # rolled back, no corruption
    assert bm.blocks[again.block_table[0]].ref_count == 1


def test_preempt_then_readmit_restores_prefix():
    bm = make_bm()
    s = make_seq(list(range(12)))
    bm.allocate_sequence(s)
    bm.register_filled_blocks(s, 12)
    bm.preempt_sequence(s)
    assert s.block_table == []
    assert s.num_computed_tokens == 0
    cached = bm.allocate_sequence(s)
    assert cached == 3 * 4                          # all full blocks restored


def test_register_uses_chained_keys():
    bm = make_bm()
    a = make_seq(list(range(8)) + [50, 51])
    bm.allocate_sequence(a)
    bm.register_filled_blocks(a, 8)
    k0 = bm._block_key(None, (0, 1, 2, 3))
    k1 = bm._block_key(k0, (4, 5, 6, 7))
    assert bm.blocks[a.block_table[0]].key == k0
    assert bm.blocks[a.block_table[1]].key == k1
    # a prompt sharing the chain but ending differently still hits block 0 only
    b = make_seq(list(range(4)) + [9, 9, 9])
    assert bm.allocate_sequence(b) == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
