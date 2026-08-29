"""Prefix-cache-aware scheduler budget (P0 fix).

The scheduler must charge only the UNCACHED tokens of a waiting prompt
against max_num_batched_tokens -- a 1000-token prompt with 900 cached hits
should consume 100 budget, not 1000, so the freed budget admits more work.
The cache probe itself must be read-only (no refcount / LRU side effects),
allocation happens strictly after the budget check, and failed allocation
must not leak blocks.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from minivllm.block_manager import BlockSpaceManager
from minivllm.scheduler import Scheduler
from minivllm.sequence import SamplingParams, Sequence


def make_sched(num_blocks=16, block_size=8, budget=16, max_num_seqs=8):
    bm = BlockSpaceManager(num_blocks=num_blocks, block_size=block_size,
                           num_layers=1, num_kv_heads=1, head_dim=4,
                           dtype=torch.float32, device="cpu",
                           enable_prefix_caching=True)
    return Scheduler(bm, max_num_seqs, budget), bm


def _prime_cache(bm, tokens, block_size):
    """Compute + register + release a prompt so its full blocks are cached."""
    s = Sequence(list(tokens), SamplingParams())
    bm.allocate_sequence(s)
    bm.register_filled_blocks(s, len(tokens))
    bm.free_sequence(s)
    return s


def test_budget_charged_only_for_uncached_tokens():
    """prompt=32, block_size=8, first 24 tokens cached, budget=16:
    admission must be decided by the 8 uncached tokens, not 32."""
    sched, bm = make_sched(budget=16)
    prompt = list(range(32))
    _prime_cache(bm, prompt[:24], block_size=8)    # only the first 3 blocks

    seq = Sequence(list(prompt), SamplingParams())
    sched.add(seq)
    out = sched.schedule()
    assert seq in out.scheduled                    # admitted on 8 <= 16 budget
    assert out.num_new_tokens == 8                 # charged 8, not 32
    assert seq.num_computed_tokens == 24           # KV of the cached prefix
    assert out.spans == [(24, 32)]


def test_multiple_cached_requests_share_one_iteration():
    """Two prompts sharing a 24-token cached base, each with 8 uncached
    suffix tokens: 8+8 uncached tokens fit in a budget of 16 TOGETHER."""
    sched, bm = make_sched(budget=16)
    base = list(range(24))
    _prime_cache(bm, base, block_size=8)           # only the shared base cached
    prompts = [base + [50, 51, 52, 53, 54, 55, 56, 57],
               base + [60, 61, 62, 63, 64, 65, 66, 67]]

    seqs = [Sequence(list(p), SamplingParams()) for p in prompts]
    for s in seqs:
        sched.add(s)
    out = sched.schedule()
    assert len(out.scheduled) == 2                 # both admitted together
    assert out.num_new_tokens == 16
    assert all(s.num_computed_tokens == 24 for s in seqs)


def test_cache_probe_is_read_only():
    """get_cached_prefix must not touch ref_counts, LRU recency or stats."""
    sched, bm = make_sched(budget=64)
    prompt = list(range(16))
    _prime_cache(bm, prompt, block_size=8)
    ref_counts = {b.block_id: b.ref_count for b in bm.blocks}
    order = list(bm.cached_blocks.keys())
    queries, hits = bm.cache_queries, bm.cache_hits

    for _ in range(3):
        assert bm.get_cached_prefix(prompt) == 16
        # a longer prompt shares only the cached prefix
        assert bm.get_cached_prefix(prompt + [1, 2, 3]) == 16
        # an unseen prompt hits nothing
        assert bm.get_cached_prefix([9] * 16) == 0

    assert {b.block_id: b.ref_count for b in bm.blocks} == ref_counts
    assert list(bm.cached_blocks.keys()) == order
    assert (bm.cache_queries, bm.cache_hits) == (queries, hits)


def test_fully_cached_prompt_charges_one_token():
    """Fully cached prompt: only the forced recompute of the last token is
    charged (needs >= 1 budget), and num_computed lands at n-1."""
    sched, bm = make_sched(budget=16)
    prompt = list(range(32))                       # 32 % 8 == 0: fully cacheable
    _prime_cache(bm, prompt, block_size=8)

    seq = Sequence(list(prompt), SamplingParams())
    sched.add(seq)
    out = sched.schedule()
    assert seq in out.scheduled
    assert out.num_new_tokens == 1
    assert seq.num_computed_tokens == 31
    assert out.spans == [(31, 32)]


def test_zero_budget_blocks_even_fully_cached_admission():
    sched, bm = make_sched(budget=16)
    prompt = list(range(32))
    _prime_cache(bm, prompt, block_size=8)

    other = Sequence([9] * 8, SamplingParams())
    sched.add(other)
    out = sched.schedule()
    other.num_computed_tokens = out.spans[0][1]
    other.output_token_ids.append(1)               # engine protocol: sampled
    # drain the budget with decode work
    for _ in range(4):
        other.output_token_ids.append(1)
    sched.max_num_batched_tokens = 0
    seq = Sequence(list(prompt), SamplingParams())
    sched.add(seq)
    out = sched.schedule()
    assert seq not in out.scheduled                # no budget -> no admission


def test_failed_allocation_rolls_back_without_leaks():
    """A partially cached prompt whose UNCACHED blocks cannot be allocated
    must fail cleanly: refcounts of the cache-hit blocks restored, cached
    prefix unharmed, free list back to its prior state, seq stays waiting."""
    bm = BlockSpaceManager(num_blocks=4, block_size=4, num_layers=1,
                           num_kv_heads=1, head_dim=4, dtype=torch.float32,
                           device="cpu", enable_prefix_caching=True)
    sched = Scheduler(bm, max_num_seqs=8, max_num_batched_tokens=64)
    _prime_cache(bm, list(range(8)), block_size=4)     # 2 blocks cached

    # holder consumes the 2 free blocks WITHOUT evicting the cache
    holder = Sequence([9] * 8, SamplingParams())
    assert bm.allocate_sequence(holder) == 0
    assert bm.num_free_blocks == 0
    assert bm.num_evictable_blocks() == 2              # cached but in-use-able? no:
    # the 2 cached blocks have ref 0 but evicting them is not needed by holder

    free_before = bm.num_free_blocks
    seq = Sequence(list(range(12)), SamplingParams())  # shares first 8 tokens
    sched.add(seq)
    # 2 cache hits + 1 cold block: the cold block cannot be allocated
    # (free=0 and the cached blocks are ref>=1 -> not evictable). With an
    # empty running list the scheduler must surface this as unschedulable
    # (a real deadlock in production) -- after a clean internal rollback.
    with pytest.raises(RuntimeError, match="does not fit in the KV pool"):
        sched.schedule()
    assert seq.block_table == []
    assert seq.num_computed_tokens == 0
    assert bm.get_cached_prefix(list(range(12))) == 8  # cache unharmed
    bm.free_sequence(holder)
    assert bm.num_free_blocks == free_before + 2


def test_preempted_sequence_readmits_through_uncached_budget():
    """A preempted seq returns with num_computed=0; its prompt blocks are
    cached, so re-admission charges only the regenerated tokens."""
    sched, bm = make_sched(num_blocks=8, block_size=8, budget=64)
    prompt = list(range(24))
    seq = Sequence(list(prompt), SamplingParams(max_tokens=8))
    sched.add(seq)
    out = sched.schedule()
    assert seq in out.scheduled
    seq.num_computed_tokens = 24
    seq.output_token_ids.extend([7, 7])            # generated 2 tokens
    bm.register_filled_blocks(seq, 24)             # prompt blocks now shared
    # preempt the way the scheduler does: out of running, back to waiting
    sched.running.remove(seq)
    bm.preempt_sequence(seq)
    from minivllm.sequence import SequenceStatus
    seq.status = SequenceStatus.WAITING
    sched.add(seq)

    out = sched.schedule()
    assert seq in out.scheduled
    # uncached = the 2 generated tokens only (prompt KV fully cached)
    assert out.num_new_tokens == 2
    assert seq.num_computed_tokens == 24
    assert out.spans == [(24, 26)]


def test_property_budget_invariants_randomized():
    """Property test: random prompts/block sizes/budgets/cached-prefix
    lengths -- per-iteration budget is honored exactly and every scheduled
    sequence keeps the structural invariants (computed <= tokens, block
    table covers computed tokens, refcounts positive, no free+ref>0)."""
    import random
    rng = random.Random(0)
    for _trial in range(30):
        bs = rng.choice([4, 8])
        n_blocks = rng.randint(4, 24)
        budget = rng.randint(4, 48)
        sched, bm = make_sched(num_blocks=n_blocks, block_size=bs, budget=budget)
        prime = [rng.randrange(256) for _ in range(rng.randint(1, 6) * bs)]
        _prime_cache(bm, prime, bs)

        live = []
        for _ in range(rng.randint(1, 4)):
            shared = rng.randint(0, len(prime) // bs) * bs
            plen = shared + rng.randint(1, 3) * bs
            prompt = prime[:shared] + [rng.randrange(256) for _ in range(plen - shared)]
            s = Sequence(prompt, SamplingParams(max_tokens=64))
            sched.add(s)
            live.append(s)
            out = sched.schedule()
            # per-iteration token budget honored exactly
            assert out.num_new_tokens <= budget
            for s2, (start, end) in zip(out.scheduled, out.spans):
                assert s2.num_computed_tokens <= s2.num_tokens
                assert start <= end <= s2.num_tokens
                assert len(s2.block_table) * bs >= end
                for bid in s2.block_table:
                    blk = bm.blocks[bid]
                    assert blk.ref_count >= 1
                    assert bid not in bm.free_ids or blk.ref_count == 0
            # engine protocol: advance scheduled seqs so they can continue
            for s2, (_, end) in zip(out.scheduled, out.spans):
                s2.num_computed_tokens = end
                if end >= s2.num_prompt_tokens:
                    s2.output_token_ids.append(7)
        for s in live:
            if not s.block_table:                  # never admitted
                assert s.num_computed_tokens == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
