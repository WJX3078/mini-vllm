"""KV capacity reservation vs physical allocation (v0.3, Phase 4).

Admission checks and books the sequence's FULL cold-prompt capacity
(reservation), while physical blocks materialize lazily, one scheduled span
at a time (allocation). Verified here:

  * a chunked-prefill admission reserves the whole prompt but materializes
    only the first chunk's blocks (peak usage drops);
  * outstanding reservations count against OTHER sequences' admission
    (full-ISL reservation blocks over-admission);
  * cache-hit blocks are excluded from the cold reservation;
  * materializing the whole prompt draws the reservation down to zero;
  * preemption releases the reservation (re-admission re-reserves);
  * reserve_full_isl=False admits aggressively and pays with over-commit;
  * eager vs lazy allocation schedule identically, lazy holds fewer blocks.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from minivllm.block_manager import BlockSpaceManager
from minivllm.scheduler import Scheduler
from minivllm.sequence import SamplingParams, Sequence


def make_sched(num_blocks, block_size=8, budget=4096, max_num_seqs=8,
               reserve_full_isl=True, lazy=True):
    bm = BlockSpaceManager(num_blocks=num_blocks, block_size=block_size,
                           num_layers=1, num_kv_heads=1, head_dim=4,
                           dtype=torch.float32, device="cpu",
                           enable_prefix_caching=True)
    return Scheduler(bm, max_num_seqs, budget, reserve_full_isl=reserve_full_isl,
                     lazy_allocation=lazy), bm


def test_lazy_admission_materializes_only_first_chunk():
    """One 100-token prompt (13 cold blocks), budget 72: admitted with a
    72-token chunk -- 9 blocks materialized, 4 exist only as a reservation
    (the eager path would hold all 13 immediately)."""
    sched, bm = make_sched(num_blocks=64, budget=72)
    seq = Sequence(list(range(100)), SamplingParams())
    sched.add(seq)
    out = sched.schedule()
    assert out.spans == [(0, 72)]
    assert len(seq.block_table) == 9
    assert seq.reserved_cold_blocks == 4
    assert bm.total_reserved_blocks == 4
    assert bm.num_used_blocks() + bm.num_free_blocks \
        + bm.num_evictable_blocks() == bm.num_blocks


def test_outstanding_reservations_block_other_admissions():
    """Pool 16; A (100 tokens, 13 cold blocks) admitted with a 24-token
    chunk: 3 blocks materialized, 10 reserved, 13 free. B (48 tokens, 6
    cold blocks) must NOT be admitted under full-ISL -- its 6-block promise
    cannot be covered once A's 10-block reservation is honored (13 - 10 =
    3 < 6)."""
    sched, bm = make_sched(num_blocks=16, budget=24, reserve_full_isl=True)
    a = Sequence(list(range(100)), SamplingParams())
    b = Sequence(list(range(48)), SamplingParams())
    sched.add(a)
    sched.add(b)
    out = sched.schedule()
    assert a in out.scheduled
    assert (len(a.block_table), a.reserved_cold_blocks) == (3, 10)
    assert b not in out.scheduled
    assert bm.total_reserved_blocks == 10


def test_aggressive_admission_over_commits_and_gets_admitted():
    """Pool 16, budget 32. A = 64 tokens (8 cold blocks): chunk (0,32) ->
    4 materialized, 4 reserved, availability for B = 16-4-4 = 8.
    B = 96 tokens (12 cold blocks):
      * full-ISL: 12 > 8 -> stays WAITING;
      * aggressive: needs only its 4-block first chunk (4 <= 8) -> admitted,
        committing 4+8 more blocks than the pool can cover (over-commit is
        exactly the aggressive trade-off; later spans may preempt)."""
    def scenario(full_isl):
        sched, bm = make_sched(num_blocks=16, budget=48,
                               reserve_full_isl=full_isl)
        a = Sequence(list(range(64)), SamplingParams())
        b = Sequence(list(range(96)), SamplingParams())
        sched.add(a)
        # iteration 1: a admitted with chunk (0,48) -> 6 blocks; full-ISL
        # books the remaining 2, aggressive books only what it checked
        out = sched.schedule()
        expected_reserved = 2 if full_isl else 0
        assert (len(a.block_table), a.reserved_cold_blocks)             == (6, expected_reserved)
        a.num_computed_tokens = 48
        a.output_token_ids.append(1)               # engine protocol: sampled
        # iteration 2: a finishes its last 16 tokens; b is considered with
        # the leftover budget
        sched.add(b)
        out = sched.schedule()
        return sched, a, b, out

    _, a, b, out = scenario(full_isl=True)
    assert a in out.scheduled and b not in out.scheduled
    # a fully materialized (9 blocks: 64 prompt + 1 sampled output); b's
    # 12-cold-block promise > 7 free -> WAITING
    assert (len(a.block_table), a.reserved_cold_blocks) == (9, 0)

    sched2, a2, b2, out2 = scenario(full_isl=False)
    assert a2 in out2.scheduled and b2 in out2.scheduled
    assert (len(a2.block_table), a2.reserved_cold_blocks) == (9, 0)
    # b enters with a 32-token chunk (4 blocks); only the checked chunk is
    # booked (drawn to 0) -- its remaining 8 cold blocks are pure
    # over-commit, preemption is the safety net
    assert (len(b2.block_table), b2.reserved_cold_blocks) == (4, 0)
    assert sched2.bm.total_reserved_blocks == 0


def test_cache_hits_are_excluded_from_cold_reservation():
    bm = BlockSpaceManager(num_blocks=32, block_size=8, num_layers=1,
                           num_kv_heads=1, head_dim=4, dtype=torch.float32,
                           device="cpu", enable_prefix_caching=True)
    # prime 8 cached tokens (1 block)
    primed = Sequence(list(range(8)), SamplingParams())
    bm.allocate_sequence(primed)
    bm.register_filled_blocks(primed, 8)
    shared_block = primed.block_table[0]
    bm.free_sequence(primed)

    seq = Sequence(list(range(40)), SamplingParams())  # 5 blocks total
    cached = bm.map_cached_prefix(seq)
    assert cached == 8
    cold = bm.cold_blocks_needed(len(seq.tokens), cached)
    assert cold == 4                                   # 5 blocks - 1 cache hit
    bm.reserve(seq, cold)
    assert bm.allocate_span(seq, cached, len(seq.tokens))
    assert seq.reserved_cold_blocks == 0               # fully materialized
    assert len(seq.block_table) == 5
    assert seq.block_table[0] == shared_block          # cache block shared
    assert bm.total_reserved_blocks == 0


def test_materializing_full_prompt_draws_reservation_to_zero():
    sched, bm = make_sched(num_blocks=32, budget=128)
    seq = Sequence(list(range(100)), SamplingParams())
    sched.add(seq)
    out = sched.schedule()                             # chunk covers all 100
    assert seq in out.scheduled
    assert seq.reserved_cold_blocks == 0
    assert bm.total_reserved_blocks == 0
    assert len(seq.block_table) == 13


def test_preemption_releases_reservation_and_readmission_rereserves():
    sched, bm = make_sched(num_blocks=12, budget=32)
    a = Sequence(list(range(64)), SamplingParams())    # 8 cold blocks
    sched.add(a)
    out = sched.schedule()                             # chunk 32 -> 4 blocks
    assert a in out.scheduled
    assert (len(a.block_table), a.reserved_cold_blocks) == (4, 4)

    # preemption (as the scheduler does) releases blocks AND reservation
    sched.running.remove(a)
    bm.preempt_sequence(a)
    from minivllm.sequence import SequenceStatus as SS
    a.status = SS.WAITING
    assert a.reserved_cold_blocks == 0
    assert a.block_table == []
    assert bm.total_reserved_blocks == 0
    sched.add(a)

    out = sched.schedule()
    assert a in out.scheduled
    assert (len(a.block_table), a.reserved_cold_blocks) == (4, 4)


def test_eager_vs_lazy_same_schedule_lazy_holds_fewer_blocks():
    """Eager (v0.2) and lazy (v0.3) allocation schedule identically; lazy
    strictly defers block materialization (12 held vs 13 for one chunked
    prompt of 100 tokens with a 96-token chunk)."""
    def run(lazy):
        sched, bm = make_sched(num_blocks=64, budget=96, lazy=lazy)
        seq = Sequence(list(range(100)), SamplingParams())
        sched.add(seq)
        out = sched.schedule()
        return list(out.spans), len(seq.block_table)

    spans_lazy, held_lazy = run(lazy=True)
    spans_eager, held_eager = run(lazy=False)
    assert spans_lazy == spans_eager
    assert held_eager == 13                            # whole prompt
    assert held_lazy == 12                             # chunk only + 1 reserved


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
