"""Unit tests for the continuous batching scheduler."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from minivllm.block_manager import BlockSpaceManager
from minivllm.scheduler import Scheduler
from minivllm.sequence import SamplingParams, Sequence, SequenceStatus


def make_sched(num_blocks=4, block_size=4, max_num_seqs=8, max_tokens_budget=64):
    bm = BlockSpaceManager(num_blocks=num_blocks, block_size=block_size,
                           num_layers=1, num_kv_heads=1, head_dim=4,
                           dtype=torch.float32, device="cpu",
                           enable_prefix_caching=True)
    return Scheduler(bm, max_num_seqs, max_tokens_budget), bm


def test_fcfs_admission_order():
    sched, bm = make_sched(num_blocks=6)
    seqs = [Sequence(list(range(6)), SamplingParams()) for _ in range(3)]
    for s in seqs:
        sched.add(s)
    out = sched.schedule()
    # 6-block pool: each seq needs ceil(6/4)=2 blocks -> all 3 fit
    assert len(out.scheduled) == 3
    assert [s.seq_id for s in out.scheduled] == [s0.seq_id for s0 in seqs]
    assert all(s.status == SequenceStatus.RUNNING for s in seqs)


def test_block_shortage_stops_admission():
    sched, bm = make_sched(num_blocks=4)            # 16 token slots
    seqs = [Sequence(list(range(6)), SamplingParams()) for _ in range(3)]
    for s in seqs:
        sched.add(s)
    out = sched.schedule()
    # seq0: 2 blocks, seq1: 2 blocks -> full; seq2 must wait
    assert len(out.scheduled) == 2
    assert sched.waiting[0] is seqs[2]


def test_token_budget_respected():
    sched, bm = make_sched(max_tokens_budget=10)
    seqs = [Sequence(list(range(6)), SamplingParams()) for _ in range(4)]
    for s in seqs:
        sched.add(s)
    out = sched.schedule()
    # budget 10 -> at most one 6-token prefill... first is 6, second 6 > 4 left
    assert len(out.scheduled) == 1
    assert out.num_new_tokens == 6


def test_decode_continuation_and_finish_removal():
    sched, bm = make_sched(num_blocks=8)
    s = Sequence(list(range(6)), SamplingParams(max_tokens=2))
    sched.add(s)
    out = sched.schedule()
    assert out.scheduled == [s]
    # simulate prefill done + one sampled token
    s.num_computed_tokens = s.num_tokens
    s.output_token_ids.append(42)
    out = sched.schedule()
    assert out.scheduled == [s]                     # decode step, needs 1 slot
    assert s.num_tokens - s.num_computed_tokens == 1
    # finish and confirm it leaves the running list
    s.status = SequenceStatus.FINISHED_LENGTH
    sched.running = [x for x in sched.running if not x.is_finished]
    out = sched.schedule()
    assert out.scheduled == []


def test_preemption_when_pool_exhausted():
    sched, bm = make_sched(num_blocks=4, block_size=4)   # 16 slots
    a = Sequence(list(range(6)), SamplingParams(max_tokens=50))
    b = Sequence(list(range(6)), SamplingParams(max_tokens=50))
    sched.add(a); sched.add(b)
    out = sched.schedule()
    assert len(out.scheduled) == 2
    # run a until both fill the pool, then a needs a new block -> b (newest) dies
    a.num_computed_tokens = 6
    b.num_computed_tokens = 6
    for step in range(10):
        a.output_token_ids.append(1)
        b.output_token_ids.append(1)
        out = sched.schedule()
        if out.num_preempted:
            break
        for s in out.scheduled:
            s.num_computed_tokens = s.num_tokens
            s.output_token_ids.append(1)
    assert out.num_preempted == 1
    assert b.status == SequenceStatus.WAITING
    assert b.num_computed_tokens == 0
    assert sched.waiting[0] is b
    # a survives and keeps decoding
    assert a in sched.running
    # freed blocks let b restart later (prefix cache restores its prompt)
    assert bm.cache_hits > 0 or bm.num_free_blocks > 0


def test_fully_cached_prompt_forces_recompute_of_last_token():
    sched, bm = make_sched(num_blocks=8)
    a = Sequence(list(range(8)), SamplingParams())
    sched.add(a)
    sched.schedule()
    a.num_computed_tokens = a.num_tokens
    bm.register_filled_blocks(a, a.num_tokens)
    bm.free_sequence(a)
    a.block_table = []

    b = Sequence(list(range(8)), SamplingParams())
    sched.add(b)
    a.status = SequenceStatus.FINISHED_LENGTH       # a is done, free its slot
    out = sched.schedule()
    assert b in out.scheduled
    assert b.num_computed_tokens == 7               # 8 - 1: recompute last token


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
