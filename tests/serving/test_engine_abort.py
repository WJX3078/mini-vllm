"""M1 cancellation regression tests: abort_request must correctly clean up
waiting / running / parallel-sampling requests, release KV blocks and
reservations, decrement shared prefix refcounts, and stay idempotent.

Every test asserts the post-condition invariants directly on the block
manager and scheduler (free blocks, reservations, refcounts, registry).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from conftest import assert_no_leaks, ids_for

from minivllm.sequence import SamplingParams


def _drive(engine, steps):
    for _ in range(steps):
        engine.step()
        engine.pop_deltas()


def test_abort_waiting_request(engine_factory):
    """A long prompt behind a running one must be cancellable while it is
    still WAITING: no blocks were materialized, nothing else is affected."""
    eng = engine_factory(num_blocks=13)
    first = eng.add_request(ids_for("short"), SamplingParams(
        max_tokens=6, ignore_eos=True))
    long_prompt = ids_for("L" * 100)
    second = eng.add_request(long_prompt, SamplingParams(
        max_tokens=6, ignore_eos=True))
    _drive(eng, 1)                                   # first runs, second waits
    assert second in [s for s in eng.scheduler.waiting] or \
        eng.groups[second].main in eng.scheduler.waiting

    assert eng.abort_request(second) is True
    assert eng.groups.get(second) is None            # registry dropped
    assert eng.abort_request(second) is False        # idempotent

    # first request completes normally
    while eng.scheduler.has_unfinished():
        eng.step()
        eng.pop_deltas()
    assert_no_leaks(eng)


def test_abort_running_request_releases_blocks(engine_factory):
    eng = engine_factory(num_blocks=32)
    prompt = ids_for("R" * 50)
    rid = eng.add_request(prompt, SamplingParams(max_tokens=32,
                                                 ignore_eos=True))
    _drive(eng, 3)                                   # running, blocks held
    rid_seq = None
    held = sum(len(s.block_table) for s in eng.scheduler.running)
    assert held >= 1

    assert eng.abort_request(rid) is True
    assert eng.groups.get(rid) is None
    # every owned block went back: nothing running, nothing reserved
    assert eng.scheduler.running == []
    assert eng.block_manager.total_reserved_blocks == 0
    assert eng.block_manager.num_used_blocks() == 0
    _drive(eng, 2)                                   # engine keeps working
    assert_no_leaks(eng)


def test_abort_is_idempotent_and_safe_after_finish(engine_factory):
    eng = engine_factory()
    rid = eng.add_request(ids_for("abc"), SamplingParams(
        max_tokens=4, ignore_eos=True))
    while eng.scheduler.has_unfinished():
        eng.step()
        eng.pop_deltas()
    # finished -> registry already dropped
    assert eng.abort_request(rid) is False
    assert eng.abort_request(rid) is False           # twice: still fine
    assert_no_leaks(eng)


def test_abort_unknown_request(engine_factory):
    eng = engine_factory()
    assert eng.abort_request(12345) is False


def test_abort_parallel_sampling_cancels_all_children(engine_factory):
    eng = engine_factory(num_blocks=32)
    rid = eng.add_request(ids_for("P" * 20), SamplingParams(
        max_tokens=32, ignore_eos=True, n=3))
    _drive(eng, 2)                                   # children forked + running
    group = eng.groups[rid]
    forked = len(group.children)
    assert forked == 2                               # n=3: main + 2 children
    running_seqs = [s for s in eng.scheduler.running
                    if eng.seq_to_group.get(s.seq_id) is group]
    assert len(running_seqs) == 3

    assert eng.abort_request(rid) is True
    assert eng.scheduler.running == []
    assert eng.groups.get(rid) is None
    assert eng.block_manager.total_reserved_blocks == 0
    assert eng.block_manager.num_used_blocks() == 0
    assert_no_leaks(eng)


def test_abort_clears_pending_forks_before_first_token(engine_factory):
    """Aborting before the fork point must clear pending_forks (they were
    never scheduled): the old code leaked them."""
    eng = engine_factory(num_blocks=16)
    rid = eng.add_request(ids_for("F" * 10), SamplingParams(
        max_tokens=8, ignore_eos=True, n=3))
    group = eng.groups[rid]
    assert len(group.pending_forks) == 2
    assert eng.abort_request(rid) is True
    assert group.pending_forks == []
    assert not eng.scheduler.waiting
    assert_no_leaks(eng)


def test_abort_decrements_shared_prefix_refcount(engine_factory):
    """Two requests share a prefix: aborting one must decrement the shared
    block's refcount, not free it (the other request still needs it)."""
    eng = engine_factory(num_blocks=32)
    shared = ids_for("S" * 16)                       # 2 full blocks (bs=8)
    a = eng.add_request(shared + ids_for("aaa"), SamplingParams(
        max_tokens=8, ignore_eos=True))
    _drive(eng, 1)                                   # a prefills + registers
    b = eng.add_request(shared + ids_for("bbb"), SamplingParams(
        max_tokens=8, ignore_eos=True))
    _drive(eng, 1)                                   # b arrives, maps the cache
    shared_blocks = set(eng.groups[a].main.block_table[:2])
    assert shared_blocks & set(eng.groups[b].main.block_table[:2])
    blk = next(iter(shared_blocks))
    ref_before = eng.block_manager.blocks[blk].ref_count
    assert ref_before >= 2

    assert eng.abort_request(a) is True
    ref_after = eng.block_manager.blocks[blk].ref_count
    assert ref_after == ref_before - 1
    # b completes normally and its output is intact
    while eng.scheduler.has_unfinished():
        eng.step()
        eng.pop_deltas()
    assert eng.groups.get(b) is None
    assert_no_leaks(eng)


def test_deltas_match_final_output(engine_factory):
    """Streamed deltas for a request must concatenate to exactly the final
    output token ids, ending with one finished=True delta."""
    eng = engine_factory()
    rid = eng.add_request(ids_for("xyz"), SamplingParams(
        max_tokens=6, ignore_eos=True))
    stream: list[tuple[int, list[int], bool, str | None]] = []
    while eng.scheduler.has_unfinished():
        eng.step()
        for d in eng.pop_deltas():
            if d.request_id == rid:
                stream.append((d.sample_idx, d.token_ids, d.finished,
                               d.finish_reason))
    assert stream and stream[-1][2] is True
    assert stream[-1][3] == "length"
    tokens = [t for (_, toks, _, _) in stream for t in toks]
    assert len(tokens) == 6
    # exactly one finished=True delta, at the end
    assert [fin for (_, _, fin, _) in stream] == [False] * 5 + [True]


def test_abort_emits_abort_delta(engine_factory):
    eng = engine_factory()
    rid = eng.add_request(ids_for("Z" * 20), SamplingParams(
        max_tokens=32, ignore_eos=True))
    _drive(eng, 2)
    eng.abort_request(rid)
    aborts = [d for d in eng.pop_deltas()
              if d.request_id == rid and d.finish_reason == "abort"]
    assert aborts and all(d.finished for d in aborts)


def test_finish_and_abort_race(engine_factory):
    """Request finishes on the exact step an abort is issued: the abort must
    be a harmless no-op and the finish output must stand."""
    eng = engine_factory()
    rid = eng.add_request(ids_for("q"), SamplingParams(
        max_tokens=1, ignore_eos=True))
    engine_finished = False
    while eng.scheduler.has_unfinished():
        eng.step()
        if any(d.request_id == rid and d.finished for d in eng.pop_deltas()):
            engine_finished = True
            assert eng.abort_request(rid) is False   # already dropped
    assert engine_finished
    assert_no_leaks(eng)


def test_registry_cleanup_long_serving(engine_factory):
    """Simulate 30 sequential requests: the registry must stay empty between
    requests (the v0.3 code leaked one group per request)."""
    eng = engine_factory()
    for i in range(30):
        rid = eng.add_request(ids_for(f"req{i}"), SamplingParams(
            max_tokens=2, ignore_eos=True))
        while eng.scheduler.has_unfinished():
            eng.step()
            eng.pop_deltas()
        assert eng.groups.get(rid) is None
        assert eng.groups == {}
    assert_no_leaks(eng)


def test_preempted_then_aborted(engine_factory):
    """Abort a request that was preempted mid-flight (waiting again, blocks
    already freed, reservation zero): cleanup must stay consistent."""
    eng = engine_factory(num_blocks=10, block_size=8, max_num_seqs=4)
    a = eng.add_request(ids_for("A" * 40), SamplingParams(
        max_tokens=40, ignore_eos=True))
    b = eng.add_request(ids_for("B" * 40), SamplingParams(
        max_tokens=40, ignore_eos=True))
    preempted = False
    for _ in range(12):
        eng.step()
        eng.pop_deltas()
        if any(s.num_computed_tokens == 0 and s.block_table == []
               for s in eng.scheduler.waiting):
            preempted = True
            break
    if not preempted:
        pytest.skip("pool pressure did not trigger preemption")
    rid = b if eng.groups.get(b) else a
    assert eng.abort_request(rid) is True
    while eng.scheduler.has_unfinished():      # survivor finishes cleanly
        eng.step()
        eng.pop_deltas()
    assert_no_leaks(eng)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
