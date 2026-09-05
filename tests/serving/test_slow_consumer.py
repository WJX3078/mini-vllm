"""Slow-consumer isolation: a client that stops reading its stream must
not block the engine or other requests.

Mechanism under test: each request's output queue is bounded
(max_queue_deltas); when the engine overflows it, `_safe_put` cancels that
request (counted in slow_client_cancellations) instead of blocking.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from conftest import assert_no_leaks, ids_for

from minivllm.sequence import SamplingParams


def make_async_engine(max_queue_deltas=2):
    from test_async_engine import make_async_engine as build
    return build(max_queue_deltas=max_queue_deltas)


def run_async(coro):
    return asyncio.run(coro)


def test_slow_consumer_does_not_block_engine():
    async def body():
        a = make_async_engine(max_queue_deltas=2)
        await a.start()
        try:
            # slow client: adds a long request and never reads the stream
            slow_rid = await a.add_request(ids_for("S" * 30), SamplingParams(
                max_tokens=256, ignore_eos=True, seed=1))

            # healthy client arrives right after: must finish completely
            fast_done = asyncio.Event()

            async def fast():
                n = 0
                async for d in a.generate(ids_for("F" * 10),
                                          SamplingParams(
                                              max_tokens=8, ignore_eos=True,
                                              seed=2)):
                    n += 1
                    if d["finished"]:
                        fast_done.set()
                return n

            fast_task = asyncio.create_task(fast())
            # the slow stream's queue overflows within a few engine steps;
            # give it time, then check the healthy client finished first
            await asyncio.wait_for(fast_done.wait(), timeout=30)
            fast_tokens = await fast_task
            assert fast_tokens == 8

            # the engine must have cancelled the slow consumer
            deadline = asyncio.get_event_loop().time() + 10
            while a._slow_client_cancellations == 0 and \
                    asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.05)
                sreq = a.requests.get(slow_rid)
                print('T:', None if sreq is None else
                      (sreq.state, sreq.output_queue.qsize()),
                      a._slow_client_cancellations,
                      len(a.engine.scheduler.running))
            assert a.requests.get(slow_rid) is None      # cleaned up
            assert_no_leaks(a.engine)
        finally:
            await a.shutdown()
    run_async(body())


def test_abandoned_stream_is_cleaned_up():
    """A generator abandoned mid-stream (client gone without abort) must
    be aborted + removed by its finally block."""
    async def body():
        a = make_async_engine()
        await a.start()
        try:
            rid = await a.add_request(ids_for("A" * 20), SamplingParams(
                max_tokens=64, ignore_eos=True, seed=3))
            agen = a.stream(rid)
            await agen.__anext__()              # consume one delta
            await agen.aclose()                 # abandon (client gone)
            await asyncio.sleep(0.1)
            assert a.requests.get(rid) is None
            assert_no_leaks(a.engine)
        finally:
            await a.shutdown()
    run_async(body())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
