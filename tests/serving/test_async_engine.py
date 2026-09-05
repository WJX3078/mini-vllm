"""M2: AsyncLLMEngine — single GPU owner loop, continuous batching under
HTTP-style concurrency, output routing, idle wakeup, shutdown.

All tests run the tiny random model on CPU; `asyncio.run` drives the event
loop explicitly (Windows-safe, no plugin dependency).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from conftest import ByteTokenizer, assert_no_leaks, ids_for
from helpers import make_tiny_pair

from minivllm import EngineConfig
from minivllm.config import ModelConfig
from minivllm.engine import LLMEngine
from minivllm.sequence import SamplingParams
from minivllm.serving.async_engine import (
    AsyncLLMEngine,
    QueueFullError,
)
from minivllm.serving.detokenizer import IncrementalDetokenizer


def make_async_engine(max_pending=64, max_queue_deltas=256, **overrides):
    hf, mine = make_tiny_pair(seed=0)
    cfg = EngineConfig(
        model="(tiny-random-qwen2)",
        block_size=overrides.pop("block_size", 8),
        num_blocks=overrides.pop("num_blocks", 64),
        max_num_seqs=overrides.pop("max_num_seqs", 8),
        max_model_len=overrides.pop("max_model_len", 256),
        max_num_batched_tokens=overrides.pop("max_num_batched_tokens", 256),
        seed=0, device="cpu", dtype="float32", **overrides)
    eng = LLMEngine(cfg, model=mine,
                    model_config=ModelConfig.from_hf_config(hf.config),
                    tokenizer=ByteTokenizer())
    return AsyncLLMEngine(cfg, max_pending_requests=max_pending,
                          max_queue_deltas=max_queue_deltas, engine=eng)


def run_async(coro):
    return asyncio.run(coro)


def test_async_single_request():
    async def body():
        a = make_async_engine()
        await a.start()
        try:
            outs = []
            async for d in a.generate(ids_for("hello"),
                                      SamplingParams(max_tokens=6,
                                                     ignore_eos=True,
                                                     seed=1)):
                outs.append(d)
            assert outs[-1]["finish_reason"] == "length"
            assert sum(len(d["token_ids"]) for d in outs) == 6
            text = "".join(d["text"] for d in outs)
            assert text == ByteTokenizer().decode(
                [t for d in outs for t in d["token_ids"]])
        finally:
            await a.shutdown()
    run_async(body())


def test_async_multi_request_and_output_routing():
    """8 concurrent requests with per-request marker prompts: outputs must
    not cross (every response's bytes belong to its own prompt stream)."""
    async def body():
        a = make_async_engine()
        await a.start()
        try:
            marks = [f"m{i}" for i in range(8)]

            async def one(mark):
                outs = []
                async for d in a.generate(ids_for(mark),
                                          SamplingParams(
                                              max_tokens=6, ignore_eos=True,
                                              seed=hash(mark) % 1000)):
                    outs.append(d)
                return mark, outs

            results = await asyncio.gather(*[one(m) for m in marks])
            for _, outs in results:
                tokens = [t for d in outs for t in d["token_ids"]]
                assert tokens, "no tokens streamed"
                assert outs[-1]["finish_reason"] == "length"
                # stream self-consistency: detokenized text equals decode of
                # its own token stream (no cross-request contamination)
                text = "".join(d["text"] for d in outs)
                assert text == ByteTokenizer().decode(tokens)
            assert len(a.requests) == 0
            assert_no_leaks(a.engine)
        finally:
            await a.shutdown()
    run_async(body())


def test_continuous_batching_under_concurrency():
    """32 requests arrive while earlier ones are still decoding: the
    scheduler must run them TOGETHER (the whole point of continuous
    batching) -- proven via the engine's peak concurrent-scheduled count.
    """
    async def body():
        a = make_async_engine()
        await a.start()
        try:
            seen_running = 0
            orig_step = a.engine.step

            def spy_step():
                nonlocal seen_running
                seen_running = max(seen_running, len(a.engine.scheduler.running))
                return orig_step()

            a.engine.step = spy_step

            async def one(i):
                async for _ in a.generate(ids_for(f"r{i}" * 3),
                                          SamplingParams(
                                              max_tokens=8, ignore_eos=True,
                                              seed=i)):
                    pass

            await asyncio.gather(*[one(i) for i in range(32)])
            assert seen_running >= 8, \
                f"continuous batching broken: max concurrent={seen_running}"
        finally:
            await a.shutdown()
    run_async(body())


def test_no_request_lost_or_duplicated():
    """128 concurrent requests: exactly one finished delta each, no loss,
    no duplicates."""
    async def body():
        a = make_async_engine(max_pending=256)
        await a.start()
        try:
            finished = []

            async def one(i):
                count = 0
                async for _ in a.generate(ids_for(f"u{i}"),
                                          SamplingParams(
                                              max_tokens=4, ignore_eos=True,
                                              seed=100 + i)):
                    count += 1
                finished.append((i, count))

            await asyncio.gather(*[one(i) for i in range(128)])
            assert sorted(i for i, _ in finished) == list(range(128))
            assert all(c >= 1 for _, c in finished)
        finally:
            await a.shutdown()
    run_async(body())


def test_queue_full_overload():
    """max_pending_requests small: overflow must raise QueueFullError (the
    HTTP layer maps it to 429) and the engine must survive."""
    async def body():
        a = make_async_engine(max_pending=2)
        await a.start()
        try:
            accepted, rejected = [], 0
            for i in range(16):
                try:
                    rid = await a.add_request(ids_for(f"o{i}"),
                                              SamplingParams(
                                                  max_tokens=4,
                                                  ignore_eos=True, seed=i))
                    accepted.append(rid)
                except QueueFullError:
                    rejected += 1
            assert rejected >= 1 and len(accepted) >= 1
            # drain the accepted requests to completion
            await asyncio.gather(*[_drain(a, rid) for rid in accepted])
            assert a.is_healthy()
            assert_no_leaks(a.engine)
        finally:
            await a.shutdown()

    run_async(body())


async def _drain(a, rid):
    async for _ in a.stream(rid):
        pass


def test_abort_via_async_engine():
    """Abort issued while a long generation runs: the stream must end with
    finish_reason="abort" well before max_tokens (the engine thread drains
    commands before every step, so the abort lands within one step)."""
    async def body():
        a = make_async_engine(max_model_len=2048)
        await a.start()
        try:
            rid = await a.add_request(ids_for("C" * 30), SamplingParams(
                max_tokens=512, ignore_eos=True, seed=5))
            got, finish = 0, None
            async for d in a.stream(rid):
                got += 1
                if got == 1:
                    await a.abort_request(rid)
                if d["finished"]:
                    finish = d["finish_reason"]
            assert finish == "abort"
            assert got < 256, f"abort took too long: {got} tokens"
            await asyncio.sleep(0.05)
            assert_no_leaks(a.engine)
        finally:
            await a.shutdown()
    run_async(body())


def test_shutdown_cancels_active_requests():
    async def body():
        a = make_async_engine()
        await a.start()
        stream_done = []

        async def one(i):
            async for _ in a.generate(ids_for("G" * 30), SamplingParams(
                    max_tokens=64, ignore_eos=True, seed=i)):
                pass
            stream_done.append(i)

        tasks = [asyncio.create_task(one(i)) for i in range(4)]
        await asyncio.sleep(0.05)             # let them start
        await a.shutdown(grace_period=0.1)    # short grace -> cancel
        await asyncio.gather(*tasks, return_exceptions=True)
        assert a.state.value == "stopped"
        assert_no_leaks(a.engine)
        assert len(a.requests) == 0
    run_async(body())


def test_incremental_detokenizer_unicode_and_boundaries():
    """The detokenizer must reconstruct EXACT text across multi-byte
    characters split over 1..4 tokens, and hold partial tails."""
    tok = ByteTokenizer()

    def stream_text(text, chunk_sizes):
        detok = IncrementalDetokenizer(tok)
        ids = ids_for(text)
        out, i = [], 0
        for size in chunk_sizes:
            chunk = ids[i:i + size]
            i += len(chunk)
            out.append(detok.push(chunk))
        tail = detok.push([], final=True)
        out.append(tail)
        return "".join(out)

    cases = [
        "plain ascii text",
        "CJK: 你好世界，中文测试。",                 # 3-byte chars
        "emoji: 👋🚀🔥 and 👨‍👩‍👧 family",           # 4-byte + ZWJ sequences
        "mixed ascii + 中文 + 🎉 end",
        "aaéñü accents",                              # 2-byte chars
    ]
    for text in cases:
        assert stream_text(text, [1] * 64) == text
        assert stream_text(text, [3] * 64) == text
        assert stream_text(text, [len(ids_for(text))]) == text
    # irregular chunks that split characters mid-way
    ids = ids_for("中文abc🎉def")
    for cut in range(1, len(ids)):
        detok = IncrementalDetokenizer(tok)
        parts = [detok.push(ids[:cut]), detok.push(ids[cut:]),
                 detok.push([], final=True)]
        assert "".join(parts) == "中文abc🎉def", f"cut={cut} -> {parts!r}"


def test_engine_thread_idle_no_busy_spin():
    """Idle engine must not spin (task: no busy loop). With no requests the
    engine thread blocks on the wake Event: loop iterations over 0.5s must
    stay in the single digits, not thousands."""
    async def body():
        a = make_async_engine()
        await a.start()
        try:
            a._loop_iterations = 0
            await asyncio.sleep(0.5)
            iters = a._loop_iterations
            assert iters <= 5, f"idle loop spun {iters}x in 0.5s"
        finally:
            await a.shutdown()
    run_async(body())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
