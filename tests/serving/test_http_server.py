"""M3-M5: OpenAI-compatible HTTP server tests over the ASGI transport.

Real FastAPI app + real AsyncLLMEngine on the tiny random model (CPU) with
the byte-level stub tokenizer -- no GPU, no model downloads. Covers the
endpoints, SSE framing/termination, unsupported-parameter rejection, queue
overload (429), slow-consumer isolation, client-disconnect cancellation,
metrics, and auth.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from conftest import assert_no_leaks, ids_for
from helpers import make_tiny_pair

from minivllm import EngineConfig
from minivllm.config import ModelConfig
from minivllm.engine import LLMEngine
from minivllm.serving.async_engine import AsyncLLMEngine


def build_app(max_pending=64, request_timeout=None, api_key=None):
    from minivllm.entrypoints.openai.api_server import create_app

    hf, mine = make_tiny_pair(seed=0)
    cfg = EngineConfig(model="tiny-random-qwen2", block_size=8, num_blocks=64,
                       max_num_seqs=8, max_model_len=256,
                       max_num_batched_tokens=256, seed=0, device="cpu",
                       dtype="float32")
    eng = LLMEngine(cfg, model=mine,
                    model_config=ModelConfig.from_hf_config(hf.config),
                    tokenizer=__import__(
                        "conftest", fromlist=["ByteTokenizer"]).ByteTokenizer())
    engine = AsyncLLMEngine(cfg, max_pending_requests=max_pending,
                            engine=eng)
    app = create_app(engine, model_name="tiny-random-qwen2",
                     request_timeout=request_timeout, api_key=api_key)
    return app, engine


def make_client(app):
    from httpx import ASGITransport, AsyncClient
    return AsyncClient(transport=ASGITransport(app=app),
                       base_url="http://test")


def with_server(fn):
    """Boot the engine + app inside one event loop and hand the client over."""
    async def body():
        app, engine = build_app()
        async with make_client(app) as client:
            await engine.start()
            try:
                return await fn(client, engine)
            finally:
                await engine.shutdown()
    return asyncio.run(body())


def test_health_ready_models():
    async def fn(client, engine):
        r = await client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "alive"
        r = await client.get("/ready")
        assert r.status_code == 200 and r.json()["ready"] is True
        r = await client.get("/v1/models")
        assert r.json()["data"][0]["id"] == "tiny-random-qwen2"
    with_server(fn)


def test_completion_non_stream():
    async def fn(client, engine):
        r = await client.post("/v1/completions", json={
            "model": "tiny-random-qwen2", "prompt": "hello",
            "max_tokens": 8, "temperature": 0.0, "seed": 3,
            "ignore_eos": True})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["usage"]["completion_tokens"] == 8
    with_server(fn)


def test_completion_token_prompt():
    async def fn(client, engine):
        r = await client.post("/v1/completions", json={
            "prompt": ids_for("tok"), "max_tokens": 4, "temperature": 0.0,
            "ignore_eos": True})
        assert r.status_code == 200
        assert r.json()["usage"]["completion_tokens"] == 4
    with_server(fn)


def test_completion_stream_sse():
    """Streaming must add up to exactly the same text as a non-streaming
    request with the same seed (random bytes mean many deltas carry held
    or replacement text -- the invariant is equivalence, not length)."""
    async def fn(client, engine):
        base = {"prompt": "stream me", "max_tokens": 6, "temperature": 0.0,
                "ignore_eos": True, "seed": 9}
        r = await client.post("/v1/completions",
                              json={**base, "stream": False})
        ref_text = r.json()["choices"][0]["text"]

        chunks = []
        async with client.stream("POST", "/v1/completions",
                                 json={**base, "stream": True}) as r2:
            assert r2.status_code == 200
            assert r2.headers["content-type"].startswith(
                "text/event-stream")
            async for line in r2.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    chunks.append(json.loads(payload))
        assert chunks, "no SSE chunks"
        texts = "".join(c["choices"][0]["text"] for c in chunks
                        if c["choices"] and "text" in c["choices"][0])
        assert texts == ref_text
        # framing: finish_reason appears exactly once, on the last chunk
        fins = [c["choices"][0]["finish_reason"] for c in chunks]
        assert fins[-1] == "length"
        assert all(f is None for f in fins[:-1])
    with_server(fn)


def test_chat_completion_and_template_missing():
    async def fn(client, engine):
        r = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4, "temperature": 0.0, "ignore_eos": True})
        # the ByteTokenizer implements apply_chat_template -> works
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"

        # a tokenizer without chat template support -> explicit 400
        engine.engine.tokenizer = object()
        r = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4})
        assert r.status_code == 400
        assert "chat template" in r.json()["error"]["message"]
    with_server(fn)


def test_unsupported_parameter_rejected():
    async def fn(client, engine):
        r = await client.post("/v1/completions", json={
            "prompt": "x", "max_tokens": 4, "logprobs": 5})
        assert r.status_code == 400
        assert "logprobs" in r.json()["error"]["message"]
    with_server(fn)


def test_prompt_too_long_rejected_400():
    async def fn(client, engine):
        r = await client.post("/v1/completions", json={
            "prompt": "x" * 300, "max_tokens": 8})
        assert r.status_code == 400
        assert "max_model_len" in r.json()["error"]["message"]
    with_server(fn)


def test_queue_overload_429():
    async def fn(client, engine):
        app2, engine2 = build_app(max_pending=2)
        async with make_client(app2) as client2:
            await engine2.start()
            accepted = rejected = 0

            async def post(i):
                nonlocal accepted, rejected
                r = await client2.post("/v1/completions", json={
                    "prompt": f"req{i}", "max_tokens": 16,
                    "temperature": 0.0, "ignore_eos": True})
                if r.status_code == 200:
                    accepted += 1
                else:
                    assert r.status_code == 429
                    assert r.json()["error"]["type"] == "server_overloaded"
                    rejected += 1

            # concurrent burst: many adds land before the engine drains
            await asyncio.gather(*[post(i) for i in range(16)])
            assert accepted >= 1 and rejected >= 1
            await engine2.shutdown()
            # original engine untouched
            assert engine.is_healthy()
    with_server(fn)


def test_concurrent_streaming_no_cross_talk():
    """16 concurrent streaming completions: every SSE stream must be self-
    consistent and terminate exactly once with [DONE]."""
    async def fn(client, engine):
        async def one(i):
            lines = []
            async with client.stream("POST", "/v1/completions", json={
                    "prompt": f"req{i} ", "max_tokens": 6, "temperature": 0.0,
                    "ignore_eos": True, "stream": True}) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        lines.append(line[6:])
            assert lines[-1] == "[DONE]"
            chunks = [json.loads(x) for x in lines[:-1] if x != "[DONE]"]
            texts = [c["choices"][0].get("text", "") for c in chunks]
            fins = [c["choices"][0]["finish_reason"] for c in chunks]
            assert fins.count("length") == 1
            assert fins[-1] == "length"
            return i, "".join(texts)
        results = await asyncio.gather(*[one(i) for i in range(16)])
        assert len(results) == 16
        assert len({i for i, _ in results}) == 16
        await asyncio.sleep(0.1)
        assert len(engine.requests) == 0     # registry: no stream leaks
    with_server(fn)


def test_metrics_endpoint_reflects_activity():
    async def fn(client, engine):
        await client.post("/v1/completions", json={
            "prompt": "count me", "max_tokens": 4, "temperature": 0.0,
            "ignore_eos": True})
        r = await client.get("/metrics")
        body = r.text
        assert "requests_total" in body
        assert "kv_blocks_total" in body
        assert "scheduler_running_sequences" in body
        assert "request_e2e_seconds_count" in body
    with_server(fn)


def test_client_disconnect_aborts_generation():
    """Abandoning a streaming response must abort the engine request (the
    GPU stops) and leave no leaks."""
    async def fn(client, engine):
        async def open_and_abandon():
            async with client.stream("POST", "/v1/completions", json={
                    "prompt": "D" * 20, "max_tokens": 128,
                    "temperature": 0.0, "ignore_eos": True}) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        # close the connection after the first chunk
                        if chunk["choices"][0].get("text"):
                            break
            return True

        await open_and_abandon()
        await asyncio.sleep(0.2)              # generator finally runs abort
        assert_no_leaks(engine.engine)
    with_server(fn)


def test_request_timeout_aborts():
    async def fn(client, engine):
        app2, engine2 = build_app(request_timeout=0.05)
        async with make_client(app2) as client2:
            await engine2.start()
            r = await client2.post("/v1/completions", json={
                "prompt": "T" * 10, "max_tokens": 200, "temperature": 0.0,
                "ignore_eos": True, "stream": True})
            lines = []
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    lines.append(line[6:])
            payloads = [x for x in lines if x != "[DONE]"]
            last = json.loads(payloads[-1])
            assert "error" in last or last["choices"][0][
                "finish_reason"] in ("length",)
            if "error" in last:
                assert last["error"]["code"] == 408
            await engine2.shutdown()
        assert engine.is_healthy()
    with_server(fn)


def test_api_key_auth():
    async def fn(client, engine):
        app2, engine2 = build_app(api_key="secret")
        async with make_client(app2) as client2:
            await engine2.start()
            r = await client2.get("/v1/models")
            assert r.status_code == 401
            r = await client2.get("/v1/models",
                                  headers={"Authorization": "Bearer secret"})
            assert r.status_code == 200
            await engine2.shutdown()
        assert engine.is_healthy()
    with_server(fn)


def test_engine_survives_many_sequential_requests():
    async def fn(client, engine):
        for i in range(20):
            r = await client.post("/v1/completions", json={
                "prompt": f"seq{i}", "max_tokens": 4, "temperature": 0.0,
                "ignore_eos": True})
            assert r.status_code == 200
        assert engine.engine.groups == {}
        assert engine.engine.block_manager.total_reserved_blocks == 0
    with_server(fn)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
