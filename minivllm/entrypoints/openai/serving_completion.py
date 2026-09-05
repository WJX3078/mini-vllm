"""Shared completion/chat serving pipeline (v0.4).

One pipeline for both endpoints: validate -> admit through the bounded
input queue (continuous batching preserved) -> SSE delta stream or
aggregated JSON. Never a per-request generate().

The engine's delta queue carries `_DeltaPayload` objects (attribute
access); `AsyncLLMEngine.stream()` yields normalized dicts -- the SSE
generator reads raw payloads, so it accepts both.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi.responses import JSONResponse, StreamingResponse

from minivllm.entrypoints.openai.protocol import error_body

logger = logging.getLogger("minivllm.server")


def _sse(chunk: dict) -> str:
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def check_unsupported(request) -> str | None:
    """Explicitly reject OpenAI parameters the engine does not implement
    (they arrive in model_extra because the protocol allows extras)."""
    extras = getattr(request, "model_extra", None) or {}
    for key in extras:
        from minivllm.entrypoints.openai.protocol import UNSUPPORTED_PARAMS
        if key in UNSUPPORTED_PARAMS:
            return (f"parameter '{key}' is not supported: "
                    f"{UNSUPPORTED_PARAMS[key]}")
    return None


def prompt_token_ids(ctx, request, prompt_text: str | list) -> list[int] | None:
    """Tokenize the (chat-template-expanded) prompt. Token ids passed
    directly skip the tokenizer."""
    if isinstance(getattr(request, "prompt", None), list) and \
            request.prompt and isinstance(request.prompt[0], int):
        return list(request.prompt)
    tok = ctx.engine.engine.tokenizer
    if tok is None:
        return []
    text = prompt_text if isinstance(prompt_text, str) else " ".join(
        str(p) for p in prompt_text)
    return tok(text).input_ids


async def serve_completion(ctx, request, kind: str, prompt_text,
                           prompt_ids: list[int]):
    """kind: "completion" (OpenAI text) or "chat" (messages)."""
    unsupported = check_unsupported(request)
    if unsupported:
        return JSONResponse(error_body(unsupported, "unsupported_parameter",
                                       400), status_code=400)
    max_len = ctx.engine.config.max_model_len
    if len(prompt_ids) + request.max_tokens > max_len:
        return JSONResponse(error_body(
            f"prompt ({len(prompt_ids)}) + max_tokens ({request.max_tokens}) "
            f"exceeds max_model_len ({max_len})", "invalid_request_error",
            400), status_code=400)

    ctx.metrics.inc("requests_total")
    ctx.metrics.inc("prompt_tokens_total", len(prompt_ids))
    created = int(time.time())
    ext_rid = ("chatcmpl-" if kind == "chat" else "cmpl-") + uuid.uuid4().hex[:24]

    from minivllm.sequence import SamplingParams
    params = SamplingParams(
        temperature=request.temperature, top_p=request.top_p,
        top_k=request.top_k, max_tokens=request.max_tokens, n=request.n,
        stop=request.normalized_stop(), ignore_eos=request.ignore_eos,
        seed=request.seed)

    internal_rid = await ctx.engine.add_request(prompt_ids, params)
    # re-key to the external id (dispatch matches on internal_id)
    req = ctx.engine.requests.pop(internal_rid, None)
    if req is None:
        return JSONResponse(error_body("request dropped by engine",
                                       "internal_error", 500),
                            status_code=500)
    req.request_id = ext_rid
    ctx.engine.requests[ext_rid] = req

    if not request.stream:
        return await _collect(ctx, req, ext_rid, created, kind,
                              len(prompt_ids))
    ctx.metrics.inc("streams_started_total")
    return StreamingResponse(
        _delta_sse(ctx, req, ext_rid, created, kind),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _collect(ctx, req, ext_rid, created, kind, prompt_tokens):
    """stream=false: still flows through the async engine's continuous
    batching; the handler only aggregates the deltas."""
    start = time.perf_counter()
    outputs: dict[int, dict] = {}
    finish_reason = None
    async for d in ctx.engine.stream(req.request_id):
        slot = outputs.setdefault(d["sample_idx"],
                                  {"token_ids": [], "text": ""})
        slot["token_ids"].extend(d["token_ids"])
        slot["text"] += d["text"]
        finish_reason = d["finish_reason"]
    e2e = time.perf_counter() - start
    ctx.metrics.observe("request_e2e_seconds", e2e)
    ctx.metrics.inc("requests_finished_total")
    gen_tokens = sum(len(o["token_ids"]) for o in outputs.values())
    ctx.metrics.inc("generation_tokens_total", gen_tokens)
    logger.info("request %s prompt_tokens=%d output_tokens=%d e2e=%.3fs "
                "finish=%s", ext_rid, prompt_tokens, gen_tokens, e2e,
                finish_reason)

    if kind == "completion":
        choices = [{"index": idx, "text": o["text"],
                    "finish_reason": finish_reason}
                   for idx, o in sorted(outputs.items())]
        obj = "text_completion"
    else:
        choices = [{"index": idx,
                    "message": {"role": "assistant", "content": o["text"]},
                    "finish_reason": finish_reason}
                   for idx, o in sorted(outputs.items())]
        obj = "chat.completion"
    return {"id": ext_rid, "object": obj, "created": created,
            "model": ctx.model_name, "choices": choices,
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": gen_tokens,
                      "total_tokens": prompt_tokens + gen_tokens}}


def _payload_field(d, name, default=None):
    """The delta queue carries `_DeltaPayload` objects; `stream()` yields
    dicts. Read a field from either."""
    if isinstance(d, dict):
        return d.get(name, default)
    return getattr(d, name, default)


async def _delta_sse(ctx, req, ext_rid, created, kind):
    """SSE generator: first chunk, incremental content, final
    finish_reason, [DONE]. Client disconnect cancels this generator, whose
    finally aborts the engine request (KV released, GPU stops)."""
    start = time.perf_counter()
    n_tokens = 0
    finished_cleanly = False
    obj = "text_completion" if kind == "completion" else "chat.completion.chunk"
    from minivllm.serving.detokenizer import IncrementalDetokenizer
    detok = IncrementalDetokenizer(ctx.engine.engine.tokenizer)
    try:
        while True:
            try:
                if ctx.request_timeout is not None:
                    remaining = ctx.request_timeout - (time.perf_counter()
                                                       - start)
                    if remaining <= 0:
                        await ctx.engine.abort_request(ext_rid)
                        ctx.metrics.inc("requests_timeout_total")
                        yield _sse(error_body("request timed out", "timeout", 408))
                        yield "data: [DONE]\n\n"
                        return
                    d = await asyncio.wait_for(req.output_queue.get(),
                                               remaining)
                else:
                    d = await req.output_queue.get()
            except asyncio.TimeoutError:
                await ctx.engine.abort_request(ext_rid)
                ctx.metrics.inc("requests_timeout_total")
                yield _sse(error_body("request timed out", "timeout", 408))
                yield "data: [DONE]\n\n"
                return

            token_ids = _payload_field(d, "token_ids", [])
            finished = _payload_field(d, "finished", False)
            sample_idx = _payload_field(d, "sample_idx", 0)
            fr = _payload_field(d, "finish_reason")
            n_tokens += len(token_ids)
            # server-side incremental detokenization: text for the tokens
            # new THIS delta (partial UTF-8 tails are held back)
            text = detok.push(token_ids, final=finished)

            if finished:
                # engine-side cancellation (slow consumer, shutdown,
                # disconnect): flush held-back text, then end without a
                # client-visible final chunk
                if fr in ("abort", "error"):
                    tail = detok.push([], final=True)
                    if tail:
                        yield _sse({"id": ext_rid, "object": obj,
                                    "created": created,
                                    "model": ctx.model_name,
                                    "choices": [{"index": sample_idx,
                                                 "text": tail,
                                                 "finish_reason": None}]})
                    yield "data: [DONE]\n\n"
                    finished_cleanly = True
                    return
                fr = fr or "stop"
                # `text` came from push(..., final=True): it already flushes
                # any detokenizer-held tail -- emit it BEFORE the finish
                # chunk so streamed text adds up to the full output
                tail = text
                if tail:
                    if kind == "completion":
                        tail_chunk = {"id": ext_rid, "object": obj,
                                      "created": created,
                                      "model": ctx.model_name,
                                      "choices": [{"index": sample_idx,
                                                   "text": tail,
                                                   "finish_reason": None}]}
                    else:
                        tail_chunk = {"id": ext_rid, "object": obj,
                                      "created": created,
                                      "model": ctx.model_name,
                                      "choices": [{"index": sample_idx,
                                                   "delta": {"content": tail},
                                                   "finish_reason": None}]}
                    yield _sse(tail_chunk)
                if kind == "completion":
                    chunk = {"id": ext_rid, "object": obj,
                             "created": created, "model": ctx.model_name,
                             "choices": [{"index": sample_idx, "text": "",
                                          "finish_reason": fr}]}
                else:
                    chunk = {"id": ext_rid, "object": obj,
                             "created": created, "model": ctx.model_name,
                             "choices": [{"index": sample_idx, "delta": {},
                                          "finish_reason": fr}]}
                yield _sse(chunk)
                yield "data: [DONE]\n\n"
                finished_cleanly = True
                return

            if not token_ids:
                continue
            if kind == "completion":
                chunk = {"id": ext_rid, "object": obj, "created": created,
                         "model": ctx.model_name,
                         "choices": [{"index": sample_idx, "text": text,
                                      "finish_reason": None}]}
            else:
                chunk = {"id": ext_rid, "object": obj, "created": created,
                         "model": ctx.model_name,
                         "choices": [{"index": sample_idx,
                                      "delta": {"content": text},
                                      "finish_reason": None}]}
            yield _sse(chunk)
    finally:
        e2e = time.perf_counter() - start
        ctx.metrics.observe("request_e2e_seconds", e2e)
        ctx.metrics.inc("generation_tokens_total", n_tokens)
        # the async-engine registry entry dies with the stream: without
        # this, finished streams leak one entry each and eventually the
        # admission cap 429s every new request
        ctx.engine.requests.pop(ext_rid, None)
        if not finished_cleanly:
            # client disconnected / generator cancelled: stop the GPU
            await ctx.engine.abort_request(ext_rid)
