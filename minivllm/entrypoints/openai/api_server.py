"""OpenAI-compatible API server: app factory + endpoints + CLI.

Pipeline (never a per-request generate()):
    HTTP -> pydantic validation -> AsyncLLMEngine.add_request
         -> single engine thread (continuous batching across ALL requests)
         -> per-request delta stream -> SSE / aggregated JSON

Run:
    python -m minivllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B

One process per GPU engine: uvicorn workers > 1 would load the model
multiple times, so the CLI pins workers=1.
"""
from __future__ import annotations

import argparse
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from minivllm.config import EngineConfig
from minivllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    error_body,
)
from minivllm.serving.async_engine import (
    AsyncLLMEngine,
    EngineUnhealthyError,
    QueueFullError,
)
from minivllm.serving.metrics import Metrics

logger = logging.getLogger("minivllm.server")


class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def create_app(engine: AsyncLLMEngine, model_name: str,
               request_timeout: float | None = None,
               api_key: str | None = None,
               enable_cors: bool = False) -> FastAPI:
    app = FastAPI(title="mini-vLLM", version="0.4.0")
    metrics = Metrics()
    ctx = SimpleNamespace(engine=engine, model_name=model_name,
                          metrics=metrics, request_timeout=request_timeout)
    app.state.ctx = ctx
    _register_engine_gauges(metrics, engine)

    if enable_cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if api_key is not None:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {api_key}":
                return JSONResponse(
                    error_body("invalid api key", "auth_error", 401),
                    status_code=401)
        return await call_next(request)

    @app.exception_handler(QueueFullError)
    async def overload_handler(request: Request, exc: QueueFullError):
        metrics.inc("requests_rejected_total")
        return JSONResponse(error_body("Server overloaded",
                                       "server_overloaded", 429),
                            status_code=429)

    @app.exception_handler(EngineUnhealthyError)
    async def unhealthy_handler(request: Request, exc: EngineUnhealthyError):
        return JSONResponse(error_body(str(exc), "engine_unhealthy", 503),
                            status_code=503)

    # ------------------------------------------------------------- lifecycle
    @app.get("/health")
    async def health():
        return {"status": "alive"}

    @app.get("/ready")
    async def ready():
        ok = engine.is_ready() and engine.is_healthy()
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [
            {"id": model_name, "object": "model", "owned_by": "mini-vllm"}]}

    @app.get("/metrics")
    async def metrics_endpoint():
        metrics.set_gauge("requests_running", len(engine.requests))
        metrics.set_gauge("requests_pending", engine._pending_adds)
        metrics.set_gauge("slow_client_cancellations_total",
                          engine._slow_client_cancellations)
        return JSONResponse(metrics.render_prometheus(),
                            media_type="text/plain; version=0.0.4")

    # ----------------------------------------------------------- completions
    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        from minivllm.entrypoints.openai.serving_completion import (
            prompt_token_ids,
            serve_completion,
        )
        prompt_ids = prompt_token_ids(ctx, body, body.prompt)
        return await serve_completion(ctx, body, "completion",
                                      body.prompt, prompt_ids or [])

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        from minivllm.entrypoints.openai.serving_completion import (
            prompt_token_ids,
            serve_completion,
        )
        tok = engine.engine.tokenizer
        if tok is None or not hasattr(tok, "apply_chat_template"):
            return JSONResponse(error_body(
                "the loaded tokenizer does not support chat templates",
                "not_supported", 400), status_code=400)
        messages = [m.model_dump() for m in body.messages]
        try:
            prompt_text = tok.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            return JSONResponse(error_body(
                f"chat template failed: {e!r}", "invalid_request_error", 400),
                status_code=400)
        prompt_ids = prompt_token_ids(ctx, body, prompt_text)
        return await serve_completion(ctx, body, "chat", prompt_text,
                                      prompt_ids or [])

    # -------------------------------------------------------------- shutdown
    @app.on_event("shutdown")
    async def _shutdown():
        await engine.shutdown()

    return app


def _register_engine_gauges(metrics: Metrics, engine: AsyncLLMEngine):
    eng = engine.engine
    metrics.set_gauge_fn("scheduler_running_sequences",
                         lambda: len(eng.scheduler.running))
    metrics.set_gauge_fn("scheduler_waiting_sequences",
                         lambda: len(eng.scheduler.waiting))
    metrics.set_gauge_fn("scheduler_preemptions_total",
                         lambda: eng.scheduler.num_preemptions)
    metrics.set_gauge_fn("kv_blocks_total",
                         lambda: eng.block_manager.num_blocks)
    metrics.set_gauge_fn("kv_blocks_free",
                         lambda: eng.block_manager.num_free_blocks)
    metrics.set_gauge_fn("kv_blocks_used",
                         lambda: eng.block_manager.num_used_blocks())
    metrics.set_gauge_fn("kv_blocks_reserved",
                         lambda: eng.block_manager.total_reserved_blocks)
    metrics.set_gauge_fn(
        "kv_cache_utilization",
        lambda: eng.block_manager.num_used_blocks()
        / max(1, eng.block_manager.num_blocks))
    metrics.set_gauge_fn("prefix_cache_hits_total",
                         lambda: eng.block_manager.cache_hits)
    metrics.set_gauge_fn("prefix_cache_queries_total",
                         lambda: eng.block_manager.cache_queries)


def main():
    ap = argparse.ArgumentParser(description="mini-vLLM OpenAI API server")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-pending-requests", type=int, default=256)
    ap.add_argument("--request-timeout", type=float, default=None,
                    help="abort requests still generating after this many "
                         "seconds")
    ap.add_argument("--shutdown-grace-period", type=float, default=10.0)
    ap.add_argument("--api-key", default=None,
                    help="require 'Authorization: Bearer <key>'")
    ap.add_argument("--allow-cors", action="store_true",
                    help="enable permissive CORS (off by default)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    # one process per GPU engine -- multiple uvicorn workers would load the
    # model multiple times; enforced here deliberately
    config = EngineConfig(
        model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, block_size=args.block_size,
        max_num_batched_tokens=args.max_num_batched_tokens)
    engine = AsyncLLMEngine(config,
                            max_pending_requests=args.max_pending_requests)

    app = create_app(engine, model_name=args.model,
                     request_timeout=args.request_timeout,
                     api_key=args.api_key, enable_cors=args.allow_cors)

    @app.on_event("startup")
    async def _start():
        await engine.start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1,
                timeout_graceful_shutdown=int(args.shutdown_grace_period))


if __name__ == "__main__":
    main()
