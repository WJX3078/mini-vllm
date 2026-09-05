"""AsyncLLMEngine: the single-owner async facade over the sync engine.

Design (see docs/V0_4_SERVING_DESIGN.md section 5):

* ONE dedicated engine thread owns the LLMEngine and is the only place
  where `step()` runs -- GPU work is strictly serial and never blocks the
  event loop.
* The event loop talks to the thread through a thread-safe input queue
  (`queue.Queue`, bounded -> backpressure) and a cancellation set.
* The thread pushes each request's deltas into a per-request
  `asyncio.Queue` via `loop.call_soon_threadsafe` -- no polling anywhere.
* Idle behaviour: with no unfinished work the thread blocks on a
  `threading.Event` (woken by new requests / shutdown) -- no busy spin.
* Shutdown: stop accepting -> drain or cancel active requests within the
  grace period -> stop the loop -> join the thread.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum

from minivllm.config import EngineConfig
from minivllm.engine import LLMEngine
from minivllm.sampling import derive_seed
from minivllm.sequence import SamplingParams


class QueueFullError(RuntimeError):
    """Input queue at capacity -- the HTTP layer maps this to 429."""


class EngineState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"
    UNHEALTHY = "unhealthy"


@dataclass
class StreamingRequest:
    """Server-side view of one request (see design doc section 4)."""

    request_id: str
    params: SamplingParams
    arrival_time: float
    state: str = "queued"                 # queued/running/finished/cancelled
    first_token_at: float | None = None
    finished_at: float | None = None
    internal_id: int | None = None
    output_queue: asyncio.Queue | None = None
    prompt_len: int = 0


@dataclass
class _EngineCommand:
    kind: str                             # "add" | "abort" | "shutdown"
    request: StreamingRequest | None = None
    params: SamplingParams | None = None
    prompt: list[int] | None = None
    target_request_id: str | None = None
    reply: queue.Queue = field(default_factory=queue.Queue)


class AsyncLLMEngine:
    """Async facade: HTTP handlers await; one thread drives the GPU."""

    def __init__(self, config: EngineConfig, max_pending_requests: int = 256,
                 max_queue_deltas: int = 256, engine: LLMEngine | None = None):
        self.config = config
        self.max_pending_requests = max_pending_requests
        self.max_queue_deltas = max_queue_deltas
        # `engine` injection is used by tests with tiny random weights;
        # production usage loads from config.model
        self.engine = engine or LLMEngine(config)
        self.metrics_hook = None          # set by the server (callable)

        self.state = EngineState.STARTING
        self.requests: dict[str, StreamingRequest] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cmd_queue: queue.Queue[_EngineCommand] = queue.Queue()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_adds = 0            # queued-but-not-yet-admitted
        self._slow_client_cancellations = 0
        self._unhealthy_reason: str | None = None
        self._loop_iterations = 0         # observability for idle behaviour

    # ------------------------------------------------------------ lifecycle
    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._engine_loop,
                                        name="minivllm-engine",
                                        daemon=True)
        self._thread.start()
        self.state = EngineState.RUNNING

    async def shutdown(self, grace_period: float = 5.0):
        """Graceful shutdown: stop accepting -> wait for active requests
        within the grace period -> abort whatever remains -> give the
        engine thread time to emit the final deltas -> stop the loop and
        join the thread."""
        if self.state == EngineState.STOPPED:
            return
        self.state = EngineState.DRAINING      # /ready flips false
        deadline = time.perf_counter() + grace_period
        while self.requests and time.perf_counter() < deadline:
            await asyncio.sleep(0.05)          # natural drain
        for rid in list(self.requests):        # cancel the remainder
            await self.abort_request(rid)
        # let the engine thread deliver the final abort deltas to streams
        deadline = time.perf_counter() + 2.0
        while self.requests and time.perf_counter() < deadline:
            await asyncio.sleep(0.02)
        self.state = EngineState.STOPPED
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._thread = None
        self.requests.clear()          # abandoned streams (clients gone)

    # ----------------------------------------------------------- public API
    async def add_request(self, prompt: list[int] | str,
                          params: SamplingParams) -> str:
        """Register a request; returns the external request id.

        Raises QueueFullError when the input queue is full (HTTP 429)."""
        if self.state not in (EngineState.RUNNING, EngineState.DRAINING):
            raise EngineUnhealthyError(self._unhealthy_reason
                                       or "engine not accepting requests")
        if self._pending_adds + len(self.requests) >= self.max_pending_requests:
            raise QueueFullError("server overloaded")
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        req = StreamingRequest(
            request_id=rid, params=params, arrival_time=time.perf_counter(),
            output_queue=asyncio.Queue(maxsize=self.max_queue_deltas),
            prompt_len=len(prompt) if isinstance(prompt, list) else -1)
        self.requests[rid] = req
        self._pending_adds += 1
        self._cmd_queue.put(_EngineCommand(kind="add", request=req,
                                           params=params, prompt=prompt))
        self._wake.set()
        return rid

    async def abort_request(self, request_id: str):
        """Cancel a request. The registry entry is kept until the engine's
        final "abort" delta reaches the stream (the stream generator pops
        it); aborting a finished/unknown id is a harmless no-op."""
        req = self.requests.get(request_id)
        if req is not None:
            req.state = "cancelled"
        self._cmd_queue.put(_EngineCommand(kind="abort",
                                           target_request_id=request_id))
        self._wake.set()

    async def stream(self, request_id: str) -> AsyncIterator:
        """Async iterator of delta dicts for an already-added request:
        {token_ids, text, finished, finish_reason, sample_idx}. Text comes
        from the incremental detokenizer. On early generator exit (client
        disconnect) the request is aborted and cleaned up."""
        from minivllm.serving.detokenizer import IncrementalDetokenizer
        req = self.requests.get(request_id)
        if req is None:
            return
        detok = IncrementalDetokenizer(self.engine.tokenizer)
        finished = False
        try:
            while True:
                delta = await req.output_queue.get()
                text = detok.push(delta.token_ids, final=delta.finished)
                finished = delta.finished
                yield {"request_id": request_id, "token_ids": delta.token_ids,
                       "text": text, "finished": delta.finished,
                       "finish_reason": delta.finish_reason,
                       "sample_idx": delta.sample_idx}
                if delta.finished:
                    return
        finally:
            self.requests.pop(request_id, None)
            if not finished:
                # consumer went away (GeneratorExit / timeout): stop the GPU
                self._cmd_queue.put(_EngineCommand(
                    kind="abort", target_request_id=request_id))
                self._wake.set()

    async def generate(self, prompt, params: SamplingParams) -> AsyncIterator:
        """Convenience: add_request + stream."""
        rid = await self.add_request(prompt, params)
        async for d in self.stream(rid):
            yield d

    async def generate_collect(self, prompt, params: SamplingParams) -> dict:
        """Convenience for non-streaming handlers: collect all deltas."""
        result: dict = {"request_id": None, "token_ids": [], "text": "",
                        "finished": False, "finish_reason": None,
                        "outputs": {}}
        async for d in self.generate(prompt, params):
            result["request_id"] = d["request_id"]
            if d["sample_idx"] not in result["outputs"]:
                result["outputs"][d["sample_idx"]] = {"token_ids": [],
                                                      "text": ""}
            acc = result["outputs"][d["sample_idx"]]
            acc["token_ids"].extend(d["token_ids"])
            acc["text"] += d["text"]
            result["finished"] = d["finished"]
            result["finish_reason"] = d["finish_reason"]
        return result

    # ------------------------------------------------------------- getters
    def is_ready(self) -> bool:
        return self.state == EngineState.RUNNING

    def is_healthy(self) -> bool:
        return self.state != EngineState.UNHEALTHY

    # -------------------------------------------------------- engine thread
    def _engine_loop(self):
        """The ONLY place engine.step() is ever called."""
        try:
            while self.state != EngineState.STOPPED:
                self._loop_iterations += 1
                self._drain_commands()
                # flush deltas on EVERY path: abort deltas are produced
                # outside step() and may be the only thing pending
                self._dispatch_deltas()
                if self.engine.scheduler.has_unfinished() or \
                        self.engine.scheduler.waiting:
                    try:
                        self.engine.step()
                    except Exception as e:            # fatal engine error
                        self._handle_engine_failure(e)
                        return
                    self._dispatch_deltas()
                elif self._pending_adds:
                    # new requests arrived but scheduled nothing (e.g. KV
                    # admission blocked) -- still make progress next round
                    self._wake.wait(timeout=0.01)
                    self._wake.clear()
                else:
                    # idle: block until a command arrives (no busy spin)
                    self._wake.wait(timeout=0.5)
                    self._wake.clear()
        except Exception as e:                        # pragma: no cover
            self._unhealthy_reason = f"engine thread crashed: {e!r}"
            self.state = EngineState.UNHEALTHY

    def _drain_commands(self):
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                return
            if cmd.kind == "shutdown":
                return
            if cmd.kind == "abort":
                iid = self._internal_id(cmd.target_request_id)
                print(f"[async] abort cmd for {cmd.target_request_id} "
                      f"-> internal {iid}")
                self.engine.abort_request(iid if iid is not None else -1)
                continue
            # add
            req = cmd.request
            try:
                prompt = cmd.prompt
                if isinstance(prompt, str) and self.engine.tokenizer:
                    prompt = self.engine.tokenizer(prompt).input_ids
                internal = self.engine.add_request(prompt or [], cmd.params)
                req.internal_id = internal
                req.state = "running"
                self._pending_adds = max(0, self._pending_adds - 1)
            except Exception as e:                # per-request failure
                self._pending_adds = max(0, self._pending_adds - 1)
                req.state = "failed"
                self._fail_request(req, f"admission failed: {e!r}")

    def _internal_id(self, external_id: str) -> int | None:
        req = self.requests.get(external_id)
        return req.internal_id if req else None

    def _dispatch_deltas(self):
        deltas = self.engine.pop_deltas()
        if not deltas or self._loop is None:
            return
        for d in deltas:
            ext = self._external_id(d.request_id)
            req = self.requests.get(ext) if ext else None
            if req is None:
                continue                    # aborted between step and dispatch
            if req.first_token_at is None and d.token_ids:
                req.first_token_at = time.perf_counter()
            payload = _DeltaPayload(d)
            try:
                self._loop.call_soon_threadsafe(
                    self._safe_put, req, payload)
            except RuntimeError:            # loop closed during shutdown
                return

    def _safe_put(self, req: StreamingRequest, payload):
        """Deliver a delta; a chronically full queue means the client is
        too slow to consume its own stream -- cancel it instead of
        blocking the GPU on its behalf."""
        try:
            req.output_queue.put_nowait(payload)
        except asyncio.QueueFull:
            self._slow_client_cancellations += 1
            rid = req.request_id
            self.requests.pop(rid, None)     # nobody will consume this
            self._cmd_queue.put(_EngineCommand(kind="abort",
                                               target_request_id=rid))
            self._wake.set()

    def _external_id(self, internal_id: int) -> str | None:
        for rid, req in self.requests.items():
            if req.internal_id == internal_id:
                return rid
        return None

    def _fail_request(self, req: StreamingRequest, reason: str):
        self.requests.pop(req.request_id, None)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._safe_put, req, _DeltaPayload.failed(reason))

    def _handle_engine_failure(self, error: Exception):
        """Fatal engine error: fail every active request, stop serving."""
        self._unhealthy_reason = f"engine failure: {error!r}"
        self.state = EngineState.UNHEALTHY
        for req in list(self.requests.values()):
            self._fail_request(req, self._unhealthy_reason)
        self.requests.clear()


class _DeltaPayload:
    """Thread -> event loop delta envelope."""

    __slots__ = ("token_ids", "finished", "finish_reason", "sample_idx",
                 "error")

    def __init__(self, d=None, error: str | None = None):
        self.error = error
        if d is None:
            self.token_ids, self.finished = [], True
            self.finish_reason, self.sample_idx = "error", 0
        else:
            self.token_ids = d.token_ids
            self.finished = d.finished
            self.finish_reason = d.finish_reason
            self.sample_idx = d.sample_idx

    @classmethod
    def failed(cls, reason: str):
        p = cls(None, error=reason)
        p.finish_reason = "error"
        return p


class EngineUnhealthyError(RuntimeError):
    pass


def _seed_for_request(base_seed: int, index: int) -> int:
    return derive_seed(base_seed, index)


def default_sampling_params(**kwargs) -> SamplingParams:
    return SamplingParams(**kwargs)
