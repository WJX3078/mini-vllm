"""LLMEngine: request intake + the step() loop that ties everything together.

    add_request() -> Sequence (with n>1 forks planned, per-request RNG seeded)
    step():
        scheduler.schedule()          # unified token budget: decodes first,
                                      # then prefill chunks (chunked prefill)
        build flat varlen batch       # each seq's [span_start, span_end) tokens
        model.forward()               # paged attention over the KV pool
        batched sampling              # grouped by sampling config, <=1 D2H/group
        append tokens                 # mid-prefill chunks do not sample yet
        register filled KV blocks     # prefix cache bookkeeping (also mid-chunk)
        fork parallel-sampling children after the parent's first token
        incremental stop checks       # token-level fast path, windowed decode
    generate(): add all requests, loop step() until everything is done.
"""
import time
from collections.abc import Sequence as SeqT
from dataclasses import dataclass, field

import torch

from minivllm.attention import SeqInput
from minivllm.block_manager import BlockSpaceManager
from minivllm.config import (
    EngineConfig,
    ModelConfig,
    estimate_num_blocks,
    resolve_device,
    resolve_dtype,
)
from minivllm.model import Qwen2ForCausalLM
from minivllm.prefix_hash import make_hash_backend
from minivllm.sampling import derive_seed, sample_tokens
from minivllm.scheduler import Scheduler
from minivllm.sequence import RequestOutput, SamplingParams, Sequence, SequenceStatus
from minivllm.stopping import StopChecker

_FINISH_REASONS = {
    SequenceStatus.FINISHED_STOPPED: "stop",
    SequenceStatus.FINISHED_LENGTH: "length",
    SequenceStatus.FINISHED_ABORTED: "abort",
}


@dataclass
class RequestOutputDelta:
    """Per-request incremental output produced by every `step()`.

    `token_ids` holds the tokens newly generated THIS step for one
    sequence of the request (`sample_idx` distinguishes parallel-sampling
    children). `finish_reason` is set on the final delta:
    "stop" | "length" | "abort"."""

    request_id: int
    sample_idx: int
    token_ids: list[int]
    finished: bool = False
    finish_reason: str | None = None


@dataclass
class RequestGroup:
    request_id: int
    prompt: str
    params: SamplingParams
    main: Sequence
    pending_forks: list[Sequence] = field(default_factory=list)
    stop_checker: StopChecker | None = None
    # all sequences of this request (children appended when forked) and how
    # many are not finished yet -- the group leaves the registry only when
    # the counter hits zero (v0.4: long-serving registry leak fix)
    seq_ids: list[int] = field(default_factory=list)
    children: list[Sequence] = field(default_factory=list)
    active_seqs: int = 0

    def register(self):
        self.active_seqs = 1 + len(self.pending_forks)
        self.seq_ids = [self.main.seq_id] + [c.seq_id
                                             for c in self.pending_forks]


class LLMEngine:
    def __init__(self, config: EngineConfig | None = None,
                 model: Qwen2ForCausalLM | None = None,
                 model_config: ModelConfig | None = None,
                 tokenizer=None,
                 **overrides):
        """`model`/`tokenizer` injection is used by tests with random tiny
        weights; production usage loads from `config.model`."""
        config = config or EngineConfig()
        for k, v in overrides.items():
            assert hasattr(config, k), f"unknown EngineConfig field {k}"
            setattr(config, k, v)
        self.config = config

        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype, self.device)
        torch.manual_seed(config.seed)

        if model is not None:
            assert model_config is not None
            self.model = model
            self.model_config = model_config
            self.tokenizer = tokenizer
            eos_ids = {model_config.eos_token_id}
        else:
            print(f"[mini-vllm] loading {config.model} on {self.device} ({self.dtype}) ...")
            self.model_config = ModelConfig.from_pretrained(config.model)
            self.model, self.model_config = self._load_model()
            self.tokenizer = self._load_tokenizer()
            eos_ids = {self.model_config.eos_token_id} | self._gen_eos_ids
        self.eos_token_ids = {e for e in eos_ids if e is not None}

        num_blocks = estimate_num_blocks(self.model_config, config, self.dtype, self.device)
        self.block_manager = BlockSpaceManager(
            num_blocks=num_blocks, block_size=config.block_size,
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
            dtype=self.dtype, device=self.device,
            enable_prefix_caching=config.enable_prefix_caching,
            hash_backend=make_hash_backend(config.hash_backend),
            hash_metadata=f"model={config.model}")
        self.scheduler = Scheduler(self.block_manager, config.max_num_seqs,
                                   config.max_num_batched_tokens,
                                   enable_chunked_prefill=config.enable_chunked_prefill,
                                   reserve_full_isl=config.scheduler_reserve_full_isl,
                                   lazy_allocation=config.lazy_block_allocation)

        # v0.3 runtime-overhead state: persistent (pinned staging, device)
        # metadata buffer pairs, grown on demand and keyed by tensor kind;
        # plus optional per-phase step timing (set `step_timings` to a dict
        # to profile; None = zero overhead)
        self.step_timings: dict | None = None
        self._buffers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # v0.4: per-step incremental outputs, consumed via pop_deltas()
        self._deltas: list[RequestOutputDelta] = []

        self.groups: dict[int, RequestGroup] = {}
        self.seq_to_group: dict[int, RequestGroup] = {}
        self._next_request_id = 0

        kv_gb = self.block_manager.pool.num_bytes() / 1024 ** 3
        print(f"[mini-vllm] KV pool: {num_blocks} blocks x {config.block_size} tokens = "
              f"{num_blocks * config.block_size} tokens, {kv_gb:.2f} GB | "
              f"kv {self.model_config.kv_bytes_per_token(self.dtype)} B/token "
              f"({self.model_config.kv_bytes_per_token(self.dtype) * config.block_size} B/block) | "
              f"chunked_prefill={config.enable_chunked_prefill}")

    # ------------------------------------------------------------- loading
    def _load_model(self):
        from transformers import AutoModelForCausalLM
        cfg = self.model_config
        hf = AutoModelForCausalLM.from_pretrained(
            self.config.model, torch_dtype=self.dtype, attn_implementation="eager")
        gen_eos = getattr(hf.generation_config, "eos_token_id", None)
        self._gen_eos_ids = set(gen_eos if isinstance(gen_eos, list) else
                                ([gen_eos] if gen_eos is not None else []))
        model = Qwen2ForCausalLM(cfg, self.device, self.dtype,
                                 attention_backend=self.config.attention_backend)
        model.load_from_hf(hf)
        del hf
        model.to(self.device)
        model.eval()
        return model, cfg

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.config.model)

    # ------------------------------------------------------------ requests
    def add_request(self, prompt: str | list[int],
                    params: SamplingParams | None = None) -> int:
        params = params or SamplingParams()
        if isinstance(prompt, str):
            token_ids = self.tokenizer(prompt).input_ids
        else:
            token_ids = list(prompt)
        if len(token_ids) + params.max_tokens > self.config.max_model_len:
            raise ValueError(
                f"prompt ({len(token_ids)}) + max_tokens ({params.max_tokens}) "
                f"exceeds max_model_len ({self.config.max_model_len})")

        request_id = self._next_request_id
        self._next_request_id += 1
        now = time.perf_counter()
        main = Sequence(token_ids, params, arrival_time=now)
        group = RequestGroup(request_id, prompt if isinstance(prompt, str) else "",
                             params, main,
                             stop_checker=StopChecker(self.tokenizer, params.stop,
                                                      self.eos_token_ids))
        self._seed_rng(main, request_id, sample_idx=0, params=params,
                       engine_seed=self.config.seed)
        # parallel sampling: children are forked after the parent's first token
        for i in range(params.n - 1):
            child = Sequence(token_ids, params, arrival_time=now)
            child.parent_seq_id = main.seq_id
            self._seed_rng(child, request_id, sample_idx=i + 1, params=params,
                           engine_seed=self.config.seed)
            group.pending_forks.append(child)
        self.groups[request_id] = group
        group.register()
        self.seq_to_group[main.seq_id] = group
        for child in group.pending_forks:
            self.seq_to_group[child.seq_id] = group
        self.scheduler.add(main)
        return request_id

    # ------------------------------------------------- v0.4: abort + deltas
    def pop_deltas(self) -> list[RequestOutputDelta]:
        """Incremental outputs produced by the last `step()` (one delta per
        sampled sequence; finished sequences carry their finish_reason)."""
        deltas, self._deltas = self._deltas, []
        return deltas

    def abort_request(self, request_id: int) -> bool:
        """Cancel a request wherever it is (waiting / running / mid-fork).

        Idempotent: unknown or already-finished ids return False and change
        nothing. Releases every owned KV block and reservation, decrements
        shared prefix refcounts, and drops the group from the registry once
        all of its sequences are done. Emits a final "abort" delta."""
        group = self.groups.get(request_id)
        if group is None:
            return False

        # 1. pending forks still sit in the waiting queue (block-free)
        for child in list(group.pending_forks):
            try:
                self.scheduler.waiting.remove(child)
            except ValueError:
                pass                      # already scheduled/finished
            child.status = SequenceStatus.FINISHED_ABORTED
            group.active_seqs = max(0, group.active_seqs - 1)
        group.pending_forks = []

        # 2. main + forked children: wherever they are, release everything
        for seq in [group.main, *group.children]:
            if seq.is_finished:
                continue
            if seq in self.scheduler.running:
                self.scheduler.running.remove(seq)
            try:
                self.scheduler.waiting.remove(seq)
            except ValueError:
                pass
            # releases blocks, shared refcounts and any reservation
            self.block_manager.free_sequence(seq)
            seq.status = SequenceStatus.FINISHED_ABORTED
            self._deltas.append(RequestOutputDelta(
                request_id, seq.sample_idx, [], finished=True,
                finish_reason="abort"))
            self._release_seq_from_group(group, seq)
        return True

    def _release_seq_from_group(self, group: RequestGroup, seq: Sequence):
        """Book one sequence as done; drop the whole group from the
        registry when its last sequence finishes (no leaks in long
        serving)."""
        group.active_seqs = max(0, group.active_seqs - 1)
        if group.active_seqs == 0:
            self.groups.pop(group.request_id, None)
            for sid in group.seq_ids:
                self.seq_to_group.pop(sid, None)

    @staticmethod
    def _seed_rng(seq: Sequence, request_id: int, sample_idx: int,
                  params: SamplingParams, engine_seed: int = 0):
        """Per-request RNG seed (stable integer mix, never salted hash()).

        * user seed given: the stream is a pure function of (seed, sample_idx)
          -- vLLM semantics: the same seeded request reproduces the same
          output no matter when/how it was submitted; parallel-sampling
          children (sample_idx) still get independent streams.
        * no user seed: derive from (engine_seed, request_id, sample_idx) so
          every request owns a stream regardless of batch composition.
        """
        seq.sample_idx = sample_idx
        if params.seed is not None:
            seq.rng_seed = derive_seed(params.seed, sample_idx)
        else:
            seq.rng_seed = derive_seed(engine_seed, request_id, sample_idx)

    # ------------------------------------------------------------ stepping
    @torch.no_grad()
    def step(self) -> list[RequestOutput]:
        """One engine iteration. Returns requests that finished in this step."""
        prof = self.step_timings              # None unless profiling (below)
        t0 = time.perf_counter() if prof is not None else 0.0
        sched = self.scheduler.schedule()
        if not sched.scheduled:
            return []
        t1 = time.perf_counter() if prof is not None else 0.0

        # ---- build the flat varlen batch from per-sequence token spans
        device = self.device
        input_ids, positions, logit_idx = [], [], []
        sample_entries = []              # (scheduled index, sequence) that sample
        t = 0
        for si, (seq, (start, end)) in enumerate(zip(sched.scheduled, sched.spans)):
            input_ids.extend(seq.tokens[start:end])
            positions.extend(range(start, end))
            if end >= seq.num_prompt_tokens:
                # this forward completes the prompt (or is a decode step), so
                # its last position yields the next-token logits. A mid-prefill
                # chunk needs no logits -- skip lm_head for it.
                logit_idx.append(t + (end - start) - 1)
                sample_entries.append((si, seq))
            t += end - start
        # persistent metadata (v0.3): each tensor goes through a pinned-CPU
        # staging fill + ONE non-blocking H2D copy -- the hot path allocates
        # no fresh device tensors. Block-table row views feed the model.
        tables_t = self._upload_tables([s.block_table
                                        for s in sched.scheduled])
        ids_t = self._upload_1d(input_ids, "ids")
        pos_t = self._upload_1d(positions, "pos")
        logits_t = self._upload_1d(logit_idx, "logits")
        seq_inputs = []
        t = 0
        for si, (seq, (start, end)) in enumerate(zip(sched.scheduled,
                                                     sched.spans)):
            seq_inputs.append(SeqInput(
                q_start=start, q_len=end - start,
                block_table=tables_t[si, :len(seq.block_table)], t0=t))
            t += end - start
        t2 = time.perf_counter() if prof is not None else 0.0

        logits = self.model(ids_t, pos_t, self.block_manager.pool, seq_inputs,
                            logits_t)
        if device == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter() if prof is not None else 0.0

        # ---- batched sampling: <=1 GPU->CPU sync per sampling-config group
        new_tokens = sample_tokens(
            logits, [seq for _, seq in sample_entries])
        t4 = time.perf_counter() if prof is not None else 0.0

        finished: list[RequestOutput] = []
        now = time.perf_counter()
        # advance EVERY scheduled sequence's KV frontier to its span end (a
        # mid-prefill chunk included) and register newly-full blocks -- only
        # prompt-completing sequences sample below
        for seq, (_, end) in zip(sched.scheduled, sched.spans):
            seq.num_computed_tokens = end
            # prefix cache: blocks that just became full are now valid to
            # share (also mid-chunk -- a preempted chunked prefill restores
            # them on re-admission)
            self.block_manager.register_filled_blocks(seq, end)

        for k, (_, seq) in enumerate(sample_entries):
            token = new_tokens[k]
            seq.output_token_ids.append(token)
            seq.record_first_token()
            seq.last_token_time = now

            self._maybe_fork_children(seq)

            group = self._group_of(seq)
            done_reason = None
            checker = group.stop_checker if group else None
            if checker is not None:
                keep = checker.check(seq.output_token_ids,
                                     seq.sampling_params.ignore_eos)
                if keep is not None:
                    if keep < len(seq.output_token_ids):
                        del seq.output_token_ids[keep:]   # stop string hit
                    done_reason = SequenceStatus.FINISHED_STOPPED
            p = seq.sampling_params
            if done_reason is None and len(seq.output_token_ids) >= p.max_tokens:
                done_reason = SequenceStatus.FINISHED_LENGTH
            if done_reason is not None:
                seq.status = done_reason
                self.block_manager.free_sequence(seq)
                out = self._build_output(seq)
                if out is not None:
                    finished.append(out)
            # v0.4: incremental delta for the serving layer
            if group is not None:
                self._deltas.append(RequestOutputDelta(
                    request_id=group.request_id,
                    sample_idx=seq.sample_idx,
                    token_ids=[token],
                    finished=done_reason is not None,
                    finish_reason=_FINISH_REASONS.get(done_reason)))
                if done_reason is not None:
                    self._release_seq_from_group(group, seq)
        if prof is not None:
            t5 = time.perf_counter()
            for key, dt in (("schedule", t1 - t0), ("metadata", t2 - t1),
                            ("forward", t3 - t2), ("sampling", t4 - t3),
                            ("bookkeeping", t5 - t4)):
                prof[key] = prof.get(key, 0.0) + dt
            prof["steps"] = prof.get("steps", 0) + 1
        # keep the running list truthful for serving gauges: finished
        # sequences leave immediately instead of lingering until the next
        # schedule() call
        self.scheduler.running = [s for s in self.scheduler.running
                                  if not s.is_finished]
        return finished

    # --------------------------------------------------- profiling support
    def _buffer_pair(self, kind: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Persistent (pinned staging, device) buffer pair for one metadata
        tensor kind -- separate pairs never alias each other."""
        pair = self._buffers.get(kind)
        if pair is None or pair[0].numel() < n:
            size = max(n, 4096)
            pair = (torch.empty(size, dtype=torch.long,
                                pin_memory=self.device == "cuda"),
                    torch.empty(size, dtype=torch.long, device=self.device))
            self._buffers[kind] = pair
        return pair

    def _upload_1d(self, values: list[int], kind: str) -> torch.Tensor:
        """values -> device int64 tensor: pinned staging fill + ONE
        non-blocking H2D copy; no fresh device allocation in the hot path."""
        n = len(values)
        st, dev = self._buffer_pair(kind, n)
        st[:n].copy_(torch.tensor(values, dtype=torch.long))
        dev[:n].copy_(st[:n], non_blocking=True)
        return dev[:n]

    def _upload_tables(self, block_tables: list[list[int]]) -> torch.Tensor:
        """Per-sequence block tables -> one persistent [S, max_nb] device
        tensor; the model consumes row views."""
        S = len(block_tables)
        max_nb = max((len(t) for t in block_tables), default=1)
        st, dev = self._buffer_pair("tables2d", S * max_nb)
        st2d = st[:S * max_nb].view(S, max_nb)
        dev2d = dev[:S * max_nb].view(S, max_nb)
        for i, table in enumerate(block_tables):
            st2d[i, :len(table)] = torch.tensor(table, dtype=torch.long)
        dev2d.copy_(st2d, non_blocking=True)
        return dev2d



    def _maybe_fork_children(self, seq: Sequence):
        group = self._group_of(seq)
        if group is None or not group.pending_forks:
            return
        if seq is not group.main or len(seq.output_token_ids) != 1:
            return
        for child in group.pending_forks:
            self.block_manager.fork_sequence(seq, child)
            child.output_token_ids = list(seq.output_token_ids)
            child.num_computed_tokens = seq.num_computed_tokens
            child.arrival_time = seq.arrival_time
            child.first_token_time = seq.first_token_time
            group.children.append(child)
            self.scheduler.add(child)
        group.pending_forks = []

    def _group_of(self, seq: Sequence) -> RequestGroup | None:
        return self.seq_to_group.get(seq.seq_id)

    def _build_output(self, seq: Sequence) -> RequestOutput | None:
        g = self._group_of(seq)
        if g is None:
            return None
        out_ids = list(seq.output_token_ids)
        text = "" if self.tokenizer is None else \
            self.tokenizer.decode(out_ids, skip_special_tokens=True)
        if self.tokenizer is not None:
            for s in g.params.stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
        return RequestOutput(
            request_id=g.request_id, prompt=g.prompt,
            outputs=[{"text": text, "token_ids": out_ids,
                      "ttft": seq.get_ttft(), "tpot": seq.get_tpot(),
                      "e2e": (seq.last_token_time or seq.arrival_time)
                      - seq.arrival_time}],
            finished=True)

    # ------------------------------------------------------- span primitive
    @torch.no_grad()
    def model_forward_span(self, seq: Sequence, start: int, end: int,
                           register: bool = False) -> torch.Tensor:
        """Process tokens[start:end] (must continue from the KV frontier) and
        return their logits [end-start, vocab]. Used by speculative decoding."""
        assert start == seq.num_computed_tokens, \
            f"non-contiguous span {start} != computed {seq.num_computed_tokens}"
        if end <= start:
            return torch.empty(0)
        assert self.block_manager.prepare_slots(seq, start, end), "no free KV slot"
        n = end - start
        ids = torch.tensor(seq.tokens[start:end], dtype=torch.long, device=self.device)
        positions = torch.arange(start, end, device=self.device)
        si = SeqInput(q_start=start, q_len=n,
                      block_table=torch.tensor(seq.block_table, dtype=torch.long,
                                               device=self.device), t0=0)
        logits = self.model(ids, positions, self.block_manager.pool, [si],
                            torch.arange(n, device=self.device))
        seq.num_computed_tokens = end
        if register:
            self.block_manager.register_filled_blocks(seq, end)
        return logits

    # ------------------------------------------------------------ high level
    def generate(self, prompts: SeqT[str | list[int]],
                 params: SamplingParams | list[SamplingParams] | None = None,
                 use_tqdm: bool = True) -> list[RequestOutput]:
        """Blocking batch generation with continuous batching under the hood."""
        from tqdm import tqdm
        n = len(prompts)
        if not isinstance(params, list):
            params = [params or SamplingParams()] * n
        assert len(params) == n
        for prompt, p in zip(prompts, params):
            self.add_request(prompt, p)

        results: dict[int, RequestOutput] = {}
        bar = tqdm(total=n, desc="generating", ncols=100, disable=not use_tqdm)
        while self.scheduler.has_unfinished():
            for out in self.step():
                prev = results.get(out.request_id)
                if prev is None:
                    results[out.request_id] = out
                else:                       # merge parallel-sampling children
                    prev.outputs.extend(out.outputs)
                bar.update(1)
            self.pop_deltas()               # sync path ignores deltas
        bar.close()
        return [results[i] for i in sorted(results)]

    # --------------------------------------------------------------- stats
    def engine_stats(self) -> dict:
        bm = self.block_manager
        return {
            "cache_hit_rate": round(bm.cache_hit_rate(), 4),
            "cache_queries": bm.cache_queries,
            "cache_hits": bm.cache_hits,
            "cow_copies": bm.cow_copies,
            "preemptions": self.scheduler.num_preemptions,
            "used_blocks": bm.num_used_blocks(),
            "free_blocks": bm.num_free_blocks,
            "evictable_blocks": bm.num_evictable_blocks(),
            "peak_blocks": bm.alloc_watermark,
            "kv_utilization": round(bm.num_used_blocks() / bm.num_blocks, 4),
            "reserved_blocks": sum(s.reserved_cold_blocks
                                   for s in self.scheduler.running),
        }
