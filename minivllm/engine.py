"""LLMEngine: request intake + the step() loop that ties everything together.

    add_request() -> Sequence (with n>1 forks planned)
    step():
        scheduler.schedule()          # continuous batching decision
        build flat varlen batch       # all new tokens of all scheduled seqs
        model.forward()               # paged attention over the KV pool
        sample + append tokens        # one token per sequence per step
        register filled KV blocks     # prefix cache bookkeeping
        fork parallel-sampling children after the parent's first token
        finish checks (EOS / max_tokens / stop strings)
    generate(): add all requests, loop step() until everything is done.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence as SeqT, Union

import torch

from minivllm.attention import SeqInput
from minivllm.block_manager import BlockSpaceManager
from minivllm.config import (EngineConfig, ModelConfig, dtype_bytes,
                             estimate_num_blocks, resolve_device, resolve_dtype)
from minivllm.model import Qwen2ForCausalLM
from minivllm.sampling import Sampler
from minivllm.scheduler import Scheduler
from minivllm.sequence import RequestOutput, SamplingParams, Sequence, SequenceStatus


@dataclass
class RequestGroup:
    request_id: int
    prompt: str
    params: SamplingParams
    main: Sequence
    pending_forks: List[Sequence] = field(default_factory=list)


class LLMEngine:
    def __init__(self, config: Optional[EngineConfig] = None,
                 model: Optional[Qwen2ForCausalLM] = None,
                 model_config: Optional[ModelConfig] = None,
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
            enable_prefix_caching=config.enable_prefix_caching)
        self.scheduler = Scheduler(self.block_manager, config.max_num_seqs,
                                   config.max_num_batched_tokens)
        self.sampler = Sampler(config.seed)

        self.groups: Dict[int, RequestGroup] = {}
        self.seq_to_group: Dict[int, RequestGroup] = {}
        self._next_request_id = 0

        kv_gb = self.block_manager.pool.num_bytes() / 1024 ** 3
        print(f"[mini-vllm] KV pool: {num_blocks} blocks x {config.block_size} tokens = "
              f"{num_blocks * config.block_size} tokens, {kv_gb:.2f} GB | "
              f"kv {self.model_config.kv_bytes_per_token(self.dtype)} B/token "
              f"({self.model_config.kv_bytes_per_token(self.dtype) * config.block_size} B/block)")

    # ------------------------------------------------------------- loading
    def _load_model(self):
        from transformers import AutoModelForCausalLM
        cfg = self.model_config
        hf = AutoModelForCausalLM.from_pretrained(
            self.config.model, torch_dtype=self.dtype, attn_implementation="eager")
        gen_eos = getattr(hf.generation_config, "eos_token_id", None)
        self._gen_eos_ids = set(gen_eos if isinstance(gen_eos, list) else
                                ([gen_eos] if gen_eos is not None else []))
        model = Qwen2ForCausalLM(cfg, self.device, self.dtype)
        model.load_from_hf(hf)
        del hf
        model.to(self.device)
        model.eval()
        return model, cfg

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.config.model)

    # ------------------------------------------------------------ requests
    def add_request(self, prompt: Union[str, List[int]],
                    params: Optional[SamplingParams] = None) -> int:
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
        group = RequestGroup(request_id, prompt if isinstance(prompt, str) else "", params, main)
        # parallel sampling: children are forked after the parent's first token
        for _ in range(params.n - 1):
            child = Sequence(token_ids, params, arrival_time=now)
            child.parent_seq_id = main.seq_id
            group.pending_forks.append(child)
        self.groups[request_id] = group
        self.seq_to_group[main.seq_id] = group
        for child in group.pending_forks:
            self.seq_to_group[child.seq_id] = group
        self.scheduler.add(main)
        return request_id

    # ------------------------------------------------------------ stepping
    @torch.no_grad()
    def step(self) -> List[RequestOutput]:
        """One engine iteration. Returns requests that finished in this step."""
        sched = self.scheduler.schedule()
        if not sched.scheduled:
            return []

        # ---- build the flat varlen batch
        device = self.device
        input_ids, positions, seq_inputs, logit_idx = [], [], [], []
        t = 0
        for seq in sched.scheduled:
            start = seq.num_computed_tokens
            end = seq.num_tokens
            input_ids.extend(seq.tokens[start:end])
            positions.extend(range(start, end))
            seq_inputs.append(SeqInput(
                q_start=start, q_len=end - start,
                block_table=torch.tensor(seq.block_table, dtype=torch.long, device=device),
                t0=t))
            logit_idx.append(t + (end - start) - 1)   # last new token
            t += end - start

        logits = self.model(
            torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(positions, dtype=torch.long, device=device),
            self.block_manager.pool, seq_inputs,
            torch.tensor(logit_idx, dtype=torch.long, device=device))

        finished: List[RequestOutput] = []
        for i, seq in enumerate(sched.scheduled):
            seq.num_computed_tokens = seq.num_tokens
            # prefix cache: blocks that just became full are now valid to share
            self.block_manager.register_filled_blocks(seq, seq.num_computed_tokens)

            p = seq.sampling_params
            token = self.sampler.sample(logits[i], p.temperature, p.top_k, p.top_p)
            seq.output_token_ids.append(token)
            seq.record_first_token()
            seq.last_token_time = time.perf_counter()

            self._maybe_fork_children(seq)

            done, reason = self._check_stop(seq)
            if done:
                seq.status = reason
                self.block_manager.free_sequence(seq)
                out = self._build_output(seq)
                if out is not None:
                    finished.append(out)
        return finished

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
            self.scheduler.add(child)
        group.pending_forks = []

    def _check_stop(self, seq: Sequence):
        p = seq.sampling_params
        if len(seq.output_token_ids) >= p.max_tokens:
            return True, SequenceStatus.FINISHED_LENGTH
        if not p.ignore_eos and seq.output_token_ids[-1] in self.eos_token_ids:
            return True, SequenceStatus.FINISHED_STOPPED
        if p.stop and self.tokenizer is not None:
            text = self.tokenizer.decode(seq.output_token_ids)
            if any(s in text for s in p.stop):
                return True, SequenceStatus.FINISHED_STOPPED
        return False, seq.status

    def _group_of(self, seq: Sequence) -> Optional[RequestGroup]:
        return self.seq_to_group.get(seq.seq_id)

    def _build_output(self, seq: Sequence) -> Optional[RequestOutput]:
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
                      "ttft": seq.get_ttft(), "tpot": seq.get_tpot()}],
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
    def generate(self, prompts: SeqT[Union[str, List[int]]],
                 params: Union[SamplingParams, List[SamplingParams], None] = None,
                 use_tqdm: bool = True) -> List[RequestOutput]:
        """Blocking batch generation with continuous batching under the hood."""
        from tqdm import tqdm
        n = len(prompts)
        if not isinstance(params, list):
            params = [params or SamplingParams()] * n
        assert len(params) == n
        for prompt, p in zip(prompts, params):
            self.add_request(prompt, p)

        results: Dict[int, RequestOutput] = {}
        bar = tqdm(total=n, desc="generating", ncols=100, disable=not use_tqdm)
        while self.scheduler.has_unfinished():
            for out in self.step():
                prev = results.get(out.request_id)
                if prev is None:
                    results[out.request_id] = out
                else:                       # merge parallel-sampling children
                    prev.outputs.extend(out.outputs)
                bar.update(1)
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
        }
