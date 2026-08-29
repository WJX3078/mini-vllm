"""SpeculativeEngine: target model + drafter, draft-then-verify loop."""
import time
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from minivllm.config import EngineConfig, ModelConfig, resolve_device, resolve_dtype
from minivllm.engine import LLMEngine
from minivllm.model import Qwen2ForCausalLM
from minivllm.sampling import probs_from_logits, sample_from_logits
from minivllm.sequence import SamplingParams, Sequence, SequenceStatus
from minivllm.spec.drafters import ModelDrafter, NGramDrafter
from minivllm.spec.worker import KVWorker


@dataclass
class SpecOutput:
    request_id: int
    text: str
    token_ids: List[int]
    ttft: float = 0.0
    tpot: float = 0.0
    # speculative stats
    num_rounds: int = 0
    num_proposed: int = 0
    num_accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted / self.num_proposed if self.num_proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        return (self.num_accepted + self.num_rounds) / self.num_rounds \
            if self.num_rounds else 0.0


class SpeculativeEngine:
    """Target model + drafter. Processes one request at a time (no continuous
    batching -- combining both is future work); throughput scales via the
    acceptance rate instead.

    drafter: "ngram" (default, no second model) or a HF model path to use as
             a small draft model. If the path equals the target's, weights are
             shared (useful for correctness testing).
    """

    def __init__(self, config: Optional[EngineConfig] = None,
                 drafter: Union[str] = "ngram",
                 num_spec_tokens: int = 4,
                 model: Optional[Qwen2ForCausalLM] = None,
                 model_config: Optional[ModelConfig] = None,
                 tokenizer=None,
                 **overrides):
        self.config = config = config or EngineConfig()
        for k, v in overrides.items():
            setattr(config, k, v)
        self.num_spec_tokens = num_spec_tokens

        # target reuses the regular engine's parts (model / pool / tokenizer)
        self.target = LLMEngine(config, model=model, model_config=model_config,
                                tokenizer=tokenizer)
        self.dtype = self.target.dtype
        self.device = self.target.device
        mc = self.target.model_config
        self.model_config = mc
        self.tokenizer = self.target.tokenizer

        # ---- drafter setup
        self.ngram_drafter: Optional[NGramDrafter] = None
        self.model_drafter: Optional[ModelDrafter] = None
        if drafter == "ngram":
            self.ngram_drafter = NGramDrafter(window=3)
            self.draft_worker = None
        else:
            draft_model = self._load_drafter_model(drafter)
            draft_cfg = draft_model[1] if isinstance(draft_model, tuple) else mc
            dm = draft_model[0] if isinstance(draft_model, tuple) else draft_model
            worker = KVWorker(dm, num_layers=draft_cfg.num_layers,
                              num_kv_heads=draft_cfg.num_kv_heads,
                              head_dim=draft_cfg.head_dim,
                              num_blocks=config.max_model_len // config.block_size + 8,
                              block_size=config.block_size,
                              dtype=self.dtype, device=self.device)
            self.draft_worker = worker
            self.model_drafter = ModelDrafter(worker)

        self._next_request_id = 0
        self.sampler = self.target.sampler

    def _load_drafter_model(self, path: str):
        if path in ("(same)", self.config.model):
            # share the target's weights (useful for tests / self-drafting)
            return self.target.model, self.target.model_config
        from transformers import AutoModelForCausalLM
        cfg = ModelConfig.from_pretrained(path)
        hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=self.dtype,
                                                  attn_implementation="eager")
        model = Qwen2ForCausalLM(cfg, self.device, self.dtype)
        model.load_from_hf(hf)
        del hf
        return model, cfg

    # ------------------------------------------------------------- sampling
    def _accept_and_bonus(self, proposals: List[int], q_probs: List[torch.Tensor],
                          target_logits: torch.Tensor, params: SamplingParams):
        """Returns (k_accepted, bonus_token). Greedy: compare argmaxes.
        Sampling: rejection sampling that preserves the target distribution."""
        gamma = len(proposals)
        if params.temperature == 0.0:
            greedy = torch.argmax(target_logits.cpu(), dim=-1)   # [gamma+1]
            k = 0
            for i in range(gamma):
                if int(greedy[i].item()) == proposals[i]:
                    k += 1
                else:
                    break
            return k, int(greedy[k].item())

        p_all = probs_from_logits(target_logits.cpu(), params.temperature,
                                  params.top_k, params.top_p)   # [gamma+1, V]
        g = self.sampler.generator
        for i in range(gamma):
            p_i, q_i = p_all[i], q_probs[i].cpu()
            ratio = min(1.0, p_i[proposals[i]].item() / q_i[proposals[i]].item())
            r = torch.rand((), generator=g).item()
            if r < ratio:
                continue
            residual = (p_i - q_i).clamp(min=0)
            residual = residual / residual.sum()
            return i, int(torch.multinomial(residual, 1, generator=g).item())
        # all proposals accepted: the bonus comes from p at the last position
        return gamma, int(torch.multinomial(p_all[gamma], 1, generator=g).item())

    # ------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, prompts, params: Union[SamplingParams, List[SamplingParams], None] = None,
                 use_tqdm: bool = True) -> List[SpecOutput]:
        from tqdm import tqdm
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        n = len(prompts)
        if not isinstance(params, list):
            params = [params or SamplingParams()] * n
        assert len(params) == n
        assert all(p.n == 1 for p in params), "speculative decoding supports n=1"

        outs = []
        bar = tqdm(total=n, desc="spec-generating", ncols=100, disable=not use_tqdm)
        for prompt, p in zip(prompts, params):
            outs.append(self._generate_one(prompt, p))
            bar.update(1)
        bar.close()
        return outs

    def _generate_one(self, prompt: Union[str, List[int]], p: SamplingParams) -> SpecOutput:
        t0 = time.perf_counter()
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer(prompt).input_ids
        else:
            prompt_ids = list(prompt)
        if len(prompt_ids) + p.max_tokens > self.config.max_model_len:
            raise ValueError("prompt + max_tokens exceeds max_model_len")

        tseq = Sequence(prompt_ids, p, arrival_time=t0)
        self.target.block_manager.allocate_sequence(tseq)

        use_ngram = self.ngram_drafter is not None
        dseq = None
        if not use_ngram:
            dseq = Sequence(prompt_ids, p, arrival_time=t0)
            self.draft_worker.allocate(dseq)

        # prefill: target computes prompt[:-1]; prompt[-1] is the first
        # "bonus" token fed together with the first proposals (uniform round)
        if len(prompt_ids) > 1:
            self._target_span(tseq, 0, len(prompt_ids) - 1, register=True)
        if use_ngram:
            self.ngram_drafter.reset()
            self.ngram_drafter.sync(tseq.tokens)
        else:
            # draft prefills prompt[:-1]; its propose() re-forwards the last
            # context token to obtain the first proposal's logits
            self.draft_worker.forward_span(dseq, 0, max(0, len(prompt_ids) - 1))

        out = SpecOutput(request_id=self._next_request_id, text="", token_ids=[])
        self._next_request_id += 1

        gamma = self.num_spec_tokens
        while len(tseq.output_token_ids) < p.max_tokens:
            out.num_rounds += 1
            accepted_len = len(tseq.tokens)          # all verified so far
            bonus = tseq.tokens[-1]                  # pending token (not in KV)
            assert tseq.num_computed_tokens == accepted_len - 1, \
                "target KV frontier drifted from the accepted stream"

            # ---- draft proposals
            room = p.max_tokens - len(tseq.output_token_ids)
            base = len(tseq.output_token_ids)
            if use_ngram:
                proposals = self.ngram_drafter.propose(tseq.tokens, min(gamma, room - 1))
                q_probs = []
            else:
                proposals, q_probs = self.model_drafter.propose(
                    dseq, accepted_len, min(gamma, max(1, room - 1)),
                    p.temperature, p.top_k, p.top_p,
                    generator=self.sampler.generator)

            # pre-append proposals so the verify forward can read their ids
            # (rolled back below for whatever the target rejects)
            tseq.output_token_ids.extend(proposals)

            # ---- verify: [bonus] + proposals in ONE target forward
            span_end = accepted_len + len(proposals)     # positions [accepted-1, span_end)
            logits = self._target_span(tseq, accepted_len - 1, span_end)
            out.num_proposed += len(proposals)

            k, bonus_tok = self._accept_and_bonus(proposals, q_probs, logits, p)
            out.num_accepted += k

            # ---- commit: keep k accepted proposals, drop rejected, add bonus
            del tseq.output_token_ids[base + k:]
            tseq.output_token_ids.append(bonus_tok)      # always fits: |props| <= room-1
            for tok in proposals[:k] + [bonus_tok]:
                tseq.record_first_token()
                tseq.last_token_time = time.perf_counter()

            # KV frontier: accepted proposals are computed (verify pass);
            # the bonus token is pending until the next round feeds it
            tseq.num_computed_tokens = accepted_len + k
            self.target.block_manager.register_filled_blocks(
                tseq, tseq.num_computed_tokens)
            if use_ngram:
                self.ngram_drafter.sync(tseq.tokens)
            else:
                # mirror the accepted stream into the draft sequence so its
                # next sync (rewind/extend) sees the bonus token it must prefill
                dseq.output_token_ids = list(tseq.output_token_ids)

            # ---- stop conditions
            last = tseq.output_token_ids[-1]
            if last in self.target.eos_token_ids and not p.ignore_eos:
                tseq.status = SequenceStatus.FINISHED_STOPPED
                break
            if len(tseq.output_token_ids) >= p.max_tokens:
                tseq.status = SequenceStatus.FINISHED_LENGTH
                break

        text = ""
        if self.tokenizer is not None:
            text = self.tokenizer.decode(tseq.output_token_ids, skip_special_tokens=True)
            for s in p.stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
        out.text = text
        out.token_ids = list(tseq.output_token_ids)
        out.ttft = tseq.get_ttft()
        out.tpot = tseq.get_tpot()

        self.target.block_manager.free_sequence(tseq)
        if dseq is not None:
            self.draft_worker.free(dseq)
        return out

    def _target_span(self, seq, start: int, end: int, register: bool = False):
        return self.target.model_forward_span(seq, start, end, register=register)

    def engine_stats(self) -> dict:
        return self.target.engine_stats()
