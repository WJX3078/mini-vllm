"""SpeculativeEngine: target model + drafter, draft-then-verify loop.

Correctness laws enforced here:

* Lossless sampling. For a deterministic drafter (n-gram) the proposal is a
  point mass q(x)=1, so the accept probability is min(1, p(x)/q(x)) = p(x)
  and the residual we resample from on rejection is norm(max(p - q, 0)) =
  p with the proposed token's mass removed. For a sampled drafter (small
  model) q is its full distribution. Both paths are the textbook speculative
  sampling scheme: every committed token is distributed exactly like p.
* Truncation at the earliest termination. A verify round commits k accepted
  proposals plus one bonus token; if an EOS (or stop string) sits inside
  that run, everything AFTER it is dropped -- output tokens, the KV
  frontier, prefix-cache registration and the drafter's view of the stream
  all stay consistent with the truncated sequence.
* Per-request RNG. Sampling draws come from the request's own generator
  (seeded in LLMEngine._seed_rng), so speculative output is reproducible
  regardless of what else the engine has sampled.
"""
import time
from dataclasses import dataclass

import torch

from minivllm.config import EngineConfig, ModelConfig
from minivllm.engine import LLMEngine
from minivllm.model import Qwen2ForCausalLM
from minivllm.sampling import filter_logits
from minivllm.sequence import SamplingParams, Sequence, SequenceStatus
from minivllm.spec.drafters import ModelDrafter, NGramDrafter
from minivllm.spec.worker import KVWorker
from minivllm.stopping import StopChecker


@dataclass
class SpecOutput:
    request_id: int
    text: str
    token_ids: list[int]
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

    def __init__(self, config: EngineConfig | None = None,
                 drafter: str = "ngram",
                 num_spec_tokens: int = 4,
                 model: Qwen2ForCausalLM | None = None,
                 model_config: ModelConfig | None = None,
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
        self.ngram_drafter: NGramDrafter | None = None
        self.model_drafter: ModelDrafter | None = None
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
    def _accept_and_bonus(self, proposals: list[int],
                          q_probs: list[torch.Tensor] | None,
                          target_logits: torch.Tensor, params: SamplingParams,
                          generator: torch.Generator):
        """Returns (k_accepted, bonus_token).

        Greedy: compare argmaxes. Sampling: rejection sampling that preserves
        the target distribution exactly.

        q_probs[i] is the drafter's distribution for proposals[i]; ``None``
        means a deterministic proposal (n-gram): q is a point mass on the
        proposed token, q(x)=1, so accept prob = p(x) and the rejection
        residual is p with the proposal's mass removed.
        """
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

        # p_all[i]: target distribution at verify position i  [gamma+1, V]
        p_all = torch.softmax(filter_logits(
            target_logits.cpu(), params.temperature,
            params.top_k, params.top_p), dim=-1)
        g = generator
        for i in range(gamma):
            p_i = p_all[i]
            x = proposals[i]
            if q_probs is not None:
                q_i = q_probs[i].detach().float().cpu()
                q_x = q_i[x].item()
                ratio = 1.0 if q_x <= 0.0 else min(1.0, p_i[x].item() / q_x)
            else:
                q_i = None
                ratio = min(1.0, p_i[x].item())      # q(x) = 1
            r = torch.rand((), generator=g).item()
            if r < ratio:
                continue
            # rejected: resample from norm(max(p - q, 0))
            residual = p_i.clone()
            if q_i is not None:
                residual -= q_i
            else:
                residual[x] = 0.0                    # subtract the point mass
            residual.clamp_(min=0)
            total = residual.sum()
            if total <= 0:
                # p == q (degenerate): sampling from p IS the correct residual
                residual = p_i
            else:
                residual = residual / total
            return i, int(torch.multinomial(residual, 1, generator=g).item())
        # all proposals accepted: the bonus comes from p at the last position
        return gamma, int(torch.multinomial(p_all[gamma], 1, generator=g).item())

    # ------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, prompts, params: SamplingParams | list[SamplingParams] | None = None,
                 use_tqdm: bool = True) -> list[SpecOutput]:
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

    def _round_keep_count(self, candidate_ids: list[int], base: int,
                          committed: list[int], params: SamplingParams,
                          checker: StopChecker | None) -> tuple:
        """How many of this round's committed tokens survive?

        Scans accepted proposals + bonus in order and cuts at the EARLIEST
        termination: EOS (kept as the final token, vLLM convention) or a
        stop string (cut before the tokens completing it). max_tokens is
        enforced last as a hard cap on the commit size.
        Returns (keep, terminated)."""
        keep = len(committed)
        terminated = False
        if not params.ignore_eos:
            for j, tok in enumerate(committed):
                if tok in self.target.eos_token_ids:
                    keep, terminated = j + 1, True
                    break
        if checker is not None:
            # stop strings compete with EOS by POSITION: earliest one wins
            cut = checker.check(candidate_ids, ignore_eos=True)
            if cut is not None and cut - base < keep:
                keep, terminated = cut - base, True
        room = params.max_tokens - base
        if keep > room:                       # length cap, not a "stop" event
            keep = room
        return keep, terminated

    def _generate_one(self, prompt: str | list[int], p: SamplingParams) -> SpecOutput:
        t0 = time.perf_counter()
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer(prompt).input_ids
        else:
            prompt_ids = list(prompt)
        if len(prompt_ids) + p.max_tokens > self.config.max_model_len:
            raise ValueError("prompt + max_tokens exceeds max_model_len")

        tseq = Sequence(prompt_ids, p, arrival_time=t0)
        # per-request RNG: this request's own stream, independent of anything
        # else the engine has sampled
        LLMEngine._seed_rng(tseq, self._next_request_id, 0, p,
                            engine_seed=self.config.seed)
        checker = StopChecker(self.tokenizer, p.stop, self.target.eos_token_ids) \
            if (self.tokenizer is not None and p.stop) else None
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
        gen = tseq.sampling_generator()

        gamma = self.num_spec_tokens
        while len(tseq.output_token_ids) < p.max_tokens:
            out.num_rounds += 1
            accepted_len = len(tseq.tokens)          # all verified so far
            # the pending bonus token (tseq.tokens[-1]) is not in the KV yet
            assert tseq.num_computed_tokens == accepted_len - 1, \
                "target KV frontier drifted from the accepted stream"

            # ---- draft proposals
            room = p.max_tokens - len(tseq.output_token_ids)
            base = len(tseq.output_token_ids)
            if use_ngram:
                proposals = self.ngram_drafter.propose(tseq.tokens, min(gamma, room - 1))
                q_probs = None                        # deterministic point mass
            else:
                proposals, q_probs = self.model_drafter.propose(
                    dseq, accepted_len, min(gamma, max(1, room - 1)),
                    p.temperature, p.top_k, p.top_p, generator=gen)

            # pre-append proposals so the verify forward can read their ids
            # (rolled back below for whatever the target rejects)
            tseq.output_token_ids.extend(proposals)

            # ---- verify: [bonus] + proposals in ONE target forward
            span_end = accepted_len + len(proposals)     # positions [accepted-1, span_end)
            logits = self._target_span(tseq, accepted_len - 1, span_end)
            out.num_proposed += len(proposals)

            k, bonus_tok = self._accept_and_bonus(proposals, q_probs, logits, p, gen)
            out.num_accepted += k

            # ---- commit: accepted proposals + bonus, cut at the earliest
            # termination inside the round (EOS/stop may sit mid-run)
            committed = proposals[:k] + [bonus_tok]
            keep, terminated = self._round_keep_count(
                tseq.output_token_ids, base, committed, p, checker)
            del tseq.output_token_ids[base:]                  # drop proposals+old bonus
            tseq.output_token_ids.extend(committed[:keep])
            for _ in committed[:keep]:
                tseq.record_first_token()
                tseq.last_token_time = time.perf_counter()

            # KV frontier: accepted proposals are computed (verify pass); the
            # bonus token is pending until the next round feeds it. If the
            # round was truncated, nothing is pending -- all kept tokens'
            # KV was computed during verification.
            tseq.num_computed_tokens = accepted_len + k if keep == len(committed) \
                else accepted_len + keep
            self.target.block_manager.register_filled_blocks(
                tseq, tseq.num_computed_tokens)
            if use_ngram:
                self.ngram_drafter.sync(tseq.tokens)
            else:
                # mirror the accepted stream into the draft sequence so its
                # next sync (rewind/extend) sees the bonus token it must prefill
                dseq.output_token_ids = list(tseq.output_token_ids)

            # ---- stop conditions
            if len(tseq.output_token_ids) >= p.max_tokens:
                tseq.status = SequenceStatus.FINISHED_LENGTH
                break
            if terminated:
                tseq.status = SequenceStatus.FINISHED_STOPPED
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
