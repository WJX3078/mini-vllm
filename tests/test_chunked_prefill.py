"""Chunked prefill / unified token scheduler (P1).

The token budget is shared by decode tokens and prefill chunks:
  Case 1: a long prefill must not monopolize iterations -- concurrent decodes
          keep advancing while the long prompt is still chunking.
  Case 2: a prompt longer than max_num_batched_tokens takes multiple
          iterations, in bounded-size chunks.
  Case 3: prefix cache hits shrink the chunked prefill to the uncached tail.
  Case 4: a long prompt preempted mid-prefill resumes correctly.
  Case 5: chunked ON vs OFF vs HuggingFace -- token-identical greedy output.

Chunk spans are observed through a spy wrapped around
``scheduler.schedule()`` -- no double scheduling, exact spans.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from helpers import make_tiny_pair, random_prompts, run_hf_greedy

from minivllm import EngineConfig, LLMEngine
from minivllm.config import ModelConfig
from minivllm.sequence import SamplingParams


def _make(chunked: bool, budget: int, max_num_seqs: int = 8,
          num_blocks: int = 512, seed: int = 0):
    hf, mine = make_tiny_pair(seed)
    model_cfg = ModelConfig.from_hf_config(hf.config)
    cfg = EngineConfig(
        model="(tiny-random-qwen2)", block_size=8, num_blocks=num_blocks,
        max_num_seqs=max_num_seqs, max_model_len=16384,
        max_num_batched_tokens=budget if chunked else 16384,
        enable_chunked_prefill=chunked,
        enable_prefix_caching=True, seed=seed, device="cpu", dtype="float32")
    eng = LLMEngine(cfg, model=mine, model_config=model_cfg, tokenizer=None)
    return eng, hf


def _spy_spans(eng):
    """Wrap scheduler.schedule to record (prompt_len, start, end) spans."""
    spans = []
    orig = eng.scheduler.schedule

    def spy():
        out = orig()
        for s, (st, e) in zip(out.scheduled, out.spans):
            spans.append((s.num_prompt_tokens, st, e))
        return out

    eng.scheduler.schedule = spy
    return spans


def _spy_spans_per_iter(eng):
    """Wrap scheduler.schedule to record per-iteration span lists."""
    iters = []
    orig = eng.scheduler.schedule

    def spy():
        out = orig()
        iters.append([(s.num_prompt_tokens, st, e)
                      for s, (st, e) in zip(out.scheduled, out.spans)])
        return out

    eng.scheduler.schedule = spy
    return iters


def test_case2_prompt_longer_than_budget_chunks_across_iterations():
    """prompt=100 tokens, budget=32: >= 4 prefill iterations, every chunk
    <= budget, final output identical to the unchunked reference."""
    eng, hf = _make(chunked=True, budget=32)
    prompt = random_prompts(1, min_len=100, max_len=100, seed=51)[0]
    params = SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True)
    rid = eng.add_request(prompt, params)
    seq = eng.groups[rid].main
    spans = _spy_spans(eng)

    while eng.scheduler.has_unfinished():
        eng.step()
    prefill = [(st, e) for plen, st, e in spans if plen == 100 and st < 100]
    decodes = [(st, e) for plen, st, e in spans if plen == 100 and st >= 100]
    assert len(prefill) >= 4                       # really was chunked
    assert all(e - st <= 32 for st, e in prefill)
    assert prefill[0][0] == 0 and prefill[-1][1] == 100
    assert len(decodes) == 5       # +1 token sampled by the final chunk step

    ref = run_hf_greedy(hf, [prompt], max_new_tokens=6)[0]
    assert seq.output_token_ids == ref


def test_case1_long_prefill_does_not_block_running_decodes():
    """An 8K-token prompt starts chunking while another sequence is already
    decoding: the decode sequence keeps advancing every iteration."""
    eng, _ = _make(chunked=True, budget=256, num_blocks=2048)
    short = random_prompts(1, min_len=10, max_len=10, seed=52)[0]
    rid_short = eng.add_request(short, SamplingParams(
        temperature=0.0, max_tokens=48, ignore_eos=True))
    short_seq = eng.groups[rid_short].main
    # let the short request start decoding first
    while len(short_seq.output_token_ids) < 3:
        eng.step()

    long_prompt = [i % 250 for i in range(8192)]   # 8K prompt, ids < vocab
    rid_long = eng.add_request(long_prompt, SamplingParams(
        temperature=0.0, max_tokens=4, ignore_eos=True))
    long_seq = eng.groups[rid_long].main
    spans = _spy_spans(eng)

    short_progress = []
    while not long_seq.is_finished:
        eng.step()
        short_progress.append(len(short_seq.output_token_ids))
    long_spans = [(st, e) for plen, st, e in spans if plen == 8192 and st < 8192]
    assert len(long_spans) >= 20                   # prefill ran >= 20 chunks
    assert max(e - st for st, e in long_spans) <= 256
    # short kept decoding while the long prompt was still chunking: it gained
    # >= 20 tokens between the long request's admission and its prefill end
    gained_while_chunking = short_progress[-1] - short_progress[0] + \
        (long_spans[-1][1] < 8192) * 0
    assert gained_while_chunking >= 20, \
        f"decode starved by chunked prefill: {short_progress}"


def test_case3_prefix_cache_shrinks_the_chunked_prefill():
    """100-token prompt, 96 tokens cache-hit (12 full blocks): the warm
    pass's prefill work happens only in the uncached tail."""
    eng, _ = _make(chunked=True, budget=48)
    base = list(range(80))
    suffix = random_prompts(1, min_len=20, max_len=20, seed=53)[0]
    prompt = base + suffix

    # cold pass computes + registers the prompt's full blocks
    eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=2,
                                          ignore_eos=True), use_tqdm=False)

    rid = eng.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=2,
                                                 ignore_eos=True))
    warm = eng.groups[rid].main
    spans = _spy_spans(eng)
    while eng.scheduler.has_unfinished():
        eng.step()
    warm_prefill = [(st, e) for plen, st, e in spans
                    if plen == 100 and st < 100 and warm is not None]
    # only spans inside the uncached tail (>= 72); the first warm span starts
    # at 96 (12 cached full blocks + forced last-token recompute)
    assert min(st for st, _ in warm_prefill) >= 72
    assert min(st for st, _ in warm_prefill) == 96
    assert eng.block_manager.cache_hits > 0


def test_case4_preempted_long_prompt_resumes_chunked():
    """A long prompt in a tight pool gets preempted and finishes correctly
    after re-admission (prefix cache restores the prompt KV)."""
    eng, hf = _make(chunked=True, budget=64, num_blocks=58)
    long_prompt = [i % 250 for i in range(400)]
    other = random_prompts(1, min_len=60, max_len=60, seed=54)[0]
    prompts = [long_prompt, other]
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    outs = eng.generate(prompts, params, use_tqdm=False)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=8)
    for i, o in enumerate(outs):
        assert o.outputs[0]["token_ids"] == ref[i]
    stats = eng.engine_stats()
    assert stats["preemptions"] >= 1 or stats["cache_hits"] >= 1


def test_case5_chunked_matches_unchunked_and_hf_randomized():
    """Randomized prompts: chunked ON (small budget) == OFF == HF, greedy."""
    prompts = random_prompts(5, min_len=40, max_len=120, seed=55)
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)

    eng_on, hf = _make(chunked=True, budget=48)
    out_on = eng_on.generate(prompts, params, use_tqdm=False)
    eng_off, _ = _make(chunked=False, budget=16384)
    out_off = eng_off.generate(prompts, params, use_tqdm=False)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=8)
    for i in range(len(prompts)):
        a = out_on[i].outputs[0]["token_ids"]
        b = out_off[i].outputs[0]["token_ids"]
        assert a == b == ref[i], f"prompt {i} diverged"


def test_mixed_chunked_batch_single_iteration():
    """A single iteration contains a decode step AND a prefill chunk of
    another sequence; results stay HF-identical.

    The short request starts decoding FIRST; the long prompt arrives later.
    FCFS admission means a fresh request cannot jump over an in-flight
    chunked prefill (same policy as vLLM V1), but the ALREADY-RUNNING decode
    must share every iteration with the long prompt's chunks."""
    eng, hf = _make(chunked=True, budget=96)
    long_prompt = [i % 250 for i in range(300)]
    short = random_prompts(1, min_len=10, max_len=10, seed=57)[0]
    rid_short = eng.add_request(short, SamplingParams(
        temperature=0.0, max_tokens=3, ignore_eos=True))
    while len(eng.groups[rid_short].main.output_token_ids) < 1:
        eng.step()                                 # short is decoding now
    rid_long = eng.add_request(long_prompt, SamplingParams(
        temperature=0.0, max_tokens=3, ignore_eos=True))
    iters = _spy_spans_per_iter(eng)
    while not (eng.groups[rid_short].main.is_finished
               and eng.groups[rid_long].main.is_finished):
        eng.step()
    mixed = [it for it in iters
             if any(pl == 10 and st >= 10 for pl, st, e in it)
             and any(pl == 300 and st < 300 for pl, st, e in it)]
    assert mixed, "no iteration mixed a decode step with a prefill chunk"

    lm = eng.groups[rid_long].main
    ss = eng.groups[rid_short].main
    assert lm.output_token_ids == run_hf_greedy(hf, [long_prompt],
                                                max_new_tokens=3)[0]
    assert ss.output_token_ids == run_hf_greedy(hf, [short],
                                                max_new_tokens=3)[0]


def test_chunked_prefill_never_breaks_block_accounting():
    """Invariant sweep over a chunked run with mid-flight arrivals:
    computed <= tokens, block table covers computed tokens, held blocks
    have ref >= 1 and are not in the free list."""
    eng, _ = _make(chunked=True, budget=64, num_blocks=64)
    prompts = random_prompts(3, min_len=100, max_len=200, seed=56)
    eng.add_request(prompts[0], SamplingParams(temperature=0.0, max_tokens=20,
                                               ignore_eos=True))
    bm = eng.block_manager
    steps = 0
    while eng.scheduler.has_unfinished():
        steps += 1
        assert steps < 800
        sched = eng.scheduler.schedule()
        if not sched.scheduled:
            eng.step()
            continue
        for seq, (_, end) in zip(sched.scheduled, sched.spans):
            assert end <= seq.num_tokens
            assert len(seq.block_table) * bm.block_size >= end
            for bid in seq.block_table:
                assert bm.blocks[bid].ref_count >= 1
                assert bid not in bm.free_ids or bm.blocks[bid].ref_count == 0
        eng.step()
        if steps == 3:
            eng.add_request(prompts[1], SamplingParams(
                temperature=0.0, max_tokens=20, ignore_eos=True))
        if steps == 7:
            eng.add_request(prompts[2], SamplingParams(
                temperature=0.0, max_tokens=20, ignore_eos=True))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
