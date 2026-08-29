"""End-to-end correctness: mini-vllm must match HuggingFace generate().

The tiny random-weight Qwen2 runs on CPU in fp32, so any real numerical
difference shows up as different greedy tokens. We test:

  1. single-token logits against HF's forward (prefill path),
  2. greedy generation token-by-token (prefill + decode + KV reuse),
  3. batched generation with continuous batching and mixed prompt lengths,
  4. prefix caching ON vs OFF must produce identical outputs,
  5. parallel sampling (n=3): all children equal the parent under greedy.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from helpers import (
    TINY,
    make_tiny_engine,
    make_tiny_pair,
    random_prompts,
    run_engine_greedy,
    run_hf_greedy,
)

from minivllm.sequence import SamplingParams


def test_logits_match_hf_forward():
    hf, mine = make_tiny_pair(seed=3)
    torch.manual_seed(7)
    ids = torch.randint(2, TINY["vocab_size"], (1, 17)).tolist()[0]
    x = torch.tensor([ids])

    with torch.no_grad():
        ref = hf(x).logits[0, -1]
        # run my model through the engine's exact paged path
        engine, _ = make_tiny_engine(seed=3)
        engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
        sched = engine.scheduler.schedule()
        # build the flat batch exactly like engine.step does
        seq = sched.scheduled[0]
        from minivllm.attention import SeqInput
        si = SeqInput(q_start=0, q_len=len(ids),
                      block_table=torch.tensor(seq.block_table, dtype=torch.long), t0=0)
        logits = engine.model(torch.tensor(ids), torch.arange(len(ids)),
                              engine.block_manager.pool, [si],
                              torch.tensor([len(ids) - 1]))
    assert torch.allclose(logits[0], ref, atol=1e-4), \
        f"max diff {(logits[0] - ref).abs().max()}"


def test_greedy_generation_matches_hf():
    engine, hf = make_tiny_engine(seed=0, enable_prefix_caching=False)
    prompts = random_prompts(6, seed=11)
    mine = run_engine_greedy(engine, prompts, max_new_tokens=12)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=12)
    for i, (a, b) in enumerate(zip(mine, ref)):
        assert a == b, f"prompt {i}: mine={a} hf={b}"


def test_batched_continuous_batching_matches_hf():
    engine, hf = make_tiny_engine(seed=0, enable_prefix_caching=False,
                                  max_num_seqs=8, num_blocks=48)
    prompts = random_prompts(8, min_len=4, max_len=30, seed=12)
    mine = run_engine_greedy(engine, prompts, max_new_tokens=10)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=10)
    for i, (a, b) in enumerate(zip(mine, ref)):
        assert a == b, f"prompt {i}: mine={a} hf={b}"


def test_prefix_cache_produces_identical_outputs():
    prompts = random_prompts(5, seed=13)
    # give all prompts a shared prefix so the cache actually hits
    shared = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    prompts = [shared + p for p in prompts]

    eng_on, _ = make_tiny_engine(seed=0, enable_prefix_caching=True)
    eng_off, _ = make_tiny_engine(seed=0, enable_prefix_caching=False)
    # batch 1 computes the shared prefix; batch 2 arrives later and reuses it
    # (requests submitted in the same step all miss -- same as real vLLM)
    out_on = (run_engine_greedy(eng_on, prompts[:2], max_new_tokens=8)
              + run_engine_greedy(eng_on, prompts[2:], max_new_tokens=8))
    out_off = (run_engine_greedy(eng_off, prompts[:2], max_new_tokens=8)
               + run_engine_greedy(eng_off, prompts[2:], max_new_tokens=8))
    assert out_on == out_off
    # batch 2 must have reused the shared prefix blocks
    assert eng_on.block_manager.cache_hits > 0
    assert eng_on.engine_stats()["cache_hit_rate"] > 0.1


def test_fully_cached_prompt_still_correct():
    engine, hf = make_tiny_engine(seed=0, enable_prefix_caching=True)
    prompt = random_prompts(1, min_len=16, max_len=16, seed=14)[0]

    first = run_engine_greedy(engine, [prompt], max_new_tokens=6)[0]
    # run the identical prompt again: prefill is fully cache-hits
    second = run_engine_greedy(engine, [prompt], max_new_tokens=6)[0]
    assert first == second
    ref = run_hf_greedy(hf, [prompt], max_new_tokens=6)[0]
    assert first == ref
    assert engine.block_manager.cache_hits > 0


def test_parallel_sampling_greedy_children_equal_parent():
    engine, hf = make_tiny_engine(seed=0, enable_prefix_caching=True, num_blocks=64)
    prompt = random_prompts(1, min_len=9, max_len=9, seed=15)[0]
    params = SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True, n=3)
    out = engine.generate([prompt], params, use_tqdm=False)[0]
    assert len(out.outputs) == 3
    ref = run_hf_greedy(hf, [prompt], max_new_tokens=10)[0]
    for o in out.outputs:
        assert o["token_ids"] == ref      # greedy forks stay identical
    assert engine.block_manager.cow_copies >= 1   # COW actually exercised


def test_preemption_under_tiny_pool_keeps_correctness():
    engine, hf = make_tiny_engine(seed=0, enable_prefix_caching=True,
                                  num_blocks=10, block_size=8, max_num_seqs=4)
    prompts = random_prompts(4, min_len=20, max_len=28, seed=16)
    mine = run_engine_greedy(engine, prompts, max_new_tokens=10)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=10)
    for i, (a, b) in enumerate(zip(mine, ref)):
        assert a == b, f"prompt {i}: mine={a} hf={b}"
    # the tiny pool forces preemption + recompute at least once
    stats = engine.engine_stats()
    assert stats["preemptions"] >= 1 or stats["cache_hits"] >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
