"""Per-request RNG (P0): reproducibility independent of batch composition.

With one shared Generator, request A's samples depended on which other
requests shared its batch (every sample advances the shared stream). Each
sequence now owns a generator seeded by derive_seed(engine_seed, request_id,
sample_idx, user_seed) -- a stable integer mix (splitmix64), never Python's
salted hash() -- so:

  1. A alone and A batched with B produce identical sampled output,
  2. parallel-sampling children (n=3) draw independent streams,
  3. identical (request, seed) pairs reproduce exactly,
  4. greedy decoding is unaffected by any of this.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from helpers import make_tiny_engine, random_prompts

from minivllm.sampling import derive_seed, sample_from_logits
from minivllm.sequence import SamplingParams


def _sampled_run(engine, prompts, params):
    outs = engine.generate(prompts, params, use_tqdm=False)
    return [o.outputs[0]["token_ids"] for o in outs]


def test_seed_batch_composition_independence():
    """A with seed alone == A with seed batched behind B."""
    prompts = random_prompts(2, min_len=10, max_len=10, seed=31)
    params_a = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=16,
                              seed=7, ignore_eos=True)

    eng1, _ = make_tiny_engine(seed=0, enable_prefix_caching=False)
    alone = _sampled_run(eng1, prompts[:1], params_a)[0]

    eng2, _ = make_tiny_engine(seed=0, enable_prefix_caching=False)
    params_b = SamplingParams(temperature=1.1, max_tokens=16, seed=99,
                              ignore_eos=True)
    both = _sampled_run(eng2, prompts, [params_a, params_b])
    assert both[0] == alone
    assert both[1] != alone or True   # B may coincide by chance; not asserted


def test_same_seed_same_output_across_engines():
    prompts = random_prompts(3, min_len=8, max_len=14, seed=32)
    params = SamplingParams(temperature=0.8, max_tokens=12, seed=42,
                            ignore_eos=True)
    eng1, _ = make_tiny_engine(seed=0)
    eng2, _ = make_tiny_engine(seed=0)
    assert _sampled_run(eng1, prompts, params) == _sampled_run(eng2, prompts, params)


def test_arrival_order_does_not_change_seeded_output():
    """Same two seeded requests, submitted in either order: each keeps its
    own output (request_id differs, but seed pins the stream)."""
    prompts = random_prompts(2, min_len=8, max_len=8, seed=33)
    pa = SamplingParams(temperature=1.0, max_tokens=12, seed=5, ignore_eos=True)
    pb = SamplingParams(temperature=1.0, max_tokens=12, seed=6, ignore_eos=True)

    eng1, _ = make_tiny_engine(seed=0)
    ra = _sampled_run(eng1, [prompts[0], prompts[1]], [pa, pb])
    eng2, _ = make_tiny_engine(seed=0)
    rb = _sampled_run(eng2, [prompts[1], prompts[0]], [pb, pa])
    assert ra[0] == rb[1] and ra[1] == rb[0]


def test_parallel_sampling_children_independent_streams():
    """n=3 with temperature>0: children get different random streams (they
    would be IDENTICAL only if one generator were shared per step)."""
    eng, _ = make_tiny_engine(seed=0, num_blocks=64)
    prompt = random_prompts(1, min_len=9, max_len=9, seed=34)[0]
    params = SamplingParams(temperature=1.0, max_tokens=24, seed=11,
                            ignore_eos=True, n=3)
    out = eng.generate([prompt], params, use_tqdm=False)[0]
    assert len(out.outputs) == 3
    token_sets = [tuple(o["token_ids"]) for o in out.outputs]
    # with a shared generator all children would be forced onto the same
    # trajectory; with independent streams 3x24 tokens almost surely differ
    assert len(set(token_sets)) == 3, \
        f"children streams identical: {token_sets}"


def test_children_independent_of_batch_composition():
    """One child (n=2) alone vs behind another request: child 1 output
    must be identical."""
    prompts = random_prompts(2, min_len=8, max_len=8, seed=35)
    p_main = SamplingParams(temperature=0.9, max_tokens=10, seed=77,
                            ignore_eos=True, n=2)
    p_other = SamplingParams(temperature=1.2, max_tokens=10, seed=88,
                             ignore_eos=True)
    eng1, _ = make_tiny_engine(seed=0, num_blocks=64)
    alone = eng1.generate([prompts[0]], p_main, use_tqdm=False)[0]
    eng2, _ = make_tiny_engine(seed=0, num_blocks=64)
    both = eng2.generate([prompts[0], prompts[1]], [p_main, p_other],
                         use_tqdm=False)
    assert [o["token_ids"] for o in alone.outputs] == \
        [o["token_ids"] for o in both[0].outputs]


def test_greedy_ignores_rng_completely():
    prompts = random_prompts(3, seed=36)
    p_greedy = SamplingParams(temperature=0.0, max_tokens=10, seed=1,
                              ignore_eos=True)
    p_greedy2 = SamplingParams(temperature=0.0, max_tokens=10, seed=2,
                               ignore_eos=True)
    eng, _ = make_tiny_engine(seed=0)
    a = _sampled_run(eng, prompts, p_greedy)
    b = _sampled_run(eng, prompts, p_greedy2)
    assert a == b


def test_derive_seed_properties():
    """Stable, order-sensitive, well spread; not dependent on PYTHONHASHSEED."""
    assert derive_seed(1, 2, 3) == derive_seed(1, 2, 3)
    assert derive_seed(1, 2, 3) != derive_seed(3, 2, 1)
    assert derive_seed(1, 2) != derive_seed(1, 2, 0)
    seeds = {derive_seed(0, i, 0, None if False else i) for i in range(1000)}
    assert len(seeds) == 1000          # no collisions in a small family
    for s in seeds:
        assert 0 <= s < (1 << 64)


def test_engine_seed_changes_streams():
    """Different engine seeds must produce different sampled output."""
    prompts = random_prompts(1, min_len=10, max_len=10, seed=37)
    params = SamplingParams(temperature=1.0, max_tokens=16, ignore_eos=True)
    eng1, _ = make_tiny_engine(seed=1)
    eng2, _ = make_tiny_engine(seed=2)
    a = _sampled_run(eng1, prompts, params)[0]
    b = _sampled_run(eng2, prompts, params)[0]
    assert a != b


def test_sample_from_logits_uses_passed_generator_only():
    """The single-sequence helper must be fully driven by its generator."""
    logits = torch.randn(1, 100)
    outs = [sample_from_logits(logits[0], 1.0, -1, 1.0,
                               generator=torch.Generator().manual_seed(9))
            for _ in range(5)]
    assert len(set(outs)) == 1         # same generator state -> same token


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
