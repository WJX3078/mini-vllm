"""Tests for speculative decoding: drafters, verify loop, distribution law."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from helpers import (
    make_tiny_spec_engine,
    random_prompts,
    run_hf_greedy,
    run_spec_greedy,
)

from minivllm.spec.drafters import NGramDrafter


# ------------------------------------------------------------------ drafters
def test_ngram_drafter_proposals():
    d = NGramDrafter(window=3)
    stream = [10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12]
    d.sync(stream)
    # suffix (10,11,12) last occurred at index 7 -> proposals follow index 9
    assert d.propose(stream, 4) == [13, 14, 10, 11]
    assert d.propose(stream, 2) == [13, 14]
    # a suffix never seen -> no proposals
    assert d.propose([1, 2, 3, 4, 1, 2, 9], 4) == []
    # incremental sync keeps working as the stream grows
    grown = stream + [13]
    d.sync(grown)
    assert d.propose(grown, 6) == [14, 10, 11, 12, 13]   # only 5 known tokens


def test_ngram_drafter_never_proposes_from_final_window():
    d = NGramDrafter(window=2)
    stream = [7, 8, 7, 8]
    d.sync(stream)
    # suffix (7,8) is the final window itself; the earlier occurrence at 0
    # proposes [7, 8, ...] wrapping around the stream
    assert d.propose(stream, 3)[:2] == [7, 8]


# --------------------------------------------------- distribution invariant
def test_rejection_sampling_preserves_target_distribution():
    """With proposals drawn from q, the committed token distribution must be
    exactly p -- the core correctness law of speculative decoding."""
    from minivllm.sequence import SamplingParams

    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram")
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.1, 0.9])
    logits = torch.stack([torch.log(p), torch.log(p)])   # rows: target dists
    params = SamplingParams(temperature=1.0, max_tokens=8)

    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(1234)
    counts = torch.zeros(2)
    trials = 20000
    for _ in range(trials):
        d0 = int(torch.multinomial(q, 1).item())
        k, bonus = eng._accept_and_bonus([d0], [q], logits, params, gen)
        committed = [d0] * k + [bonus]
        counts[committed[0]] += 1
    empirical = counts / trials
    assert torch.allclose(empirical, p, atol=0.02), \
        f"empirical {empirical.tolist()} != target {p.tolist()}"


# --------------------------------------------------------- end-to-end greedy
def test_spec_ngram_greedy_matches_plain_greedy():
    eng, hf = make_tiny_spec_engine(seed=0, drafter="ngram", num_spec_tokens=4)
    prompts = random_prompts(5, seed=21)
    outs = run_spec_greedy(eng, prompts, max_new_tokens=12)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=12)
    for i, o in enumerate(outs):
        assert o.token_ids == ref[i], \
            f"prompt {i}: spec={o.token_ids} hf={ref[i]}"


def test_spec_model_drafter_full_acceptance():
    """Draft == target weights: every proposal must be accepted, so output is
    identical to plain greedy and each round yields gamma+1 tokens."""
    eng, hf = make_tiny_spec_engine(seed=0, drafter="(same)", num_spec_tokens=4)
    assert eng.model_drafter is not None
    prompts = random_prompts(4, seed=22)
    outs = run_spec_greedy(eng, prompts, max_new_tokens=12)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=12)
    for i, o in enumerate(outs):
        assert o.token_ids == ref[i]
        assert o.acceptance_rate > 0.95, f"acceptance {o.acceptance_rate}"
        assert o.tokens_per_round > 3.5          # gamma=4 -> 5 tokens/round


def test_spec_sampling_runs_and_matches_greedy_length():
    """Sampling path (temperature>0) must run the rejection-sampling loop and
    produce exactly max_tokens tokens."""
    eng, _ = make_tiny_spec_engine(seed=0, drafter="(same)", num_spec_tokens=3)
    from minivllm.sequence import SamplingParams
    prompts = random_prompts(2, seed=23)
    params = SamplingParams(temperature=1.0, max_tokens=16, ignore_eos=True)
    outs = eng.generate(prompts, params, use_tqdm=False)
    for o in outs:
        assert len(o.token_ids) == 16
        assert o.num_rounds >= 2                 # several verify rounds happened
        assert 0.0 < o.acceptance_rate <= 1.0


def test_spec_with_prefix_caching_still_correct():
    """Repeated prompt: prefix cache hits must not corrupt speculative output."""
    eng, hf = make_tiny_spec_engine(seed=0, drafter="(same)", num_spec_tokens=4)
    prompt = random_prompts(1, min_len=20, max_len=20, seed=24)[0]
    first = run_spec_greedy(eng, [prompt], max_new_tokens=10)[0]
    second = run_spec_greedy(eng, [prompt], max_new_tokens=10)[0]
    ref = run_hf_greedy(hf, [prompt], max_new_tokens=10)[0]
    assert first.token_ids == second.token_ids == ref
    assert eng.target.block_manager.cache_hits > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
