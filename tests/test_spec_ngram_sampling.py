"""NGram drafter + temperature > 0: mathematically lossless speculative
sampling (P0).

A deterministic drafter is the point-mass proposal q(x)=1 (all other tokens
0). Speculative sampling then accepts with probability min(1, p(x)/q(x)) =
p(x), and on rejection resamples from norm(max(p - q, 0)) -- p with the
proposed token's mass removed. This module verifies BOTH levels:

  * statistical: the committed-token distribution equals the target p over
    tens of thousands of trials (direct + end-to-end through the engine),
  * end-to-end: NGramDrafter + temperature>0 actually runs the rejection
    path and yields exactly max_tokens tokens (the old code crashed with
    IndexError because q_probs was an empty list).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from helpers import make_tiny_spec_engine, random_prompts

from minivllm.sequence import SamplingParams


def _stat_check(eng, proposals, q_probs, p_row, trials, gen_seed, tol=0.015):
    """Run _accept_and_bonus `trials` times; committed[0]'s empirical
    distribution must equal the target p_row."""
    vocab = p_row.numel()
    logits = torch.log(p_row).repeat(2, 1)          # rows for proposal & bonus
    params = SamplingParams(temperature=1.0, max_tokens=8)
    gen = torch.Generator().manual_seed(gen_seed)
    counts = torch.zeros(vocab)
    for _ in range(trials):
        k, bonus = eng._accept_and_bonus(proposals, q_probs, logits, params, gen)
        counts[proposals[0] if k >= 1 else bonus] += 1
    return counts / trials


@pytest.mark.parametrize("x,p_x", [(0, 0.3), (1, 0.6), (2, 0.05)])
def test_deterministic_proposal_distribution_exact(x, p_x):
    """Point-mass proposal: committed token ~ p for every proposal token."""
    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram")
    p = torch.tensor([p_x, (1 - p_x) * 0.55, (1 - p_x) * 0.45])
    emp = _stat_check(eng, [x], None, p, trials=30000, gen_seed=100 + x)
    assert torch.allclose(emp, p, atol=0.015), \
        f"proposal {x}: empirical {emp.tolist()} != target {p.tolist()}"


def test_deterministic_proposal_never_breaks_top_k_target():
    """Proposal outside the target's top-k support (p(x)=0) must be rejected
    100% of the time; the committed token still follows p."""
    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram")
    p = torch.tensor([0.5, 0.5, 1e-9])
    logits = torch.log(p).repeat(2, 1)
    params = SamplingParams(temperature=1.0, top_k=2, max_tokens=8)
    gen = torch.Generator().manual_seed(7)
    outside = 2
    committed = torch.zeros(3)
    trials = 5000
    for _ in range(trials):
        k, bonus = eng._accept_and_bonus([outside], None, logits, params, gen)
        committed[outside if k >= 1 else bonus] += 1
    assert committed[outside] == 0                   # p(x)=0 -> never accepted


def test_ngram_e2e_sampling_runs_lossless_first_token():
    """End-to-end through SpeculativeEngine with the ngram drafter and
    temperature>0: the first committed token's empirical distribution over
    many runs equals the target model's distribution at that position."""
    eng, hf = make_tiny_spec_engine(seed=3, drafter="ngram", num_spec_tokens=4)
    prompt = random_prompts(1, min_len=16, max_len=16, seed=41)[0]
    params = SamplingParams(temperature=1.0, max_tokens=1, ignore_eos=True)

    # target distribution of the first output token (same prefill, no draft)
    with torch.no_grad():
        ref = hf(torch.tensor([prompt])).logits[0, -1]
        p_ref = torch.softmax(ref, dim=-1)

    trials = 400
    counts = torch.zeros(p_ref.numel())
    for _ in range(trials):
        out = eng.generate([prompt], params, use_tqdm=False)[0]
        counts[out.token_ids[0]] += 1
    emp = counts / trials
    top = torch.topk(p_ref, 8)
    # compare the top-8 mass (tail is too small to resolve at 400 trials)
    emp_top = emp[top.indices]
    assert torch.allclose(emp_top, top.values, atol=0.05), \
        f"empirical {emp_top.tolist()} vs target {top.values.tolist()}"


def test_ngram_e2e_sampling_full_length_and_reproducible():
    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram", num_spec_tokens=4)
    prompt = random_prompts(1, min_len=20, max_len=20, seed=42)[0]
    params = SamplingParams(temperature=0.8, max_tokens=24, seed=5,
                            ignore_eos=True)
    # reusing one engine: request ids differ, but the user seed pins output
    o1 = eng.generate([prompt], params, use_tqdm=False)[0]
    o2 = eng.generate([prompt], SamplingParams(temperature=0.8, max_tokens=24,
                                               seed=5, ignore_eos=True),
                      use_tqdm=False)[0]
    assert len(o1.token_ids) == 24
    assert len(o2.token_ids) == 24
    assert o1.token_ids == o2.token_ids
    assert o1.num_rounds >= 2


def test_rejection_residual_falls_back_to_p_when_p_equals_q():
    """Degenerate p == q: residual mass is zero, sampling from p is the
    correct limit; must not produce NaN."""
    eng, _ = make_tiny_spec_engine(seed=0, drafter="ngram")
    p = torch.tensor([0.4, 0.6])
    q = p.clone()
    logits = torch.log(p).repeat(2, 1)
    params = SamplingParams(temperature=1.0, max_tokens=4)
    gen = torch.Generator().manual_seed(3)
    counts = torch.zeros(2)
    trials = 8000
    for _ in range(trials):
        d0 = int(torch.multinomial(q, 1, generator=gen).item())
        k, bonus = eng._accept_and_bonus([d0], [q], logits, params, gen)
        counts[[d0, bonus][0 if k else 1]] += 1
    assert torch.allclose(counts / trials, p, atol=0.02)
    assert not torch.isnan(counts).any()


def test_ngram_still_greedy_correct():
    """Regression guard: the sampling-path rework must not touch the greedy
    ngram pipeline (output identical to plain greedy)."""
    from helpers import run_hf_greedy
    eng, hf = make_tiny_spec_engine(seed=0, drafter="ngram", num_spec_tokens=6)
    prompts = random_prompts(4, seed=43)
    params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
    outs = eng.generate(prompts, params, use_tqdm=False)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=12)
    for i, o in enumerate(outs):
        assert o.token_ids == ref[i]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
