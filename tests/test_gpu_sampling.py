"""GPU-native sampling (v0.3): inverse-CDF on the device, O(B) transfers.

The v0.2 batched sampler moved the whole [B, vocab] probability matrix to
the CPU (one D2H, but O(B*V) bytes ~ 38 MB/step at B=64, V=150k fp32). The
v0.3 sampler draws ONE uniform per sequence from its private CPU generator,
uploads [B] floats, runs filter -> softmax -> cumsum -> inverse CDF on the
device, and downloads [B] token ids.

Verified here:
  * distribution exactness (vocab=4, p=[.1,.2,.3,.4], 50k draws),
  * RNG independence: A alone == A batched with B..E (own generator),
  * same-seed reproducibility,
  * mixed-config batches equal the per-sequence reference,
  * GPU path == CPU path token-for-token for a fixed seed (gpu marker).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from minivllm.sampling import (
    _inverse_cdf,
    sample_from_logits,
    sample_tokens,
)
from minivllm.sequence import SamplingParams


class _FakeSeq:
    def __init__(self, params: SamplingParams, seed: int):
        self.sampling_params = params
        self._gen = torch.Generator().manual_seed(seed)

    def sampling_generator(self):
        return self._gen


def test_distribution_exact_vocab4():
    """p = [0.1, 0.2, 0.3, 0.4], 50k draws -> empirical ~ p."""
    p = torch.tensor([0.1, 0.2, 0.3, 0.4])
    logits = torch.log(p)[None]                       # [1, 4]
    counts = torch.zeros(4)
    trials = 50_000
    for _ in range(trials // 1000):                   # 1000-token batches
        logits_rep = logits.repeat(1000, 1)
        # distinct seeds per row -> iid uniforms across the batch
        batch = [_FakeSeq(SamplingParams(temperature=1.0), seed=i)
                 for i in range(1000)]
        toks = sample_tokens(logits_rep, batch)
        for t in toks:
            counts[t] += 1
    emp = counts / trials
    assert torch.allclose(emp, p, atol=0.015), \
        f"empirical {emp.tolist()} != target {p.tolist()}"


def test_uniforms_map_through_cdf_correctly():
    """Direct inverse-CDF checks: u below p0 picks 0; u at the top picks 3;
    u=1-eps clamps into range."""
    p = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    assert _inverse_cdf(p, torch.tensor([0.05]))[0] == 0
    assert _inverse_cdf(p, torch.tensor([0.15]))[0] == 1
    assert _inverse_cdf(p, torch.tensor([0.65]))[0] == 3
    assert _inverse_cdf(p, torch.tensor([0.99999]))[0] == 3
    # u exactly on a boundary rounds consistently (first cdf > u)
    assert _inverse_cdf(p, torch.tensor([0.10]))[0] == 1


def test_rng_independence_batch_composition():
    """A alone vs A first-in-batch: same generator seed -> same token."""
    torch.manual_seed(3)
    logits_a = torch.randn(1, 64)
    logits_b = torch.randn(3, 64)
    params = SamplingParams(temperature=0.9, top_p=0.95)
    alone = sample_tokens(logits_a, [_FakeSeq(params, seed=123)])
    batch = sample_tokens(torch.cat([logits_a, logits_b]),
                          [_FakeSeq(params, seed=123)]
                          + [_FakeSeq(SamplingParams(temperature=1.3), seed=i)
                             for i in range(3)])
    assert alone == batch[:1]


def test_same_seed_reproducible():
    torch.manual_seed(4)
    logits = torch.randn(4, 32)
    params = [SamplingParams(temperature=0.8, top_k=5)] * 4
    a = sample_tokens(logits, [_FakeSeq(params[i], seed=500 + i)
                               for i in range(4)])
    b = sample_tokens(logits, [_FakeSeq(params[i], seed=500 + i)
                               for i in range(4)])
    assert a == b


def test_mixed_configs_match_single_sequence_reference():
    torch.manual_seed(5)
    logits = torch.randn(5, 128)
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=0.7, top_p=0.9),
        SamplingParams(temperature=1.0, top_k=5),
        SamplingParams(temperature=1.2),
        SamplingParams(temperature=0.7, top_p=0.9),
    ]
    seqs = [_FakeSeq(params[i], seed=800 + i) for i in range(5)]
    got = sample_tokens(logits, seqs)
    for i, prm in enumerate(params):
        ref = sample_from_logits(logits[i], prm.temperature, prm.top_k,
                                 prm.top_p,
                                 generator=torch.Generator().manual_seed(800 + i))
        assert got[i] == ref, f"seq {i}: {got[i]} != {ref}"


def test_zero_probability_tail_never_sampled():
    """Top-k filtered tokens have p=0; a u beyond the kept mass must clamp
    into the support, never select a zero-probability token."""
    logits = torch.full((1, 10), -100.0)
    logits[0, :3] = torch.tensor([0.0, 1.0, 2.0])     # top-3 support
    probs = torch.softmax(logits, dim=-1)
    toks = _inverse_cdf(probs, torch.tensor([0.999999, 0.5, 0.0]))
    for t in toks.tolist():
        assert t in (0, 1, 2)


@pytest.mark.gpu
@pytest.mark.triton
def test_gpu_path_matches_cpu_path_same_seed():
    """Same seed, CPU vs CUDA device: the uniform comes from a CPU
    generator, so both paths should agree token-for-token (up to cumsum
    rounding, which this well-separated distribution avoids)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(6)
    logits = torch.randn(8, 32000)
    params = [SamplingParams(temperature=0.8, top_p=0.92) for _ in range(8)]
    cpu = sample_tokens(logits, [_FakeSeq(params[i], seed=900 + i)
                                 for i in range(8)])
    gpu = sample_tokens(logits.cuda(), [_FakeSeq(params[i], seed=900 + i)
                                        for i in range(8)])
    assert cpu == gpu


@pytest.mark.gpu
def test_gpu_sampling_distribution_end_to_end():
    """End-to-end on CUDA: batched inverse-CDF empirically matches p."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    p = torch.tensor([0.1, 0.2, 0.3, 0.4])
    logits = torch.log(p).cuda()[None].repeat(2048, 1)
    batch = [_FakeSeq(SamplingParams(temperature=1.0), seed=1_000_000 + i)
             for i in range(2048)]
    toks = sample_tokens(logits, batch)
    counts = torch.bincount(torch.tensor(toks), minlength=4).float()
    emp = counts / counts.sum()
    assert torch.allclose(emp, p, atol=0.02), \
        f"gpu empirical {emp.tolist()} != {p.tolist()}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
