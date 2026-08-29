"""Batched sampling (P1): grouped execution with <=1 D2H per group.

sample_tokens() must produce, per sequence, exactly what the single-sequence
reference (sample_from_logits) produces given the same generator state --
grouping by (temperature, top_k, top_p) may not change any draw -- and
greedy sequences must equal the batched argmax regardless of which other
sampling configs share the batch.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from minivllm.sampling import (
    derive_seed,
    probs_batch_from_logits,
    sample_from_logits,
    sample_tokens,
)
from minivllm.sequence import SamplingParams


class _FakeSeq:
    """Minimal sequence stand-in: params + an independent generator."""

    def __init__(self, params: SamplingParams, seed: int):
        self.sampling_params = params
        self._seed = seed
        self._gen = torch.Generator().manual_seed(seed)

    def sampling_generator(self):
        return self._gen


def _make_batch(params_list, seed0=1000):
    return [_FakeSeq(p, seed0 + i) for i, p in enumerate(params_list)]


def test_greedy_equals_batched_argmax():
    torch.manual_seed(0)
    logits = torch.randn(8, 64)
    params = [SamplingParams(temperature=0.0) for _ in range(8)]
    seqs = _make_batch(params)
    assert sample_tokens(logits, seqs) == torch.argmax(logits, dim=-1).tolist()


def test_mixed_configs_match_per_seq_reference():
    """A batch mixing greedy / temperature+top_p / top_k sequences: every
    sampled token equals the per-sequence reference with the same RNG state."""
    torch.manual_seed(1)
    logits = torch.randn(6, 128)
    params = [
        SamplingParams(temperature=0.0),                    # greedy
        SamplingParams(temperature=0.7, top_p=0.9),         # group A
        SamplingParams(temperature=0.7, top_p=0.9),         # group A
        SamplingParams(temperature=1.0, top_k=5),           # group B
        SamplingParams(temperature=1.0, top_k=5),           # group B
        SamplingParams(temperature=0.7, top_p=0.9),         # group A
    ]
    seqs = _make_batch(params)
    got = sample_tokens(logits, seqs)

    for i, p in enumerate(params):
        ref = sample_from_logits(logits[i], p.temperature, p.top_k, p.top_p,
                                 generator=torch.Generator().manual_seed(1000 + i))
        assert got[i] == ref, f"seq {i}: {got[i]} != reference {ref}"


def test_batch_composition_does_not_change_draws():
    """Adding another sequence to the batch must not change any existing
    sequence's sampled token (same generator seed -> same result)."""
    torch.manual_seed(2)
    params = [SamplingParams(temperature=0.8, top_p=0.95) for _ in range(3)]
    logits_small = torch.randn(3, 64)
    logits_big = torch.cat([logits_small, torch.randn(1, 64)])

    got_small = sample_tokens(logits_small, _make_batch(params))
    got_big = sample_tokens(logits_big, _make_batch(params))
    assert got_small == got_big[:3]


def test_batched_probs_match_single_probs():
    torch.manual_seed(3)
    logits = torch.randn(4, 32)
    pb = probs_batch_from_logits(logits, 0.8, -1, 0.9)
    for i in range(4):
        ps = torch.softmax(
            __import__("minivllm.sampling", fromlist=["filter_logits"])
            .filter_logits(logits[i], 0.8, -1, 0.9), dim=-1)
        assert torch.allclose(pb[i], ps, atol=1e-6)


def test_greedy_group_shared_with_sampled_group():
    """Greedy sequences are unaffected by sampled neighbours in the same
    call and stay argmax-exact."""
    torch.manual_seed(4)
    logits = torch.randn(3, 16)
    params = [SamplingParams(temperature=1.2),
              SamplingParams(temperature=0.0),
              SamplingParams(temperature=0.5, top_k=2)]
    seqs = _make_batch(params)
    got = sample_tokens(logits, seqs)
    assert got[1] == int(torch.argmax(logits[1]).item())


def test_seeds_are_independent_of_derivation_path():
    """derive_seed for user-seeded requests depends only on (seed, idx)."""
    assert derive_seed(7, 0) == derive_seed(7, 0)
    assert derive_seed(7, 0) != derive_seed(7, 1)
    assert derive_seed(7, 0) != derive_seed(8, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
