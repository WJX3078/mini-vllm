"""Sampling: temperature / top-k / top-p, plus helpers used by speculative decoding."""
from typing import List, Optional

import torch


def filter_logits(logits: torch.Tensor, temperature: float = 1.0,
                  top_k: int = -1, top_p: float = 1.0) -> torch.Tensor:
    """Apply temperature, top-k, top-p. Returns logits of a truncated distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for filter_logits (use argmax for greedy)")
    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # keep tokens whose cumulative prob (inclusive) <= top_p; always keep the first
        remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    return logits


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0,
                       top_k: int = -1, top_p: float = 1.0,
                       generator: Optional[torch.Generator] = None) -> int:
    """Sample one token id. temperature==0 means greedy argmax."""
    if temperature == 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = filter_logits(logits, temperature, top_k, top_p)
    probs = torch.softmax(logits, dim=-1)
    # multinomial on CPU: keeps the sampler generator device-independent
    return int(torch.multinomial(probs.detach().cpu(), num_samples=1,
                                 generator=generator).item())


def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0,
                      top_k: int = -1, top_p: float = 1.0) -> torch.Tensor:
    """The full sampling distribution (used as p/q in speculative decoding)."""
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs[int(torch.argmax(logits))] = 1.0
        return probs
    logits = filter_logits(logits, temperature, top_k, top_p)
    return torch.softmax(logits, dim=-1)


class Sampler:
    """Per-sequence sampler."""

    def __init__(self, seed: int = 0):
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def sample(self, logits: torch.Tensor, temperature: float, top_k: int,
               top_p: float) -> int:
        return sample_from_logits(logits, temperature, top_k, top_p,
                                  generator=self.generator)
