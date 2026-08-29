"""Sampling: temperature / top-k / top-p, per-request RNG, batched sampling.

Three concerns live here:

1. Filter math (temperature / top-k / top-p) -- shared by engine decoding and
   speculative-decoding verification. All helpers accept a leading batch dim.

2. Per-request RNG. A single shared Generator would make request A's sampled
   tokens depend on which other requests happen to share its batch (each
   sample advances the shared stream). Instead every Sequence owns a
   torch.Generator seeded by a *stable* integer mix of
   (engine_seed, request_id, sample_idx, user_seed). Python's builtin hash()
   is salted per process and must never be used for this. Consequence: the
   same request+seed reproduces the same tokens regardless of batch
   composition, and parallel-sampling children (different sample_idx) get
   independent streams.

3. Batched sampling. Per-sequence ``.item()`` / ``.cpu()`` calls force a
   GPU->CPU sync per sequence per step; with B sequences that is B syncs
   per decode step. ``sample_tokens`` groups sequences by sampling
   configuration, runs filter+softmax once per group on the GPU, and does at
   most ONE D2H transfer per group. The final per-sequence multinomial runs
   on CPU against that group's probs, using each sequence's own generator --
   so batching never changes the RNG semantics.
"""
from collections.abc import Sequence as SeqT

import torch

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


def _mix64(z: int) -> int:
    """splitmix64 finalizer -- fast, well-distributed 64-bit integer mixing."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def derive_seed(*parts: int) -> int:
    """Deterministic 64-bit seed from integer components.

    Order-sensitive and collision-resistant enough for seed derivation;
    stable across processes (unlike Python's salted hash()).
    """
    h = 0
    for p in parts:
        h = _mix64((h + (int(p) & _MASK64) + _GOLDEN) & _MASK64)
    return h


# ------------------------------------------------------------------ filtering
def filter_logits(logits: torch.Tensor, temperature: float = 1.0,
                  top_k: int = -1, top_p: float = 1.0) -> torch.Tensor:
    """Apply temperature, top-k, top-p. Returns logits of a truncated
    distribution. Works on [..., vocab] (batched or single)."""
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
                       generator: torch.Generator | None = None) -> int:
    """Sample one token id (single sequence). temperature==0 means greedy."""
    if temperature == 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = filter_logits(logits, temperature, top_k, top_p)
    probs = torch.softmax(logits, dim=-1)
    # multinomial on CPU: keeps the generator device-independent
    return int(torch.multinomial(probs.detach().float().cpu(), num_samples=1,
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


def probs_batch_from_logits(logits: torch.Tensor, temperature: float,
                            top_k: int, top_p: float) -> torch.Tensor:
    """Batched probs_from_logits: logits [B, vocab] -> probs [B, vocab]."""
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs[torch.arange(logits.size(0), device=logits.device),
              torch.argmax(logits, dim=-1)] = 1.0
        return probs
    return torch.softmax(filter_logits(logits, temperature, top_k, top_p), dim=-1)


# ------------------------------------------------------------ batched sampling
def sample_tokens(logits: torch.Tensor, seqs: SeqT) -> list[int]:
    """Sample one token per sequence for a whole batch with minimal syncs.

    logits: [S, vocab] on any device. seqs[i] must expose
    ``sampling_params`` and ``sampling_generator()``.

    Sequences are grouped by (temperature, top_k, top_p):
      * greedy group: one batched argmax, one D2H for the whole group;
      * sampled group: one batched filter+softmax on the GPU, ONE D2H of the
        group's probs, then per-sequence CPU multinomial against the group's
        own generator. Per-request RNG independence is preserved exactly:
        each sequence consumes only its own generator stream.
    """
    groups: dict[tuple, list[int]] = {}
    for i, seq in enumerate(seqs):
        p = seq.sampling_params
        groups.setdefault((p.temperature, p.top_k, p.top_p), []).append(i)

    out: list[int] = [0] * len(seqs)
    for (temperature, top_k, top_p), idxs in groups.items():
        sub = logits[idxs]                                    # [g, vocab] (GPU)
        if temperature == 0.0:
            toks = torch.argmax(sub, dim=-1)
        else:
            probs = probs_batch_from_logits(sub, temperature, top_k, top_p)
            probs_cpu = probs.detach().float().cpu()          # the one D2H
            for row, i in enumerate(idxs):
                out[i] = int(torch.multinomial(
                    probs_cpu[row], num_samples=1,
                    generator=seqs[i].sampling_generator()).item())
            continue
        out_ids = toks.cpu().tolist()                         # the one D2H
        for row, i in enumerate(idxs):
            out[i] = int(out_ids[row])
    return out
