"""Sampling: temperature / top-k / top-p, per-request RNG, GPU-native
batched sampling.

Four concerns live here:

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

3. GPU-native batched sampling. Per-sequence ``.item()`` forces B GPU->CPU
   syncs per decode step; the v0.2 fix moved the whole [B, vocab] probability
   matrix to the CPU (one D2H -- but O(B*V) bytes: at B=64, V=150k fp32 that
   is ~38 MB *per step*). This version keeps everything on the sampling
   device and moves only O(B) data:

       CPU  : each sequence's own generator draws ONE uniform  u_i
       H2D  : [B] float32 uniforms
       GPU  : filter (temperature/top-k/top-p) -> softmax -> cumulative sum
              -> inverse CDF:  token_i = #{ j : cdf_j <= u_i }
       D2H  : [B] int64 token ids

   so the D2H traffic drops from O(B*V) to O(B). RNG semantics are
   unchanged: the uniform still comes from each sequence's private CPU
   generator, so batch composition cannot influence any request.

4. One sampling primitive everywhere. ``sample_from_logits`` (single
   sequence, used by speculative drafting/verification) and the batched
   ``sample_tokens`` draw the SAME u from the generator and apply the SAME
   inverse-CDF rule, so equal generator state => equal token on both paths.
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


# ------------------------------------------------------- inverse-CDF sampling
def _inverse_cdf(probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """Map uniforms [B] through the CDF of probs [B, V]: token_i is the
    first index whose cumulative probability exceeds u_i. For u ~ U[0,1)
    this samples exactly from probs. The final clamp covers float rounding
    at u -> 1 (cdf[-1] may fall a few ulp short of 1.0)."""
    cdf = probs.cumsum(dim=-1)
    tokens = (cdf <= uniforms[:, None]).sum(dim=-1)
    return tokens.clamp_max_(probs.shape[-1] - 1)


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0,
                       top_k: int = -1, top_p: float = 1.0,
                       generator: torch.Generator | None = None) -> int:
    """Sample one token id (single sequence). temperature==0 means greedy.

    Draws exactly one uniform from `generator` and applies the inverse-CDF
    rule -- the same primitive the batched GPU path uses, so equal
    generator state yields equal tokens on both paths."""
    if temperature == 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = filter_logits(logits, temperature, top_k, top_p)
    probs = torch.softmax(logits, dim=-1).float()
    u = torch.rand((), generator=generator).to(probs.device)
    return int(_inverse_cdf(probs[None], u[None])[0].item())


def sample_from_probs(probs: torch.Tensor,
                      generator: torch.Generator) -> int:
    """Inverse-CDF sample from a ready probability row (speculative
    decoding's rejection-residual / bonus draws). One uniform per call."""
    u = torch.rand((), generator=generator)
    return int(_inverse_cdf(probs.float()[None], u[None])[0].item())


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
    """Sample one token per sequence for a whole batch, GPU-native.

    logits: [S, vocab] on any device. seqs[i] must expose
    ``sampling_params`` and ``sampling_generator()``.

    Data movement per sampling-config group (v0.2 -> v0.3):
        greedy : one batched argmax, one [G] D2H              (unchanged)
        sampled: filter+softmax+cumsum on device; the ONLY
                 cross-device traffic is [G] float32 H2D
                 (uniforms from each sequence's private CPU
                 generator) + [G] int64 D2H (token ids) --
                 O(B), never O(B*vocab).

    Per-request RNG independence is exact: each sequence's token is a pure
    function of (its logits row, the uniform from its own generator).
    """
    groups: dict[tuple, list[int]] = {}
    for i, seq in enumerate(seqs):
        p = seq.sampling_params
        groups.setdefault((p.temperature, p.top_k, p.top_p), []).append(i)

    out: list[int] = [0] * len(seqs)
    device = logits.device
    for (temperature, top_k, top_p), idxs in groups.items():
        sub = logits[idxs]                                    # [g, vocab]
        if temperature == 0.0:
            toks = torch.argmax(sub, dim=-1)
        else:
            probs = probs_batch_from_logits(sub, temperature, top_k,
                                            top_p).float()
            # one uniform per sequence, from its OWN CPU generator
            uniforms = torch.tensor(
                [torch.rand((), generator=seqs[i].sampling_generator()).item()
                 for i in idxs], dtype=torch.float32, device=device)
            toks = _inverse_cdf(probs, uniforms)
        # the group's single D2H: [G] token ids
        ids = toks.cpu().tolist()
        for row, i in enumerate(idxs):
            out[i] = int(ids[row])
    return out
