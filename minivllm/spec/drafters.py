"""Drafters: cheap token proposers for speculative decoding.

* NGramDrafter  -- prompt-lookup decoding. Finds the previous occurrence of
  the current suffix in the context and proposes what followed it. Needs no
  model, excels when the output copies/repeats the input (summarization,
  coding, multi-turn echoes).

* ModelDrafter  -- a full (small) language model driven through the same
  paged-KV worker machinery as the target.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # type-checking only; avoids a runtime import cycle
    from minivllm.spec.worker import KVWorker


class NGramDrafter:
    """Proposes tokens by n-gram matching against the accepted context.

    Maintains gram -> [start indices] over the accepted stream. A proposal
    source must occur strictly before the final window (the final w-gram is
    the query itself, proposing from it would yield nothing).
    """

    def __init__(self, window: int = 3):
        self.window = max(1, window)
        self.table = {}                       # w-gram tuple -> [start indices]
        self._synced = 0                      # tokens already indexed

    def reset(self):
        self.table.clear()
        self._synced = 0

    def sync(self, tokens: list[int]):
        """Track the accepted context (stream only ever grows)."""
        w = self.window
        first = max(0, self._synced - w + 1)
        for i in range(first, len(tokens) - w + 1):
            self.table.setdefault(tuple(tokens[i:i + w]), []).append(i)
        self._synced = len(tokens)

    def propose(self, tokens: list[int], gamma: int) -> list[int]:
        """Up to `gamma` tokens that followed the current suffix last time."""
        w = self.window
        if len(tokens) < w or gamma <= 0:
            return []
        occurrences = self.table.get(tuple(tokens[-w:]))
        if not occurrences:
            return []
        start = max((i for i in occurrences if i < len(tokens) - w), default=None)
        if start is None:
            return []
        out_start = start + w
        return tokens[out_start:out_start + gamma]


class ModelDrafter:
    """A small LM that drafts `gamma` tokens greedily (or by sampling).

    It owns its own KV pool + block manager; a Sequence object carries the
    draft state. `rewind` drops the draft KV beyond the accepted prefix --
    stale slots in exclusive blocks are simply overwritten later.
    """

    def __init__(self, worker: "KVWorker"):
        self.worker = worker

    @torch.no_grad()
    def propose(self, seq, context_len: int, gamma: int,
                temperature: float = 0.0, top_k: int = -1, top_p: float = 1.0,
                generator: torch.Generator | None = None
                ) -> tuple[list[int], list[torch.Tensor]]:
        """Draft gamma tokens conditioned on tokens[0:context_len).

        KV sync: make the draft's computed frontier exactly context_len-1,
        then re-forward the last context token -- its logits produce the first
        proposal (a 1-token recompute, same trick as forced-recompute of a
        fully cached prompt). Returns (tokens, q_probs) where q_probs[i] is
        the draft's probability of tokens[i] for rejection sampling."""
        from minivllm.sampling import probs_from_logits, sample_from_logits

        w = self.worker
        frontier = context_len - 1
        if seq.num_computed_tokens > frontier:
            seq.num_computed_tokens = frontier             # logical rewind
        elif seq.num_computed_tokens < frontier:
            w.forward_span(seq, seq.num_computed_tokens, frontier)

        tokens: list[int] = []
        probs: list[torch.Tensor] = []
        # logits at position frontier-1 give the first proposal (position frontier)
        logits = w.forward_span(seq, frontier, context_len)[-1]
        for i in range(gamma):
            if temperature == 0.0:
                tok = int(torch.argmax(logits).item())
            else:
                tok = sample_from_logits(logits, temperature, top_k, top_p,
                                         generator=generator)
            q = probs_from_logits(logits, temperature, top_k, top_p)
            tokens.append(tok)
            probs.append(q)
            seq.output_token_ids.append(tok)               # grows seq.tokens
            if i + 1 < gamma:
                # process proposal i (at position context_len+i) to get the
                # distribution for proposal i+1
                logits = w.forward_span(seq, context_len + i,
                                        context_len + i + 1)[-1]
        return tokens, probs
