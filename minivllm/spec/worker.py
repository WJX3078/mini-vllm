"""Single-model paged-KV worker used by the speculative engine.

Wraps (model, BlockSpaceManager) and provides span-forward primitives over
one Sequence's KV state. The target engine and the draft model each get one
worker; the draft worker is cheap because its model is small (or shared).
"""
import torch

from minivllm.attention import SeqInput
from minivllm.block_manager import BlockSpaceManager


class KVWorker:
    def __init__(self, model, num_layers: int, num_kv_heads: int, head_dim: int,
                 num_blocks: int, block_size: int, dtype, device: str,
                 enable_prefix_caching: bool = False):
        self.model = model
        self.bm = BlockSpaceManager(num_blocks=num_blocks, block_size=block_size,
                                    num_layers=num_layers, num_kv_heads=num_kv_heads,
                                    head_dim=head_dim, dtype=dtype, device=device,
                                    enable_prefix_caching=enable_prefix_caching)
        self.device = device

    def allocate(self, seq) -> int:
        cached = self.bm.allocate_sequence(seq)
        assert cached is not None, "KV worker allocation failed"
        return cached

    def free(self, seq):
        self.bm.free_sequence(seq)

    @torch.no_grad()
    def forward_span(self, seq, start: int, end: int,
                     register: bool = False) -> torch.Tensor:
        """Compute K/V for tokens[start:end] (must continue from the current
        num_computed frontier) and return their logits [end-start, vocab]."""
        assert start == seq.num_computed_tokens, \
            f"non-contiguous span {start} != computed {seq.num_computed_tokens}"
        if end <= start:
            return torch.empty(0)
        assert self.bm.prepare_slots(seq, start, end), "no free KV slot"
        device = self.device
        n = end - start
        ids = torch.tensor(seq.tokens[start:end], dtype=torch.long, device=device)
        positions = torch.arange(start, end, device=device)
        si = SeqInput(q_start=start, q_len=n,
                      block_table=torch.tensor(seq.block_table, dtype=torch.long,
                                               device=device),
                      t0=0)
        logits = self.model(ids, positions, self.bm.pool, [si],
                            torch.arange(n, device=device))
        seq.num_computed_tokens = end
        if register:
            self.bm.register_filled_blocks(seq, end)
        return logits
