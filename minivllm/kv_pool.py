"""The physical KV cache pool.

One big pre-allocated tensor holds every layer's K and V for every physical
block, exactly like vLLM's paged KV cache tensor:

    pool.data[block_id, layer, {0=K,1=V}, kv_head, slot_in_block, head_dim]

A sequence never owns contiguous memory: it references physical blocks through
its block table, and attention gathers K/V block by block.
"""
import torch


class KVCachePool:
    def __init__(self, num_blocks: int, num_layers: int, num_kv_heads: int,
                 block_size: int, head_dim: int, dtype: torch.dtype, device: str):
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim

        self.data = torch.zeros(
            num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device)

    def layer_kv(self, layer: int):
        """Views shaped [num_blocks, kv_heads, block_size, head_dim] for one layer."""
        return self.data[:, layer, 0], self.data[:, layer, 1]

    def copy_block(self, src: int, dst: int):
        """Used by copy-on-write: duplicate one block across all layers."""
        self.data[dst].copy_(self.data[src])

    def num_bytes(self) -> int:
        return self.data.numel() * self.data.element_size()
