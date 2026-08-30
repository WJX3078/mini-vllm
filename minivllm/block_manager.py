"""Block-level KV cache manager -- the memory layer of PagedAttention.

Design (mirrors vLLM's BlockSpaceManager, simplified):

* Physical pool: `num_blocks` pre-allocated blocks of `block_size` token slots
  each (all layers).  A block is the smallest unit of allocation.

* Per sequence:  block_table = [phys_block_0, phys_block_1, ...]
  Logical block i of the sequence maps to physical block block_table[i].
  Attention gathers K/V through this mapping -- no contiguous reservation.

* Reference counting: several sequences may map the same physical block
  (prefix cache hits, parallel sampling forks). ref_count tracks holders.

* Copy-on-write: a *partially filled shared* block must be copied before a
  sequence writes into it (e.g. two forks appending different tokens). Full
  blocks are never written in place, so they can be shared freely.

* Prefix caching: for every FULL block we compute a chained hash key
      key_i = (key_{i-1}, tuple(tokens[i*bs : (i+1)*bs])),  key_{-1} = ()
  The chain makes equal keys <=> equal token prefixes. A dict key->block lets
  a new sequence reuse previously computed KV (num_computed_tokens jumps to
  the end of the matched prefix). Unreferenced cached blocks are evicted LRU.

* Preemption support: free_sequence() drops a sequence's refs; cached blocks
  survive (ref==0, evictable), so a preempted sequence re-admitted later gets
  most of its prompt KV back from the cache (recompute-preemption).
"""
from collections import OrderedDict

from minivllm.kv_pool import KVCachePool
from minivllm.prefix_hash import HashKey, PrefixHashBackend, TupleBackend
from minivllm.sequence import Sequence


class BlockAllocationError(RuntimeError):
    pass


class PhysicalBlock:
    __slots__ = ("block_id", "ref_count", "key", "last_access")

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.key: HashKey | None = None  # prefix-cache key when registered
        self.last_access = 0

    @property
    def is_cached(self) -> bool:
        return self.key is not None


class BlockSpaceManager:
    def __init__(self, num_blocks: int, block_size: int, num_layers: int,
                 num_kv_heads: int, head_dim: int, dtype, device: str,
                 enable_prefix_caching: bool = True,
                 hash_backend: PrefixHashBackend | None = None,
                 hash_metadata: str = ""):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_prefix_caching = enable_prefix_caching
        self.hash_backend = hash_backend or TupleBackend()
        self.hash_metadata = hash_metadata
        self.pool = KVCachePool(num_blocks, num_layers, num_kv_heads,
                                block_size, head_dim, dtype, device)

        self.blocks: list[PhysicalBlock] = [PhysicalBlock(i) for i in range(num_blocks)]
        # free list: pop() from the end => lowest ids are handed out first
        self.free_ids: list[int] = list(range(num_blocks - 1, -1, -1))
        # LRU dict: prefix-cache key -> block. move_to_end on every touch.
        self.cached_blocks: OrderedDict[tuple, PhysicalBlock] = OrderedDict()

        # stats
        self.total_reserved_blocks = 0   # sum of outstanding reservations
        self.cache_queries = 0
        self.cache_hits = 0
        self.cow_copies = 0
        self.alloc_watermark = 0   # max concurrently used blocks

        self._clock = 0

    # ------------------------------------------------------------------ stats
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_ids)

    def num_evictable_blocks(self) -> int:
        return sum(1 for b in self.cached_blocks.values() if b.ref_count == 0)

    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks - self.num_evictable_blocks()

    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.cache_queries if self.cache_queries else 0.0

    # ----------------------------------------------------------- internals
    def _touch(self, blk: PhysicalBlock):
        self._clock += 1
        blk.last_access = self._clock

    def _new_block(self) -> PhysicalBlock:
        """Take a block from the free list, evicting LRU cached blocks if needed."""
        if not self.free_ids:
            for key, blk in list(self.cached_blocks.items()):
                if blk.ref_count == 0:
                    del self.cached_blocks[key]
                    blk.key = None
                    self.free_ids.append(blk.block_id)
                    break
            else:
                raise BlockAllocationError("KV cache out of memory")
        blk = self.blocks[self.free_ids.pop()]
        self._touch(blk)
        return blk

    def _key_for(self, parent_key: HashKey | None,
                 tokens: tuple[int, ...]) -> HashKey:
        return self.hash_backend.hash_block(parent_key, tokens, self.hash_metadata)

    # ------------------------------------------------- read-only cache probe
    def get_cached_prefix(self, tokens: list[int]) -> int:
        """Length of the token prefix whose FULL blocks are already in the
        prefix cache -- WITHOUT touching ref_counts or LRU order.

        The scheduler calls this to learn how many tokens really need
        computing BEFORE spending token budget or allocating blocks
        (budget -> admission -> acquire, in that order). Only full blocks
        participate, so the result is always a multiple of block_size and
        the (possibly partial) tail block is never counted.
        """
        if not self.enable_prefix_caching:
            return 0
        bs = self.block_size
        parent = None
        hits = 0
        for i in range(len(tokens) // bs):
            key = self._key_for(parent, tuple(tokens[i * bs:(i + 1) * bs]))
            blk = self.cached_blocks.get(key)
            if blk is None:
                break
            parent = key
            hits += 1
        return hits * bs

    # ---------------------------------------------------------- allocation
    def map_cached_prefix(self, seq: Sequence) -> int:
        """Map the cached full-block prefix into the sequence's block table.

        Only refcounts move -- NO new physical blocks are allocated. Returns
        the number of prompt tokens whose KV is already available (always a
        multiple of block_size; the partial tail block is never cached).
        Stops at the first cache miss, so a prefix maps contiguously.
        """
        assert not seq.block_table, "sequence already has blocks"
        if not self.enable_prefix_caching:
            return 0
        bs = self.block_size
        tokens = seq.tokens
        parent: tuple | None = None
        table: list[int] = []
        keys: list[HashKey | None] = []
        hits = 0
        for i in range(len(tokens) // bs):
            key = self._key_for(parent, tuple(tokens[i * bs:(i + 1) * bs]))
            self.cache_queries += 1
            cached = self.cached_blocks.get(key)
            if cached is None:
                break
            cached.ref_count += 1
            self._touch(cached)
            self.cached_blocks.move_to_end(key)
            self.cache_hits += 1
            hits += 1
            table.append(cached.block_id)
            keys.append(key)
            parent = key
        seq.block_table = table
        seq.block_keys = keys
        return hits * bs

    # ------------------------------------------------------ KV reservation
    def cold_blocks_needed(self, n_tokens: int, cached_len: int) -> int:
        """Cold (not-cache-hit) blocks required to materialize the whole
        prompt: the sequence's full ISL capacity minus cache hits."""
        bs = self.block_size
        return (n_tokens - cached_len + bs - 1) // bs

    def blocks_available_after_mapping(self, cached_len: int) -> int:
        """Capacity a NEW admission may still promise: free + evictable
        blocks, minus blocks that mapping this sequence's cache hits will
        promote out of the evictable pool, minus the OUTSTANDING
        reservations of already-admitted sequences (their promises are
        future demand that must stay coverable)."""
        return (self.num_free_blocks + self.num_evictable_blocks()
                - cached_len // self.block_size - self.total_reserved_blocks)

    def reserve(self, seq: Sequence, cold_blocks: int):
        """Book the sequence's full cold-prompt capacity. Reservation is an
        admission promise, not storage: physical blocks are materialized
        lazily by allocate_span and draw the reservation down."""
        seq.reserved_cold_blocks = cold_blocks
        self.total_reserved_blocks += cold_blocks

    def release_reservation(self, seq: Sequence):
        self.total_reserved_blocks -= seq.reserved_cold_blocks
        seq.reserved_cold_blocks = 0

    def allocate_span(self, seq: Sequence, start: int | None = None,
                      end: int | None = None) -> bool:
        """Materialize physical blocks covering token span [start, end)
        (defaults: the not-yet-computed tail). Draws down the sequence's
        reservation by the number of newly appended blocks; COW copies do
        not count (they duplicate existing storage). Returns False (no
        mutation) if the pool cannot provide the blocks."""
        before = len(seq.block_table)
        if not self.prepare_slots(seq, start, end):
            return False
        if seq.reserved_cold_blocks:
            draw = min(seq.reserved_cold_blocks,
                       len(seq.block_table) - before)
            seq.reserved_cold_blocks -= draw
            self.total_reserved_blocks -= draw
        self.alloc_watermark = max(self.alloc_watermark, self.num_used_blocks())
        return True

    def allocate_sequence(self, seq: Sequence) -> int | None:
        """EAGER admission: map the cached prefix, reserve the cold capacity
        and materialize the whole prompt in one go.

        Returns the cached prefix token count, or None on failure (fully
        rolled back: cache-hit refs dropped, reservation zero, table empty).
        Used by the speculative engine and unit tests; the scheduler uses
        the lazy map -> reserve -> allocate_span path instead.
        """
        cached = self.map_cached_prefix(seq)
        n = len(seq.tokens)
        cold = self.cold_blocks_needed(n, cached)
        if cold > self.blocks_available_after_mapping(cached):
            self.free_sequence(seq)               # unmap the cache hits
            return None
        self.reserve(seq, cold)
        if not self.allocate_span(seq, cached, n):
            self.release_reservation(seq)
            self.free_sequence(seq)
            return None
        return cached

    def fork_sequence(self, parent: Sequence, child: Sequence):
        """Parallel sampling: child shares every block with the parent.

        The (possibly partial) tail block is shared too; whoever appends to it
        first triggers copy-on-write.
        """
        assert parent.block_table, "fork parent has no blocks"
        child.block_table = list(parent.block_table)
        child.block_keys = list(parent.block_keys)
        for bid in child.block_table:
            self.blocks[bid].ref_count += 1

    def free_sequence(self, seq: Sequence):
        """Release a sequence's blocks. Registered (full) blocks stay in the
        prefix cache with ref 0 and are evicted LRU when memory is needed."""
        for bid in seq.block_table:
            blk = self.blocks[bid]
            assert blk.ref_count > 0, f"block {bid} ref={blk.ref_count}"
            blk.ref_count -= 1
            if blk.ref_count == 0 and not blk.is_cached:
                self.free_ids.append(blk.block_id)
        seq.block_table = []
        seq.block_keys = []
        self.total_reserved_blocks -= seq.reserved_cold_blocks
        seq.reserved_cold_blocks = 0

    # ------------------------------------------------------------ appending
    def prepare_slots(self, seq: Sequence, start: int | None = None,
                      end: int | None = None) -> bool:
        """Make sure the block table covers token positions [start, end)
        (defaults: [num_computed, num_tokens)), copy-on-writing shared partial
        blocks. Returns False if not enough free blocks (no mutation then)."""
        if start is None:
            start = seq.num_computed_tokens
        if end is None:
            end = seq.num_tokens
        bs = self.block_size
        need_new = max(0, (end + bs - 1) // bs - len(seq.block_table))
        cow_blocks = set()
        for pos in range(start, end):
            bi = pos // bs
            if bi < len(seq.block_table) and \
                    self.blocks[seq.block_table[bi]].ref_count > 1:
                cow_blocks.add(bi)
        if need_new + len(cow_blocks) > self.num_free_blocks + self.num_evictable_blocks():
            return False

        for bi in sorted(cow_blocks):          # copy-on-write
            old = self.blocks[seq.block_table[bi]]
            new = self._new_block()
            new.ref_count = 1
            old.ref_count -= 1
            if old.ref_count == 0 and not old.is_cached:
                self.free_ids.append(old.block_id)
            self.pool.copy_block(old.block_id, new.block_id)
            seq.block_table[bi] = new.block_id
            # same content => same key chain; a copied block is never registered
            self.cow_copies += 1

        for _ in range(need_new):
            blk = self._new_block()
            blk.ref_count = 1
            seq.block_table.append(blk.block_id)
            seq.block_keys.append(None)
        return True

    # ------------------------------------------------------- prefix caching
    def register_filled_blocks(self, seq: Sequence, upto_tokens: int):
        """After KV for tokens [0, upto_tokens) has been computed, register any
        newly-full blocks in the prefix cache. Content-registered blocks make
        reuse safe: a block enters the cache only after its KV exists."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        for i in range(upto_tokens // bs):
            if i >= len(seq.block_table):
                break
            blk = self.blocks[seq.block_table[i]]
            if blk.is_cached:
                continue
            key = seq.block_keys[i]
            if key is None:                    # block just became full during decode
                key = self._key_for(seq.block_keys[i - 1] if i > 0 else None,
                                    tuple(seq.tokens[i * bs:(i + 1) * bs]))
                seq.block_keys[i] = key
            if key not in self.cached_blocks:
                blk.key = key
                self.cached_blocks[key] = blk
                self._touch(blk)
            # else: another block already owns this key; keep ours private

    # ------------------------------------------------------------- preemption
    def preempt_sequence(self, seq: Sequence):
        """Recompute-preemption: free all blocks; on re-admission the prefix
        cache restores the prompt KV, so only generated tokens are recomputed."""
        self.free_sequence(seq)
        seq.num_computed_tokens = 0
