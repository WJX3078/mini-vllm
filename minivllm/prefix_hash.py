"""Pluggable prefix-cache hash backends.

A prefix-cache key must satisfy: key_i == key_j  <=>  the token prefixes
[0, block_i) and [0, block_j) are identical AND both were produced under the
same `metadata` (model identity today). The chain structure gives the
prefix property; the backend picks the concrete representation:

* TupleBackend   -- chained tuples with the metadata as a synthetic root:
                   key = (("metadata", meta), blk0, blk1, ...). Equality is
                   structural comparison, so correctness is airtight (no
                   collisions), but each key holds its whole ancestor chain:
                   O(depth) memory and O(depth) compare on long contexts.

* SHA256Backend  -- fixed-size 32-byte digests:
                   H_i = SHA256(H_{i-1} || tokens || metadata). O(1) memory
                   and compare per key. Cryptographic collisions are the
                   same trade-off production vLLM makes with partial hashes.

`metadata` folds engine-scoped facts into the root of the chain (model
identity today). Production systems additionally salt the hash with LoRA
adapter ids, modality-specific preprocessing signatures, and tenant salts:
two requests must only share KV if *everything that influenced the KV
computation* is identical, otherwise a hit silently serves wrong values.
"""
import hashlib
import struct
from collections.abc import Hashable

#: A prefix-cache key: tuple-chains for TupleBackend, 32-byte digests for
#: SHA256Backend. Hashable covers both.
HashKey = Hashable


class PrefixHashBackend:
    """hash_block(parent_hash, block_tokens, metadata) -> new hash key."""

    name = "base"

    def hash_block(self, parent_hash: HashKey | None,
                   block_tokens: tuple[int, ...], metadata: str) -> HashKey:
        raise NotImplementedError


class TupleBackend(PrefixHashBackend):
    """Chained tuples rooted at ("metadata", metadata): model identity is
    part of every key, so different engines/models can never collide."""

    name = "tuple"

    def hash_block(self, parent_hash: tuple | None,
                   block_tokens: tuple[int, ...], metadata: str) -> tuple:
        if parent_hash is None:
            return (("metadata", metadata), block_tokens)
        return (parent_hash, block_tokens)


class SHA256Backend(PrefixHashBackend):
    """32-byte digest chain: H(parent || tokens || metadata)."""

    name = "sha256"

    def hash_block(self, parent_hash: bytes | None,
                   block_tokens: tuple[int, ...], metadata: str) -> bytes:
        h = hashlib.sha256()
        if parent_hash:
            h.update(parent_hash)
        h.update(struct.pack(f"<{len(block_tokens)}q", *block_tokens))
        h.update(metadata.encode("utf-8"))
        return h.digest()


def make_hash_backend(name: str) -> PrefixHashBackend:
    backends = {"tuple": TupleBackend, "sha256": SHA256Backend}
    if name not in backends:
        raise ValueError(f"unknown prefix hash backend {name!r}; "
                         f"choose from {sorted(backends)}")
    return backends[name]()
