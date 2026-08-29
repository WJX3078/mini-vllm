"""Pluggable prefix-cache hash backends.

A prefix-cache key must satisfy: key_i == key_j  <=>  the token prefixes
[0, block_i) and [0, block_j) are identical. The chain structure
(parent_hash, block_tokens) gives this structurally; the backend only picks
the concrete representation:

* TupleBackend   -- nested Python tuples. Equality is structural comparison,
                   so correctness is trivially airtight (no collisions), but
                   each key holds the whole ancestor chain: O(depth) memory
                   and O(depth) compare on long contexts.

* SHA256Backend  -- fixed-size 32-byte digest chaining
                   H_i = SHA256(parent || tokens || metadata). O(1) memory
                   per key, O(1) compare. Cryptographic collisions are the
                   same trade-off production vLLM makes when it uses
                   (partial) hashing for cache keys.

`metadata` folds engine-scoped facts into the root of the chain (model
identity today). Production systems additionally salt the hash with LoRA
adapter ids, modality-specific preprocessing signatures, and tenant salts:
two requests must only share KV if *everything that influenced the KV
computation* is identical, otherwise a hit silently serves wrong values.
"""
import hashlib
import struct


class PrefixHashBackend:
    """hash_block(parent_hash, block_tokens, metadata) -> new hash key."""

    name = "base"

    def hash_block(self, parent_hash, block_tokens: tuple[int, ...],
                   metadata: str) -> object:
        raise NotImplementedError


class TupleBackend(PrefixHashBackend):
    """Chained tuples: key = (parent_key, tokens). Structural equality."""

    name = "tuple"

    def hash_block(self, parent_hash: tuple | None,
                   block_tokens: tuple[int, ...], metadata: str) -> tuple:
        return (parent_hash if parent_hash is not None else (), block_tokens)


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
