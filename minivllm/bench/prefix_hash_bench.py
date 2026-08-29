"""Prefix-hash backend microbenchmark (P2).

Measures hash-chain construction, lookup, and key memory overhead for the
TupleBackend vs SHA256Backend over 1K / 8K / 32K-token contexts.

Run:  python -m minivllm.bench.prefix_hash_bench [--contexts 1024,8192,32768]
"""
import argparse
import sys

sys.getsizeof  # noqa: B018 -- keep linters aware we use it below


def _deep_size(key) -> int:
    """Approximate retained memory of a hash key (tuples nest)."""
    seen, total, stack = set(), 0, [key]
    while stack:
        k = stack.pop()
        if id(k) in seen:
            continue
        seen.add(id(k))
        total += sys.getsizeof(k)
        if isinstance(k, tuple):
            stack.extend(k)
        elif isinstance(k, (bytes, str)):
            pass
    return total


def bench_backend(backend_name: str, context_len: int, block_size: int,
                  reps: int = 20):
    from minivllm.prefix_hash import SHA256Backend, TupleBackend
    backend = {"tuple": TupleBackend, "sha256": SHA256Backend}[backend_name]()
    tokens = list(range(256)) * (context_len // 256 + 1)
    tokens = tokens[:context_len]
    metadata = "model=bench"

    n_blocks = context_len // block_size
    # ---- construction: build the full chain
    best_build = float("inf")
    keys = []
    for _ in range(reps):
        import time
        t0 = time.perf_counter()
        keys = []
        parent = None
        for i in range(n_blocks):
            parent = backend.hash_block(parent,
                                        tuple(tokens[i * block_size:
                                                    (i + 1) * block_size]),
                                        metadata)
            keys.append(parent)
        dt = time.perf_counter() - t0
        best_build = min(best_build, dt)

    # ---- lookup: dict hits along the chain
    import time
    table = {k: i for i, k in enumerate(keys)}
    best_lookup = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        hits = sum(1 for k in keys if k in table)
        dt = time.perf_counter() - t0
        best_lookup = min(best_lookup, dt)
    assert hits == n_blocks

    mem = _deep_size(keys[-1]) + sum(sys.getsizeof(k) for k in keys)
    return best_build * 1e3, best_lookup * 1e3, mem / 1024


def main():
    ap = argparse.ArgumentParser(description="prefix hash microbenchmark")
    ap.add_argument("--contexts", default="1024,8192,32768")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    print(f"prefix hash backends (block_size={args.block_size}, "
          f"best of {args.reps})")
    print(f"{'context':>8} {'backend':>8} {'build ms':>10} "
          f"{'lookup ms':>10} {'key mem KiB':>12}")
    for ctx in (int(x) for x in args.contexts.split(",")):
        for backend in ("tuple", "sha256"):
            build_ms, lookup_ms, mem = bench_backend(backend, ctx,
                                                     args.block_size,
                                                     args.reps)
            print(f"{ctx:>8} {backend:>8} {build_ms:>10.3f} "
                  f"{lookup_ms:>10.3f} {mem:>12.1f}")


if __name__ == "__main__":
    main()
