"""Triton PagedAttention decode microbenchmark (kernel-only).

Compares the PyTorch gather+SDPA path against the Triton page-aware kernel
across context lengths and batch sizes, reporting latency, speedup, and the
estimated KV bytes each decode attention must read
(2 * ctx * kv_heads * head_dim * dtype_bytes per sequence -- the weight
matrix is bandwidth-bound, so bytes moved IS the story).

Run:  python -m minivllm.bench.paged_attention_bench
"""
import argparse

import torch

from minivllm.kernels.paged_attention import (
    paged_attention_decode_torch,
    paged_attention_decode_triton,
    triton_available,
)


def bench(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps * 1000      # us


def main():
    ap = argparse.ArgumentParser(description="paged attention microbenchmark")
    ap.add_argument("--contexts", default="128,512,2048,4096,8192")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--num-q-heads", type=int, default=14)   # Qwen2.5-0.5B
    ap.add_argument("--num-kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if not triton_available():
        raise SystemExit("Triton not available -- kernel cannot run here")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device = "cuda"
    H, KVH, D, bs = (args.num_q_heads, args.num_kv_heads,
                     args.head_dim, args.block_size)
    dtype_bytes = 2
    print(f"paged attention decode kernel | {args.num_q_heads}Q/{KVH}KV "
          f"head_dim={D} block={bs} {args.dtype} | {torch.cuda.get_device_name(0)}")
    print(f"{'ctx':>6} {'B':>4} {'torch us':>10} {'triton us':>10} "
          f"{'speedup':>8} {'KV MiB/step':>12}")

    for ctx in (int(x) for x in args.contexts.split(",")):
        for B in (int(x) for x in args.batches.split(",")):
            nb = (ctx + bs - 1) // bs
            nblk = B * nb
            q = torch.randn(B, H, D, device=device, dtype=dtype)
            k = torch.randn(nblk, KVH, bs, D, device=device, dtype=dtype)
            v = torch.randn(nblk, KVH, bs, D, device=device, dtype=dtype)
            tables = torch.stack([
                torch.randperm(nblk)[:nb] for _ in range(B)]).to(device)
            lens = torch.full((B,), ctx, device=device)

            t_torch = bench(lambda q=q, k=k, v=v, tables=tables, lens=lens:
                            paged_attention_decode_torch(
                                q, k, v, tables, lens, D ** -0.5), args.reps)
            t_triton = bench(lambda q=q, k=k, v=v, tables=tables, lens=lens:
                             paged_attention_decode_triton(
                                 q, k, v, tables, lens, D ** -0.5), args.reps)
            kv_mib = B * ctx * KVH * 2 * D * dtype_bytes / 1024 ** 2
            print(f"{ctx:>6} {B:>4} {t_torch:>10.1f} {t_triton:>10.1f} "
                  f"{t_torch / t_triton:>7.2f}x {kv_mib:>12.1f}")


if __name__ == "__main__":
    main()
