"""GPU runtime step profiler (v0.3).

Instruments one engine run and breaks each decode step into:
    schedule    : scheduler.schedule()          (pure Python)
    metadata    : batch/list building + pinned-staging H2D uploads
    forward     : model forward (GPU, synced)
    sampling    : grouped GPU-native sampling (incl. its D2H)
    bookkeeping : frontier advance / cache registration / stop checks

Usage:
    python -m minivllm.bench.profile_runtime [--model ...] [--batch 8]
        [--input-len 128] [--output-len 64] [--backend auto]
"""
import argparse
import time

import torch

from minivllm import EngineConfig, LLMEngine
from minivllm.sampling import derive_seed
from minivllm.sequence import SamplingParams


def profile(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = LLMEngine(EngineConfig(
        model=args.model, device=device,
        max_num_seqs=args.batch, max_model_len=4096,
        max_num_batched_tokens=max(2048, args.input_len + args.output_len),
        attention_backend=args.backend))
    # make the engine record per-phase timings
    engine.step_timings = {}

    prompts = []
    for i in range(args.batch):
        torch.manual_seed(i)
        ids = torch.randint(100, 30000, (args.input_len,)).tolist()
        prompts.append(ids)
    params = [SamplingParams(temperature=0.0, max_tokens=args.output_len,
                             ignore_eos=True, seed=derive_seed(i))
              for i in range(args.batch)]

    for p in prompts:
        engine.add_request(p, params[0])

    t0 = time.perf_counter()
    while engine.scheduler.has_unfinished():
        engine.step()
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    prof = engine.step_timings
    steps = prof.get("steps", 1)
    tokens = args.batch * args.output_len
    rows = [("schedule", "schedule"), ("metadata", "metadata"),
            ("forward (GPU, synced)", "forward"),
            ("sampling", "sampling"), ("bookkeeping", "bookkeeping")]
    print(f"\nprofile: {args.model} on {device} | batch={args.batch} "
          f"in={args.input_len} out={args.output_len} | "
          f"attention={engine.model.attention_backend}")
    print(f"steps={steps} total wall={wall:.2f}s "
          f"({tokens / wall:.1f} tok/s, {wall / steps * 1000:.1f} ms/step)")
    print(f"{'phase':<24} {'ms/step':>9} {'% of step':>10}")
    total_phase = sum(prof.get(k, 0.0) for _, k in rows)
    for label, key in rows:
        ms = prof.get(key, 0.0) / steps * 1000
        share = prof.get(key, 0.0) / total_phase * 100 if total_phase else 0
        print(f"{label:<24} {ms:>9.2f} {share:>9.0f}%")
    unaccounted = wall - total_phase
    print(f"{'(unaccounted sync/gaps)':<24} "
          f"{unaccounted / steps * 1000:>9.2f} "
          f"{unaccounted / total_phase * 100 if total_phase else 0:>9.0f}%")


def main():
    ap = argparse.ArgumentParser(description="mini-vllm runtime profiler")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--backend", default="auto",
                    help="attention backend: auto / torch / triton")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    profile(args)


if __name__ == "__main__":
    main()
