"""Prefix caching demo: N requests sharing one long system prefix.

The 2nd batch of requests hits the cache: prefill shrinks to the unmatched
suffix, so TTFT drops. Compare --no-cache.

Run:  python examples/prefix_cache_demo.py [--no-cache] [--cpu]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minivllm import EngineConfig, LLMEngine, SamplingParams

SYSTEM = (
    "You are a helpful assistant that answers in the style of a pirate. "
    "Always mention the sea, ships and treasure in every answer. "
    "Keep answers short. Here are the rules you must never break: "
) * 6


def run(engine, prompts, tag):
    params = SamplingParams(temperature=0.0, max_tokens=32)
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, params, use_tqdm=False)
    wall = time.perf_counter() - t0
    ttfts = [o["ttft"] for out in outputs for o in out.outputs]
    stats = engine.engine_stats()
    print(f"[{tag}] {len(prompts)} requests, wall={wall:.2f}s, "
          f"mean TTFT={sum(ttfts)/len(ttfts)*1000:.0f}ms, "
          f"cache_hit_rate={stats['cache_hit_rate']:.1%}, "
          f"hits={stats['cache_hits']}/{stats['cache_queries']}")
    return wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    engine = LLMEngine(EngineConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu" if args.cpu else "auto",
        enable_prefix_caching=not args.no_cache,
        max_model_len=2048,
    ))

    first = [SYSTEM + q for q in ["What is 2+2?", "Name a color.", "Say hi."]]
    second = [SYSTEM + q for q in ["What is 3+3?", "Name an animal.", "Say bye."]]

    print("batch 1 (cold):")
    run(engine, first, "cold  ")
    print("batch 2 (same long system prefix):")
    run(engine, second, "warm  ")


if __name__ == "__main__":
    main()
