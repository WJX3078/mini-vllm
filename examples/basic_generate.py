"""Basic mini-vllm generation: batch prompts, continuous batching, n>1.

Run:  python examples/basic_generate.py [--cpu]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minivllm import EngineConfig, LLMEngine, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    engine = LLMEngine(EngineConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu" if args.cpu else "auto",
        max_num_seqs=8,
        max_model_len=1024,
        enable_prefix_caching=True,
    ))

    prompts = [
        "The capital of France is",
        "Write a haiku about paged attention:",
        "1, 2, 3,",
        "Q: Why is KV cache paging useful?\nA:",
    ]
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=48)
    outputs = engine.generate(prompts, params)

    print("\n" + "=" * 60)
    for out in outputs:
        for i, o in enumerate(out.outputs):
            tag = f"[req{out.request_id} sample{i}]"
            print(f"{tag} TTFT={o['ttft']*1000:.0f}ms TPOT={o['tpot']*1000:.1f}ms")
            print(f"    prompt: {out.prompt[:50]}...")
            print(f"    output: {o['text']!r}")
    print("=" * 60)
    print("engine stats:", engine.engine_stats())


if __name__ == "__main__":
    main()
