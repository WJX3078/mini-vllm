"""Speculative decoding demo (works with a single downloaded model).

* n-gram drafter: proposes tokens that followed the same suffix earlier in
  the context -- fast when the output echoes the input (lists, repetition).
* draft-model drafter: pass an HF path (e.g. a 0.5B draft for a 1.5B target);
  pass "same" to share the target's weights (correctness demo only).

Run:
  python examples/spec_demo.py --drafter ngram
  python examples/spec_demo.py --drafter Qwen/Qwen2.5-0.5B-Instruct --target Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minivllm import EngineConfig, SamplingParams
from minivllm.spec.spec_engine import SpeculativeEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--drafter", default="ngram",
                    help="'ngram', 'same', or an HF model path")
    ap.add_argument("--gamma", type=int, default=6,
                    help="number of speculative tokens per round")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = EngineConfig(model=args.target, device="cpu" if args.cpu else "auto",
                       max_model_len=2048, enable_prefix_caching=False)

    # a repeating-sequence prompt: the continuation echoes earlier context,
    # which is exactly what n-gram lookup drafts well (high acceptance)
    rng = random.Random(0)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "today", "tomorrow"]
    prompt = ("Memorize this sequence and keep repeating it from the start: "
              + " ".join(rng.choice(words) for _ in range(80)) + " ... repeat: ")

    params = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)

    # ---- baseline: plain greedy decoding (single stream)
    spec = SpeculativeEngine(cfg, drafter="ngram", num_spec_tokens=0)
    t0 = time.perf_counter()
    base = spec.generate([prompt], params, use_tqdm=False)[0]
    base_wall = time.perf_counter() - t0

    # ---- speculative
    spec = SpeculativeEngine(cfg, drafter=args.drafter,
                             num_spec_tokens=args.gamma)
    t0 = time.perf_counter()
    out = spec.generate([prompt], params, use_tqdm=False)[0]
    spec_wall = time.perf_counter() - t0

    print(f"output: {out.text!r}")
    print(f"[plain]     {len(base.token_ids)} tokens in {base_wall:.2f}s "
          f"({len(base.token_ids)/base_wall:.1f} tok/s)")
    print(f"[spec x{args.drafter} gamma={args.gamma}] "
          f"{len(out.token_ids)} tokens in {spec_wall:.2f}s "
          f"({len(out.token_ids)/spec_wall:.1f} tok/s) | "
          f"acceptance={out.acceptance_rate:.1%} "
          f"tokens/round={out.tokens_per_round:.2f}")
    assert out.token_ids == base.token_ids, "greedy outputs must match"
    print("==> greedy outputs identical; latency speedup: "
          f"{base_wall/spec_wall:.2f}x")


if __name__ == "__main__":
    main()
