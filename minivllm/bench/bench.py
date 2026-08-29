"""Benchmark: mini-vllm (paged KV + continuous batching + prefix caching)
vs HuggingFace `model.generate`.

Metrics
-------
* throughput   : total generated tokens / wall time (all requests)
* TTFT         : time from request arrival to its first generated token
* TPOT         : (t_last - t_first) / (num_output_tokens - 1) per request
* cache hit %  : prefix-cache hits / lookups (mini-vllm only)

Methodology notes
-----------------
HF batched generate() gives every request in a batch the same schedule, so
per-request TTFT is not meaningful there; we report batch-level latency for
the HF baseline and true per-request TTFT/TPOT for mini-vllm. For a fair
single-stream comparison run both with --batch-size 1.
"""
import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import List

import torch


# ----------------------------------------------------------------- workload
def build_text_workload(num_prompts: int, input_len: int,
                        shared_prefix_len: int, seed: int = 0):
    """Text prompts: a shared system prefix + random suffix words."""
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "omega", "kappa", "sigma",
             "epsilon", "theta", "lambda", "zeta", "tau", "rho", "psi"]
    prefix = " ".join(rng.choice(words) for _ in range(shared_prefix_len))
    prompts = []
    for _ in range(num_prompts):
        suffix_len = max(1, input_len - shared_prefix_len)
        suffix = " ".join(rng.choice(words) for _ in range(suffix_len))
        prompts.append((prefix + " " + suffix).strip() if prefix else suffix)
    return prompts


# ------------------------------------------------------------------ metrics
@dataclass
class BenchResult:
    name: str
    wall_time: float
    num_requests: int
    total_output_tokens: int
    ttfts: List[float]
    tpots: List[float]
    extra: dict

    def summary(self) -> str:
        def pct(xs, p):
            if not xs:
                return float("nan")
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(len(xs) * p))]

        thr = self.total_output_tokens / self.wall_time
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        lines = [
            f"[{self.name}] requests={self.num_requests} "
            f"out_tokens={self.total_output_tokens} wall={self.wall_time:.2f}s",
            f"  throughput      : {thr:.1f} tok/s",
            f"  TTFT  mean/p50/p99: {mean(self.ttfts)*1000:.1f} / "
            f"{pct(self.ttfts, .5)*1000:.1f} / {pct(self.ttfts, .99)*1000:.1f} ms",
            f"  TPOT  mean/p50/p99: {mean(self.tpots)*1000:.1f} / "
            f"{pct(self.tpots, .5)*1000:.1f} / {pct(self.tpots, .99)*1000:.1f} ms",
        ]
        for k, v in self.extra.items():
            lines.append(f"  {k:<15} : {v}")
        return "\n".join(lines)


def result_from_engine(name, outputs, wall_time, extra=None):
    ttfts, tpots, total = [], [], 0
    for out in outputs:
        for o in out.outputs:
            ttfts.append(o["ttft"])
            tpots.append(o["tpot"])
            total += len(o["token_ids"])
    return BenchResult(name, wall_time, len(outputs), total, ttfts, tpots,
                       extra or {})


# ----------------------------------------------------------------- runners
def generate_staggered(engine, prompts, params, max_in_flight):
    """Drive the engine with at most `max_in_flight` unfinished requests,
    adding the next one whenever a slot frees up. Emulates realistic request
    arrivals (requests land in different scheduling rounds, so prefix caching
    gets a chance to hit, unlike one synchronous submit)."""
    results = {}
    inflight = 0
    next_to_add = 0
    n = len(prompts)

    def merge(out):
        prev = results.get(out.request_id)
        if prev is None:
            results[out.request_id] = out
        else:
            prev.outputs.extend(out.outputs)

    while next_to_add < n and inflight < max_in_flight:
        engine.add_request(prompts[next_to_add], params)
        inflight += 1
        next_to_add += 1
    while engine.scheduler.has_unfinished():
        for out in engine.step():
            merge(out)
            inflight -= 1
        while inflight < max_in_flight and next_to_add < n:
            engine.add_request(prompts[next_to_add], params)
            inflight += 1
            next_to_add += 1
    return [results[i] for i in sorted(results)]


def run_mini_vllm(model_path, prompts, max_tokens, temperature, max_num_seqs,
                  block_size, enable_prefix_caching, device, dtype,
                  max_in_flight=8):
    from minivllm import EngineConfig, LLMEngine, SamplingParams

    engine = LLMEngine(EngineConfig(
        model=model_path, device=device, dtype=str(dtype).split(".")[-1],
        block_size=block_size, max_num_seqs=max_num_seqs,
        max_model_len=4096, max_num_batched_tokens=4096,
        enable_prefix_caching=enable_prefix_caching))
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                            ignore_eos=True)
    t0 = time.perf_counter()
    outputs = generate_staggered(engine, prompts, params, max_in_flight)
    wall = time.perf_counter() - t0
    stats = engine.engine_stats()
    extra = {
        "cache_hit_rate": f"{stats['cache_hit_rate']:.1%}",
        "preemptions": stats["preemptions"],
        "cow_copies": stats["cow_copies"],
        "peak_blocks": f"{stats['peak_blocks']}/{engine.block_manager.num_blocks}",
    }
    result = result_from_engine("mini-vllm", outputs, wall, extra)
    del engine
    return result


def run_hf(model_path, prompts, max_tokens, temperature, batch_size, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device).eval()
    pad_id = tok.pad_token_id or tok.eos_token_id

    ttfts, tpots, total = [], [], 0
    t0 = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, padding_side="left")
        enc = {k: v.to(device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]
        gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0,
                          pad_token_id=pad_id)
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        t_batch = time.perf_counter()
        out = model.generate(**enc, **gen_kwargs)
        t_batch = time.perf_counter() - t_batch
        n = len(batch)
        # batch-level latency split: TTFT ~ 1 step, TPOT ~ the rest
        gen_len = out.shape[1] - input_len
        total += gen_len * n
        approx_ttft = t_batch / gen_len
        approx_tpot = (t_batch - approx_ttft) / max(1, gen_len - 1)
        ttfts.extend([approx_ttft] * n)
        tpots.extend([approx_tpot] * n)
    wall = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    return BenchResult("huggingface", wall, len(prompts), total, ttfts, tpots,
                       {"batch_size": batch_size,
                        "note": "TTFT/TPOT are batch-level approximations"})


def compare(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    prompts = build_text_workload(args.num_prompts, args.input_len,
                                  args.shared_prefix_len, args.seed)
    print(f"workload: {args.num_prompts} prompts x ~{args.input_len} in / "
          f"{args.output_len} out tokens | shared_prefix={args.shared_prefix_len}")

    results = []
    r = run_mini_vllm(args.model, prompts, args.output_len, args.temperature,
                      args.max_num_seqs, args.block_size,
                      args.enable_prefix_caching, device, dtype,
                      max_in_flight=args.max_in_flight)
    print(r.summary()); results.append(r)

    r = run_hf(args.model, prompts, args.output_len, args.temperature,
               args.hf_batch_size, device, dtype)
    print(r.summary()); results.append(r)

    mini, hf = results
    speedup = (mini.total_output_tokens / mini.wall_time) / \
              (hf.total_output_tokens / hf.wall_time)
    print(f"\n==> mini-vllm throughput speedup over HF generate: {speedup:.2f}x")
    return results


def bench_speculative(args):
    """Speculative decoding: same model greedy, with and without drafting."""
    from minivllm import EngineConfig, LLMEngine, SamplingParams
    from minivllm.spec.spec_engine import SpeculativeEngine
    import time as _time

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    if args.spec_drafter == "ngram":
        # copy-style workload makes n-gram lookup shine
        rng = random.Random(0)
        base = [rng.choice(["the", "quick", "brown", "fox", "jumps", "over",
                            "lazy", "dog", "today", "tomorrow"]) for _ in range(64)]
        prompts = [" ".join(base) for _ in range(args.num_prompts)]
        print("workload: repeating-text prompts (n-gram friendly)")
    else:
        prompts = build_text_workload(args.num_prompts, args.input_len, 0)

    cfg = EngineConfig(model=args.model, device=device,
                       dtype=str(dtype).split(".")[-1],
                       block_size=args.block_size, max_num_seqs=1,
                       max_model_len=4096, max_num_batched_tokens=4096,
                       enable_prefix_caching=False)

    # baseline: plain engine, single stream
    engine = LLMEngine(cfg)
    params = SamplingParams(temperature=0.0, max_tokens=args.output_len,
                            ignore_eos=True)
    t0 = _time.perf_counter()
    base_out = engine.generate(prompts[:1], params, use_tqdm=False)
    base_wall = _time.perf_counter() - t0
    base_tokens = len(base_out[0].outputs[0]["token_ids"])
    print(f"[plain single-stream] {base_tokens} tokens in {base_wall:.2f}s "
          f"-> {base_tokens / base_wall:.1f} tok/s")
    del engine

    # speculative
    spec = SpeculativeEngine(cfg, drafter=args.spec_drafter,
                             num_spec_tokens=args.num_spec_tokens)
    t0 = _time.perf_counter()
    outs = spec.generate(prompts[:1], params, use_tqdm=False)
    spec_wall = _time.perf_counter() - t0
    o = outs[0]
    print(f"[speculative x{args.spec_drafter} gamma={args.num_spec_tokens}] "
          f"{len(o.token_ids)} tokens in {spec_wall:.2f}s -> "
          f"{len(o.token_ids) / spec_wall:.1f} tok/s | "
          f"acceptance={o.acceptance_rate:.1%} tokens/round={o.tokens_per_round:.2f}")
    print(f"==> latency speedup: {base_wall / spec_wall:.2f}x")
    assert o.token_ids == base_out[0].outputs[0]["token_ids"], \
        "speculative output must equal plain greedy output"


def main():
    ap = argparse.ArgumentParser(description="mini-vllm benchmark")
    ap.add_argument("--mode", choices=["compare", "spec"], default="compare")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--shared-prefix-len", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--hf-batch-size", type=int, default=4)
    ap.add_argument("--max-in-flight", type=int, default=8,
                    help="max unfinished requests kept in the engine "
                         "(lower = more staggered arrivals)")
    ap.add_argument("--enable-prefix-caching", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec-drafter", default="ngram",
                    help="'ngram' or an HF model path used as the drafter")
    ap.add_argument("--num-spec-tokens", type=int, default=4)
    args = ap.parse_args()

    if args.mode == "compare":
        compare(args)
    else:
        bench_speculative(args)


if __name__ == "__main__":
    main()
