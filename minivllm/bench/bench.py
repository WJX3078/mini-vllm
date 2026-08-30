"""Benchmark: mini-vllm vs HuggingFace (and vLLM when installed).

Methodology (what makes these numbers trustworthy)
--------------------------------------------------
* CUDA is synchronized before/after every timed region; wall time is real
  GPU-completed time, not kernel-launch time.
* Every configuration runs WARMUP runs first (default 3), then MEASURED
  runs (default 5). Reported throughput is the MEDIAN across measured runs;
  TTFT / TPOT / E2E percentiles (p50/p95/p99) pool all measured requests.
* Fairness: both engines see IDENTICAL token-id prompts, the same
  tokenizer, dtype, device, sampling params and output length. By default
  the benchmark runs greedy with ignore_eos on BOTH sides so every request
  generates exactly `output_len` tokens (wall-clock is then a pure engine
  comparison; EOS behavior is workload-controlled, not engine-controlled).
* HF latency is measured with a custom generation loop (use_cache), so TTFT
  is the real first forward step and TPOT the real per-step time -- not
  batch_latency / num_tokens. For batched HF runs, TTFT is the batch's
  first step (shared by the batch members) and is flagged as approximate.
* vLLM baseline is used only if `import vllm` works; otherwise it is
  skipped with a note (the benchmark never hard-depends on it).

Metrics
-------
throughput tok/s, requests/s, TTFT / TPOT / E2E (mean, p50, p95, p99),
prefix-cache hit rate, preemptions, peak KV blocks + utilization (mini-vllm);
acceptance rate / tokens per round / speedup (speculative mode).
"""
import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass, field

import torch


# ----------------------------------------------------------------- workload
@dataclass
class Workload:
    num_prompts: int
    input_len: int
    output_len: int
    shared_prefix_ratio: float = 0.0
    seed: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    concurrency: int = 8           # mini-vllm max in-flight requests
    base_seed: int = 1234          # per-request RNG seed: base_seed + i


@dataclass
class BenchResult:
    engine: str
    wall_time: float
    num_requests: int
    total_output_tokens: int
    ttfts: list[float] = field(default_factory=list)
    tpots: list[float] = field(default_factory=list)
    e2es: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    approximate: dict = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        return self.total_output_tokens / self.wall_time if self.wall_time else 0.0

    @property
    def requests_per_s(self) -> float:
        return self.num_requests / self.wall_time if self.wall_time else 0.0


def build_token_workload(tok, wl: Workload):
    """Token-id prompts: a shared prefix (ratio * input_len tokens from a
    fixed random pool) + per-request random suffix. Building at token level
    guarantees both engines see byte-identical inputs."""
    rng = random.Random(wl.seed)
    vocab_hint = getattr(tok, "vocab_size", 32000) or 32000
    safe_max = min(vocab_hint - 1, 30000)
    shared_len = int(wl.input_len * wl.shared_prefix_ratio)
    shared = [rng.randrange(100, safe_max) for _ in range(shared_len)]
    prompts = []
    for _ in range(wl.num_prompts):
        suffix = [rng.randrange(100, safe_max)
                  for _ in range(wl.input_len - shared_len)]
        prompts.append(shared + suffix)
    return prompts


# ------------------------------------------------------------------ metrics
def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def dist(xs: list[float]) -> dict:
    if not xs:
        return {"mean": float("nan"), "p50": float("nan"),
                "p95": float("nan"), "p99": float("nan")}
    return {"mean": statistics.fmean(xs), "p50": pct(xs, 50),
            "p95": pct(xs, 95), "p99": pct(xs, 99)}


def fmt_ms(x: float) -> str:
    return f"{x * 1000:.1f}"


def result_to_json(r: BenchResult, wl: Workload, model: str, device: str,
                   dtype: str) -> dict:
    return {
        "engine": r.engine,
        "model": model,
        "device": device,
        "dtype": dtype,
        "workload": {
            "num_prompts": wl.num_prompts, "input_len": wl.input_len,
            "output_len": wl.output_len,
            "shared_prefix_ratio": wl.shared_prefix_ratio,
            "temperature": wl.temperature, "top_p": wl.top_p,
            "top_k": wl.top_k, "seed": wl.seed,
            "ignore_eos": True,
        },
        "throughput_tok_s": round(r.throughput, 2),
        "requests_s": round(r.requests_per_s, 3),
        "wall_time_s": round(r.wall_time, 3),
        "ttft": {k: round(v * 1000, 2) for k, v in dist(r.ttfts).items()},
        "tpot": {k: round(v * 1000, 3) for k, v in dist(r.tpots).items()},
        "e2e": {k: round(v * 1000, 2) for k, v in dist(r.e2es).items()},
        "prefix_cache": r.extra.get("prefix_cache"),
        "kv_cache": r.extra.get("kv_cache"),
        "preemptions": r.extra.get("preemptions"),
        "approximate": r.approximate or None,
        "extra": {k: v for k, v in r.extra.items()
                  if k not in ("prefix_cache", "kv_cache", "preemptions")},
    }


def print_result(r: BenchResult):
    d_t, d_p, d_e = dist(r.ttfts), dist(r.tpots), dist(r.e2es)
    lines = [
        f"[{r.engine}] requests={r.num_requests} "
        f"out_tokens={r.total_output_tokens} wall={r.wall_time:.2f}s",
        f"  throughput : {r.throughput:.1f} tok/s | {r.requests_per_s:.2f} req/s",
        f"  TTFT ms    : mean {fmt_ms(d_t['mean'])} | p50 {fmt_ms(d_t['p50'])}"
        f" | p95 {fmt_ms(d_t['p95'])} | p99 {fmt_ms(d_t['p99'])}",
        f"  TPOT ms    : mean {fmt_ms(d_p['mean'])} | p50 {fmt_ms(d_p['p50'])}"
        f" | p95 {fmt_ms(d_p['p95'])} | p99 {fmt_ms(d_p['p99'])}",
        f"  E2E   s    : mean {d_e['mean']:.2f} | p50 {d_e['p50']:.2f}"
        f" | p95 {d_e['p95']:.2f} | p99 {d_e['p99']:.2f}",
    ]
    for k, v in r.extra.items():
        if k not in ("prefix_cache", "kv_cache"):
            lines.append(f"  {k:<15} : {v}")
    if "prefix_cache" in r.extra and r.extra["prefix_cache"]:
        pc = r.extra["prefix_cache"]
        lines.append(f"  prefix cache    : hit {pc['hit_rate']:.1%} "
                     f"({pc['hits']}/{pc['queries']})")
    if "kv_cache" in r.extra and r.extra["kv_cache"]:
        kv = r.extra["kv_cache"]
        lines.append(f"  kv cache        : peak {kv['peak_blocks']}/"
                     f"{kv['total_blocks']} blocks ({kv['utilization']:.1%} at end)")
    for k, v in r.approximate.items():
        lines.append(f"  ~ {k}: {v}")
    print("\n".join(lines))


def merge_runs(results: list[BenchResult], name: str) -> BenchResult:
    """Median wall time across runs; latency percentiles pooled over all
    measured requests of all runs."""
    med_wall = statistics.median(r.wall_time for r in results)
    ttfts = [x for r in results for x in r.ttfts]
    tpots = [x for r in results for x in r.tpots]
    e2es = [x for r in results for x in r.e2es]
    base = results[0]
    thr = [r.total_output_tokens / r.wall_time for r in results]
    return BenchResult(
        engine=name, wall_time=med_wall, num_requests=base.num_requests,
        total_output_tokens=int(statistics.median(
            r.total_output_tokens for r in results)),
        ttfts=ttfts, tpots=tpots, e2es=e2es,
        extra={**base.extra, "throughput_runs": [round(t, 1) for t in thr],
               "throughput_median": round(statistics.median(thr), 1)},
        approximate=base.approximate)


# ------------------------------------------------------------- mini-vllm
def generate_staggered(engine, prompts, params_list, max_in_flight):
    """Drive the engine with at most `max_in_flight` unfinished requests,
    adding the next one whenever a slot frees up (realistic arrivals: later
    requests land in different scheduling rounds, so prefix caching can
    hit)."""
    results = {}
    inflight, next_to_add = 0, 0
    n = len(prompts)

    def merge(out):
        prev = results.get(out.request_id)
        if prev is None:
            results[out.request_id] = out
        else:
            prev.outputs.extend(out.outputs)

    while next_to_add < n and inflight < max_in_flight:
        engine.add_request(prompts[next_to_add], params_list[next_to_add])
        inflight, next_to_add = inflight + 1, next_to_add + 1
    while engine.scheduler.has_unfinished():
        for out in engine.step():
            merge(out)
            inflight -= 1
        while inflight < max_in_flight and next_to_add < n:
            engine.add_request(prompts[next_to_add], params_list[next_to_add])
            inflight, next_to_add = inflight + 1, next_to_add + 1
    return [results[i] for i in sorted(results)]


def _params_for(wl: Workload):
    from minivllm import SamplingParams
    return [SamplingParams(
        temperature=wl.temperature, top_p=wl.top_p, top_k=wl.top_k,
        max_tokens=wl.output_len, ignore_eos=True, seed=wl.base_seed + i)
        for i in range(wl.num_prompts)]


def run_mini_vllm_once(model_path, prompts, wl: Workload, device, dtype_str,
                       max_num_seqs, block_size, enable_prefix_caching,
                       max_in_flight, max_model_len,
                       max_num_batched_tokens=0,
                       enable_chunked_prefill=True,
                       num_blocks=None) -> BenchResult:
    from minivllm import EngineConfig, LLMEngine

    engine = LLMEngine(EngineConfig(
        model=model_path, device=device, dtype=dtype_str,
        block_size=block_size, max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens or max(2048, max_model_len),
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        num_blocks=num_blocks))
    params = _params_for(wl)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = generate_staggered(engine, prompts, params, max_in_flight)
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    ttfts, tpots, e2es, total = [], [], [], 0
    for out in outputs:
        for o in out.outputs:
            ttfts.append(o["ttft"])
            tpots.append(o["tpot"])
            e2es.append(o.get("e2e", o["ttft"]))
            total += len(o["token_ids"])
    stats = engine.engine_stats()
    extra = {
        "prefix_cache": {"hit_rate": stats["cache_hit_rate"],
                         "hits": stats["cache_hits"],
                         "queries": stats["cache_queries"]},
        "kv_cache": {"peak_blocks": stats["peak_blocks"],
                     "total_blocks": engine.block_manager.num_blocks,
                     "utilization": stats["kv_utilization"]},
        "preemptions": stats["preemptions"],
        "cow_copies": stats["cow_copies"],
    }
    del engine
    if device == "cuda":
        torch.cuda.empty_cache()
    return BenchResult("mini-vllm", wall, len(outputs), total, ttfts, tpots,
                       e2es, extra)


# ---------------------------------------------------------------- HuggingFace
@torch.no_grad()
def hf_generate_profiled(model, batch_ids, wl: Workload, device,
                         ignore_eos: bool):
    """Custom HF generation loop: REAL TTFT (completion of the first
    forward) and per-step TPOT, same sampling params + ignore_eos semantics
    as mini-vllm. Returns per-request (ttft, tpot, e2e, n_tokens).

    With batch_size > 1 the first forward is shared by the whole batch, so
    per-request TTFT is flagged approximate; with batch_size == 1 it is
    exact."""
    from minivllm.sampling import sample_from_logits

    gens = [torch.Generator().manual_seed(wl.base_seed + i)
            for i in range(len(batch_ids))]
    cfg_eos = model.config.eos_token_id
    eos_ids = set(cfg_eos) if isinstance(cfg_eos, list) else {cfg_eos}

    input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    past = None
    finished = [False] * len(batch_ids)
    first_t = [None] * len(batch_ids)
    last_t = [None] * len(batch_ids)
    n_out = [0] * len(batch_ids)

    gen_start = time.perf_counter()
    cur = input_ids
    for _step in range(wl.output_len):
        if device == "cuda":
            torch.cuda.synchronize()
        out = model(cur, attention_mask=attention_mask,
                    past_key_values=past, use_cache=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t_done = time.perf_counter()
        past = out.past_key_values

        logits = out.logits[:, -1, :].float()
        next_ids, any_active = [], False
        for i in range(len(batch_ids)):
            if finished[i]:
                next_ids.append(eos_ids and min(eos_ids))
                continue
            any_active = True
            tok_i = sample_from_logits(logits[i], wl.temperature, wl.top_k,
                                       wl.top_p, generator=gens[i])
            next_ids.append(tok_i)
            if first_t[i] is None:
                first_t[i] = t_done       # real first-token availability
            last_t[i] = t_done
            n_out[i] += 1
            if not ignore_eos and tok_i in eos_ids:
                finished[i] = True
        cur = torch.tensor(next_ids, dtype=torch.long, device=device)[:, None]
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((len(batch_ids), 1), dtype=torch.long,
                        device=device)], dim=1)
        if not any_active:
            break

    res = []
    for i in range(len(batch_ids)):
        ttft = first_t[i] - gen_start if first_t[i] is not None else 0.0
        tpot = ((last_t[i] - first_t[i]) / (n_out[i] - 1)
                if n_out[i] > 1 else 0.0)
        e2e = last_t[i] - gen_start if last_t[i] is not None else 0.0
        res.append((ttft, tpot, e2e, n_out[i]))
    return res


def run_hf_throughput(model_path, prompts, wl: Workload, device, dtype,
                      batch_size, ignore_eos: bool = True) -> BenchResult:
    """THROUGHPUT runner: stock `model.generate`, synchronized only at the
    timed-region boundaries -- no per-token syncs, so HF's own kernel
    scheduling is measured as-is. Latency fields stay empty (the latency
    profiler owns TTFT/TPOT)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device).eval()
    pad_id = model.config.eos_token_id
    if isinstance(pad_id, list):
        pad_id = pad_id[0]

    total = 0
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        input_ids = torch.tensor(prompts[i:i + batch_size],
                                 dtype=torch.long, device=device)
        kwargs = dict(max_new_tokens=wl.output_len, do_sample=wl.temperature > 0,
                      pad_token_id=pad_id, temperature=wl.temperature,
                      top_p=wl.top_p)
        if wl.top_k > 0:
            kwargs["top_k"] = wl.top_k
        if ignore_eos:
            kwargs["eos_token_id"] = None
        torch.manual_seed(wl.base_seed)     # HF generate has one global RNG
        out = model.generate(input_ids, **kwargs)
        total += (out.shape[1] - input_ids.shape[1]) * input_ids.shape[0]
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return BenchResult(
        "huggingface", wall, len(prompts), total,
        extra={"batch_size": batch_size,
               "runner": "model.generate (throughput)"},
        approximate={"throughput": "stock generate; latency metrics come "
                                   "from the profiling runner"})


def run_hf_latency(model_path, prompts, wl: Workload, device, dtype,
                   batch_size, ignore_eos: bool = True) -> BenchResult:
    """LATENCY PROFILER: custom use_cache loop with a sync per step -- real
    TTFT/TPOT at the cost of inflated wall time. Its throughput must NOT be
    used for speed comparisons against mini-vllm."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device).eval()

    ttfts, tpots, e2es, total = [], [], [], 0
    for i in range(0, len(prompts), batch_size):
        rows = hf_generate_profiled(model, prompts[i:i + batch_size], wl,
                                    device, ignore_eos)
        for (ttft, tpot, e2e, n) in rows:
            ttfts.append(ttft)
            tpots.append(tpot)
            e2es.append(e2e)
            total += n
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    approximate = {"runner": "profiling runner (per-step sync) -- TTFT/TPOT "
                             "are real; IGNORE its throughput"}
    if batch_size > 1:
        approximate["TTFT"] = (f"approximate: first step shared by batch "
                               f"members (batch={batch_size})")
    return BenchResult("huggingface-latency", 0.0, len(prompts), total,
                       ttfts, tpots, e2es, {"batch_size": batch_size},
                       approximate)


# -------------------------------------------------------------------- vLLM
def make_vllm_runner(model_path, wl: Workload, prompts, enable_prefix_caching: bool,
                     max_model_len: int, max_num_seqs: int, block_size: int):
    """Optional baseline: only when vLLM imports and initializes cleanly.
    Sampling params mirror the workload (temperature/top_p/top_k/max_tokens/
    ignore_eos/seed per request); dtype/max_model_len/prefix caching match
    mini-vllm's config. Returns run_once(prompts) or None."""
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_path, dtype="float16",
                  max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                  block_size=block_size,
                  enable_prefix_caching=enable_prefix_caching,
                  gpu_memory_utilization=0.75)
    except Exception as e:                     # not installed / init failure
        print(f"[vllm] unavailable ({type(e).__name__}: {e}) -- skipped")
        return None
    sp = [SamplingParams(temperature=wl.temperature, top_p=wl.top_p,
                         top_k=wl.top_k if wl.top_k > 0 else None,
                         max_tokens=wl.output_len, ignore_eos=True,
                         seed=wl.base_seed + i)
          for i in range(len(prompts))]

    def run_once(run_prompts):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompt_token_ids=run_prompts, sampling_params=sp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        total = sum(len(o.outputs[0].token_ids) for o in outs)
        return BenchResult("vllm", wall, len(run_prompts), total, [], [], [],
                           {"note": "per-request latency metrics not "
                                    "collected (throughput-only baseline)"})

    return run_once


# ------------------------------------------------------------------ drivers
def prepare_dtype_str(device):
    return "float16" if device == "cuda" else "float32"


def compare(args, wl: Workload):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    dtype_str = prepare_dtype_str(device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = build_token_workload(tok, wl)
    max_model_len = max(wl.input_len + wl.output_len + 64, 1024)
    budget = args.max_num_batched_tokens or max(2048, max_model_len)

    print(f"\n=== workload: {wl.num_prompts} prompts | in~{wl.input_len} "
          f"out={wl.output_len} | shared_prefix={wl.shared_prefix_ratio:.0%} | "
          f"concurrency={wl.concurrency} | greedy="
          f"{wl.temperature == 0.0} | budget={budget} "
          f"(chunked prefill {'ON' if budget < max_model_len else 'n/a'}) ===")

    results = []
    # ---- mini-vllm: warmup N + measured N -> median
    for _ in range(args.warmup):
        run_mini_vllm_once(args.model, prompts, wl, device, dtype_str,
                           args.max_num_seqs, args.block_size,
                           args.enable_prefix_caching, wl.concurrency,
                           max_model_len, budget)
    runs = [run_mini_vllm_once(args.model, prompts, wl, device, dtype_str,
                               args.max_num_seqs, args.block_size,
                               args.enable_prefix_caching, wl.concurrency,
                               max_model_len, budget)
            for _ in range(args.runs)]
    mini = merge_runs(runs, "mini-vllm")
    print_result(mini)
    results.append(mini)

    # ---- HuggingFace throughput: SAME warmup N + measured N -> median
    # (stock generate, no per-token syncs)
    for _ in range(args.warmup):
        run_hf_throughput(args.model, prompts[:args.hf_batch_size], wl, device,
                          dtype, args.hf_batch_size)
    hf_runs = [run_hf_throughput(args.model, prompts, wl, device, dtype,
                                 args.hf_batch_size)
               for _ in range(args.runs)]
    hf = merge_runs(hf_runs, "huggingface")
    print_result(hf)
    results.append(hf)

    # ---- HuggingFace latency profiler: ONE run, never used for throughput
    hf_lat = run_hf_latency(args.model, prompts, wl, device, dtype,
                            args.hf_batch_size)
    print_result(hf_lat)
    results.append(hf_lat)

    # ---- vLLM baseline: SAME warmup N + measured N -> median
    if device == "cuda" and not args.no_vllm:
        runner = make_vllm_runner(args.model, wl, prompts,
                                  args.enable_prefix_caching, max_model_len,
                                  args.max_num_seqs, args.block_size)
        if runner is not None:
            for _ in range(args.warmup):
                runner(prompts)
            v = merge_runs([runner(prompts) for _ in range(args.runs)], "vllm")
            print_result(v)
            results.append(v)

    base = results[0]
    for other in results[1:]:
        if other.engine == "huggingface-latency" or other.throughput <= 0:
            continue
        print(f"\n==> mini-vllm throughput vs {other.engine}: "
              f"{base.throughput / other.throughput:.2f}x")
    return results


def bench_speculative(args):
    """Speculative decoding: same model greedy, with and without drafting.
    Correctness is asserted (spec output == plain greedy output)."""
    from minivllm import EngineConfig, LLMEngine, SamplingParams
    from minivllm.spec.spec_engine import SpeculativeEngine

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
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(args.model)
        prompts = [" ".join(map(str, p)) for p in build_token_workload(
            tk, Workload(args.num_prompts, args.input_len, args.output_len))]

    cfg = EngineConfig(model=args.model, device=device,
                       dtype=str(dtype).split(".")[-1],
                       block_size=args.block_size, max_num_seqs=1,
                       max_model_len=4096, max_num_batched_tokens=4096,
                       enable_prefix_caching=False)

    params = SamplingParams(temperature=0.0, max_tokens=args.output_len,
                            ignore_eos=True)

    def plain_once():
        engine = LLMEngine(cfg)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = engine.generate(prompts[:1], params, use_tqdm=False)
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        del engine
        return out, wall

    def spec_once():
        spec = SpeculativeEngine(cfg, drafter=args.spec_drafter,
                                 num_spec_tokens=args.num_spec_tokens)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = spec.generate(prompts[:1], params, use_tqdm=False)
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        o = outs[0]
        del spec
        return o, wall

    base_out, base_wall = plain_once()           # warmup
    for _ in range(args.warmup):
        plain_once()
    walls = []
    for _ in range(args.runs):
        _, w = plain_once()
        walls.append(w)
    base_wall = statistics.median(walls)
    base_tokens = len(base_out[0].outputs[0]["token_ids"])
    print(f"[plain single-stream] {base_tokens} tokens | median "
          f"{base_wall:.2f}s -> {base_tokens / base_wall:.1f} tok/s "
          f"(runs={args.runs})")

    spec_out, _ = spec_once()                    # warmup + correctness
    assert spec_out.token_ids == base_out[0].outputs[0]["token_ids"], \
        "speculative output must equal plain greedy output"
    for _ in range(args.warmup):
        spec_once()
    swalls, acc, tpr = [], [], []
    for _ in range(args.runs):
        o, w = spec_once()
        swalls.append(w)
        acc.append(o.acceptance_rate)
        tpr.append(o.tokens_per_round)
    spec_wall = statistics.median(swalls)
    print(f"[speculative x{args.spec_drafter} gamma={args.num_spec_tokens}] "
          f"{len(spec_out.token_ids)} tokens | median {spec_wall:.2f}s -> "
          f"{len(spec_out.token_ids) / spec_wall:.1f} tok/s | "
          f"acceptance={statistics.median(acc):.1%} "
          f"tokens/round={statistics.median(tpr):.2f}")
    print(f"==> latency speedup (median of {args.runs} runs): "
          f"{base_wall / spec_wall:.2f}x")


def bench_sampling(args):
    """Sampling microbenchmark: three implementations.

      per-seq   : sample_from_logits per sequence (B syncs/step)   [v0.1]
      cpu-probs : one D2H of the [B, V] probability matrix         [v0.2]
      gpu-native: [B] uniforms H2D + inverse CDF + [B] ids D2H     [v0.3]

    Reports latency, estimated cross-device bytes per step, and speedup.
    The v0.2->v0.3 story is bandwidth, not just time:
        v0.2: O(B*V) D2H  (B=64, V=150k, fp32 -> ~38 MB / step)
        v0.3: O(B) H2D + O(B) D2H (~0.8 KB / step)
    """
    from minivllm.sampling import (
        probs_batch_from_logits,
        sample_from_logits,
        sample_tokens,
    )
    from minivllm.sequence import SamplingParams, Sequence

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"sampling microbenchmark on {device} (best of 20)")
    print(f"{'B':>4} {'vocab':>7} {'per-seq ms':>11} {'cpu-probs ms':>13} "
          f"{'gpu-native ms':>14} {'D2H KiB old->new':>18} {'speedup':>8}")

    def make_seqs(B, seed0=1000):
        seqs = []
        for i in range(B):
            p = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1)
            s = Sequence([1], p)
            s.rng_seed = seed0 + i
            seqs.append(s)
        return seqs

    for vocab in (32_000, 64_000, 150_000):
        for B in (1, 4, 16, 32, 64, 128):
            torch.manual_seed(0)
            logits = torch.randn(B, vocab, device=device)
            seqs = make_seqs(B)

            def per_seq(B=B, logits=logits, seqs=seqs):
                for i in range(B):
                    sample_from_logits(logits[i], 0.7, -1, 0.9,
                                       generator=seqs[i].sampling_generator())

            def cpu_probs(B=B, logits=logits, seqs=seqs):
                # v0.2 path: batched filter+softmax, then the whole
                # [B, V] probability matrix crosses to the CPU
                probs = probs_batch_from_logits(logits, 0.7, -1, 0.9).float()
                probs_cpu = probs.cpu()
                for row in range(B):
                    torch.multinomial(probs_cpu[row], 1,
                                      generator=seqs[row].sampling_generator())

            def gpu_native(logits=logits, seqs=seqs):
                sample_tokens(logits, seqs)

            def bench(fn, reps=20):
                for _ in range(3):
                    fn()
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    fn()
                if device == "cuda":
                    torch.cuda.synchronize()
                return (time.perf_counter() - t0) / reps * 1000

            t1, t2, t3 = bench(per_seq), bench(cpu_probs), bench(gpu_native)
            old_kib = B * vocab * 4 / 1024               # fp32 probs D2H
            new_kib = (B * 4 + B * 8) / 1024             # uniforms H2D + ids D2H
            print(f"{B:>4} {vocab:>7} {t1:>11.2f} {t2:>13.2f} {t3:>14.2f} "
                  f"{old_kib:>10.0f}->{new_kib:>5.1f} {t2 / t3:>7.1f}x")


# --------------------------------------------------------------------- CLI
# -------------------------------------------------------- scheduler bench
def bench_scheduler(args):
    """Scheduler-focused workloads (mini-vllm only), each run with chunked
    prefill ON and OFF:

      A decode-heavy      : many short prompts, long outputs
      B long-prefill+mix  : long prompts staggered behind running decodes
      C prefix-cache hit  : 90% shared prefix, cache ON
      D KV pressure       : small pool -> preemption / recompute
    """
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = prepare_dtype_str(device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    max_model_len = 8192

    scenarios = {
        "A-decode-heavy": Workload(32, 128, 128, 0.0, args.seed,
                                   concurrency=32),
        "B-long-prefill-mix": Workload(8, 4096, 64, 0.0, args.seed,
                                       concurrency=8),
        "C-prefix-cache-hit": Workload(32, 256, 32, 0.9, args.seed,
                                       concurrency=16),
        "D-kv-pressure": Workload(16, 512, 64, 0.0, args.seed,
                                  concurrency=16),
    }
    budget = args.max_num_batched_tokens or 1024
    pressure_blocks = 512 if device == "cuda" else 512   # small on purpose

    json_out = {"model": args.model, "mode": "scheduler", "scenarios": {}}
    for name, wl in scenarios.items():
        for chunked in (True, False):
            num_blocks = pressure_blocks if name.startswith("D") else None
            tag = f"{name}{'/chunked' if chunked else '/whole'}"
            print(f"\n=== scheduler scenario {tag} | budget={budget} ===")
            prompts = build_token_workload(tok, wl)
            for _ in range(args.warmup):
                run_mini_vllm_once(args.model, prompts, wl, device, dtype_str,
                                   args.max_num_seqs, args.block_size, True,
                                   wl.concurrency, max_model_len, budget,
                                   chunked, num_blocks)
            runs = [run_mini_vllm_once(args.model, prompts, wl, device,
                                       dtype_str, args.max_num_seqs,
                                       args.block_size, True, wl.concurrency,
                                       max_model_len, budget, chunked,
                                       num_blocks)
                    for _ in range(args.runs)]
            r = merge_runs(runs, f"mini-vllm[{tag}]")
            print_result(r)
            jr = result_to_json(r, wl, args.model, device, dtype_str)
            jr["scenario"] = name
            jr["chunked_prefill"] = chunked
            json_out["scenarios"].setdefault(name, []).append(jr)
    return json_out


def main():
    ap = argparse.ArgumentParser(description="mini-vllm benchmark")
    ap.add_argument("--mode", choices=["compare", "spec", "sampling",
                                       "scheduler"], default="compare")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--input-len", type=int, default=128,
                    help="single value or comma list (matrix sweep)")
    ap.add_argument("--output-len", type=int, default=64,
                    help="single value or comma list (matrix sweep)")
    ap.add_argument("--shared-prefix-ratio", type=float, default=0.0,
                    help="single value or comma list: 0 / 0.5 / 0.9 ...")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="mini-vllm max in-flight; single or comma list")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=0,
                    help="token budget per step (0 = auto: max(2048, "
                         "max_model_len)). Set BELOW input_len to actually "
                         "exercise chunked prefill, e.g. --input-len 2048 "
                         "--max-num-batched-tokens 256")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--hf-batch-size", type=int, default=1)
    ap.add_argument("--enable-prefix-caching", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--no-vllm", action="store_true",
                    help="do not try the vLLM baseline")
    ap.add_argument("--output", default=None,
                    help="write results to this JSON file")
    ap.add_argument("--spec-drafter", default="ngram",
                    help="'ngram' or an HF model path used as the drafter")
    ap.add_argument("--num-spec-tokens", type=int, default=8)
    args = ap.parse_args()

    def csv(x):
        return [float(v) if "." in str(v) else int(v)
                for v in str(x).split(",")]

    if args.mode == "sampling":
        bench_sampling(args)
        return

    if args.mode == "spec":
        bench_speculative(args)
        return

    if args.mode == "scheduler":
        json_out = bench_scheduler(args)
        if args.output and json_out:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(json_out, f, indent=2, ensure_ascii=False)
            print(f"\nresults written to {args.output}")
        return

    json_out = {"model": args.model, "results": []}
    for input_len in csv(args.input_len):
        for output_len in csv(args.output_len):
            for ratio in csv(args.shared_prefix_ratio):
                for conc in csv(args.concurrency):
                    wl = Workload(num_prompts=args.num_prompts,
                                  input_len=int(input_len),
                                  output_len=int(output_len),
                                  shared_prefix_ratio=float(ratio),
                                  seed=args.seed,
                                  temperature=args.temperature,
                                  top_p=args.top_p, top_k=args.top_k,
                                  concurrency=int(conc))
                    results = compare(args, wl)
                    device = args.device or ("cuda" if torch.cuda.is_available()
                                             else "cpu")
                    dtype_str = prepare_dtype_str(device)
                    for r in results:
                        json_out["results"].append(result_to_json(
                            r, wl, args.model, device, dtype_str))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(json_out, f, indent=2, ensure_ascii=False)
        print(f"\nresults written to {args.output}")


if __name__ == "__main__":
    main()
