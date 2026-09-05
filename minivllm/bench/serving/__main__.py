"""HTTP serving benchmark: closed-loop concurrency + open-loop Poisson QPS
against a running mini-vLLM server (or an OpenAI-compatible endpoint).

Run (server in another terminal):
    python -m minivllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B
Then:
    python -m minivllm.bench.serving --base-url http://localhost:8000 \
        --num-prompts 64 --concurrency 8 --input-len 128 --output-len 64
    python -m minivllm.bench.serving --request-rate 8 ...   (open loop)
    python -m minivllm.bench.serving --workload mixed       (80% short/20% long)

Latency is measured CLIENT-VISIBLE (first SSE chunk = TTFT); the server is
never synchronized from the benchmark side.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import time
from dataclasses import dataclass, field

import httpx

_COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()     or "?"


@dataclass
class RequestStat:
    ttft: float | None = None
    e2e: float = 0.0
    output_tokens: int = 0
    queue_wait: float | None = None
    ok: bool = False
    rejected: bool = False
    error: str | None = None


@dataclass
class BenchResult:
    workload: str
    mode: str
    level: int
    stats: list = field(default_factory=list)
    wall: float = 0.0

    def summarize(self) -> dict:
        ok = [s for s in self.stats if s.ok]
        rejected = [s for s in self.stats if s.rejected]
        failed = [s for s in self.stats if not s.ok and not s.rejected]
        ttfts = [s.ttft for s in ok if s.ttft is not None]
        tpots: list[float] = []
        # TPOT needs per-request token timing; the client measures e2e and
        # derives mean TPOT = (e2e - ttft) / (out_tokens - 1)
        for s in ok:
            if s.ttft and s.output_tokens > 1:
                tpots.append((s.e2e - s.ttft) / (s.output_tokens - 1))
        e2es = [s.e2e for s in ok]
        qwait = [s.queue_wait for s in ok if s.queue_wait is not None]

        def dist(xs):
            if not xs:
                return None
            xs = sorted(xs)
            def pct(p):
                return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]
            return {"mean": statistics.fmean(xs), "p50": pct(50),
                    "p90": pct(90), "p95": pct(95), "p99": pct(99)}

        out_tokens = sum(s.output_tokens for s in ok)
        return {
            "workload": self.workload, "mode": self.mode, "level": self.level,
            "requests": len(self.stats), "ok": len(ok),
            "rejected": len(rejected), "failed": len(failed),
            "wall_s": round(self.wall, 2),
            "request_throughput": round(len(ok) / self.wall, 2) if self.wall else 0,
            "output_token_throughput": round(out_tokens / self.wall, 1)
            if self.wall else 0,
            "ttft_s": dist(ttfts), "tpot_s": dist(tpots), "e2e_s": dist(e2es),
            "queue_wait_s_p95": (dist(qwait) or {}).get("p95"),
        }


def _pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None


def make_prompts(workload: str, num: int, input_len: int, output_len: int,
                 seed: int, base_url: str, model: str):
    """Token-id prompts encoded as plain text chunks the server tokenizes;
    the shared prefix is literal repeated text so the prefix cache hits."""
    rng = random.Random(seed)
    word = "alpha beta gamma delta omega kappa sigma lambda "
    items = []
    for i in range(num):
        if workload == "mixed":
            long = rng.random() < 0.2
            il = min(4096, 4096 - 64 - 64)    # keep il+ol <= max_model_len
            ol = 64
        else:
            il, ol = input_len, output_len
        il = min(il, 4096 - ol - 64)          # server max_model_len guard
        shared = int(il * {"none": 0.0, "p50": 0.5, "p90": 0.9}.get(
            workload, 0.0)) if workload in ("none", "p50", "p90") else \
            int(il * 0.9) if workload == "prefix-hit" else 0
        text = (word * (shared // len(word) + 1))[:shared * 5 // 5]
        suffix = " ".join(rng.choice(["red", "green", "blue", "cold", "warm"])
                          for _ in range(max(1, il - shared)))
        items.append({"prompt": (text + " " + suffix).strip(),
                      "max_tokens": ol, "temperature": 0.0,
                      "ignore_eos": True, "seed": seed + i})
    return items


async def stream_one(client: httpx.AsyncClient, base_url: str, model: str,
                     body: dict, stat: RequestStat, timeout: float):
    t0 = time.perf_counter()
    try:
        async with client.stream(
                "POST", f"{base_url}/v1/completions",
                json={**body, "model": model, "stream": True},
                timeout=timeout) as r:
            if r.status_code == 429:
                stat.rejected = True
                stat.e2e = time.perf_counter() - t0
                return
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                choice = chunk["choices"][0] if chunk.get("choices") else {}
                text = choice.get("text")
                if text is None:
                    text = (choice.get("delta") or {}).get("content")
                if text and stat.ttft is None:
                    stat.ttft = time.perf_counter() - t0
                if text:
                    stat.output_tokens += 1
                if choice.get("finish_reason"):
                    stat.e2e = time.perf_counter() - t0
        stat.ok = True
    except Exception as e:                            # noqa: BLE001
        stat.error = repr(e)
        stat.e2e = time.perf_counter() - t0


async def run_closed_loop(client, base_url, model, prompts, concurrency,
                          timeout, pacer=None):
    stats = [RequestStat() for _ in prompts]
    queue_i = 0
    start = time.perf_counter()

    async def worker():
        nonlocal queue_i
        while True:
            if pacer is not None:
                await pacer()
            i = queue_i
            queue_i += 1
            if i >= len(prompts):
                return
            await stream_one(client, base_url, model, prompts[i], stats[i],
                             timeout)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return stats, time.perf_counter() - start


async def run_open_loop(client, base_url, model, prompts, request_rate,
                        timeout, seed=0):
    args_seed_holder = [seed]
    stats = [RequestStat() for _ in prompts]
    start = time.perf_counter()

    rng = random.Random(args_seed_holder[0])

    async def one(i, delay):
        await asyncio.sleep(delay)
        await stream_one(client, base_url, model, prompts[i], stats[i],
                         timeout)

    # Poisson arrivals: inter-arrival ~ Exp(rate)
    delays = [rng.expovariate(request_rate) for _ in range(len(prompts))]
    delays = [sum(delays[:i + 1]) for i in range(len(delays))]  # schedule times
    await asyncio.gather(*[one(i, delays[i]) for i in range(len(prompts))])
    return stats, time.perf_counter() - start


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


async def bench(args):
    timeout = args.request_timeout or 600.0
    async with httpx.AsyncClient(timeout=timeout) as client:
        # readiness + warmup
        for _ in range(20):
            try:
                r = await client.get(f"{args.base_url}/ready")
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        warm = make_prompts("short", 2, 64, 8, 0, args.base_url, args.model)
        for w in warm:
            await client.post(f"{args.base_url}/v1/completions",
                              json={**w, "model": args.model})
        print(f"server ready ({args.base_url}); warmup done")

        results = []
        for level in args.levels:
            for workload in args.workloads.split(","):
                prompts = make_prompts(workload, args.num_prompts,
                                       args.input_len, args.output_len,
                                       args.seed, args.base_url, args.model)
                if args.request_rate:
                    mode, level_use = "open-loop-qps", args.request_rate
                    stats, wall = await run_open_loop(
                        client, args.base_url, args.model, prompts,
                        args.request_rate, timeout)
                else:
                    mode, level_use = "closed-loop-concurrency", level
                    stats, wall = await run_closed_loop(
                        client, args.base_url, args.model, prompts, level,
                        timeout)
                res = BenchResult(f"{workload}", mode, level_use, stats, wall)
                results.append(res)
                s = res.summarize()
                print(f"\n[{workload} | {mode}={level_use}] "
                      f"ok={s['ok']} rejected={s['rejected']} "
                      f"wall={s['wall_s']}s")
                print(f"  request/s={s['request_throughput']} "
                      f"out tok/s={s['output_token_throughput']}")
                for k in ("ttft_s", "tpot_s", "e2e_s"):
                    d = s[k]
                    if d:
                        print(f"  {k[:-2]}: mean={d['mean']*1000:.0f}ms "
                              f"p50={d['p50']*1000:.0f} p95={d['p95']*1000:.0f} "
                              f"p99={d['p99']*1000:.0f}")
        return results


def main():
    ap = argparse.ArgumentParser(description="mini-vllm serving benchmark")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--concurrency", default="1,4,16",
                    help="closed-loop levels, comma separated")
    ap.add_argument("--request-rate", type=float, default=0.0,
                    help="open-loop Poisson QPS (0 = closed loop)")
    ap.add_argument("--workload", dest="workloads", default="short",
                    help="short | medium | long | mixed | prefix-hit "
                         "(comma separated)")
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None, help="write JSON here")
    args = ap.parse_args()

    args.levels = [int(x) for x in str(args.concurrency).split(",")]
    results = asyncio.run(bench(args))

    report = {
        "hardware": {"gpu": gpu_info()},
        "model": args.model, "git_commit": _COMMIT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {"base_url": args.base_url,
                          "request_rate": args.request_rate,
                          "input_len": args.input_len,
                          "output_len": args.output_len,
                          "seed": args.seed},
        "results": [r.summarize() for r in results],
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nresults written to {args.output}")


if __name__ == "__main__":
    main()
