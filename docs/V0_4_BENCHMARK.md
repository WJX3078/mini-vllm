# mini-vLLM v0.4 — Serving Benchmark Report

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop 8GB |
| OS | Windows 10.0.26200（win32） |
| Python | 3.10.9 |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0（triton-windows） |
| Model | Qwen/Qwen2.5-0.5B（本地缓存） |
| dtype | float16 |
| KV block size | 16 |
| max_num_seqs | 16 |
| max_num_batched_tokens | 2048 |
| prefix cache | enabled |
| Server | `python -m minivllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B --max-num-seqs 16 --max-num-batched-tokens 2048` |
| Client | `python -m minivllm.bench.serving ...`（httpx，client-visible 延迟） |
| git commit | e7ddbfe 之后（v0.4） |

所有延迟为 client-visible（HTTP SSE 首 chunk = TTFT）；benchmark 不对
server 侧做任何 synchronize。每档先 warmup 2 请求。

## Closed-loop 并发（short：~128 in / 64 out，48 prompts）

| 并发 | 吞吐 (req/s) | 输出吞吐 (tok/s) | TTFT mean/p50/p99 (ms) | E2E mean/p99 (ms) |
|---|---|---|---|---|
| 1 | 0.29 | 17.9 | 143 / 139 / 187 | 3516 / 4176 |
| 8 | 1.90 | 119.7 | 197 / 195 / 224 | 4199 / 4308 |
| 32 | 3.15 | 198.7 | 3684 / 4984 / 5690 | 8502 / 10361 |

结论：并发 1→8 吞吐 **6.7x**（continuous batching 生效）；并发 32 时
16 个 running 槽位饱和，TTFT 因排队升至 ~5s（排队论预期行为，非异常）。

## Open-loop Poisson QPS=8（short，24 prompts）

arrival ~ Exp(8/s)（v0.4 修复后真实泊松）：
req/s 2.47（到达率 8 超过服务能力时排队，符合 open-loop 语义——
这是它的价值：暴露过载下的 tail）
TTFT mean 1082ms / p50 240ms / p95 2940ms / p99 2945ms
E2E mean 5467ms / p99 6965ms

## Mixed workload（80% short 128/64 + 20% long ~4K/64，并发 8，12 prompts）

ok=12（v0.4 早期超长 prompt 被 400 正确拒绝后已 clamp）
out tok/s 9.8；TTFT mean 5933ms / p99 12221ms；E2E p99 52.7s

长 prefill 的 20% 请求主导了队尾延迟：chunked prefill 切片仍要排队算完
（server 端 budget=2048，一个 4K prompt ≈ 2 个 iteration）。这正是
mixed workload 要暴露的 tail 行为；改进方向是 prefill 优先级/预算公平
（vLLM 的 preemption-of-prefills）。

## Prefix-hit workload（90% 共享前缀，24×~1.3K in，并发 8）

ok=24；TTFT mean 2452ms / p99 2718ms；输出 9.3 tok/s
（server metrics：prefix_cache_hits/queries 数据可在 /metrics 观察；
单流 9.3 tok/s 的瓶颈是 HTTP SSE 逐 chunk + CPU 0.5B decode，非 KV）。

## Engine e2e 对照（同机，HTTP 之外的引擎直驱，v0.3 数据）

batch=8/128in/64out：engine 直驱 94.8 tok/s（triton backend）vs
HTTP serving 输出 ~119 tok/s @ 并发 8（HTTP 路径并发更高是因为
serving 测试 server max_num_seqs=16 且请求持续到达填满批）。
HTTP 层开销：client-visible TTFT − engine TTFT ≈ 5-15ms/请求
（本地回环 + SSE flush）。

## 复现

```bash
python -m minivllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B --max-num-seqs 16 --max-num-batched-tokens 2048
python -m minivllm.bench.serving --base-url http://127.0.0.1:8321 \
    --model Qwen/Qwen2.5-0.5B --num-prompts 48 --concurrency 1,8,32 \
    --workload short --output results.json
python -m minivllm.bench.serving --request-rate 8 --workload mixed ...
```

## 已知失真

* 笔记本 GPU 功耗状态波动会造成轮次间 ±20% 吞吐漂移（多轮中位数已缓解）；
* HTTP bench 的"output_tokens"按 SSE 文本 chunk 计数（chunk≈token，
  byte 级 tokenizer 下与真实 token 数有 ±10% 误差）；
* vLLM 对比基线：本机未安装 vLLM —— **UNVERIFIED，requires vLLM install**。
