# mini-vLLM v0.4 — From-Scratch LLM Inference **Serving** Runtime

[![CI](https://github.com/WJX3078/mini-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/WJX3078/mini-vllm/actions/workflows/ci.yml)

一个为**学习推理系统**而写的迷你 vLLM，基于 Qwen2.5-0.5B。
v0.2 调度与正确性 → v0.3 GPU runtime（Triton PagedAttention / GPU 采样 /
KV reservation）→ **v0.4 Production-style Serving**：AsyncLLMEngine、
OpenAI 兼容 HTTP API、SSE 流式、取消/背压/可观测性/HTTP serving benchmark。

```
HTTP (OpenAI API)
  │  POST /v1/completions · /v1/chat/completions（stream=true/false）
  ▼
AsyncLLMEngine ──────────── 单 GPU 线程，唯一 step() 所有者
  │  bounded input queue ── 满 → 429（背压）
  │  per-request bounded output queue ── 慢客户端 → 取消隔离（不丢 token）
  ▼
Continuous Batching Scheduler（token budget · chunked prefill · KV reservation）
  ▼
Paged KV + Triton PagedAttention + GPU-native Sampling
  ▼
增量 detokenizer → SSE（data: {...} / data: [DONE]）→ Client
```

```bash
pip install -e ".[serve]"
python -m minivllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B

curl http://localhost:8000/v1/completions -H "Content-Type: application/json"   -d '{"prompt": "1, 2, 3,", "max_tokens": 8, "temperature": 0}'
# → {"choices":[{"text":" 4, 5, 6","finish_reason":"length"}], "usage": {...}}

curl -N http://localhost:8000/v1/completions -H "Content-Type: application/json"   -d '{"prompt": "hi", "max_tokens": 16, "stream": true}'     # SSE 流式
```

关键 serving 语义（全部有测试）：
* **HTTP 永不绕过调度器**：所有请求进同一个 continuous-batching 引擎线程；
* **取消**：客户端断开 / 超时 / 显式 abort → KV 释放、reservation 清零、
  前缀 refcount 递减，全部幂等（12 个取消回归测试直接断言块管理不变量）；
* **背压**：入口 bounded queue（429）+ 每请求 bounded 输出队列
  （慢客户端被取消而非阻塞 GPU，计入 `slow_client_cancellations_total`）；
* **可观测**：/metrics（Prometheus 文本：request/scheduler/KV/prefix +
  TTFT/E2E 分位数）、/health、/ready、结构化日志（不打 prompt 正文）；
* 增量 detokenizer：有界窗口 + 稳定边界校验，部分 UTF-8 尾部持有，
  流式文本 == 非流式文本（CJK/emoji 拆字测试）。

设计文档：[docs/V0_4_SERVING_DESIGN.md](docs/V0_4_SERVING_DESIGN.md) ·
Adversarial Review：[docs/V0_4_REVIEW.md](docs/V0_4_REVIEW.md) ·
Benchmark：[docs/V0_4_BENCHMARK.md](docs/V0_4_BENCHMARK.md)

---

## 引擎与 GPU Runtime（v0.2/v0.3 能力，v0.4 serving 的地基）

## GPU Runtime 架构（v0.3）

```
Request
  │
  ▼
Scheduler ──────────────  Unified Token Budget（decode 优先 + prefill chunk）
  │  ├─ Token Budget       enable_chunked_prefill=False 时 prompt 必须整段放下
  │  ├─ KV Reservation     准入检查并登记完整 cold-prompt 容量（cache hit 不计）
  │  └─ Block Allocation   物理块按 scheduled span 惰性物化（reserve ≠ allocate）
  ▼
Metadata Builder ────────  pinned staging + 非阻塞 H2D，热路径零新建张量
  │
  ▼
GPU Runtime
  ├── Qwen Forward         varlen 平铺批（RoPE/SDPA/Flash 风格逐层）
  ├── PagedAttention       triton backend：kernel 直接按 block table 遍历物理块
  │   （decode q_len=1）   torch backend：gather + SDPA（fallback，所有设备）
  └── GPU-native Sampler   CPU 各自 RNG 出 u → H2D [B] → GPU top-k/top-p +
                           inverse CDF → D2H [B] token ids
  │
  ▼
Token IDs
```

**三个概念，不要混淆：**

| 概念 | 是什么 | 本项目位置 |
|---|---|---|
| **Paged KV Cache** | 内存管理：KV 切块、block table、引用计数/COW/LRU | `block_manager.py` + `kv_pool.py` |
| **Paged Attention** | kernel 层：attention 直接按页访问分块 KV，不物化连续 K/V | `kernels/paged_attention.py` |
| **Continuous Batching** | 调度策略：iteration 级重组批、chunked prefill、抢占 | `scheduler.py` |

v0.2 只有第一层 + gather/SDPA；**v0.3 的 Triton kernel 让"paged attention"名副其实**。

| 机制 | 代码入口 |
|---|---|
| Block 级 KV Cache（映射/引用计数/COW/LRU/抢占重算） | `minivllm/block_manager.py` |
| **KV Reservation ≠ Allocation**（准入登记容量，span 惰性物化） | `block_manager.py` + `scheduler.py` |
| Continuous batching + Chunked prefill（统一 token 预算） | `minivllm/scheduler.py` |
| Prefix caching（哈希链，metadata 入根，tuple/sha256 可插拔） | `block_manager.py` + `prefix_hash.py` |
| **Triton PagedAttention decode kernel**（online softmax，GQA） | `minivllm/kernels/` |
| **GPU-native sampling**（inverse CDF，D2H 从 O(B·V) 到 O(B)） | `minivllm/sampling.py` |
| Speculative decoding（无损拒绝采样 + committed-stream stop 检查） | `minivllm/spec/` |
| Per-request RNG（批组成无关）/ 增量 stop 检查 | `sampling.py` / `stopping.py` |
| **Runtime step profiler**（schedule/metadata/forward/sampling 分解） | `bench/profile_runtime.py` |

正确性保证（**889 个测试**；CPU CI 覆盖 Python 3.10/3.11/3.12，GPU/Triton 测试用 `gpu`/`triton` marker 与 CI 隔离）：

* fp32 与 HuggingFace `generate` **逐 token 全等**（随机小模型端到端）；
* **Correctness Matrix**：attention backend (torch/triton) × greedy/sampling × 前缀缓存 × chunked prefill × MHA/GQA × 单请求/连续批（`tests/test_correctness_matrix.py`）；
* **Triton kernel 数值矩阵**：3 batch × 7 context（含 1/15/16/17/511 等边界）× 3 block size × 3 head 布局 × 2 head_dim × fp16/bf16 = **758 个组合全部对齐** PyTorch reference（`tests/test_paged_attention.py`）；
* GPU 采样分布统计检验（50k draws）+ GPU/CPU 同 seed 逐 token 一致；
* 投机解码分布无损（数万次试验）+ **stop 串只看 committed 流**（v0.3 修复）；
* property 不变量扫描 + KV reservation 记账一致性。

## 快速开始

```bash
pip install -e .                       # 或 pip install -r requirements.txt
pytest tests/ -q                       # CPU 测试，无需下载模型
ruff check .

# GPU 测试（本机有 RTX 4060 时）
pytest tests/ -m gpu                   # 含 Triton kernel 正确性矩阵

# 真实模型（~1GB，首次自动下载；国内可 export HF_ENDPOINT=https://hf-mirror.com）
python examples/basic_generate.py
python examples/prefix_cache_demo.py                # warm TTFT 1105ms→168ms
python examples/spec_demo.py                        # n-gram 投机解码 4.2x

# Benchmarks
python -m minivllm.bench.bench --mode compare --num-prompts 16 \
    --input-len 128 --output-len 64 --concurrency 16
python -m minivllm.bench.bench --mode compare --input-len 2048 --output-len 64 \
    --max-num-batched-tokens 256        # 明确触发 chunked prefill
python -m minivllm.bench.bench --mode scheduler    # 4 个调度场景 × chunked ON/OFF
python -m minivllm.bench.bench --mode spec --num-spec-tokens 8
python -m minivllm.bench.bench --mode sampling     # 采样 D2H 对比
python -m minivllm.bench.paged_attention_bench     # kernel 微基准
python -m minivllm.bench.profile_runtime           # 逐阶段 step 分解
```

---

## 1. Block 级 KV Cache + KV Reservation（内存层）

### Paged KV 的动机

传统连续 KV cache 要求每序列 K/V 逻辑连续：DynamicCache 逐步 `cat` 有重组开销，StaticCache 按最大长度预留。Paged KV 切成固定 block，block table 映射任意物理块——多序列并发、前缀共享、抢占从此都是块级操作。

机制（`BlockSpaceManager`）：按需分配（内部碎片 ≤1 块）、引用计数、半满块 COW、ref=0 缓存块 LRU 驱逐、recompute 抢占（配合前缀缓存重调度只重算生成部分）。

### v0.3：Reservation ≠ Allocation

v0.2 的 chunked prefill "计算上分片，但准入一次分配整段 prompt 的块"——一个 8K prompt 只算了 256 token 就占着 8K 的显存，提前挤压其他请求并诱发缓存驱逐。v0.3 把两件事拆开：

```
Admission
    ↓ 只读 cache probe（get_cached_prefix，不动 ref_count）
    ↓ 预算检查（只按未缓存 token 扣 max_num_batched_tokens）
    ↓ 容量检查 + 登记：cold_blocks = ceil((prompt - cache_hit) / block_size)
      （cache-hit 块不占新容量；其他序列的未兑现 reservation 计入占用）
Reservation（记账，不是显存）
    ↓
物理块按 scheduled span 惰性物化（allocate_span 逐块扣减 reservation）
```

* `scheduler_reserve_full_isl=True`（默认）：准入检查**整段**冷容量——不过度准入、无 KV 抖动；`=False`：只检查首 chunk，激进准入，代价是可能抢占（有 A/B 开关）；
* **抢占释放 reservation**，重新准入重新登记，记账永远一致（`total_reserved_blocks` 全局校验）；
* 数据（8×8K prompt、22K token 池、budget 256）：full-ISL reservation 把并发准入压到 2 个 prompt（峰值 1002/1400 块，**零抢占**——多出的 6 个在 waiting 排队而不是挤爆池子）；lazy 与 eager 的稳态峰值收敛（准入的 prompt 终会物化完全），lazy 的收益在瞬态占用（首 iteration 只持有首个 chunk 的块，约 3%）与策略解耦本身。

### KV 大小计算（面试必考）

```
每 token KV = 2 (K+V) × n_layers × n_kv_heads × head_dim × dtype_bytes
Qwen2.5-0.5B fp16: 2 × 24 × 2 × 64 × 2B = 12 KB/token（GQA 压缩 7 倍）
```

## 2. Scheduler：统一 Token 预算

```
budget = max_num_batched_tokens
────────────────────────────────────────────────────
① decode 优先   running 各取 1 token；池满 → 抢占队尾（recompute）
② prefill chunk running 未完成 prefill 续算
③ 新请求准入    chunk = min(未缓存余量, 剩余预算)；cache hit 不扣预算
────────────────────────────────────────────────────
enable_chunked_prefill=False：③ 要求整段未缓存余量 ≤ 剩余预算，
否则保持 WAITING——真正的"legacy 语义"（v0.3 修复：之前开关只影响
预算钳制，调度器仍会切 chunk）。
```

## 3. Prefix Caching

块粒度哈希链，**metadata（模型身份）入根**（v0.3 修复：TupleBackend 之前忽略 metadata，README 声称与实现不符）：

```
TupleBackend:  key_i = (key_{i-1}, block_i)，根 = ("metadata", model)
SHA256Backend: key_i = SHA256(key_{i-1} || tokens || metadata)
```

32K context 实测（bs=16）：tuple lookup 159ms/O(链长) vs sha256 0.15ms/O(1)，内存 567KB vs 130KB。生产系统还会把 LoRA adapter id、多模态签名、租户 salt 纳入 metadata——"影响 KV 计算的一切"都必须相同才允许共享。

安全细节：只有满块进缓存；先算后注册（chunked prefill 逐 chunk 注册）；探测严格只读；准入失败自动回滚。

## 4. GPU-native Sampling（D2H 从 O(B·V) 到 O(B)）

三代采样路径的系统故事：

```
v0.1 per-seq   : 每序列 argmax().item()/multinomial → B 次 GPU sync/步
v0.2 cpu-probs : 一次 D2H 搬整张 [B,V] 概率矩阵 → 1 次同步但 O(B·V) 字节
                 （B=64, V=150k, fp32 ≈ 37.5 MB/步！）
v0.3 gpu-native: 每序列私有 CPU Generator 出一个均匀数 u_i
                 → H2D [B] float32
                 → GPU: filter(top-k/top-p) → softmax → cumsum → inverse CDF
                 → D2H [B] int64 token ids
```

关键点：**u 仍然来自每序列自己的 RNG 流**——采样结果是 (logits 行, u) 的纯函数，批组成无关性分毫不动（有专项测试）。inverse CDF 的数值边界（u→1 时 cdf 差几个 ulp）用 `clamp_max(vocab-1)` 兜住。

实测（RTX 4060，best of 20，temperature/top-p 路径）：

| B × V | v0.2 cpu-probs | v0.3 gpu-native | 跨设备字节/步 |
|---|---|---|---|
| 16 × 150k | 35.6 ms | 3.0 ms（11.9x） | 9375 KB → 0.2 KB |
| 64 × 150k | 160.4 ms | 19.5 ms（**8.2x**） | 37500 KB → 0.8 KB |
| 128 × 150k | 344.0 ms | 35.6 ms（9.7x） | 75000 KB → 1.5 KB |

（B=1 时分组开销略负 ~-15%，诚实记录。）

## 5. Triton PagedAttention Decode Kernel

范围严格限定 decode（q_len=1）；fp16/bf16 存储、fp32 累积；MHA + GQA（`kv_head = q_head // group_size`，constexpr 断言整除）。

```
torch backend（fallback）:  paged KV → gather 连续 K/V（物化！）→ SDPA
triton backend            :  Q → block_table → kernel 直接遍历物理块
                             → block-wise online softmax → 加权 V 和 → out
```

kernel 要点（`kernels/paged_attention.py`）：

* **不物化连续 K/V**：grid=(batch, head)，程序内按 block table 逐物理块加载 K/V；
* **online softmax**：跨块维护 running max `m`、exp-sum `l`、加权 `acc`——`m_new=max(m,block_max); α=exp(m−m_new); p=exp(s−m_new); l=α·l+Σp; acc=α·acc+Σp·V`，从不保存完整 attention score；
* 长上下文收益的本质：torch 路径的 gather 要物化 `[B, nb, kvh, bs, D]` 中间张量（B=64/ctx=8192 fp16 ≈ 256 MiB/层/步），Triton 直接页寻址，显存流量即理论 KV 读取量。

**数值正确性矩阵（758 组合全对齐 PyTorch reference）**：batch {1,4,16} × context {1,15,16,17,128,511,1024} × block {8,16,32} × MHA/GQA{8Q/2KV, 16Q/4KV} × head_dim {64,128} × {fp16,bf16}，fp16 atol/rtol≈1e-2。

**kernel 微基准**（14Q/2KV/head_dim=64/fp16，kernel-only，best of 50）：

| ctx \ B | 1 | 8 | 32 | 64 |
|---|---|---|---|---|
| 512 | 4.2x | 7.0x | 9.1x | 10.1x |
| 2048 | 8.3x | 10.3x | 9.2x | 9.1x |
| 8192 | 3.0x | 8.3x | 8.8x | **75.6x**（torch 路径 778ms 崩于物化，Triton 10.3ms） |

**端到端**（batch=8/in=128/out=64，真实模型）：triton backend 94.8 tok/s vs torch backend 81.7 tok/s（**+16%**），greedy 输出逐 token 一致（`test_correctness_matrix` GPU 行）。

`attention_backend: "auto" | "torch" | "triton"`——auto 在 CUDA+Triton 可用时选 triton，否则自动回退 torch；CPU 测试永不依赖 Triton。

## 6. Speculative Decoding（无损 + committed-stream stop）

draft-then-verify：接受概率 min(1, p/q)，拒绝从 norm(max(p−q,0)) 重采样；确定性 drafter（n-gram）是 q=onehot 特例（接受率=p(x)），统计检验证明输出分布严格等于 p。

**v0.3 修复 stop 串的真实调用语义**：stop 检查的候选流必须是 `history + accepted + bonus`（committed），而不是 `history + 全部 proposals`——

* rejected proposal 里出现完整 stop 串 → **不能停**（修复前会假停）；
* bonus token 补全 stop 串 → **必须停**（修复前漏停）；
* EOS/stop/max_tokens 同轮竞争 → 取 token 位置最早者；
* 极端情况 bonus 首个 commit 即触发 stop → 零 token 输出合法（`get_ttft` 返回 0）。

四种场景都有 regression 测试（脚本化 drafter 注入可控 proposals）。

实测（RTX 4060，单流 greedy，n-gram γ=8，5 轮中位）：acceptance 52%、2.56 tokens/round → **2.04x**（复述 demo 场景 4.2x）。

## 7. Runtime Overhead：Profiler 与持久化 metadata

`python -m minivllm.bench.profile_runtime`（batch=8/in=128/out=64，triton backend）：

```
steps=64  wall=5.40s  94.8 tok/s  84.4 ms/step
schedule     1.04 ms/step   1%
metadata     0.61 ms/step   1%     ← pinned staging + 非阻塞 H2D
forward      82.15 ms/step 97%     ← GPU-bound（小模型 decode 的真实分布）
sampling     0.39 ms/step   0%     ← v0.3 GPU-native 采样
bookkeeping  0.14 ms/step   0%
```

结论被数据固定：**0.5B 模型 batch=8 时引擎是 GPU-bound 的**，CPU 侧总开销 <3ms/步；进一步优化空间在 kernel/带宽（v0.2 时代"每步 15ms Python 开销"的故事已由 chunked prefill + GPU 采样 + Triton kernel 分解并解决）。metadata 持久化把每步新建张量/H2D 次数从 3+2S 次降到每 tensor 类别一次（复用 pinned staging + 非阻塞拷贝）。

**CUDA Graph**（架构就绪、默认关闭、未实测）：`enable_cuda_graph` 预留；只适合 all-decode batch（固定 shape 的 metadata buffer 已就位），prefill/varlen/spec 不适合；本机收益预期有限（CPU 侧仅 ~2ms/步），留作 v0.4 实验——不编造数据。

## 8. Benchmark 实测（RTX 4060 Laptop 8GB · fp16 · Qwen2.5-0.5B）

方法学（`bench.py` 头部声明）：计时区间两端 `torch.cuda.synchronize()`；**所有引擎统一 warmup N + measured N**（v0.3 修复：HF/vLLM 之前只跑 1 次），吞吐取中位数，TTFT/TPOT/E2E p50/p95/p99 合并统计；相同 token-id prompt/tokenizer/dtype/采样参数/输出长度（greedy + ignore_eos，输出长度 workload 控制）；**HF 吞吐与延迟拆分**（v0.3）——throughput runner 用原生 `model.generate`（仅计时区间同步，不逐 token sync），latency profiler 用自定义 use_cache 循环逐步计时并标注 profiling（绝不拿它的吞吐做对比）；vLLM 基线参数对齐（dtype/max_model_len/prefix caching/sampling/seed），未安装则跳过。

| workload（greedy, ignore_eos） | mini-vllm | HuggingFace | 加速 |
|---|---|---|---|
| 16×(128in/64out)，HF batch=1 | 78.3 tok/s（中位） | 12.8 tok/s | 6.1x |
| 32×(256in/32out) 共享前缀 87.5% | 125.7 tok/s（命中 65.6%） | 51.5 tok/s（batch=4） | 2.4x |
| 8×(2048in/32out) | TTFT p50 **1.96s** | TTFT 4.04s（batch=8） | TTFT 2.1x |
| engine e2e（batch=8 triton vs torch backend） | 94.8 vs 81.7 tok/s | — | +16% |

调度器专项（`--mode scheduler`，chunked ON vs OFF × 4 场景，真机数据）：

| 场景 | chunked | whole-prompt | 说明 |
|---|---|---|---|
| A decode-heavy（32×128/128） | 187.6 tok/s | **232.8 tok/s** | 无压力时 chunking 的 Python 开销可见（诚实数据） |
| B long-prefill（8×4096/64） | **11.5 tok/s** | 10.4 tok/s | 混批收益 + TTFT 改善 |
| C prefix-hit 90%（32×256/32） | 178.2（命中 92.5%） | 184.4（87.5%） | 命中后 prefill 已很短，差距收窄 |
| D KV 压力（512 块小池） | 1 次抢占 | 2 次抢占 | 两种模式都触发 recompute 路径 |

结论：chunked prefill 不是免费午餐——混入长 prefill 时它救 TTFT，但无压力时每 chunk 的调度开销会吃掉 ~20% 吞吐；这就是 `enable_chunked_prefill` 做成开关、benchmark 给出对比的意义。JSON 输出（`--output`）含全部 workload/metrics 字段便于绘图。

诚实短板：batch=1 单流吞吐低于 HF（Python 循环 vs C++ generate 循环）；Triton kernel 在 ctx=128/B=1 的小 kernel 场景收益有限（launch 开销主导）；本机为笔记本 GPU，绝对数值随功耗波动，以自测为准。

## 9. 正确性验证

1. **单元**（无模型依赖）：块管理/reservation 记账/哈希链双 backend/调度器预算与抢占/增量 stop/inverse-CDF 边界；
2. **端到端 HF 对齐**：chunked×缓存×RNG×并行采样×抢占全组合逐 token 全等；
3. **Correctness Matrix**：backend×模式×缓存×chunking 16 CPU 组合 + GPU triton==torch 行；
4. **Kernel 数值矩阵**：758 组合（上文）；GQA 映射/块表乱序专用测试；
5. **统计检验**：采样分布（4-token 50k）、投机解码无损（数万次）、GPU==CPU 同 seed；
6. **property 不变量**：随机配置扫描 `computed≤tokens`、块表覆盖、reservation 收支平衡、free-list 一致。

## 10. 面试要点速查

**Paged KV Cache 和 Paged Attention 的区别？** 前者是内存管理层（块化+block table+引用计数），后者是 kernel 层（按页直接计算 attention）。有前者没后者=要 gather 物化（v0.2）；两者齐备=kernel 直接页寻址（v0.3）。

**为什么 gather+SDPA 不算真正的 PagedAttention？** 多一次全量 K/V 物化：额外显存流量+中间张量（B=64/ctx=8192 一层 256MiB/步）。真实收益要看 kernel 微基准：同场景 Triton 75x。

**为什么每序列 `.item()` 拖慢 decode？** 每次都强制 GPU→CPU 同步，流水线被打断 B 次/步。

**为什么把 B×V 概率矩阵搬到 CPU 仍然不行？** 同步次数降了，但字节数是 O(B·V)——B=64/V=150k 每步 37.5MB PCIe 流量。正确形态是 O(B)：传 u 上去、传 id 下来，CDF 在 GPU 上算。

**怎么在保持 per-request RNG 下做 GPU batch sampling？** 随机数在 CPU 各自 Generator 产生（一个 u），只把 u 传上 GPU 做逆变换采样。采样结果=(该序列 logits, 该序列 u) 的纯函数。

**Chunked prefill 为什么改善 decode latency？** 长 prompt 不再独占 iteration——decode 先各拿 1 token，剩余预算给 prefill chunk，TTFT 与 TPOT 解耦（2048-token 场景 TTFT 4.0s→2.0s）。

**为什么 chunked prefill 不能无限 admission？** 计算可分片，但 KV 容量是硬约束。所以 v0.3 准入时登记整段容量（reservation）——不超额准入，物理块才敢惰性分配。

**Reservation 和 allocation 为什么分开？** 前者是准入控制的记账（未来最多要多少），后者是当下显存占用。混在一起=要么提前占显存压别人（eager），要么无界准入导致抢占风暴（无 reservation 的 lazy）。分离后两个策略独立可调（`scheduler_reserve_full_isl` × `lazy_block_allocation`）。

**Online softmax 为什么省？** 分块遍历时只维护 (m, l, acc) 三个标量/向量状态，attention score 从不整段存在——显存 O(1) 于块数，这是 flash attention 的核心思想。

**GQA 的 head 映射？** `kv_head = q_head // group_size`，group_size=H/kvh 编译期常量；数值上等价于 K/V repeat_interleave。

**Triton kernel 为什么长 context 收益大？** gather 物化成本 ∝ ctx（且是完整中间张量），kernel 页寻址成本 ∝ KV 理论读取量；ctx 越长差比越大（8192 时 75x）。

**为什么 throughput 和 latency profiling 要分开测？** 逐 token synchronize 的 profiler 会人为放大同步开销——拿它测吞吐等于惩罚对手。吞吐用原生 generate（只在区间两端同步），延迟才用逐步 profiler 并明确标注。

**CUDA Graph 能解决什么？** 消 kernel launch + CPU 调度开销。本机 profile 显示 CPU 侧已 <3ms/步（GPU-bound），收益有限；且只适合 shape 固定的 all-decode 批——prefill/varlen 不行。这不是万能药。

**与 production vLLM 的关键差距？** 无 TP/PP 多卡、无 FP8 KV、CUDA Graph 未实测、投机解码未与 continuous batching 组合、chunked logits/lm_head 分块缺失、无异步 server。但内存层/调度层/采样/kernel 四层的机制与权衡已同构且可用数据回答。

## 项目结构

```
mini-vllm/
├── minivllm/
│   ├── config.py            # EngineConfig（backend/chunking/reservation 策略）
│   ├── sequence.py          # Sequence / SamplingParams / RNG / reservation
│   ├── kv_pool.py           # 物理块池 [blocks, layers, 2, kvh, bs, dim]
│   ├── block_manager.py     # ★ 分页内存 + 只读探测 + reservation/lazy allocation
│   ├── prefix_hash.py       # 可插拔 hash backend（tuple/sha256，metadata 入根）
│   ├── attention.py         # RoPE / 分页 KV 读写 / 批量 SDPA / GQA（torch 路径）
│   ├── kernels/
│   │   └── paged_attention.py  # ★ Triton decode kernel（online softmax, GQA）+ torch 参考
│   ├── model.py             # Qwen2 varlen 前向（backend 分派：triton/torch）
│   ├── scheduler.py         # ★ 统一预算 + chunked prefill + reservation 语义
│   ├── engine.py            # step 循环 / 持久化 metadata / stop 检查 / profiler 钩子
│   ├── sampling.py          # ★ derive_seed + GPU-native inverse-CDF 批量采样
│   ├── stopping.py          # 增量 stop（token fast path + 窗口 fallback）
│   ├── spec/                # 投机解码（ngram/model drafter，committed-stream stop）
│   └── bench/
│       ├── bench.py         # compare / spec / sampling / scheduler 四模式
│       ├── paged_attention_bench.py
│       ├── prefix_hash_bench.py
│       └── profile_runtime.py   # 逐阶段 step profiler
├── tests/                   # 889 个测试（CPU 全量 + gpu/triton marker 隔离）
├── examples/
├── pyproject.toml           # pip install -e . + ruff + pytest markers
└── .github/workflows/ci.yml # py3.10-3.12 · CPU · ruff + pytest(-m "not gpu")
```

## 已知限制

- CUDA Graph：架构就绪（固定 shape metadata buffer）、`enable_cuda_graph` 预留、**未实测**——本机 CPU 侧开销已 <3ms/步，收益预期有限，不编数据；
- Triton kernel 只覆盖 decode（q_len=1）；prefill 走 torch 路径（flash attention 型 prefill kernel 是下一个大项）；
- 投机解码未与 continuous batching 组合；
- stop string 窗口 decode 在极端 BPE 边界下理论上可能差一个 token 的截断精度；
- bf16 kernel 测试依赖 GPU bf16 支持（不支持时自动 skip）。
