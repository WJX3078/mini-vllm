# mini-vLLM：从零实现的推理引擎（Paged KV Cache · Continuous Batching · Chunked Prefill · Prefix Caching · Speculative Decoding）

[![CI](https://github.com/WJX3078/mini-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/WJX3078/mini-vllm/actions/workflows/ci.yml)

一个为**学习推理系统**而写的迷你 vLLM，基于 Qwen2.5-0.5B，纯 PyTorch 实现（无 CUDA kernel），把推理岗面试高频的机制全部亲手写了一遍：

| 机制 | 对应 vLLM 概念 | 代码入口 |
|---|---|---|
| Block 级 KV Cache（逻辑块→物理块映射、引用计数、COW、LRU 驱逐） | PagedAttention 的内存层 | `minivllm/block_manager.py` |
| Continuous batching + **Chunked prefill**（统一 token 预算的 iteration 级调度 + 抢占重算） | `Scheduler` (V1-style) | `minivllm/scheduler.py` |
| Prefix caching（块粒度哈希链复用，**可插拔 hash backend**） | `enable_prefix_caching` | `block_manager.py` + `prefix_hash.py` |
| Speculative decoding（n-gram / 小模型 draft + 一步验证，**分布无损**） | SpecDecode worker | `minivllm/spec/` |
| **Per-request RNG**（请求级随机流，批组成无关） | V1 sampler 语义 | `sampling.py` + `sequence.py` |
| **Batched sampling**（按采样配置分组，每组 ≤1 次 GPU→CPU 同步） | V1 batched sampler | `minivllm/sampling.py` |
| **增量 stop 检查**（token 序列 fast path + 窗口 decode fallback） | `stop_checker` | `minivllm/stopping.py` |

正确性保证（**92 个测试，CI 覆盖 Python 3.10/3.11/3.12**）：

* fp32 下与 HuggingFace `generate` **逐 token 完全一致**（随机小模型端到端对齐）；
* 前缀缓存开/关、chunked prefill 开/关、抢占恢复、并行采样 COW 输出全部一致；
* 投机解码用**统计检验**证明采样分布严格等于目标分布（含 n-gram + temperature>0，2~5 万次试验）；
* EOS/stop 出现在 **proposal 中途**时逐 token 截断，KV/输出/缓存三方一致；
* property-style 随机配置扫描：`computed ≤ tokens`、块表覆盖、引用计数、free-list 一致性等不变量。

---

## 快速开始

```bash
pip install -e .                       # 或 pip install -r requirements.txt
pytest tests/ -q                       # 92 个测试，无需下载模型（随机小权重）
ruff check .                           # lint（CI 同款）

# 需要真实模型（~1GB，首次自动下载；国内可 export HF_ENDPOINT=https://hf-mirror.com）
python examples/basic_generate.py                   # 基础生成 + 引擎统计
python examples/prefix_cache_demo.py                # 前缀缓存：warm 批 TTFT 1105ms→168ms
python examples/spec_demo.py                        # n-gram 投机解码 4.2x（复述类任务）

# Benchmark（吞吐 / TTFT / TPOT / E2E，中位数 + p50/p95/p99，JSON 输出）
python -m minivllm.bench.bench --mode compare --num-prompts 16 \
    --input-len 128 --output-len 64 --concurrency 16
python -m minivllm.bench.bench --mode spec --spec-drafter ngram --num-spec-tokens 8
python -m minivllm.bench.bench --mode sampling      # 采样同步开销 microbenchmark
python -m minivllm.bench.prefix_hash_bench          # hash backend microbenchmark
```

架构上模型是配置驱动的，任何 Qwen2/Qwen2.5 尺寸都能跑（draft model 传另一个 HF 模型路径即可）。

---

## 1. Block 级 KV Cache（PagedAttention 的核心）

### 问题：传统 KV cache 为什么浪费

传统连续 KV cache 要求**每个序列的 K/V 在逻辑上连续存储**：DynamicCache 随生成逐步 `cat` 增长，带来反复的内存重组与分配开销；StaticCache 则要提前按最大长度预留。两者都难以支持多序列动态并发、跨请求共享前缀和抢占——Paged KV 把 K/V 切成固定大小 block，用 block table 把逻辑块映射到任意物理块，从根本上解决这三件事（这也是 vLLM 论文的核心动机）。

### 方案：像操作系统管内存一样管 KV

```
逻辑视图（每序列一个 block table）          物理池（一次性预分配的大张量）
seq A: [4] [7] [9]  ← 逻辑块0/1/2          pool.data[block_id, layer, K|V, kv_head, slot, dim]
seq B: [5] [8]                             任意物理块可被任意序列通过 block table 引用，
seq C: [4] [6]      ← A、C 共享块4（前缀复用） ref_count > 1 即共享
```

关键机制（都在 `BlockSpaceManager`）：

* **按需分配**：序列每写满一个块才拿下一个块，内部碎片 ≤ 1 块（block_size=16 时最多浪费 15 个 token 槽）。
* **引用计数**：块被多个序列映射时 `ref_count > 1`。
* **Copy-on-Write**：共享的**半满块**被写入前先复制（并行采样的 fork 场景）；**满块永不原地写**，可无限共享。
* **LRU 驱逐**：`ref_count==0` 的缓存块按 LRU 淘汰，回收物理块。
* **抢占重算（recompute preemption）**：显存耗尽时从 running 队尾踢序列、释放块；重新调度时前缀缓存直接命中旧块，只重算少量 token。

### KV cache 大小计算（必考）

```
每 token KV = 2 (K+V) × n_layers × n_kv_heads × head_dim × dtype_bytes
```

Qwen2.5-0.5B（24 层，GQA 14 个 Q 头 / **2 个 KV 头**，head_dim=64，fp16）：

```
2 × 24 × 2 × 64 × 2B = 12 KB/token，一个 16-token 块 = 192 KB
```

`config.py` 里的 `ModelConfig.kv_bytes_per_token()` 就是这个公式；引擎启动时打印实际数值。对比：若没有 GQA（14 个 KV 头），同样的模型每 token 要 84 KB —— **GQA 把 KV cache 压缩了 7 倍**，这就是 MHA→GQA→MQA 演进的动机（MQA = 1 个 KV 头，压到极限）。

### 本项目的 attention 与真实 vLLM 的关系

本项目的 decode 路径把整个 batch 的 K/V 写入、按 block table 聚合（gather）、SDPA 注意力各合并成**一次批量操作**；prefill 路径对每个序列 gather + SDPA。真实 vLLM 用单个融合 CUDA kernel（paged attention kernel）直接在分块内存上算 flash attention，省掉 gather 的物化。**内存布局和语义完全一致，差别只在 kernel 融合**。

---

## 2. Scheduler：统一 token 预算（Continuous Batching + Chunked Prefill）

### static batching 的问题

HF `generate` 把一批请求绑定到批生命周期：短的先完成也只能干等长的（下图 `x` 为空转）。请求长度方差越大浪费越多。

```
static:      |req1 ██████████|
             |req2 ████████████████|   ← req1 完成后槽位空转
             |req3 ████|

continuous:  |req1 ██████████|req4 ██████|
             |req2 ████████████████|
             |req3 ████|req5 ██████████|    ← 每个 iteration 都重新组批
```

### 统一 token 预算怎么分配（`scheduler.py`）

每个 engine step 有一个总预算 `max_num_batched_tokens`，**decode token 和 prefill chunk 共享**，按优先级填充：

```
budget = max_num_batched_tokens
────────────────────────────────────────────────────────────────
① decode 优先   running 序列各取 1 token（FCFS）
                KV 池给不出槽位 → 抢占队尾（recompute 策略）
② prefill chunk running 中未完成的 prefill 继续推进（FCFS）
③ 新请求准入    waiting 队头按剩余预算 chunk 化准入
                chunk = min(剩余 prompt, 剩余预算)
────────────────────────────────────────────────────────────────
例：budget=2048，running 里 A/B/C 在 decode（各 1 token），
    剩余 2045 给新请求 D（prompt 8192）→ 本轮只算 D 的 2045，
    下一轮继续 —— 长 prompt 永远不会把 decode 卡住超过一个 iteration。
```

**Prefix cache 感知准入**（重要的正确性/效率修复）：准入前先做一次**只读**缓存探测（不动 ref_count / LRU），预算只按**真正要算的 token 数**扣除——1000 token 的 prompt 命中 900，只扣 100 预算，省出的预算能多收请求。预算检查通过后才正式 `allocate`（分配失败在 block manager 内部回滚，无泄漏）；完全命中的 prompt 强制重算最后一个 token 以产出 logits。

**chunked prefill 与块分配的关系**：准入时仍一次性预留整个 prompt 的块（**预留 ≠ 计算**，`num_computed_tokens` 跟踪实际计算进度）。这保证"准进来的必装得下"，不会死锁，且完整复用抢占机制；真实 vLLM V1 更激进（按 chunk 惰性分配 + 可抢占 prefill），见"与真实 vLLM 的差距"。

**调度语义不变量**（有测试盯着）：已 running 的 decode 每轮必推进；waiting FCFS 不饿死；`spans` 是本轮每序列真正计算的 `[start, end)`。

---

## 3. Prefix Caching（相同前缀复用 KV 块）

**动机**：多轮对话的 system prompt、few-shot 前缀、RAG 文档在请求间完全相同，重复 prefill 纯属浪费。

**块粒度哈希链**（`block_manager.py` + 可插拔 `prefix_hash.py`）：每个**满块**的 key 由内容和前缀链决定：

```
TupleBackend:  key_i = (key_{i-1}, tuple(tokens[i*bs:(i+1)*bs]))   结构相等，无碰撞
SHA256Backend: key_i = SHA256(key_{i-1} || tokens || metadata)     定长 32B，O(1) 比较
```

`metadata` 把模型身份编进哈希根；生产系统还会把 **LoRA adapter id、多模态预处理签名、租户 salt** 纳入——两段 KV 只有在"影响 KV 计算的一切"都相同时才允许共享，否则命中即出错。

两种 backend 的实测权衡（`python -m minivllm.bench.prefix_hash_bench`，32K context，bs=16）：

```
            build ms   lookup ms   key 内存 KiB
tuple          1.53     159.25        567        ← 查找 O(链长)，长上下文退化
sha256         4.43       0.15        130        ← 查找 O(1)，构建略贵
```

安全细节（面试可讲的点）：

* **只有满块进缓存**，半满尾块永远私有 —— 内容不完整不能共享；
* **先算后注册**：块的前向算完才插入缓存表，杜绝"复用到尚未算好的块"（chunked prefill 逐 chunk 注册，被抢占的长 prompt 重调度时自动恢复）；
* 复用块永不原地写；缓存探测**严格只读**，准入失败自动回滚；
* 命中的块即使原序列已结束也保留（ref=0，LRU 可驱逐）。

实测（RTX 4060 Laptop，32 请求 × 256 token 输入、224 token 共享前缀，错峰到达）：**缓存命中率 65.6%，warm 批 TTFT mean 1105ms → 168ms**。

---

## 4. Speculative Decoding（投机解码，分布无损）

### 算法（`spec/spec_engine.py`）

一轮 = 3 步：

```
1. DRAFT   便宜 drafter（n-gram 查表 / 小模型自回归）给出 γ 个候选
2. VERIFY  目标模型一次 forward 吃下 [上轮 bonus] + γ 个候选，
           在每个位置都拿到分布 p_i —— 一次 forward 验证 γ+1 个位置
3. ACCEPT  贪心：目标 argmax 与候选一致就收；
           采样：以 min(1, p(x)/q(x)) 接受，拒绝时从 norm(max(p-q,0)) 重采样
           —— 数学上保证输出分布与目标模型完全相同，无损
```

**确定性 drafter 的处理**（常见的错法是直接崩）：n-gram 的 proposal 是点分布 q(x)=1，此时接受概率 = min(1, p(x))，拒绝后从 **p 去掉 x 的质量**后重采样——这是标准 speculative sampling 在 q 退化为 one-hot 时的特例，统计检验（数万次试验）证明每个 commit token 的分布仍严格等于 p。若 p(x)=0（如被 top-k 过滤掉）则 100% 拒绝，residual 退化为 p；若 p==q（residual 全零）正确回退为从 p 采样。

**中途截断**：一轮 commit 的是"k 个被接受候选 + 1 个 bonus"，若 **EOS / stop string 落在这串 token 中间**，从最早的终止点截断——输出 token、KV frontier、前缀缓存注册、drafter 的流视图四方一致，终止后的 proposal/bonus 绝不泄漏进输出（专项测试覆盖）。

两种 drafter（`spec/drafters.py`）：

* **NGramDrafter**（prompt-lookup decoding）：在后文找当前后缀上一次出现的位置，把当时后续的 token 抄过来。零模型成本，复述/翻译/代码类任务命中率很高；
* **ModelDrafter**：一个小语言模型（比如 0.5B 给 1.5B 打草稿），走同一套 paged-KV worker，每轮同步/回滚 draft KV。

KV 记账（最容易写错的部分，本项目已测）：目标侧 KV 前沿 = 已验收前缀（bonus 挂起）；被拒候选留下的脏槽位只存在于**独占块**，下一轮直接覆写；**缓存表只注册到"最后被验收的 token"为止**，脏数据永不出借。

实测（RTX 4060，单流 greedy，n-gram γ=8，5 轮取中位）：acceptance 52%、2.56 tokens/round → **2.04x 延迟加速**，输出与普通 greedy 逐 token 一致（断言验证）；demo 场景（复述任务）acceptance 100%、4.2x。

---

## 5. Per-request RNG 与 Batched Sampling（性能工程）

### 为什么共享 Generator 是错的

所有序列共用一个 `torch.Generator` 时，**每次采样都会推进共享流**：A 的输出取决于"谁和它同批、什么顺序采样"。正确做法（vLLM V1 语义）：每个序列一个独立 generator，seed 来自**稳定的整数混合**（splitmix64，Python `hash()` 每进程加盐，绝不能用）：

```
用户指定 seed：rng_seed = mix(seed, sample_idx)          → 同一 seed 重复运行/换批结果不变
未指定 seed：  rng_seed = mix(engine_seed, request_id, sample_idx) → 请求间互不干扰
n>1 并行采样： sample_idx 不同 → 子序列独立随机流
```

greedy 完全不触碰 RNG，不受影响（都有测试）。

### 为什么 per-seq `.item()` 很慢，batched sampling 快在哪

每个 decode step 里，每个序列各自 `.argmax().item()` / `.cpu()` 会强制一次 **GPU→CPU 同步**：CPU 停在那等 GPU 把这一步算完，流水线被打断 N 次。`sample_tokens()` 按 `(temperature, top_k, top_p)` 分组：组内一次批量 filter+softmax、**每组最多一次 D2H**，multinomial 在 CPU 用各自的 generator——**RNG 语义和批独立性分毫不动**，同步次数从 N 次降到组数（通常 1）次。实测（RTX 4060，`--mode sampling`）：

```
B=16: per-seq 24.6 ms → batched 9.6 ms（2.6x）
B=64: per-seq 106 ms  → batched 35.8 ms（3.0x）
```

（B=1 时分组开销略负，约 -20%，可忽略。）

### stop string 的增量检查

每生成一个 token 就 `decode(全部历史)` 是 O(T²) 的 tokenizer 开销。`stopping.py` 的做法：stop 串预编码成 token 序列做**后缀匹配 fast path**；匹配不上时只 decode **尾部窗口**（窗口 ≥ 2×最长 stop 序列，保证跨 token 边界的 stop 一定落在窗口内），命中后按 stop 首字符位置回映射到 token 数截断。长输出的 stop 检查从"整段重解码"变成 O(窗口)。

---

## 6. Benchmark 实测（RTX 4060 Laptop 8GB · fp16 · Qwen2.5-0.5B-Instruct）

方法学（`bench.py` 头部同款声明）：计时前后 `torch.cuda.synchronize()`；每组配置 **3 次 warmup + 5 次测量**，吞吐取**中位数**，TTFT/TPOT/E2E 报 p50/p95/p99（所有测量轮的请求合并统计）；**两侧使用完全相同的 token-id prompt、tokenizer、dtype、采样参数与输出长度**（greedy + ignore_eos，输出长度由 workload 控制，保证墙钟时间纯比引擎）；HF 的 TTFT/TPOT 用**自定义 use_cache 循环逐步计时**（真实首 token 延迟，不再用 batch_latency/tokens 冒充；batch>1 时标注 approximate）。vLLM 装了就自动加为第三条基线，没装优雅跳过。

| workload（greedy, ignore_eos） | mini-vllm | HuggingFace | 加速 |
|---|---|---|---|
| 16×(128in/64out)，HF batch=1 | **78.3 tok/s**（中位） | 12.8 tok/s | **6.1x** |
| 32×(256in/32out)，共享前缀 87.5%，缓存开 | **125.7 tok/s**（命中 65.6%） | 51.5 tok/s（batch=4） | **2.4x** |
| 8×(2048in/32out)，concurrency=8 | **43.4 tok/s**，TTFT p50 **1.96s** | 40.7 tok/s，TTFT **4.04s**（batch=8） | 1.07x / **TTFT 2.1x** |

长输入场景正是 chunked prefill 的价值：HF 的静态批必须先把 16K token 一次性 prefill 完（TTFT 4 秒），mini-vllm 把 prefill 切片混进 decode 流，首批请求 ~1.2s 就开始出 token；代价是 decode 步与 chunk 同迭代时 TPOT 略升（72→115ms），这就是 chunked prefill 的经典 trade-off，数据里看得见。

投机解码（单流 greedy，n-gram γ=8，5 轮中位）：acceptance 52%、2.56 tokens/round → **2.04x 延迟加速**（复述类 demo 场景 4.2x）。

诚实的短板：**batch=1 单流吞吐低于 HF**（本项目每步 ~15ms 的 Python/调度开销 vs HF 的 C++ 循环），0.5B 小模型 decode 步骤极短时开销占比最大；真实 vLLM 用 CUDA graph + 融合 kernel 把这部分压到近乎零。**结论：continuous batching 的收益来自批量摊薄开销，batch 越大相对 HF 优势越大**。另外本机为笔记本 GPU，绝对数值会随功耗状态波动——请以复现命令自测为准。

复现：

```bash
python -m minivllm.bench.bench --mode compare --num-prompts 16 --input-len 128 \
    --output-len 64 --concurrency 16 --hf-batch-size 1
python -m minivllm.bench.bench --mode compare --num-prompts 32 --input-len 256 \
    --output-len 32 --shared-prefix-ratio 0.875 --enable-prefix-caching \
    --concurrency 16 --hf-batch-size 4
python -m minivllm.bench.bench --mode compare --num-prompts 8 --input-len 2048 \
    --output-len 32 --concurrency 8 --hf-batch-size 8
python -m minivllm.bench.bench --mode spec --spec-drafter ngram --num-spec-tokens 8
# 全部支持 --output results.json（含 workload/metrics 全量字段，便于绘图）
```

---

## 7. 正确性验证（怎么证明写得对）

1. **单元测试**（无模型依赖，秒级）：块分配/回收、LRU 驱逐、COW 语义、哈希链（双 backend）、调度器 FCFS/预算/抢占、**prefix cache 预算只按未缓存 token 扣除**（含只读探测、准入失败回滚、全命中强制重算）、**统一预算/chunk 边界**、增量 stop 检查。
2. **随机小模型端到端**：2 层 64 维随机 Qwen2 同权重灌进我的实现和 HF，**fp32 CPU 下 greedy 输出逐 token 全等**——覆盖连续批处理、**chunked prefill 开/关/随机预算**、前缀缓存开/关、并行采样 n=3、极小池抢占。
3. **Per-request RNG**：单独运行 vs 与他人同批、重复运行、到达顺序——seeded 请求输出全部一致；n=3 子序列随机流独立；greedy 不受影响。
4. **投机解码**：greedy 全等；**n-gram + temperature>0** 分布无损（数万次统计检验，含点分布 proposal、top-k 支持集外 proposal、p==q 退化）；**EOS/stop 出现在 proposal 中途**逐 token 截断（单元 + 端到端）。
5. **Property 不变量扫描**：随机 (block_size, 池大小, 预算, chunked, 缓存) 配置 × 随机 prompt，每步检查 `computed ≤ tokens`、块表覆盖计算前沿、refcount ≥ 1、不在 free list、缓存块只含已算满块，结束序列不占私有块——同时输出仍与 HF 全等。
6. **真实模型 fp32 CPU**：首 token logits `max|Δ|=2.8e-5`；fp16 CUDA 多数 prompt 全等（分叉为 fp16 累计精度，vLLM 与 HF 间同样存在）。

---

## 8. 面试要点速查（每个点都对应本项目代码）

**Q: 你的 scheduler 为什么这么设计？**
统一 token 预算（decode 优先 + prefill chunk），一次 forward 服务尽量多 token；decode 是延迟敏感的多数派所以先分配；prefill 可切，decode 的 1 token 不可切。

**Q: Prefix Cache hit 后，scheduler 怎么知道真正要算多少 token？**
准入前只读探测（`get_cached_prefix`，不动 ref_count）→ 未缓存 token 数过预算 → 通过才 `allocate`。全命中时强制重算最后一个 token 拿 logits。

**Q: 长 prompt 为什么不会阻塞 decode？**
chunked prefill：decode 先各拿 1 token，剩余预算给 prefill chunk，8K prompt 分 ~32 轮算完。实测 2048-token 场景 TTFT 4.0s → 2.0s。

**Q: KV 不够怎么抢占？抢占后为什么能恢复？**
recompute 策略：从 running 队尾踢（最新请求），块全部释放。恢复靠前缀缓存：prompt 满块都注册过（chunked prefill 逐 chunk 注册），重调度只重算生成部分。

**Q: 多个请求为什么不会互相影响 sampling RNG？**
每序列独立 generator，seed 由 splitmix64 稳定混合派生；用户 seed 时不掺 request_id（vLLM 语义：同 seed 复现同输出，n>1 子序列靠 sample_idx 区分）。有批组成无关性测试。

**Q: Speculative Decoding 为什么分布无损？**
接受概率 min(1, p/q) + 拒绝后从 norm(max(p-q,0)) 重采样，每个输出 token 边际分布恒为 p；确定性 drafter 是 q=onehot 的特例（接受率=p(x)）。统计检验直测分布。

**Q: GPU decode 路径真正的 CPU synchronization 在哪里？**
每序列采样结果的 `.item()/.cpu()`。批分组采样把 N 次同步降到每组 1 次（实测 B=64 提速 3.0x）；此外调度器/块管理的纯 Python 逻辑是第二步瓶颈（→ CUDA graph 的动机）。

**Q: Paged KV 和真正的 Paged Attention kernel 有什么区别？**
本项目：按 block table gather 成连续 K/V 再 SDPA（多一次物化）；真实 vLLM：融合 kernel 直接遍历物理块 + online softmax（省 gather、省启动）。内存语义相同，性能差在 kernel 融合与启动开销。

**Q: 与真实 vLLM 还差什么？**
融合 paged-attention kernel（FlashDecoding）、CUDA graph、TP/PP 多卡、FP8 KV、按 chunk 惰性块分配（本项目准入时整 prompt 预留——保守但无死锁，V1 是惰性分配+可抢占 prefill）、投机解码 × continuous batching 组合、异步 server / OpenAI API。

---

## 项目结构

```
mini-vllm/
├── minivllm/
│   ├── config.py            # EngineConfig（含 chunked prefill / hash backend）+ KV 大小计算
│   ├── sequence.py          # Sequence / SamplingParams / per-request RNG 状态
│   ├── kv_pool.py           # 物理块池大张量 [blocks, layers, 2, kvh, bs, dim]
│   ├── block_manager.py     # ★ 逻辑块→物理块、引用计数、COW、LRU、只读缓存探测、抢占
│   ├── prefix_hash.py       # 可插拔 hash backend（tuple / sha256）
│   ├── attention.py         # RoPE / 分页 KV 写入+聚合 / 批量 decode SDPA / GQA
│   ├── model.py             # Qwen2（RMSNorm/SwiGLU/RoPE），varlen 平铺批前向
│   ├── scheduler.py         # ★ 统一 token 预算：decode 优先 + chunked prefill + FCFS + 抢占
│   ├── engine.py            # step 主循环（span 化）/ 请求组 / fork / stop 检查 / 统计
│   ├── sampling.py          # ★ filter/probs + derive_seed + 分组批量采样（每组 ≤1 次同步）
│   ├── stopping.py          # 增量 stop 检查（token fast path + 窗口 decode）
│   ├── spec/
│   │   ├── drafters.py      # NGramDrafter（prompt lookup）/ ModelDrafter
│   │   ├── worker.py        # 单序列 paged-KV worker（draft 复用）
│   │   └── spec_engine.py   # ★ draft-then-verify + 无损拒绝采样 + 中途截断
│   └── bench/
│       ├── bench.py         # compare / spec / sampling 三模式，中位数+percentile+JSON
│       └── prefix_hash_bench.py
├── tests/                   # 92 个测试（单元 + 端到端 + 统计检验 + property 不变量）
├── examples/                # 生成 / 前缀缓存 / 投机解码 demo
├── pyproject.toml           # pip install -e . + ruff + pytest 配置
└── .github/workflows/ci.yml # Python 3.10/3.11/3.12 · CPU · ruff + pytest
```

## 已知限制

- 纯 PyTorch 前向，无融合 kernel / CUDA graph → 单流吞吐低于 HF（见 benchmark 一节）；
- 投机解码未与 continuous batching 组合（vLLM 早期同样分开处理），批内 n=1；
- chunked prefill 的块分配是准入时整 prompt 预留（保守、无死锁），非 V1 的逐 chunk 惰性分配；
- stop string 的窗口 decode 在极端 BPE 边界情况下理论上可能差一个 token 的截断精度（窗口内已尽力精确）；
- transformers 5.x 加载（老版本 transformers 未测）。
