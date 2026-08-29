# mini-vLLM：从零实现的推理引擎（Block 级 KV Cache · Continuous Batching · Prefix Caching · Speculative Decoding）

一个为**学习推理系统**而写的迷你 vLLM，基于 Qwen2.5-0.5B，纯 PyTorch 实现（约 2000 行，无 CUDA kernel），把推理岗面试高频的四大机制全部亲手写了一遍：

| 机制 | 对应 vLLM 概念 | 代码入口 |
|---|---|---|
| Block 级 KV Cache（逻辑块→物理块映射、引用计数、COW、LRU 驱逐） | PagedAttention 的内存层 | `minivllm/block_manager.py` |
| Continuous batching（iteration 级调度 + 抢占重算） | `Scheduler` | `minivllm/scheduler.py` |
| Prefix caching（块粒度哈希链复用） | `enable_prefix_caching` | `block_manager.py` 的哈希链 |
| Speculative decoding（n-gram / 小模型 draft + 一步验证） | SpecDecode worker | `minivllm/spec/` |

正确性保证：**fp32 下与 HuggingFace `generate` 逐 token 完全一致**（29 个测试，含随机小模型端到端对齐、前缀缓存开关输出一致、并行采样 COW、抢占正确性、spec 解码分布保持性检验）。

---

## 快速开始

```bash
pip install -r requirements.txt        # torch(CUDA) + transformers
pytest tests/ -q                       # 29 个测试，无需下载模型（随机小权重）

# 需要真实模型（~1GB，首次自动下载；国内可 export HF_ENDPOINT=https://hf-mirror.com）
python examples/basic_generate.py                   # 基础生成 + 引擎统计
python examples/prefix_cache_demo.py                # 前缀缓存：warm 批 TTFT 385ms→110ms
python examples/spec_demo.py                        # n-gram 投机解码，acceptance ~85%

# Benchmark（吞吐 / TTFT / TPOT，对比 HF generate）
python -m minivllm.bench.bench --mode compare --num-prompts 16 --enable-prefix-caching
python -m minivllm.bench.bench --mode spec --spec-drafter ngram --num-spec-tokens 8
```

架构上模型是配置驱动的，任何 Qwen2/Qwen2.5 尺寸都能跑（draft model 传另一个 HF 模型路径即可）。

---

## 1. Block 级 KV Cache（PagedAttention 的核心）

### 问题：传统 KV cache 为什么浪费

HF 式推理为每个请求预留 `[max_seq_len, layers, kv_heads, head_dim]` 的连续 KV 空间。实际生成长度不可预知，内部碎片 + 外部碎片 + 为并发预留，实际利用率通常只有 **20%~40%**。显存被 KV 塞满 → 并发数（batch size）上不去 → 吞吐上不去。

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

## 2. Continuous Batching（iteration 级调度）

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

### 实现（`scheduler.py` + `engine.step()`）

每个 engine step（一次 forward）：

1. **running 序列继续 decode**（每序列 1 个新 token）；池子给不出槽位 → 从队尾抢占；
2. **waiting 队列 FCFS 准入**：token 预算（`max_num_batched_tokens`）+ 并发上限（`max_num_seqs`）内尽量多收，准入时走前缀缓存分配；
3. prefill 与 decode **混在一个 varlen 批**里：全批 token 拼成一条 `[total_tokens, hidden]` 平铺流做投影/RoPE，attention 按序列用自己的 block table 算——新请求在 iteration 粒度加入，不用等整批清空。

调度器每次 `schedule()` 输出的就是本 iteration 的批次 —— 这就是 "iteration-level scheduling"。

---

## 3. Prefix Caching（相同前缀复用 KV 块）

**动机**：多轮对话的 system prompt、few-shot 前缀、RAG 文档在请求间完全相同，重复 prefill 纯属浪费。

**块粒度哈希链**（`block_manager.py`）：每个**满块**的 key 由内容和前缀链决定：

```
key_i = (key_{i-1}, tuple(tokens[i*bs : (i+1)*bs])),   key_root = ()
```

链式结构保证 **key 相等 ⇔ 前缀完全相同**（避免哈希碰撞误判）。新请求 admission 时逐块查表，命中则 `ref_count++` 直接映射旧块，`num_computed_tokens` 直接跳到命中前缀末尾 —— prefill 只算不匹配的后缀。

安全细节（面试可讲的点）：

* **只有满块进缓存**，半满尾块永远私有 —— 内容不完整不能共享；
* **先算后注册**：块的前向算完才插入缓存表，杜绝"别人复用到尚未算好的块"；
* 复用块永不原地写（满了才入表，写入只发生在新块/私有块）；
* 命中的块即使原序列已结束也保留（ref=0，LRU 可驱逐），**被抢占的序列重新调度时自动恢复大部分 KV**。

实测（RTX 4060，32 请求 × 256 token 输入、224 token 共享前缀，错峰到达）：**缓存命中率 44%~62%，warm 批 TTFT 从 385ms 降到 110ms**。

---

## 4. Speculative Decoding（投机解码）

### 算法（`spec/spec_engine.py`）

一轮 = 3 步：

```
1. DRAFT   便宜 drafter（n-gram 查表 / 小模型自回归）给出 γ 个候选
2. VERIFY  目标模型一次 forward 吃下 [上轮 bonus] + γ 个候选，
           在每个位置都拿到分布 p_i —— 一次 forward 验证 γ+1 个位置
3. ACCEPT  贪心：目标 argmax 与候选一致就收；
           采样：以 min(1, p(x)/q(x)) 接受，拒绝时从 norm(max(p-q,0)) 重采样
           —— 数学上保证输出分布与目标模型**完全相同**，无损
```

每轮产出 `k+1` 个 token（k 个被接受的候选 + 1 个免费 bonus），目标模型成本从"每 token 一次 forward"摊薄到"每 k+1 token 一次 forward"。

两种 drafter（`spec/drafters.py`）：

* **NGramDrafter**（prompt-lookup decoding）：在后文找当前后缀上一次出现的位置，把当时后续的 token 抄过来。零模型成本，复述/翻译/代码类任务命中率很高；
* **ModelDrafter**：一个小语言模型（比如 0.5B 给 1.5B 打草稿），走同一套 paged-KV worker，每轮同步/回滚 draft KV。

KV 记账（最容易写错的部分，本项目已测）：目标侧 KV 前沿 = 已验收前缀（bonus 挂起）；被拒候选留下的脏槽位只存在于**独占块**，下一轮直接覆写；**缓存表只注册到"最后被验收的 token"为止**，脏数据永不出借。测试 `test_rejection_sampling_preserves_target_distribution` 用 2 万次模拟验证了 p/q 拒绝采样下产出分布严格等于 p。

实测：n-gram drafter（γ=8），acceptance 52%~85%、2.5~4.6 tokens/round，单流延迟 **~2x 加速**，输出与普通 greedy 完全一致（断言验证）。

---

## 5. Benchmark 实测（RTX 4060 Laptop 8GB · fp16 · Qwen2.5-0.5B-Instruct）

吞吐（16~32 请求 continuous batching，HF 为 `generate` 按批 4）：

```
16×(128in/64out) 独占 prompt   mini-vllm 229 tok/s  vs  HF 91 tok/s    → 2.51x
32×(256in/32out) 共享前缀 224  mini-vllm 199 tok/s  vs  HF 99 tok/s    → 2.03x（命中率 44%）
```

延迟（共享前缀 + 前缀缓存，错峰到达 in-flight=4）：

```
TTFT  mean 794ms → 220ms（p50 715ms → 128ms，命中后 prefill 只算后缀）
TPOT  ~52ms（受每步 Python 开销下限约束）
```

投机解码（单流 greedy，n-gram γ=8）：acceptance 52%、2.56 tokens/round → **1.99x 延迟加速**。

诚实的短板：**batch=1 单流吞吐约为 HF 的 0.8 倍**。0.5B 模型在 GPU 上 decode 步骤极快（~10ms），每步 ~15ms 的 Python/调度开销成为主导；真实 vLLM 用 CUDA graph + 融合 kernel 把这部分压到近乎零。**结论：continuous batching 的收益来自批量摊薄开销，batch 越大相对 HF 优势越大**。

复现：

```bash
python -m minivllm.bench.bench --mode compare --num-prompts 16 --max-num-seqs 16
python -m minivllm.bench.bench --mode compare --num-prompts 32 --input-len 256 \
    --shared-prefix-len 224 --enable-prefix-caching --max-in-flight 16
python -m minivllm.bench.bench --mode spec --spec-drafter ngram --num-spec-tokens 8
```

---

## 6. 正确性验证（怎么证明写得对）

1. **单元测试**（无模型依赖，秒级）：块分配/回收、LRU 驱逐、COW 语义（fork 后写入复制、内容相等）、哈希链（同前缀命中/异前缀不命中）、调度器 FCFS/预算/抢占、全命中 prompt 强制重算末 token。
2. **随机小模型端到端**：构造一个 2 层 64 维的随机 Qwen2（`tests/helpers.py`），同一份权重分别灌进我的实现和 HF，**fp32 CPU 下 greedy 输出逐 token 全等**（连续批处理、前缀缓存开/关、并行采样 n=3、极小池抢占场景全覆盖）。
3. **真实模型 fp32 CPU**：首 token logits `max|Δ|=2.8e-5`，top-5 完全一致。
4. **fp16 CUDA 生成**：多数 prompt 完全一致；个别 prompt 从中段某个 token 分叉（分叉前文本相同连贯）—— 这是两种等价 kernel 路径在 fp16 上的累计精度分叉，vLLM 与 HF 之间同样存在，fp32 下消失。
5. **投机解码**：draft=目标权重时 acceptance≈100%、输出与 greedy 全等；拒绝采样分布保持性统计检验。

---

## 7. 面试要点速查（每个点都对应本项目代码）

**Q: KV cache 为什么按块分页？块多大合适？**
碎片 vs 元数据/内核效率的权衡。块太小 → block table 长、kernel 启动开销大；太大 → 内部碎片大（平均浪费 block_size/2）。vLLM 默认 16。

**Q: 0.5B 模型 1GB 显存能放多少 KV？**
fp16 每 token 12KB（本模型），1GB ≈ 8.5 万 token ≈ 21k token/请求时同时服务 4 个。顺手能推出 GQA 的 7 倍压缩。

**Q: Prefix cache 的 key 怎么设计？哈希碰撞怎么办？**
链式 key（父块 key + 本块内容），dict 相等性是结构比较，无碰撞问题；工程上 vLLM 用 partial hash 也要处理碰撞 → 同样靠整链比较。

**Q: 什么时候需要 COW？**
只有"共享的半满块被写入"需要：并行采样 fork 后各自续写。满块永不被原地写，所以前缀复用不需要 COW。

**Q: 显存不够了怎么办（preemption）？**
swap（把块挪 CPU）或 recompute（丢 KV 重新算）。本项目实现 recompute：配合 prefix cache，重新调度时 prompt 部分直接命中，只重算已生成部分，成本很低。

**Q: Continuous batching 和 chunked prefill 的关系？**
前者解决"何时收新请求"（iteration 级），后者解决"长 prompt 阻塞 decode"（把 prefill 切成 ≤ 预算的片混进 decode 批）。本项目已支持 prefill/decode 混批，chunked prefill 是自然扩展。

**Q: 投机解码为什么无损？什么情况收益最大？**
拒绝采样构造（accept prob min(1,p/q)，拒绝后从 residual 采样）使每个输出 token 的边际分布仍是 p。收益 = f(draft 与目标的一致率、drafter 便宜程度、γ)；batch 大时 verify 变贵，收益下降。

**Q: 与真实 vLLM 还差什么？**
融合 paged-attention kernel（FlashDecoding）、CUDA graph 消启动开销、TP/PP 多卡、量化 KV（FP8）、chunked prefill、结构化输出、chunked logits/lm_head 分块。内存管理层与调度层的设计是同构的 —— 这正是本项目练手的价值。

---

## 项目结构

```
mini-vllm/
├── minivllm/
│   ├── config.py            # EngineConfig / KV cache 大小计算 / 显存估算
│   ├── sequence.py          # Sequence / SamplingParams / RequestOutput
│   ├── kv_pool.py           # 物理块池大张量 [blocks, layers, 2, kvh, bs, dim]
│   ├── block_manager.py     # ★ 逻辑块→物理块、引用计数、COW、LRU、哈希链、抢占
│   ├── attention.py         # RoPE / 分页 KV 写入+聚合 / 批量 decode SDPA / GQA
│   ├── model.py             # Qwen2（RMSNorm/SwiGLU/RoPE），varlen 平铺批前向
│   ├── scheduler.py         # ★ continuous batching：decode 优先 + FCFS 准入 + 抢占
│   ├── engine.py            # step 主循环 / 请求组 / fork / 采样 / 流式输出
│   ├── sampling.py          # temperature/top-k/top-p + 投机解码用分布工具
│   ├── spec/
│   │   ├── drafters.py      # NGramDrafter（prompt lookup）/ ModelDrafter
│   │   ├── worker.py        # 单序列 paged-KV worker（draft 复用）
│   │   └── spec_engine.py   # ★ draft-then-verify 循环 + 拒绝采样
│   └── bench/bench.py       # 吞吐/TTFT/TPOT 基准 + 错峰到达模拟
├── tests/                   # 29 个测试（单元 + 端到端 + 统计检验）
└── examples/                # 生成 / 前缀缓存 / 投机解码 demo
```

## 已知限制

- 纯 PyTorch 前向，无融合 kernel / CUDA graph → 单流吞吐低于 HF（见 benchmark 一节）；
- 投机解码未与 continuous batching 组合（vLLM 早期同样分开处理），批内 n=1；
- 不支持 chunked prefill、流式 token 级 API、stop 序列的 token 级精确截断；
- transformers 5.x 加载（老版本 transformers 未测）。
