# 简历成果提取（mini-vLLM v0.2 → v0.4）

## 30 秒版本

从零实现的单 GPU LLM 推理服务运行时（纯 PyTorch，约 5000 行）：分页 KV
cache + 容量预留、continuous batching + chunked prefill、Triton
PagedAttention decode kernel、GPU 原生采样、OpenAI 兼容异步 serving
（SSE 流式/取消/背压/可观测），889→926 项测试与真机 RTX 4060 数据支撑。

## 1 分钟版本

- **动机**：把"推理岗面试高频机制"从名词变成可运行、可测量、可回答追问
  的代码——每一层（内存/调度/GPU 执行/同步/kernel/测量）都亲手实现。
- **架构**：HTTP → AsyncLLMEngine（单 GPU 线程 owner + bounded 队列）
  → continuous batching 调度器（token budget/chunked prefill/KV
  reservation）→ 分页 KV + Triton kernel → GPU 采样 → 增量
  detokenizer → SSE。
- **核心优化**：GPU-native 采样把跨设备流量从 O(B·V) 降到 O(B)
  （B=64/V=150k：160ms→19.5ms/步）；Triton decode kernel 免去连续 K/V
  物化（B=64/8K ctx：778ms→10.3ms）；KV reservation 与惰性分配解耦
  （8×8K 并发零抢占）；持久化 metadata + pinned H2D。
- **benchmark**：全部真机数据（RTX 4060）：batch 1→32 吞吐线性扩展、
  serving 输出 199 tok/s @并发32、TTFT/E2E 分位数、4 场景调度对比、
  758 组合 kernel 数值矩阵。

## 简历三条 bullet

• 系统架构：从零实现单 GPU LLM serving runtime（~5k 行 Python）：异步
  HTTP 前端（OpenAI 兼容、SSE 流式）→ 单 owner 引擎线程 → continuous
  batching/chunked prefill 调度器 → 分页 KV（预留/分配解耦、前缀缓存、
  抢占重算）→ Triton PagedAttention + GPU 采样；889→926 项测试
  （含 758 组合 kernel 数值矩阵与取消/泄漏不变量回归）。

• 核心优化：GPU 原生采样将每步跨设备流量由 O(B×V) 降至 O(B)（150k 词表
  × B=64 下 8.2x）；Triton 分页注意力 kernel 消除连续 K/V 物化
  （8K 上下文 B=64 提速 75x，端到端 +16%）；KV 容量预留与惰性物理分配
  解耦（8×8K 并发零抢占）；入口/出口双有界队列实现背压与慢客户端隔离。

• 性能结果：真机 RTX 4060 实测 serving 输出 199 tok/s（并发 32）、
  engine 吞吐随并发近线性（1→8 并发 6.7x）、流式 TTFT p50 ~200ms；
  adversarial review 抓出并修复 async registry 泄漏（256 上限后全线 429）
  与 abort id 竞态两处 P0；所有数字可由仓库内 benchmark 命令复现。
