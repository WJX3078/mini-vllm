# mini-vLLM v0.4 — Serving Runtime Design

状态：设计基线（与 v0.3 代码逐点核对后编写）。实现过程中若与本文冲突，
以代码实际行为为准并回改本文。

## 1. Existing Architecture（现有架构）

v0.3 的同步核心由以下组件构成：

```
LLMEngine (engine.py)
 ├── add_request(prompt, params) -> request_id:int   登记 RequestGroup
 ├── step() -> list[RequestOutput]                   一个 iteration；
 │                                                   返回本步 finish 的请求
 ├── generate(...)                                   阻塞批接口（serving 不用）
 ├── abort_request        —— v0.4 新增
 └── groups: {request_id: RequestGroup}
        RequestGroup(request_id, prompt, params, main, pending_forks, stop_checker)
             main: Sequence（status / num_computed_tokens / block_table /
                              block_keys / reserved_cold_blocks / RNG）
        seq_to_group: {seq_id: RequestGroup}

Scheduler (scheduler.py)
 ├── waiting: deque[Sequence]      未准入（含 fork children、被抢占重排）
 ├── running: list[Sequence]       已准入（decode 或 mid-prefill chunk）
 ├── schedule() -> SchedulerOutput(scheduled, spans, counts, num_preempted)
 └── _preempt_newest()             池满时从队尾抢占（recompute 策略）

BlockSpaceManager (block_manager.py)
 ├── map_cached_prefix / can_reserve→cold_blocks_needed /
 │   blocks_available_after_mapping / reserve / release_reservation
 ├── allocate_span(seq, start, end)  惰性物化 + reservation 扣减
 ├── free_sequence(seq)              释放块引用 + reservation 归零
 ├── preempt_sequence(seq)           = free_sequence + computed=0
 └── total_reserved_blocks           全局预订计数（准入公式的一部分）

KVCachePool、Sampler（GPU-native 采样）、StopChecker（增量 stop）、
prefix_hash backend、Triton PagedAttention kernel、ModelConfig/EngineConfig
```

### 标识符与生命周期（现状）

* `request_id`：engine 内部自增 int；`RequestGroup` 是唯一事实源。
* `sequence_id`：`Sequence._next_id` 全局自增；一个 request 可对应
  1..n 个 Sequence（parallel sampling）。
* 生命周期（同步视角）：
  `add_request` → main 进 `scheduler.waiting`（children 在组内挂起，
  parent 首个 token 后 fork 进 waiting）→ `schedule()` 准入（map cache 前缀
  + reserve + 首 chunk 物化）→ running → 每 `step()` 采 1 token →
  EOS/stop/max_tokens 触发 `free_sequence` + 从 registry 删除 →
  `RequestOutput` 返回。
* KV 状态：waiting（0 块或仅 cache 前缀映射）、running（span 物化 +
  reservation 记账）、finished（free_sequence，缓存块留 ref=0 可驱逐）。

## 2. Request / Sequence Lifecycle（v0.4 状态机）

服务层新增 `ServingRequestState`（engine 的 SequenceStatus 保持不变，
两层各自有事实源，通过 abort/finish 事件同步）：

```
CREATED ──→ QUEUED ──→ RUNNING ──→ FINISHED(stop|length)
   │           │          │
   │           │          ├──→ CANCELLED   (client disconnect / timeout / abort)
   │           │          └──→ FAILED       (局部异常)
   │           └──→ CANCELLED
   └──→ REJECTED            (input queue 满 → HTTP 429)
```

每个 HTTP 请求在服务层持有：

```python
@dataclass
class ServingRequest:
    request_id: str            # "cmpl-xxxxxxxx" / "chatcmpl-xxxxxxxx"（外部）
    internal_id: int | None    # engine 的 int request_id（admission 后回填）
    arrival_time / first_token_at / finished_at: float
    state: ServingRequestState
    params: SamplingParams
    output_queue: asyncio.Queue[RequestOutputDelta]   # bounded（backpressure）
    abort_event: asyncio.Event
    detokenizer: IncrementalDetokenizer
```

外部 id ↔ 内部 id 的映射只存在于服务层 registry；不改 scheduler 的 int id。

## 3. Current Streaming Blockers（3.3 审计：step() 是否适合异步 serving）

1. **一次 step 返回什么？** `list[RequestOutput]`，只含本步 finish 的请求
   （`_build_output` 构造，含全量 token_ids/text）。
2. **是否只返回 finished？** 是。RUNNING 请求的新 token 只落在
   `seq.output_token_ids`，调用方不可见。
3. **如何获得每步新增 token？** 现状需对比前后 `len(output_token_ids)`——
   属于 hack。
4. **需要新 API？** 需要。→ v0.4 在 engine 层新增
   `RequestOutputDelta(request_id, sample_idx, token_ids, finished,
   finish_reason)`：`step()` 内部在每个采样点收集 delta，
   `engine.pop_deltas()` 在每步后取走。文本增量由服务层
   `IncrementalDetokenizer` 负责（engine 保持 tokenizer 可注入/可为 None，
   CPU 测试不依赖 tokenizer）。
5. **取消 running 安全吗？** 安全点存在：`step()` 之外的时刻，
   scheduler.running 中的 Sequence 可以被移除——但必须同时
   `free_sequence`（块 + reservation）并处理 COW 共享（`free_sequence`
   已经正确递减 ref_count，共享块留给他者，私有块回 free list）。
6. **取消 waiting 安全吗？** 安全：直接从 deque 移除；未准入序列没有
   物理块，若已有 cache 前缀映射则必须 `free_sequence` 回滚 ref_count
   （v0.4 的 abort 覆盖）。
7. **n>1 如何取消？** 取消整个 RequestGroup：main（running/waiting）+
   pending_forks（waiting）+ 已 fork children（running）全部处理。
8. **speculative decoding 如何取消？** SpeculativeEngine 是单请求阻塞
   API（无连续批处理），v0.4 serving 不暴露 spec 路径（HTTP 只走
   LLMEngine 连续批处理）；文档明确说明。
9. **取消后块如何释放？** 统一走 `bm.free_sequence(seq)`（递减 ref_count、
   归还私有块、reservation 清零）；缓存块保留在 cache（ref=0 可 LRU 驱逐，
   refcount 语义与正常 finish 完全一致）。
10. **pending forks 会泄漏吗？** 现状：`pending_forks` 只在
    `_maybe_fork_children` 清空；若请求在 fork 前完成/取消会泄漏。→
    abort 与 finish 路径都必须清空 pending_forks（v0.4 修复点）。

**结论**：同步引擎的调度/内存语义完全可复用；缺的是
delta 输出、abort API 和异步外壳。不 hack HTTP 层，全部在 engine/serving
层新增。

## 4. Proposed v0.4 Architecture

```
HTTP (FastAPI/Starlette)
   → protocol 校验（pydantic）+ 外部 request id
   → AsyncLLMEngine.add_request()          [async, bounded input queue]
   → per-request asyncio.Queue[RequestOutputDelta]（bounded）
   → SSE / 聚合响应
                 ▲
                 │ loop.call_soon_threadsafe（线程→事件循环）
Engine Thread（唯一 GPU 所有者）
   → drain input / cancellations
   → LLMEngine.step()（同步 GPU 调用，单线程串行）
   → pop_deltas() → dispatch
   → 无请求时 await/Event 等待唤醒（禁 busy spin）
```

新增文件：

```
minivllm/serving/__init__.py
minivllm/serving/request.py        ServingRequest / 状态机
minivllm/serving/detokenizer.py    IncrementalDetokenizer（有界窗口增量解码）
minivllm/serving/async_engine.py   AsyncLLMEngine（engine 线程 + 桥接）
minivllm/serving/metrics.py        轻量 metrics（Prometheus 文本格式）
minivllm/entrypoints/__init__.py
minivllm/entrypoints/openai/__init__.py
minivllm/entrypoints/openai/protocol.py      pydantic 请求/响应模型
minivllm/entrypoints/openai/serving_completion.py
minivllm/entrypoints/openai/serving_chat.py
minivllm/entrypoints/openai/api_server.py    app 工厂 + CLI
minivllm/bench/serving/__init__.py
minivllm/bench/serving/client.py   压测 HTTP 客户端
minivllm/bench/serving/workload.py workload 生成
minivllm/bench/serving/metrics.py  延迟统计
minivllm/bench/serving/__main__.py CLI
docs/V0_4_*.md
tests/serving/...
```

修改文件：`engine.py`（deltas + abort）、`sequence.py`（如有字段）、
`config.py`（serving 相关配置）、`pyproject.toml`（serve/bench extras、
版本 0.4.0）、`README.md`。

## 5. Async Concurrency Model

**选择：dedicated engine thread + 单步串行。** 理由：

* `engine.step()` 是同步 CUDA 调用（~10-100ms）；直接跑在事件循环里会
  冻结所有 HTTP 处理（SSE flush、新请求接受都停摆）；
* `asyncio.to_thread(step)` 每步一次线程切换 + 异步语义复杂化，且
  "step 必须串行"的约束只能靠锁保证——不如专职线程清晰；
* 线程内**严格单循环**：任何时刻至多一个 `step()` 在执行（任务 1.1）；
  与 GPU 的交互全部在该线程内，PyTorch CUDA 行为最稳定；
* 线程 → 事件循环的输出派发用 `loop.call_soon_threadsafe`（非阻塞）；
  事件循环 → 线程的输入/取消用 `queue.Queue` + `threading.Event` 唤醒
  （无 busy spin，空闲时阻塞在 Event.wait）。

优雅关闭：`shutdown()` → 置停机位 → 停止接受新请求 → 等待
`shutdown_grace_period` 内 active 请求完成 → 超时则 abort 全部 → join
线程。

## 6. Request State Machine

见第 4 节。转换点：

* CREATED→QUEUED：HTTP 校验通过、进入 input queue；
* QUEUED→REJECTED：input queue 满（HTTP 429，结构化错误体）；
* QUEUED→RUNNING：engine 线程 admission（add_request 成功）；
* RUNNING→FINISHED：finish_reason ∈ {stop, length}；
* QUEUED/RUNNING→CANCELLED：abort（disconnect/timeout/显式）；幂等；
* RUNNING→FAILED：单请求局部异常（采样参数/tokenizer 等），引擎健康；
* 引擎级致命错误（CUDA fatal/不变量破坏）：标记 engine unhealthy，
  全部 active 请求 FAILED，/ready 置 false，停止收流量。

## 7. Cancellation Design

`LLMEngine.abort_request(request_id) -> bool`（幂等，engine 线程内调用）：

1. registry 无此 id → False（已 finish/cancel 竞态，无害）；
2. `group.pending_forks` 非空 → 逐个从 waiting 移除（这些无块）；
3. main 在 waiting → 移除；若有 block_table（预映射 cache 前缀）→
   `free_sequence`；
4. main 在 running → 移出 running；`free_sequence`（释放私有块、递减
   共享 ref、reservation 清零）；
5. children（n>1 已 fork）在 running → 同 4；在 waiting → 移除；
6. 清空 pending_forks、registry、seq_to_group；状态置 ABORTED；
7. 返回 True，服务层向 output queue 投递 `finished=True,
   finish_reason="abort"` 的最终 delta（客户端断开场景可不投递）。

竞态覆盖：token 刚生成 + 取消（delta 已投递，abort 幂等）；刚 finish +
abort（registry 已删 → False）；重复 abort（幂等）。所有路径跑
cancellation regression 测试（直接断言 free blocks / total_reserved /
refcount / running / waiting / registry）。

## 8. KV Cleanup Semantics

统一入口 `bm.free_sequence`：递减共享块 ref_count（缓存块 ref=0 留在
cache 供 LRU 驱逐，refcount 语义与正常 finish 完全一致）、私有块回
free list、`total_reserved_blocks` 扣减、reservation 清零。异常路径
（FAILED）复用同一清理。Invariant 测试在每组用例后断言：
waiting/running 空、total_reserved==0、free+used+evictable==num_blocks、
registry 空。

## 9. HTTP API Design

* `POST /v1/completions`：prompt (str|list[str]|token ids)、max_tokens、
  temperature、top_p、top_k、stop、seed、stream、n；
  **显式拒绝** engine 不支持的 OpenAI 参数（logprobs、best_of、
  presence/frequency_penalty、tools、response_format…）→ 400 + 说明；
* `POST /v1/chat/completions`：messages → `tokenizer.apply_chat_template`
  （tokenizer 不支持时返回清晰 400，不硬编码模板）；
* `GET /v1/models`、`GET /health`（进程活着）、`GET /ready`
  （模型加载完 + 引擎运行 + 接受请求）、`GET /metrics`（Prometheus 文本）；
* SSE：`data: {chunk}\n\n` + `data: [DONE]`；finish_reason: stop/length
  （abort 场景客户端已断开，不发 final chunk；显式 abort 请求按内部状态
  记 cancelled）；错误：结构化 JSON（`{"error": {message, type, code}}`）；
* API key（可选 `--api-key`，Bearer 校验）、CORS 默认关闭（`--allow-cors`）；
* prompt 长度预校验 `prompt+max_tokens ≤ max_model_len` → 400。

## 10. Backpressure Design

* **入口**：input queue `maxsize=max_pending_requests`；满 → 立即
  429（`server_overloaded`），绝不无限排队；
* **出口**：每请求 output queue bounded（`max_queue_deltas`，默认 256）；
  engine 派发用 `put_nowait`，队列满 → 该请求标记 slow，engine 线程
  abort 它（不丢 token、不阻塞其他请求）→ `slow_client_cancellations_total`
  计数。GPU 永远不等慢客户端；
* **超时**：`--request-timeout`；引擎线程定期检查，超时 → abort + KV 清理。

## 11. Metrics Design

自实现轻量 registry（不引 prometheus_client）：
request counters（total/running/waiting/finished/cancelled/rejected/failed/
slow_client）、engine counters（steps/tokens/prompt_tokens）、scheduler
gauges（running/waiting/preemptions）、KV gauges（total/free/used/
reserved/utilization）、prefix cache（hits/misses/hit_tokens）、延迟直方图
（e2e/ttft/tpot/queue_wait，p50/p90/p95/p99）。`/metrics` 输出 Prometheus
text format。日志用标准 `logging`：每请求一行 INFO（id/prompt_tokens/
output_tokens/ttft/e2e/finish_reason），不打印 prompt 正文。

## 12. Benchmark Plan

`minivllm/bench/serving`：closed-loop 固定并发 {1,2,4,8,16,32} + open-loop
Poisson QPS {1,2,4,8,16,32}；workload：short(128/64)、medium(512/128)、
long(4096/64)、mixed(80% short + 20% long)、prefix(0/50/90% 共享)；
指标：请求/输出 token 吞吐、TTFT/TPOT/E2E mean+p50/p90/p95/p99、queue
wait p95、rejected 数、KV 峰值利用率、prefix hit、抢占数；JSON 输出
（硬件/GPU/模型/dtype/配置/git commit/时间戳）。真实 serving 不做逐请求
`cuda.synchronize`（不污染服务路径）；warmup 不计入。vLLM 对比基线可选，
无则 skip。逐项遵守"Measure → Identify → Optimize → Re-benchmark"。

## 13-15. Risks

* engine 线程与事件循环的桥接错误（丢 delta/重复派发）→ 派发幂等 +
  单调 token 序列断言；
* pinned buffer 跨线程复用竞争 → H2D 只发生在 engine 线程（现状保持）；
* 异步测试在 Windows 的事件循环兼容性 → 测试用 `asyncio.run` 显式驱动；
* CI 无 GPU/无真实 tokenizer → serving 测试用 tiny 随机模型 + stub
  tokenizer（char 级），Unicode/BPE 增量解码用规则 stub 精确构造；
* 取消/完成竞态 → 幂等 abort + registry 真相源 + invariant 测试。

## 16. Milestone Plan

1. M1 Engine streaming foundation：RequestOutputDelta + pop_deltas +
   abort_request + pending_forks 泄漏修复 + cancellation regression 测试。
2. M2 AsyncLLMEngine：engine 线程、bounded input queue、registry、
   idle 唤醒、shutdown；continuous batching 证明测试（32 并发同批）。
3. M3 HTTP server：completions/chat/models/health/ready + SSE。
4. M4 Reliability：disconnect/429/slow consumer/timeout/graceful
   shutdown/异常隔离。
5. M5 Observability：metrics + logging。
6. M6 Serving benchmark（closed/open loop、mixed、prefix、JSON）。
7. M7 真 GPU benchmark 与性能 review。
8. M8 Adversarial review（docs/V0_4_REVIEW.md，P0/P1 全修）。
9. 终稿：README v0.4、V0_4_BENCHMARK.md、RESUME_BULLETS.md、
   INTERVIEW_GUIDE_V0_4.md。
