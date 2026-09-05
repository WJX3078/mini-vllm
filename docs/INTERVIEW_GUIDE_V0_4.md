# v0.4 Serving — 面试追问指南

每题答案都能落到仓库具体代码/测试/数据上。

**为什么 HTTP handler 不能直接调用 generate？**
`generate()` 是阻塞批接口：每个 HTTP 请求独立 generate = 每请求独占 GPU
跑完才轮到下一个，continuous batching 完全失效。正确形态是 handler 只把
请求放入队列，唯一的 engine 循环以 iteration 粒度把所有在线请求批进同一
次 forward。测试 `test_continuous_batching_under_concurrency` 证明 32 个
并发请求的峰值同批 running ≥ 8（max_num_seqs=8 的上限）。

**Async Engine 如何保证只有一个 GPU step loop？**
`AsyncLLMEngine` 拥有一个专职 engine 线程，`step()` 只在该线程的循环里
调用；async 侧只有 add/abort 命令入队。没有锁竞争、没有 to_thread 的
调度抖动，PyTorch CUDA 语义最稳。

**Continuous batching 如何和 HTTP concurrency 结合？**
HTTP 并发 = input queue 的到达流；engine 每步 drain 队列、schedule()
重新组批——新请求 iteration 粒度插入，早完成的立即让位。

**HTTP 请求怎样进入 Scheduler？**
`AsyncLLMEngine.add_request`（事件循环）→ `queue.Queue` → engine 线程
`_drain_commands` → `LLMEngine.add_request`（同一个 Sequence/registry
事实源）→ scheduler.waiting。

**Streaming token 怎么回到正确客户端？**
engine 每步产出 `RequestOutputDelta(request_id, sample_idx, ...)`；
engine 线程按 internal id 反查 registry 里的 StreamingRequest，经
`call_soon_threadsafe` 投进该请求私有的 `asyncio.Queue`；SSE 生成器
await 自己的队列。跨请求串线在结构上不可能（队列一对一）。

**如何避免不同 request output 串线？**
同上 + 测试：`test_async_multi_request_and_output_routing` 校验每个流
文本 == 自己 token 流的 decode；128 并发无丢失/重复测试。

**客户端断开后怎样取消 generation？**
SSE 生成器被取消（Starlette 断开处理）→ finally：弹 registry +
向 engine 线程发 abort（**携带 internal id**——registry 条目已弹出，
不携带会查不到，这是 review 抓出的 P0 竞态）→ engine 从
waiting/running 移除并 free_sequence。

**取消 running request 时 KV Cache 怎么释放？**
`free_sequence`：私有块回 free list、共享块 ref_count-1、reservation
清零（`total_reserved_blocks` 同步扣减）。缓存块留在 cache（ref=0）供
LRU 驱逐——语义与正常 finish 完全一致。

**Prefix Cache block refcount 怎么处理？**
`test_abort_decrements_shared_prefix_refcount`：两请求共享前缀块
ref=2，abort 一个后 ref=1，另一个正常完成。

**为什么 output queue 要 bounded？** 慢客户端为什么不阻塞 GPU？
无界队列 = 慢客户端的 delta 无限堆积，且背压会沿队列传导到 engine 的
派发点。bounded（默认 256）+ 满则取消该请求：GPU 永不为慢读者空转，
`slow_client_cancellations_total` 可观测。token 不静默丢弃——是整请求
取消。

**如何实现 backpressure？queue 满为什么返回 429？**
入口 bounded queue + 立即拒绝：过载时明确失败好过无限排队（客户端可
重试/降级）。429 + 结构化错误体（`server_overloaded`）是 HTTP 的标准
过载语义。测试 `test_queue_full_overload`。

**TTFT / TPOT？** TTFT=请求到达到首个 token（服务端记录 arrival 与
first_token_at，client-visible TTFT 由 benchmark 测首 SSE chunk）；
TPOT=(e2e−ttft)/(out_tokens−1)。/metrics 出分位数。

**HTTP 层增加多少 TTFT？** 本地回环实测 client-visible TTFT 与 engine
TTFT 差 ~5-15ms（SSE flush 一次）。`profile_runtime` 可分解引擎内耗时。

**为什么一个 GPU server 不能开多个 uvicorn worker？** 每 worker 是独立
进程 = 独立加载一份模型 + 独立 KV 池，8GB 卡直接 OOM；且多进程无法共享
continuous batching。CLI 钉死 workers=1。

**如何 graceful shutdown？** /ready 置 false（停止引流）→ 等待
`shutdown_grace_period` 内自然完成 → abort 剩余（KV 全清）→ 等 abort
delta 冲刷 → 停线程 join → 清 registry。测试 `test_shutdown_cancels_active_requests`。

**为什么 CUDA OOM 不能作为 admission control？** OOM 是全局性失败：
爆的时刻可能连带其他请求的 workspace 分配；且 PyTorch 缓存分配器下
恢复不可控。正确做法是准入时用 reservation 检查并登记容量（本项目的
`scheduler_reserve_full_isl`），OOM 只作为 bug 处理。

**KV Reservation 怎么帮助避免 OOM？** 准入时检查并登记整个 cold-prompt
容量（cache hit 不计、其他序列的未兑现 reservation 计入占用），物理块
按 span 惰性物化——"准进来的必定装得下"，KV 从不超卖。

**Chunked Prefill 对真实 serving 的意义？** 长 prompt 不再独占
iteration：decode 先各拿 1 token，剩余预算给 prefill chunk。
serving mixed workload 实测 TTFT p99 12.2s（20% 4K prompt 排队）——
它同时暴露 tail 问题，引出 prefill 抢占的改进方向。

**为什么 mixed workload 比固定 batch benchmark 更重要？** 生产流量长短
混合；固定 batch 会隐藏 head-of-line 和排队 tail。mixed 数据里 TTFT
p99/p50 = 2.4x（vs short 的 1.3x），这是只有 mixed 能暴露的。

**Open-loop QPS vs closed-loop concurrency？** closed：N 个 worker 各
发完再发（不超 N 在飞）——测饱和吞吐；open：按到达率发请求（在飞数无
上限）——测过载下的排队与 tail。8 QPS open-loop 的 TTFT p50 240ms 但
p99 2945ms——tail 只有 open-loop 看得见。

**p99 为什么对 serving 很重要？** LLM 响应是长流式交互，一个坏请求拖住
整个会话体验；SLA 通常按 p99 定。chunked prefill/reservation/抢占策略
的每个决定都该看 p99 而不是均值（本项目每个 benchmark 都输出 p50-p99）。
