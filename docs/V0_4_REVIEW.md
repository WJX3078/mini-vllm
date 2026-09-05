# mini-vLLM v0.4 — Adversarial Review

Review 立场：假设实现者写了严重 bug，对 v0.4 全部 diff 重新从零审查。
每一项都实际验证过（读代码 + 跑诊断脚本），不是猜测。

## P0（correctness / leak / crash）

### P0-1 Async registry 泄漏（已修复，benchmark 期间实证发现）
`_delta_sse`（流式路径）直接消费原始 delta 队列，其 finally 只 abort 不
弹 `AsyncLLMEngine.requests` 条目；`AsyncLLMEngine.abort_request` 又改为
保留条目等最终 delta。结果：每个完成的流式请求泄漏一条 registry 记录，
256 条（max_pending）之后所有新请求被 429。
**实证**：真机 benchmark 中 `requests_running=256`、
`requests_rejected_total=194`、engine 层 scheduler/KV 全空。
**修复**：`_delta_sse` finally 弹出 registry 条目（流结束 = 条目死亡）；
`_safe_put` 溢出取消路径同样弹出；回归测试 `concurrent_streaming` 断言
`len(engine.requests)==0`。

### P0-2 abort id race（已修复，slow-consumer 测试实证发现）
`stream()` 的 finally 先弹 registry 再发 abort cmd——engine 线程按
`target_request_id` 反查 internal id 时条目已不在，abort 落空（-1 → no-op），
被放弃的请求继续烧 GPU 直到自然完成。
**实证**：`aclose()` 诊断脚本中 `abort cmd -> internal None`。
**修复**：`_EngineCommand` 增加 `internal_id` 字段；finally 直接携带
internal id（此时已知），`_drain_commands` 优先使用。

### P0-3 spec stop 语义（继承自 v0.3 已修，本轮补测试）
stop 检查只看 committed 流（v0.3 修复）；本轮补充 4 个 regression 测试
（脚本化 drafter 构造 rejected-proposal / bonus-completes-stop 场景）。

## P1（performance / reliability）

### P1-1 非流式路径无 request timeout（已修复）
超时只在 SSE 生成器实现；`stream=false` 的聚合路径会无限等待。
修复：`_collect` 逐步 `wait_for(anext, remaining)`，超时 abort + 408。

### P1-2 open-loop benchmark 的 "Poisson" 是均匀间隔（已修复）
`delay = i / rate` 是 deterministic 均匀到达，不是 Poisson。
修复：inter-arrival ~ `Exp(rate)`（`random.expovariate`），累计调度。

### P1-3 finished 序列滞留 scheduler.running（已修复，v0.4 早期）
running 列表只在下一次 schedule() 开头清理 → serving gauge 虚高一个
周期。修复：step() 末尾清理。

## P2（记录，不阻塞）
* `_external_id` 对 registry 线性扫描——每次 delta O(N)；规模小可接受，
 量大时应维护 internal→external dict。
* slow-client 的 abort cmd 可能与 SSE finally 的 abort cmd 重复——
  engine.abort_request 幂等，无害。
* `_cmd_queue` 无界——abort 风暴才会增长；admission 有界，可接受。
* StreamingRequest.state 不会推进到 "finished"（最终状态在 delta 的
  finish_reason 里）——冗余字段，保留供调试。
* 非 streaming 的 `_collect` 里 `n>1` 只聚合不做逐 choice 的 SSE
  （chat streaming 的 choices index 已支持）。

## P3（cosmetic）
* `@app.on_event` deprecated（FastAPI 建议 lifespan）；功能正常。
* auth middleware 在 CORS 之前——浏览器预检 OPTIONS 会被 401
  （CORS 默认关闭，仅显式开启时相关）。
* 日志里 prompt 不落盘 ✓（审查确认无泄漏点）。

## 回归结果（修复后）
```
ruff check .                    → All checks passed
pytest tests/ -q                → 926 passed
pytest tests/serving/ -q        → 49 passed（engine 12 + async 9 + slow 2
                                   + http 16 …含全部 cancellation/leak 矩阵）
```
