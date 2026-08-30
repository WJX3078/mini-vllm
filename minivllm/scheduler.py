"""Unified token scheduler: continuous batching + chunked prefill.

Every engine step has one token budget, ``max_num_batched_tokens``, shared by
ALL work scheduled in that iteration. The scheduler fills it in priority
order:

  1. DECODE tokens of running sequences (1 token each, FCFS). If the KV pool
     cannot provide a slot, the newest sequence is preempted (recompute
     policy: blocks freed, seq returns to the FRONT of the waiting queue and
     later restarts from the prefix cache).
  2. PREFILL CHUNKS: first the still-incomplete prefills of running
     sequences, then fresh WAITING admissions -- both FCFS, each taking
     ``min(remaining_work, remaining_budget)`` tokens. A long prompt is
     therefore spread over several iterations and can never block decodes
     for more than one iteration (head-of-line blocking is gone).
     With ``enable_chunked_prefill=False`` the legacy semantics hold: a
     waiting prompt is admitted only if its WHOLE uncached remainder fits
     the budget (no partial prompt prefill; mid-prefill running sequences
     cannot exist in that mode).

Prefix-cache-aware admission (the budget fix): for a waiting sequence we
first ask the block manager -- read-only, no refcount changes -- how many of
its prompt tokens already live in the prefix cache. Only the *uncached*
tokens are charged against the token budget, so cache hits translate
directly into extra admission/batching capacity. Blocks are acquired only
after the budget check passes; a failed allocation rolls back inside the
block manager (no leaks). A fully-cached prompt still recomputes its last
token so the forward pass produces the logits for sampling.

Chunked prefill and block allocation: admission reserves the whole prompt's
blocks up front (allocation != computation; ``num_computed_tokens`` tracks
what is actually computed). Reserving up front keeps admission control
conservative and deadlock-free (what we admit always fits), and reuses the
preemption machinery unchanged; the per-chunk lazy path in ``prepare_slots``
covers generated tokens. Real vLLM V1 goes further (lazy per-chunk
allocation + premption of prefills) -- noted as future work.
"""
from collections import deque
from dataclasses import dataclass, field

from minivllm.block_manager import BlockSpaceManager
from minivllm.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    scheduled: list[Sequence] = field(default_factory=list)
    # token span [start, end) each scheduled sequence must compute THIS step
    # (a chunk for mid-prefill sequences; [n-1, n) for decodes). Parallel to
    # `scheduled`.
    spans: list[tuple[int, int]] = field(default_factory=list)
    num_prefill_seqs: int = 0
    num_decode_seqs: int = 0
    num_new_tokens: int = 0
    num_preempted: int = 0


class Scheduler:
    def __init__(self, block_manager: BlockSpaceManager, max_num_seqs: int,
                 max_num_batched_tokens: int,
                 enable_chunked_prefill: bool = True,
                 reserve_full_isl: bool = True,
                 lazy_allocation: bool = True):
        self.bm = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        # Reservation policy: True = admission checks and books the whole
        # cold-prompt capacity (conservative, no over-admission); False =
        # only the first chunk's blocks are checked (aggressive admission,
        # more preemptions possible). Physical allocation policy: lazy =
        # blocks materialize per scheduled span; eager = the whole prompt
        # materializes at admission (v0.2 behavior, kept for A/B tests).
        self.reserve_full_isl = reserve_full_isl
        self.lazy_allocation = lazy_allocation
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or any(not s.is_finished for s in self.running)

    # ------------------------------------------------------------ preemption
    def _preempt_newest(self, out: SchedulerOutput) -> Sequence:
        victim = self.running.pop()
        self.num_preemptions += 1
        out.num_preempted += 1
        self.bm.preempt_sequence(victim)
        victim.status = SequenceStatus.WAITING
        self.waiting.appendleft(victim)
        return victim

    # -------------------------------------------------------------- schedule
    def schedule(self) -> SchedulerOutput:
        budget = self.max_num_batched_tokens
        out = SchedulerOutput()
        self.running = [s for s in self.running if not s.is_finished]

        # ---- 1. running decodes (FCFS): 1 new token each
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            if seq.is_prefill_step:        # unfinished prefill -> phase 2
                i += 1
                continue
            need = seq.num_tokens - seq.num_computed_tokens
            assert need >= 1, f"seq {seq.seq_id} has no new tokens to run"
            if need > budget:
                break                    # budget fully consumed by decodes
            if not self.bm.allocate_span(seq):
                victim = self._preempt_newest(out)
                if victim is seq:
                    continue             # seq left the list; do not advance
                continue                 # retry seq[i] against a smaller batch
            out.scheduled.append(seq)
            out.spans.append((seq.num_computed_tokens, seq.num_tokens))
            out.num_decode_seqs += 1
            out.num_new_tokens += need
            budget -= need
            i += 1

        # ---- 2. prefill chunks of running sequences (FCFS).
        # With chunked prefill disabled, mid-prefill running sequences
        # cannot exist: admission below only accepts prompts that fit the
        # budget whole (invariant asserted at admission).
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            if not seq.is_prefill_step:
                i += 1
                continue
            start = seq.num_computed_tokens
            end = seq.num_tokens if not self.enable_chunked_prefill \
                else min(seq.num_tokens, start + budget)
            assert end - start <= budget or self.enable_chunked_prefill
            if not self.bm.allocate_span(seq, start, end):
                victim = self._preempt_newest(out)
                if victim is seq:
                    continue             # re-examine running[i] (new seq now)
                continue                 # retry with freed blocks
            out.scheduled.append(seq)
            out.spans.append((start, end))
            out.num_prefill_seqs += 1
            out.num_new_tokens += end - start
            budget -= end - start
            i += 1

        # ---- 3. admit waiting sequences (FCFS), chunked prefill
        while self.waiting and len(self.running) < self.max_num_seqs \
                and budget > 0:
            seq = self.waiting[0]

            if seq.block_table:
                # Forked child (parallel sampling): it already shares the
                # parent's blocks and its KV frontier is copied from the
                # parent -- the cache probe below does NOT apply (the block
                # table, not the global cache, defines what it computed).
                start = seq.num_computed_tokens
                chunk = min(seq.num_tokens - start, budget)
                if chunk < 1:
                    break
                if not self.bm.allocate_span(seq, start, start + chunk):
                    break
            else:
                # 3a. read-only prefix-cache probe: how many prompt tokens
                # are already computed and shareable? Only UNCACHED tokens
                # cost budget.
                cached_len = self.bm.get_cached_prefix(seq.tokens)
                start = min(cached_len, seq.num_tokens - 1)
                # a fully cached prompt must still compute its last token
                # once, so the forward produces logits (rewritten KV
                # identical to what the shared block already holds).

                # 3b. budget check BEFORE acquiring any block. With chunked
                # prefill disabled, the WHOLE remaining prompt must fit the
                # budget -- otherwise the request stays WAITING (no partial
                # prompt prefill, ever).
                remaining = seq.num_tokens - start
                if not self.enable_chunked_prefill and remaining > budget:
                    break
                chunk = min(remaining, budget)
                if chunk < 1:
                    break

                # 3c. KV CAPACITY RESERVATION != ALLOCATION. The cold-prompt
                # capacity (cache hits excluded) is checked -- and, under
                # reserve_full_isl, booked for the whole prompt -- before
                # any block is taken. Physical blocks materialize lazily,
                # one scheduled span at a time (eager mode kept for A/B
                # benchmarking).
                cold = self.bm.cold_blocks_needed(seq.num_tokens, cached_len)
                available = self.bm.blocks_available_after_mapping(cached_len)
                need_now = (start + chunk + self.bm.block_size - 1)                     // self.bm.block_size - cached_len // self.bm.block_size
                if self.reserve_full_isl:
                    # promise the whole prompt capacity, checked upfront
                    if cold > available:
                        break            # would over-admit: FCFS stop
                    booked = cold
                else:
                    # aggressive: only the first chunk is checked and booked;
                    # later spans proceed unchecked (preemption is the net)
                    if need_now > available:
                        break
                    booked = need_now

                if self.lazy_allocation:
                    self.bm.map_cached_prefix(seq)
                    # single-threaded scheduler: the mapping equals the
                    # probe above
                    self.bm.reserve(seq, booked)
                    if not self.bm.allocate_span(seq, start, start + chunk):
                        # first chunk does not fit: undo mapping + reservation
                        self.bm.release_reservation(seq)
                        self.bm.free_sequence(seq)
                        break
                else:
                    cached = self.bm.allocate_sequence(seq)   # eager path
                    if cached is None:
                        break
                    start = min(cached, seq.num_tokens - 1)
                    chunk = min(seq.num_tokens - start, budget)
                    seq.num_computed_tokens = start

            if self.lazy_allocation:
                seq.num_computed_tokens = start
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            out.scheduled.append(seq)
            out.spans.append((start, start + chunk))
            if start < seq.num_prompt_tokens:
                out.num_prefill_seqs += 1
            else:
                out.num_decode_seqs += 1
            out.num_new_tokens += chunk
            budget -= chunk

        if not out.scheduled and self.waiting:
            head = self.waiting[0]
            if not self.running:
                raise RuntimeError(
                    f"cannot schedule seq {head.seq_id}: prompt of {head.num_tokens} "
                    f"tokens does not fit in the KV pool "
                    f"({self.bm.num_blocks} blocks x {self.bm.block_size} tokens). "
                    f"Increase num_blocks / max_model_len or reduce block_size.")
        return out
