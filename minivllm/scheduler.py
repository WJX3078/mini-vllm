"""Continuous batching scheduler (iteration-level scheduling).

Every engine step the scheduler decides which sequences run:

  1. RUNNING sequences keep decoding. Each needs slots for its new token(s);
     if the KV pool cannot provide one, the newest sequence is preempted
     (recompute policy: its blocks are freed, it goes back to the front of the
     waiting queue and later restarts from the prefix cache).
  2. WAITING sequences are admitted FCFS while the token budget and the
     max_num_seqs cap allow, allocating blocks (prefix-cache lookup included).

A single step may mix prefills and decodes in one batch -- new requests join
at iteration granularity, never waiting for the whole batch to drain.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

from minivllm.block_manager import BlockSpaceManager
from minivllm.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    scheduled: List[Sequence] = field(default_factory=list)
    num_prefill_seqs: int = 0
    num_decode_seqs: int = 0
    num_new_tokens: int = 0
    num_preempted: int = 0


class Scheduler:
    def __init__(self, block_manager: BlockSpaceManager, max_num_seqs: int,
                 max_num_batched_tokens: int):
        self.bm = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or any(not s.is_finished for s in self.running)

    def schedule(self) -> SchedulerOutput:
        budget = self.max_num_batched_tokens
        out = SchedulerOutput()
        self.running = [s for s in self.running if not s.is_finished]

        # ---- 1. running decodes (FCFS), preempting the newest on OOM
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            need = seq.num_tokens - seq.num_computed_tokens
            assert need >= 1, f"seq {seq.seq_id} has no new tokens to run"
            if need > budget:
                break
            if not self.bm.prepare_slots(seq):
                victim = self.running[-1]
                self.num_preemptions += 1
                out.num_preempted += 1
                self.bm.preempt_sequence(victim)
                victim.status = SequenceStatus.WAITING
                self.running.pop()
                self.waiting.appendleft(victim)
                if victim is seq:
                    continue          # do not advance: seq left the list
                continue              # re-examine seq[ i ] against a smaller batch
            out.scheduled.append(seq)
            out.num_new_tokens += need
            out.num_decode_seqs += 1 if not seq.is_prefill_step else 0
            out.num_prefill_seqs += 1 if seq.is_prefill_step else 0
            budget -= need
            i += 1

        # ---- 2. admit waiting prefills (FCFS)
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            new_tokens = seq.num_tokens - seq.num_computed_tokens
            if new_tokens > budget:
                break
            if seq.block_table:
                # Forked child (parallel sampling): already shares the parent's
                # blocks, it only needs a slot for its next decode token.
                if not self.bm.prepare_slots(seq):
                    break
            else:
                cached = self.bm.allocate_sequence(seq)   # None => rolled back
                if cached is None:
                    break                                  # out of blocks, FCFS stop
                if cached == seq.num_tokens:
                    # Fully-cached prompt: force-recompute the last token so the
                    # forward produces its logits. The rewritten K/V is identical
                    # to what the shared block already holds, so this is safe.
                    cached -= 1
                seq.num_computed_tokens = cached
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            out.scheduled.append(seq)
            out.num_prefill_seqs += 1 if seq.is_prefill_step else 0
            out.num_decode_seqs += 0 if seq.is_prefill_step else 1
            out.num_new_tokens += new_tokens
            budget -= new_tokens

        if not out.scheduled and self.waiting:
            head = self.waiting[0]
            if not self.running:
                raise RuntimeError(
                    f"cannot schedule seq {head.seq_id}: prompt of {head.num_tokens} "
                    f"tokens does not fit in the KV pool "
                    f"({self.bm.num_blocks} blocks x {self.bm.block_size} tokens). "
                    f"Increase num_blocks / max_model_len or reduce block_size.")
        return out
