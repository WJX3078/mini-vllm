"""Request / sequence abstractions and sampling parameters."""
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import torch


class SequenceStatus(Enum):
    WAITING = auto()       # in the waiting queue, not yet allocated
    RUNNING = auto()       # has KV blocks, participates in batching
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH = auto()    # hit max_tokens
    FINISHED_ABORTED = auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in (SequenceStatus.FINISHED_STOPPED,
                          SequenceStatus.FINISHED_LENGTH,
                          SequenceStatus.FINISHED_ABORTED)


@dataclass
class SamplingParams:
    temperature: float = 1.0       # 0.0 => greedy
    top_p: float = 1.0
    top_k: int = -1                # -1 disables
    max_tokens: int = 64
    n: int = 1                     # parallel sampling: n sequences forked from one prompt
    stop: list[str] = field(default_factory=list)
    ignore_eos: bool = False
    seed: int | None = None

    def __post_init__(self):
        assert self.max_tokens > 0
        assert self.n >= 1
        if self.temperature == 0.0:
            # greedy ignores the rest
            self.top_p = 1.0
            self.top_k = -1


class Sequence:
    """One generation path: a prompt plus the tokens generated so far.

    KV-cache bookkeeping lives here:
      * block_table[i]  -> physical block id holding tokens [i*bs, (i+1)*bs)
      * block_keys[i]   -> prefix-cache hash key of that block's *content*
                           (None until the block is full)
      * num_computed_tokens -> tokens whose K/V already sit in the pool
    """

    _next_id = 0

    def __init__(self, prompt_token_ids: list[int], sampling_params: SamplingParams,
                 arrival_time: float | None = None):
        self.seq_id = Sequence._next_id
        Sequence._next_id += 1

        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: list[int] = []
        self.sampling_params = sampling_params

        self.status = SequenceStatus.WAITING
        self.num_computed_tokens = 0

        # Managed by BlockSpaceManager
        self.block_table: list[int] = []
        self.block_keys: list[tuple | None] = []

        # Timing (for TTFT / TPOT metrics)
        self.arrival_time = arrival_time if arrival_time is not None else time.perf_counter()
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None

        # Fork bookkeeping (parallel sampling): id of the parent sequence
        self.parent_seq_id: int | None = None

        # Per-request RNG (see minivllm/sampling.py): the engine derives a
        # stable 64-bit seed from (engine_seed, request_id, sample_idx,
        # user_seed) and each sequence lazily owns an independent Generator,
        # so sampled output never depends on batch composition.
        self.rng_seed: int | None = None
        self.sample_idx: int = 0          # 0 = main output, 1..n-1 = forks
        self._generator: torch.Generator | None = None

    def sampling_generator(self) -> torch.Generator:
        """This sequence's own CPU generator (independent of other requests)."""
        if self._generator is None:
            self._generator = torch.Generator()
            self._generator.manual_seed(self.rng_seed if self.rng_seed is not None else 0)
        return self._generator

    # ---- token views -------------------------------------------------------
    @property
    def tokens(self) -> list[int]:
        """prompt + generated so far."""
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    def get_new_tokens(self) -> list[int]:
        """Tokens not yet written to the KV pool."""
        return self.tokens[self.num_computed_tokens:]

    def get_len(self) -> int:
        return self.num_tokens

    def get_last_token_id(self) -> int:
        return self.tokens[-1]

    # ---- status ------------------------------------------------------------
    @property
    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    @property
    def is_prefill_step(self) -> bool:
        """True if this sequence still has uncomputed prompt tokens."""
        return self.num_computed_tokens < self.num_prompt_tokens

    def record_first_token(self):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    # ---- metrics -----------------------------------------------------------
    def get_ttft(self) -> float:
        assert self.first_token_time is not None
        return self.first_token_time - self.arrival_time

    def get_tpot(self) -> float:
        """Time per output token (averaged over tokens after the first)."""
        n_out = len(self.output_token_ids)
        if n_out < 2 or self.first_token_time is None:
            return 0.0
        return (self.last_token_time - self.first_token_time) / (n_out - 1)

    def __repr__(self):
        return (f"Sequence(id={self.seq_id}, len={self.num_tokens}, "
                f"computed={self.num_computed_tokens}, status={self.status.name}, "
                f"blocks={len(self.block_table)})")


@dataclass
class RequestOutput:
    """Final result for one request (contains all n sampled sequences)."""

    request_id: int
    prompt: str
    outputs: list[dict] = field(default_factory=list)  # [{text, token_ids, ttft, tpot}]
    finished: bool = False
