"""Serving-layer shared fixtures: a tiny CPU engine + a byte-level stub
tokenizer so HTTP/streaming tests can verify real text without downloading
any model.

Token ids ARE the UTF-8 bytes of the text: multi-byte characters (CJK = 3
bytes, emoji = 4) genuinely split across tokens and decoding a partial tail
yields the replacement char -- exactly the incremental-detokenizer edge
cases that must be tested, with no model download.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from helpers import make_tiny_pair  # noqa: E402 (repo tests path)


class ByteTokenizer:
    """Byte-level stub: token ids ARE the UTF-8 bytes of the text."""

    def __init__(self):
        self.pad_token_id = 0

    def __call__(self, text):
        # HF convention: return an object carrying input_ids
        class _Enc:
            pass
        e = _Enc()
        e.input_ids = self.encode(text)
        return e

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        return text if not tokenize else self(text).input_ids


@pytest.fixture
def tiny_pair():
    return make_tiny_pair(seed=0)


@pytest.fixture
def engine_factory(tiny_pair):
    """Build an LLMEngine on the tiny random weights + ByteTokenizer."""

    def factory(**overrides):
        from minivllm import EngineConfig
        from minivllm.config import ModelConfig
        from minivllm.engine import LLMEngine

        hf, mine = tiny_pair
        cfg = EngineConfig(
            model="(tiny-random-qwen2)",
            block_size=overrides.pop("block_size", 8),
            num_blocks=overrides.pop("num_blocks", 64),
            max_num_seqs=overrides.pop("max_num_seqs", 8),
            max_model_len=overrides.pop("max_model_len", 256),
            max_num_batched_tokens=overrides.pop("max_num_batched_tokens", 256),
            enable_prefix_caching=overrides.pop("enable_prefix_caching", True),
            enable_chunked_prefill=overrides.pop("enable_chunked_prefill",
                                                 True),
            seed=overrides.pop("seed", 0), device="cpu", dtype="float32",
            **overrides)
        eng = LLMEngine(cfg, model=mine,
                        model_config=ModelConfig.from_hf_config(hf.config),
                        tokenizer=ByteTokenizer())
        return eng
    return factory


def ids_for(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def assert_no_leaks(engine):
    """Post-run invariants: queues empty, registry empty, reservations zero,
    pool accounting consistent. Prefix-cache blocks may legitimately stay
    cached (ref 0)."""
    bm = engine.block_manager
    assert not engine.scheduler.waiting
    assert not engine.scheduler.running
    assert engine.groups == {} and engine.seq_to_group == {}
    assert bm.total_reserved_blocks == 0
    assert bm.num_free_blocks + bm.num_used_blocks() \
        + bm.num_evictable_blocks() == bm.num_blocks
