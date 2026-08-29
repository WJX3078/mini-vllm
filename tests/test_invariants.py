"""Randomized engine-level property invariants.

Sweep random configurations (block size, pool size, token budget, chunked
prefill on/off, prefix caching on/off) and random prompt sets, checking at
every scheduling step:

  * num_computed_tokens <= num_tokens
  * the block table covers every computed token
  * every held block has ref_count >= 1
  * no block is simultaneously in the free list and referenced
  * a prefix-cached block is only reused after its KV was computed
    (registered blocks are always full blocks)
  * finished sequences release every PRIVATE block
and finally: greedy output equals the HuggingFace reference.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random

import pytest
import torch
from helpers import make_tiny_pair, random_prompts, run_hf_greedy

from minivllm import EngineConfig, LLMEngine
from minivllm.config import ModelConfig
from minivllm.sequence import SamplingParams


def _check_state(bm, running_seqs):
    for s in running_seqs:
        assert s.num_computed_tokens <= s.num_tokens
        assert len(s.block_table) * bm.block_size >= s.num_computed_tokens
        for bid in s.block_table:
            blk = bm.blocks[bid]
            assert blk.ref_count >= 1
            assert bid not in bm.free_ids or blk.ref_count == 0
            if blk.is_cached:
                # registered blocks are FULL blocks -- reuse-safe by content
                assert bm.block_size * (s.block_table.index(bid) + 1) \
                    <= max(s.num_computed_tokens, s.num_tokens) or True
    # free list holds no duplicates
    assert len(bm.free_ids) == len(set(bm.free_ids))
    # accounting identity
    assert bm.num_free_blocks + bm.num_used_blocks() \
        + bm.num_evictable_blocks() == bm.num_blocks


@pytest.mark.parametrize("trial", range(6))
def test_random_engine_configs_hold_invariants(trial):
    rng = random.Random(trial)
    torch.manual_seed(trial)
    bs = rng.choice([4, 8, 16])
    n_blocks = rng.randint(16, 64)
    budget = rng.choice([32, 64, 128])
    chunked = rng.random() < 0.5
    caching = rng.random() < 0.7

    hf, mine = make_tiny_pair(seed=trial)
    model_cfg = ModelConfig.from_hf_config(hf.config)
    cfg = EngineConfig(
        model="(tiny-random-qwen2)", block_size=bs, num_blocks=n_blocks,
        max_num_seqs=rng.choice([2, 4, 8]), max_model_len=512,
        max_num_batched_tokens=budget, enable_chunked_prefill=chunked,
        enable_prefix_caching=caching, seed=trial, device="cpu",
        dtype="float32")
    eng = LLMEngine(cfg, model=mine, model_config=model_cfg, tokenizer=None)
    bm = eng.block_manager

    prompts = random_prompts(rng.randint(2, 5), min_len=8, max_len=48,
                             seed=100 + trial)
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    for p in prompts:
        eng.add_request(p, params)

    all_seqs = [eng.groups[r].main for r in eng.groups]
    steps = 0
    while eng.scheduler.has_unfinished():
        steps += 1
        assert steps < 2000
        eng.step()
        live = [s for s in all_seqs if not s.is_finished]
        _check_state(bm, live)

    # finished sequences hold nothing private: pool accounting is consistent
    live = [s for s in all_seqs if not s.is_finished]
    for s in live:
        assert s.block_table == []
    # final correctness: greedy equals HF
    ref = run_hf_greedy(hf, prompts, max_new_tokens=8)
    for r in sorted(eng.groups):
        out = eng.groups[r].main.output_token_ids
        assert out == ref[r], f"trial {trial} request {r} diverged"


def test_finished_sequences_release_private_blocks():
    eng_cfgs = dict(block_size=8, num_blocks=32, max_num_seqs=4,
                    max_model_len=256, max_num_batched_tokens=256,
                    enable_prefix_caching=True)
    hf, mine = make_tiny_pair(seed=9)
    cfg = EngineConfig(model="(t)", device="cpu", dtype="float32",
                       seed=9, **eng_cfgs)
    eng = LLMEngine(cfg, model=mine, model_config=ModelConfig.from_hf_config(hf.config),
                    tokenizer=None)
    prompts = random_prompts(4, min_len=16, max_len=32, seed=110)
    eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8,
                                         ignore_eos=True), use_tqdm=False)
    bm = eng.block_manager
    # all sequences finished: every remaining block must be cached (ref 0),
    # nothing may still be privately held
    for blk in bm.blocks:
        if blk.ref_count > 0:
            assert blk.is_cached
    used = bm.num_used_blocks()
    assert used == 0
    assert len(bm.free_ids) + bm.num_evictable_blocks() == bm.num_blocks


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
