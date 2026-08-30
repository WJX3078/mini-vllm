"""End-to-end correctness matrix (v0.3).

Attention backend (torch / triton) x decoding mode (greedy / sampling) x
prefix cache (off / on) x chunked prefill (off / on) x head layout (MHA /
GQA) x batch shape (single / continuous batch).

CPU rows run on the tiny random Qwen2 in fp32 and compare token-identical
output against HuggingFace. GPU rows run the Triton backend on CUDA in
fp32 (marked `gpu`) and compare token-identical against the torch backend
on the same device -- fp32 keeps both attention paths numerically close
enough for identical greedy tokens.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from helpers import make_tiny_pair, random_prompts, run_hf_greedy

from minivllm import EngineConfig, LLMEngine
from minivllm.config import ModelConfig
from minivllm.kernels.paged_attention import triton_available
from minivllm.sequence import SamplingParams


def _engine(backend, cache, chunked, max_num_seqs=8, num_blocks=64,
            device="cpu", dtype="float32", seed=0):
    hf, mine = make_tiny_pair(seed)
    cfg = EngineConfig(
        model="(tiny-random-qwen2)", block_size=8, num_blocks=num_blocks,
        max_num_seqs=max_num_seqs, max_model_len=256,
        max_num_batched_tokens=96 if chunked else 256,
        enable_chunked_prefill=chunked, enable_prefix_caching=cache,
        attention_backend=backend, seed=seed, device=device, dtype=dtype)
    if device == "cuda":
        mine = mine.to(device)
    eng = LLMEngine(cfg, model=mine,
                    model_config=ModelConfig.from_hf_config(hf.config),
                    tokenizer=None)
    return eng, hf


@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("sampling", [False, True])
def test_cpu_matrix_matches_hf(backend, cache, chunked, sampling):
    """16 CPU combinations: token-identical to HF greedy (sampling rows use
    a fixed per-request seed and only assert shape/reproducibility)."""
    eng, hf = _engine(backend, cache, chunked)
    prompts = random_prompts(4, min_len=12, max_len=30, seed=77)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=10)
    if sampling:
        params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=10,
                                seed=42, ignore_eos=True)
        outs = eng.generate(prompts, params, use_tqdm=False)
        for o in outs:
            assert len(o.outputs[0]["token_ids"]) == 10
        # same seeds again -> identical sampled output
        eng2, _ = _engine(backend, cache, chunked)
        outs2 = eng2.generate(prompts, params, use_tqdm=False)
        assert [o.outputs[0]["token_ids"] for o in outs] == \
            [o.outputs[0]["token_ids"] for o in outs2]
    else:
        params = SamplingParams(temperature=0.0, max_tokens=10,
                                ignore_eos=True)
        outs = eng.generate(prompts, params, use_tqdm=False)
        for i, o in enumerate(outs):
            assert o.outputs[0]["token_ids"] == ref[i], \
                f"backend={backend} cache={cache} chunked={chunked} req {i}"


def test_cpu_gqa_tiny_layout_matches_hf():
    """The tiny model IS GQA (4Q/2KV) -- every row above already covers it;
    this explicit alias keeps the matrix self-documenting."""
    eng, hf = _engine("torch", cache=True, chunked=True)
    prompts = random_prompts(2, min_len=10, max_len=16, seed=78)
    ref = run_hf_greedy(hf, prompts, max_new_tokens=8)
    outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8,
                                                ignore_eos=True),
                        use_tqdm=False)
    for i, o in enumerate(outs):
        assert o.outputs[0]["token_ids"] == ref[i]


@pytest.mark.gpu
def test_gpu_triton_backend_matches_torch_backend():
    """On CUDA (fp32 so both attention paths agree numerically): the Triton
    engine must produce token-identical greedy output to the torch engine,
    across cache/chunked toggles and a continuous batch."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not triton_available():
        pytest.skip("Triton not available")
    prompts = random_prompts(6, min_len=12, max_len=40, seed=79)
    params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
    for cache in (False, True):
        for chunked in (False, True):
            eng_t, _ = _engine("torch", cache, chunked, device="cuda")
            eng_k, _ = _engine("triton", cache, chunked, device="cuda")
            out_t = eng_t.generate(prompts, params, use_tqdm=False)
            out_k = eng_k.generate(prompts, params, use_tqdm=False)
            for i, (a, b) in enumerate(zip(out_t, out_k)):
                assert a.outputs[0]["token_ids"] == b.outputs[0]["token_ids"], \
                    f"cache={cache} chunked={chunked} req {i} diverged"


@pytest.mark.gpu
def test_gpu_triton_sampling_reproducible():
    if not torch.cuda.is_available() or not triton_available():
        pytest.skip("no CUDA device / no Triton")
    eng, _ = _engine("triton", cache=True, chunked=True, device="cuda")
    prompts = random_prompts(3, min_len=10, max_len=20, seed=80)
    params = SamplingParams(temperature=0.9, top_p=0.92, max_tokens=12,
                            seed=7, ignore_eos=True)
    a = eng.generate(prompts, params, use_tqdm=False)
    b = eng.generate(prompts, params, use_tqdm=False)
    for x, y in zip(a, b):
        assert x.outputs[0]["token_ids"] == y.outputs[0]["token_ids"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
