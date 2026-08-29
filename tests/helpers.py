"""Shared test utilities: a tiny random-weight Qwen2 that runs anywhere.

Lets us test the whole engine (paging, scheduling, prefix cache, sampling,
speculative decoding) against HuggingFace reference outputs without
downloading any model.
"""
import torch
from transformers import Qwen2Config
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM

from minivllm.config import EngineConfig, ModelConfig
from minivllm.engine import LLMEngine
from minivllm.model import Qwen2ForCausalLM

TINY = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,        # GQA: 4 query heads, 2 KV heads
    head_dim=16,
    max_position_embeddings=512,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    attention_bias=True,
    eos_token_id=0,
)


def make_tiny_pair(seed: int = 0):
    """(hf_model, my_model) sharing identical weights, on CPU in fp32."""
    torch.manual_seed(seed)
    hf_cfg = Qwen2Config(**TINY)
    hf = HFQwen2ForCausalLM(hf_cfg).eval()

    cfg = ModelConfig.from_hf_config(hf_cfg)
    mine = Qwen2ForCausalLM(cfg, device="cpu", dtype=torch.float32)
    mine.load_from_hf(hf)
    return hf, mine


def make_tiny_engine(seed: int = 0, num_blocks: int = 64, block_size: int = 8,
                     max_num_seqs: int = 8, enable_prefix_caching: bool = True,
                     max_model_len: int = 256, temperature: float = 0.0):
    hf, mine = make_tiny_pair(seed)
    model_cfg = ModelConfig.from_hf_config(hf.config)
    engine_cfg = EngineConfig(
        model="(tiny-random-qwen2)",
        block_size=block_size,
        num_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        seed=seed,
        device="cpu",
        dtype="float32",
    )
    engine = LLMEngine(engine_cfg, model=mine, model_config=model_cfg, tokenizer=None)
    return engine, hf


def random_prompts(n: int, min_len: int = 6, max_len: int = 24, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    prompts = []
    for _ in range(n):
        L = int(torch.randint(min_len, max_len + 1, (1,), generator=g))
        ids = torch.randint(2, TINY["vocab_size"], (L,), generator=g).tolist()
        prompts.append(ids)
    return prompts


def run_hf_greedy(hf_model, prompts, max_new_tokens: int = 12):
    """Reference greedy generation with HF generate()."""
    outs = []
    for ids in prompts:
        x = torch.tensor([ids], dtype=torch.long)
        out = hf_model.generate(
            x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1,
            eos_token_id=None, pad_token_id=1)
        outs.append(out[0, x.shape[1]:].tolist())
    return outs


def run_engine_greedy(engine, prompts, max_new_tokens: int = 12):
    from minivllm.sequence import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True)
    results = engine.generate(prompts, params, use_tqdm=False)
    return [r.outputs[0]["token_ids"] for r in results]


def make_tiny_spec_engine(seed: int = 0, drafter: str = "ngram",
                          num_spec_tokens: int = 4, enable_prefix_caching: bool = True):
    """SpeculativeEngine on the tiny random weights + the HF reference model."""
    from minivllm.config import EngineConfig, ModelConfig
    from minivllm.spec.spec_engine import SpeculativeEngine

    hf, mine = make_tiny_pair(seed)
    model_cfg = ModelConfig.from_hf_config(hf.config)
    cfg = EngineConfig(
        model="(tiny-random-qwen2)",
        block_size=8, num_blocks=64, max_num_seqs=4,
        max_model_len=256, max_num_batched_tokens=256,
        enable_prefix_caching=enable_prefix_caching,
        seed=seed, device="cpu", dtype="float32")
    eng = SpeculativeEngine(cfg, drafter=drafter, num_spec_tokens=num_spec_tokens,
                            model=mine, model_config=model_cfg, tokenizer=None)
    return eng, hf


def run_spec_greedy(engine, prompts, max_new_tokens: int = 12):
    from minivllm.sequence import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True)
    outs = engine.generate(prompts, params, use_tqdm=False)
    return outs
