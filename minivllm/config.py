"""Engine / model configuration and KV-cache size math."""
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig


@dataclass
class EngineConfig:
    """Top-level configuration for LLMEngine."""

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "auto"            # "auto" | "float16" | "bfloat16" | "float32"
    device: str = "auto"           # "auto" | "cuda" | "cpu"

    # Paged KV cache
    block_size: int = 16           # tokens per physical KV block
    num_blocks: int | None = None          # override pool size (in blocks)
    gpu_memory_utilization: float = 0.75      # fraction of VRAM for weights+KV+activations

    # Scheduler (continuous batching + chunked prefill)
    max_num_seqs: int = 16                    # max sequences in a batch
    max_model_len: int = 1024                 # max tokens per sequence (prompt + output)
    max_num_batched_tokens: int = 2048        # token budget per engine step, shared by
                                              # decode tokens + prefill chunks
    enable_chunked_prefill: bool = True       # False = legacy: a whole prompt must fit
                                              # in one scheduling iteration

    enable_prefix_caching: bool = True
    hash_backend: str = "tuple"               # prefix-cache key backend: "tuple" | "sha256"
    seed: int = 1234

    def __post_init__(self):
        if not self.enable_chunked_prefill and \
                self.max_num_batched_tokens < self.max_model_len:
            # legacy mode: a whole prompt must fit in one scheduling budget
            self.max_num_batched_tokens = self.max_model_len


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
    }[dtype]


@dataclass
class ModelConfig:
    """Flat view of a HF Qwen2 config + derived KV-cache constants."""

    hf_config: Any = None
    model_path: str = ""

    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0          # query heads
    num_kv_heads: int = 0       # key/value heads (GQA when < num_heads)
    head_dim: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 0
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    eos_token_id: int | None = None

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        hf = AutoConfig.from_pretrained(model_path)
        return cls.from_hf_config(hf, model_path=model_path)

    @classmethod
    def from_hf_config(cls, hf, model_path: str = "") -> "ModelConfig":
        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        eos = hf.eos_token_id
        if isinstance(eos, list):
            eos = eos[0]
        # transformers >= 5 moved rope_theta into rope_parameters/rope_scaling
        rope_theta = getattr(hf, "rope_theta", None)
        if rope_theta is None:
            rp = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None)
            rope_theta = rp.get("rope_theta", 10000.0) if isinstance(rp, dict) else 10000.0
        return cls(
            hf_config=hf,
            model_path=model_path,
            hidden_size=hf.hidden_size,
            num_layers=hf.num_hidden_layers,
            num_heads=hf.num_attention_heads,
            num_kv_heads=getattr(hf, "num_key_value_heads", hf.num_attention_heads),
            head_dim=head_dim,
            intermediate_size=hf.intermediate_size,
            vocab_size=hf.vocab_size,
            max_position_embeddings=getattr(hf, "max_position_embeddings", 32768),
            rope_theta=rope_theta,
            rms_norm_eps=getattr(hf, "rms_norm_eps", 1e-6),
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
            attention_bias=getattr(hf, "attention_bias", False),
            eos_token_id=eos,
        )

    # ---- KV cache size math (a classic interview question) -----------------
    #
    # Per token:  2 (K and V) * num_layers * num_kv_heads * head_dim * dtype_bytes
    # For Qwen2.5-0.5B in fp16:
    #   2 * 24 * 2 * 64 * 2 = 12,288 B = 12 KB/token
    # One 16-token block: 192 KB.
    #
    # With MHA (num_kv_heads == num_heads) the KV cache is much larger: that is
    # exactly why GQA/MQA exist -- they shrink the KV cache by num_heads/num_kv_heads.

    def kv_bytes_per_token(self, dtype: torch.dtype) -> int:
        dtype_bytes = torch.tensor([], dtype=dtype).element_size()
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * dtype_bytes

    def kv_bytes_per_block(self, block_size: int, dtype: torch.dtype) -> int:
        return self.kv_bytes_per_token(dtype) * block_size

    def __str__(self) -> str:
        gqa = f"GQA {self.num_heads}Q/{self.num_kv_heads}KV" if self.num_kv_heads != self.num_heads \
            else "MHA"
        return (f"{self.model_path}: {self.num_layers}L hidden={self.hidden_size} "
                f"heads={self.num_heads} kv_heads={self.num_kv_heads} head_dim={self.head_dim} "
                f"vocab={self.vocab_size} ({gqa})")


def estimate_num_blocks(model_config: ModelConfig, engine_config: EngineConfig,
                        dtype: torch.dtype, device: str) -> int:
    """Pick the number of KV blocks the pool should pre-allocate.

    On CUDA: budget = total VRAM * utilization - weights - slack, then divide by
    bytes-per-block. On CPU: size for max_num_seqs * max_model_len tokens.
    """
    if engine_config.num_blocks is not None:
        return engine_config.num_blocks

    bytes_per_block = model_config.kv_bytes_per_block(engine_config.block_size, dtype)

    if device == "cuda":
        # Weights are expected to be loaded already, so `free` excludes them.
        free_b, _ = torch.cuda.mem_get_info()
        slack = 512 * 1024 * 1024  # activations, fragmentation, CUDA context
        kv_budget = int(free_b * engine_config.gpu_memory_utilization) - slack
        num_blocks = max(64, kv_budget // bytes_per_block)
        # never allocate for more than we can serve
        cap = (engine_config.max_num_seqs * engine_config.max_model_len
               // engine_config.block_size) + 64
        return min(num_blocks, cap)
    else:
        tokens = engine_config.max_num_seqs * engine_config.max_model_len
        return max(64, tokens // engine_config.block_size + 64)


def dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def estimate_params(c: ModelConfig) -> int:
    h, layers, kvh, hd, inter, vocab = (
        c.hidden_size, c.num_layers, c.num_kv_heads, c.head_dim,
        c.intermediate_size, c.vocab_size)
    qo = c.num_heads * hd
    per_layer = (h * (qo + 2 * kvh * hd)          # attention projections (qkv)
                 + qo * h                          # o_proj
                 + 3 * h * inter)                  # mlp
    return layers * per_layer + vocab * h + vocab * qo  # + embed + lm_head(rough)
