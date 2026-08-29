"""mini-vLLM: a minimal vLLM-style LLM inference engine for learning purposes.

Implements from scratch, in pure PyTorch:
  * Block-level KV cache management (PagedAttention's memory layer)
  * Continuous batching + chunked prefill (unified token scheduler)
  * Prefix caching (block-granularity KV reuse, pluggable hash backends)
  * Speculative decoding (n-gram / draft-model, lossless draft-then-verify)
  * Per-request RNG and batched sampling (<=1 GPU sync per sampling group)
"""

from minivllm.config import EngineConfig, ModelConfig
from minivllm.engine import LLMEngine
from minivllm.sequence import SamplingParams

__all__ = ["EngineConfig", "ModelConfig", "LLMEngine", "SamplingParams"]
__version__ = "0.2.0"
