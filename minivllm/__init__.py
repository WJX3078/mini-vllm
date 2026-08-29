"""mini-vLLM: a minimal vLLM-style LLM inference engine for learning purposes.

Implements from scratch, in pure PyTorch:
  * Block-level KV cache management (PagedAttention's memory layer)
  * Continuous batching (iteration-level scheduling)
  * Prefix caching (block-granularity KV reuse with hash chains)
  * Speculative decoding (n-gram / draft-model, draft-then-verify)
"""

from minivllm.config import EngineConfig, ModelConfig
from minivllm.engine import LLMEngine
from minivllm.sequence import SamplingParams

__all__ = ["EngineConfig", "ModelConfig", "LLMEngine", "SamplingParams"]
__version__ = "0.1.0"
