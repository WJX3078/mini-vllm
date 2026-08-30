"""Triton kernels (optional GPU fast paths)."""
from minivllm.kernels.paged_attention import (
    paged_attention_decode_torch,
    paged_attention_decode_triton,
    triton_available,
)

__all__ = ["paged_attention_decode_torch", "paged_attention_decode_triton",
           "triton_available"]
