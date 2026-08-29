"""Qwen2-family model with paged KV cache, config-driven (any Qwen2/2.5 size).

The forward pass is written for continuous batching:
  * all scheduled sequences' new tokens are concatenated into one flat
    [total_tokens, hidden] stream (a "varlen" batch),
  * projections/RoPE/norms run once over the whole stream,
  * attention is computed per sequence through its block table,
  * only the positions whose logits are needed go through lm_head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from minivllm.attention import (
    SeqInput,
    _sdpa,
    apply_rope,
    gather_kv,
    paged_attention,
    rope_inv_freq,
    write_kv_to_pool,
)
from minivllm.config import ModelConfig


class RMSNorm(nn.Module):
    """Bit-for-bit the same math as transformers' Qwen2RMSNorm."""

    def __init__(self, hidden_size: int, eps: float, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(input_dtype)


class Qwen2Layer(nn.Module):
    def __init__(self, cfg: ModelConfig, dtype, device):
        super().__init__()
        h, hd = cfg.hidden_size, cfg.head_dim
        q_out = cfg.num_heads * hd
        kv_out = cfg.num_kv_heads * hd
        # fused qkv projection (weights concatenated at load time)
        self.qkv_proj = nn.Parameter(torch.empty(q_out + 2 * kv_out, h, dtype=dtype, device=device))
        self.qkv_bias = nn.Parameter(torch.zeros(q_out + 2 * kv_out, dtype=dtype, device=device)) \
            if cfg.attention_bias else None
        self.o_proj = nn.Parameter(torch.empty(h, q_out, dtype=dtype, device=device))
        self.gate_proj = nn.Parameter(torch.empty(cfg.intermediate_size, h, dtype=dtype, device=device))
        self.up_proj = nn.Parameter(torch.empty(cfg.intermediate_size, h, dtype=dtype, device=device))
        self.down_proj = nn.Parameter(torch.empty(h, cfg.intermediate_size, dtype=dtype, device=device))
        self.input_layernorm = RMSNorm(h, cfg.rms_norm_eps, dtype, device)
        self.post_attention_layernorm = RMSNorm(h, cfg.rms_norm_eps, dtype, device)

        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = hd


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Parameter(
            torch.empty(cfg.vocab_size, cfg.hidden_size, dtype=dtype, device=device))
        self.layers = nn.ModuleList(Qwen2Layer(cfg, dtype, device) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.lm_head: nn.Parameter | None = None
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden_size,
                                                    dtype=dtype, device=device))
        self.register_buffer(
            "inv_freq", rope_inv_freq(cfg.head_dim, cfg.rope_theta, device), persistent=False)
        # init (overwritten by load_from_hf; kept sane for random-weight tests)
        for p in self.parameters():
            std = 0.02
            nn.init.normal_(p, mean=0.0, std=std)

    # ---------------------------------------------------------------- load
    @torch.no_grad()
    def load_from_hf(self, hf_model: nn.Module):
        """Copy weights from a transformers Qwen2ForCausalLM.

        Attention-bias presence is detected from the state dict itself --
        some configs (e.g. transformers 5.x) omit the `attention_bias` flag
        even when the checkpoint carries biases.
        """
        sd = hf_model.state_dict()
        self.embed_tokens.copy_(sd["model.embed_tokens.weight"])
        for i, layer in enumerate(self.layers):
            pre = f"model.layers.{i}."
            q = sd[pre + "self_attn.q_proj.weight"]
            k = sd[pre + "self_attn.k_proj.weight"]
            v = sd[pre + "self_attn.v_proj.weight"]
            layer.qkv_proj.copy_(torch.cat([q, k, v], dim=0))
            has_bias = pre + "self_attn.q_proj.bias" in sd
            if has_bias:
                if layer.qkv_bias is None:
                    layer.qkv_bias = nn.Parameter(torch.zeros_like(layer.qkv_proj[:, 0]))
                layer.qkv_bias.copy_(torch.cat([
                    sd[pre + "self_attn.q_proj.bias"],
                    sd[pre + "self_attn.k_proj.bias"],
                    sd[pre + "self_attn.v_proj.bias"]], dim=0))
            layer.o_proj.copy_(sd[pre + "self_attn.o_proj.weight"])
            layer.gate_proj.copy_(sd[pre + "mlp.gate_proj.weight"])
            layer.up_proj.copy_(sd[pre + "mlp.up_proj.weight"])
            layer.down_proj.copy_(sd[pre + "mlp.down_proj.weight"])
            layer.input_layernorm.weight.copy_(sd[pre + "input_layernorm.weight"])
            layer.post_attention_layernorm.weight.copy_(sd[pre + "post_attention_layernorm.weight"])
        self.norm.weight.copy_(sd["model.norm.weight"])
        if self.lm_head is not None:
            self.lm_head.copy_(sd["lm_head.weight"])

    @classmethod
    def from_pretrained(cls, model_path: str, device: str, dtype: torch.dtype):
        from transformers import AutoModelForCausalLM
        cfg = ModelConfig.from_pretrained(model_path)
        hf = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        model = cls(cfg, device, dtype)
        model.load_from_hf(hf)
        del hf
        return model, cfg

    # ------------------------------------------------------------- forward
    def _decode_batch_context(self, pool, seq_inputs):
        """Shared prep for the all-decode fast path: one padded block-table
        tensor for the whole batch, plus the physical block / slot index of
        each sequence's single new token."""
        device = seq_inputs[0].block_table.device
        S = len(seq_inputs)
        tables = [si.block_table for si in seq_inputs]
        max_nb = max(t.numel() for t in tables)
        table_t = torch.empty(S, max_nb, dtype=torch.long, device=device)
        for i, t in enumerate(tables):
            table_t[i, :t.numel()] = t
            if t.numel() < max_nb:               # pad with a valid block id
                table_t[i, t.numel():] = t[-1]
        return table_t

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                pool, seq_inputs: list[SeqInput],
                logits_indices: torch.Tensor) -> torch.Tensor:
        """input_ids/positions: [T] flat over all sequences' new tokens.
        logits_indices: [S] flat token indices whose logits we need.
        Returns [S, vocab]."""
        T = input_ids.shape[0]
        x = F.embedding(input_ids, self.embed_tokens)
        bs = pool.block_size
        H, kvh, D = self.layers[0].num_heads, self.layers[0].num_kv_heads, self.layers[0].head_dim

        # all-decode fast path: 1 new token per sequence -> the whole batch's
        # KV write + gather + attention collapses into single batched ops
        all_decode = len(seq_inputs) == T and all(si.q_len == 1 for si in seq_inputs)
        table_t = phys = slots = max_ctx = None
        if all_decode:
            table_t = self._decode_batch_context(pool, seq_inputs)
            blk_idx = (positions // bs).clamp(max=table_t.shape[1] - 1)[:, None]
            phys = table_t.gather(1, blk_idx)[:, 0]              # [S]
            slots = positions % bs                               # [S]
            max_ctx = max(si.ctx_len for si in seq_inputs)

        for li, layer in enumerate(self.layers):
            # ---- attention block
            residual = x
            h = layer.input_layernorm(x)
            qkv = h @ layer.qkv_proj.T
            if layer.qkv_bias is not None:
                qkv = qkv + layer.qkv_bias
            q_out = layer.num_heads * layer.head_dim
            kv_out = layer.num_kv_heads * layer.head_dim
            q = qkv[:, :q_out].view(T, layer.num_heads, layer.head_dim)
            k = qkv[:, q_out:q_out + kv_out].view(T, layer.num_kv_heads, layer.head_dim)
            v = qkv[:, q_out + kv_out:].view(T, layer.num_kv_heads, layer.head_dim)
            q, k = apply_rope(q, k, positions, self.inv_freq)

            k_view, v_view = pool.layer_kv(li)
            if all_decode:
                # one indexed write for every sequence's new K/V
                k_view[phys, :, slots, :] = k
                v_view[phys, :, slots, :] = v
                # one gather + one SDPA for the whole batch
                kk = k_view[table_t]                             # [S, nb, kvh, bs, D]
                vv = v_view[table_t]
                kk = kk.permute(0, 2, 1, 3, 4).reshape(T, kvh, -1, D)[:, :, :max_ctx]
                vv = vv.permute(0, 2, 1, 3, 4).reshape(T, kvh, -1, D)[:, :, :max_ctx]
                # each query attends to tokens 0..positions[i] (its own context)
                mask = (torch.arange(max_ctx, device=x.device)[None, None, None, :]
                        <= positions[:, None, None, None])       # [S,1,1,ctx]
                out = _sdpa(q[:, :, None],                       # [S, H, 1, D]
                            kk, vv, mask, D ** -0.5, H, kvh)
                attn_out = out[:, :, 0, :].reshape(T, q_out)
            else:
                attn_out = torch.empty(T, q_out, dtype=x.dtype, device=x.device)
                for si in seq_inputs:
                    lo = si.t0                        # flat-batch index
                    hi = lo + si.q_len
                    # 1) page-in the new tokens' K/V
                    write_kv_to_pool(k_view, v_view, k[lo:hi], v[lo:hi],
                                     positions[lo:hi], si.block_table, bs)
                    # 2) gather the whole context and attend
                    k_ctx, v_ctx = gather_kv(k_view, v_view, si.block_table,
                                             si.ctx_len, bs)
                    out = paged_attention(q[lo:hi], k_ctx, v_ctx,
                                          si.q_start, layer.num_heads)
                    attn_out[lo:hi] = out.reshape(hi - lo, q_out)
            x = residual + attn_out @ layer.o_proj.T

            # ---- MLP block
            residual = x
            h = layer.post_attention_layernorm(x)
            gate = F.silu(h @ layer.gate_proj.T)
            up = h @ layer.up_proj.T
            x = residual + (gate * up) @ layer.down_proj.T

        x = self.norm(x)
        w = self.lm_head if self.lm_head is not None else self.embed_tokens
        return x[logits_indices] @ w.T

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
