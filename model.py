"""~20M 参数的中文 decoder-only Transformer。

Llama/Qwen 风格: pre-RMSNorm, RoPE 旋转位置编码, SwiGLU FFN, 无 bias,
embedding 与 lm_head 权重绑定。
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    """返回 (cos, sin): [seq_len, head_dim/2]，fp32 计算保证精度。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # [seq, hd/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [batch, seq, heads, head_dim]; cos/sin: [seq, head_dim/2]

    对 (x[i], x[i+head_dim/2]) 二元组做旋转变换。
    """
    seq = x.size(1)
    x1, x2 = x.float().chunk(2, dim=-1)                    # 各 [b, s, h, hd/2]
    cos = cos[:seq].view(1, seq, 1, -1)
    sin = sin[:seq].view(1, seq, 1, -1)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, rope: tuple) -> torch.Tensor:
        bsz, seq, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq, self.num_heads, self.head_dim)
        k = k.view(bsz, seq, self.num_heads, self.head_dim)
        v = v.view(bsz, seq, self.num_heads, self.head_dim)
        q = apply_rope(q, *rope)
        k = apply_rope(k, *rope)
        # [bsz, heads, seq, head_dim]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(bsz, seq, dim)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, rope: tuple) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniChineseLLM(nn.Module):
    """decoder-only 小模型, 拼接式语言建模 (GPT 风格, labels 左移一位)。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.register_buffer(
            "rope_cos", torch.zeros(0), persistent=False)
        self.register_buffer("rope_sin", torch.zeros(0), persistent=False)
        self._ensure_rope(cfg.max_seq_len)
        self.apply(self._init_weights)
        # 残差投影采用缩放初始化 (GPT-2 做法)
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.num_layers))

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _ensure_rope(self, seq_len: int):
        need = max(seq_len, self.cfg.max_seq_len)
        if self.rope_cos.size(0) < need:
            cos, sin = build_rope_cache(
                need, self.cfg.hidden_size // self.cfg.num_heads,
                self.cfg.rope_base, self.tok_emb.weight.device,
                dtype=torch.float32,
            )
            self.rope_cos, self.rope_sin = cos, sin

    def forward(
        self,
        idx: torch.Tensor,                       # [batch, seq] 输入 token id
        targets: Optional[torch.Tensor] = None,  # [batch, seq] 目标, -100 忽略
    ) -> tuple:
        seq = idx.size(1)
        if seq > self.cfg.max_seq_len:
            raise ValueError(
                f"输入长度 {seq} 超过 max_seq_len={self.cfg.max_seq_len}")
        self._ensure_rope(seq)
        rope = (self.rope_cos, self.rope_sin)
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, rope)
        x = self.norm(x)
        if self.lm_head is None:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """自回归采样。eos_id 命中时停止 (每个序列独立返回已生成部分)。"""
        self.eval()
        device = idx.device
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                cut = cum - probs > top_p          # 保留第一个超过 top_p 的词
                sorted_logits = sorted_logits.masked_fill(cut, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)   # [batch, 1]
            next_tok = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_tok, eos_id if eos_id is not None else 0),
                next_tok,
            )
            idx = torch.cat([idx, next_tok], dim=1)
            if eos_id is not None:
                finished |= next_tok.squeeze(1) == eos_id
                if finished.all():
                    break
        return idx


def init_model(cfg: ModelConfig, device: str = "cpu") -> MiniChineseLLM:
    model = MiniChineseLLM(cfg)
    model = model.to(device)
    return model
