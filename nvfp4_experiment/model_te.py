"""
NanoGPT architecture with Transformer Engine Linear layers.

Drop-in replacement for model.py. Replaces nn.Linear in:
  - CausalSelfAttention (c_attn, c_proj)
  - MLP                 (c_fc,   c_proj)

lm_head stays as nn.Linear to preserve weight-tying with the token embedding.
All other components (LayerNorm, Dropout, Embedding) are unchanged.

Within a te.fp8_autocast(enabled=True, fp8_recipe=NVFP4BlockScaling()) context
the te.Linear layers quantise activations + weights to NVFP4 (E2M1 block-scaled)
and use Blackwell's 5th-gen Tensor Cores for FP4 GEMM.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import transformer_engine.pytorch as te


# ── LayerNorm (unchanged from nanoGPT) ───────────────────────────────────────

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (nn.LayerNorm doesn't support bias=False)."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


# ── TE-backed Attention ───────────────────────────────────────────────────────

class CausalSelfAttentionTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn      = te.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj      = te.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop   = nn.Dropout(config.dropout)
        self.resid_drop  = nn.Dropout(config.dropout)
        self.n_head      = config.n_head
        self.n_embd      = config.n_embd
        self.dropout     = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


# ── TE-backed MLP ─────────────────────────────────────────────────────────────

class MLPTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = te.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = te.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ── Transformer Block ─────────────────────────────────────────────────────────

class BlockTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionTE(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLPTE(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ── Config & Model ────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    block_size: int   = 1024
    vocab_size: int   = 50304
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    dropout:    float = 0.0
    bias:       bool  = True


class GPTTE(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([BlockTE(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # Keep lm_head as nn.Linear to preserve weight-tying
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 paper)
        for name, module in self.named_modules():
            if name.endswith('c_proj') and hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.normal_(module.weight,
                                      mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"GPTTE parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(T, dtype=torch.long, device=idx.device)

        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss   = None

        return logits, loss

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay_params    = [p for n, p in params.items() if p.dim() >= 2]
        nodecay_params  = [p for n, p in params.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        print(f"Decay params:    {sum(p.numel() for p in decay_params):,}")
        print(f"No-decay params: {sum(p.numel() for p in nodecay_params):,}")
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=True)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_c  = idx[:, -self.config.block_size:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx
