"""A readable GPT-style decoder-only Transformer implemented with PyTorch."""
import math
import torch
from torch import nn
from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.context_length, cfg.context_length), persistent=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
            y = torch.softmax(scores, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DeonTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        b, t = input_ids.shape
        if t > self.cfg.context_length:
            raise ValueError(f"Sequence length {t} exceeds context length {self.cfg.context_length}")
        pos = torch.arange(t, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.cfg.context_length:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids
