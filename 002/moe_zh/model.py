"""Minimal MoE causal language model for Simplified Chinese smoke training."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_zh.balance import load_balancing_loss
from moe_zh.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, nh, T, hd
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class ExpertFFN(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.expert_ffn_mult * cfg.n_embd
        self.fc1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class MoELayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_expert = cfg.n_expert
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(cfg) for _ in range(cfg.n_expert)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            out: (B, T, C)
            router_probs: (B, T, E)
            expert_indices: (B, T, K)
        """
        B, T, C = x.shape
        logits = self.router(x)  # B,T,E
        probs = F.softmax(logits, dim=-1)
        top_v, top_i = torch.topk(probs, k=self.top_k, dim=-1)  # B,T,K
        # normalize top-k weights
        top_w = top_v / top_v.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(x)
        # token-wise routing (clear, fine for tiny smoke model)
        flat_x = x.reshape(B * T, C)
        flat_out = torch.zeros_like(flat_x)
        flat_i = top_i.reshape(B * T, self.top_k)
        flat_w = top_w.reshape(B * T, self.top_k)

        for k in range(self.top_k):
            idx_k = flat_i[:, k]
            w_k = flat_w[:, k].unsqueeze(-1)
            for e in range(self.n_expert):
                mask = idx_k == e
                if not mask.any():
                    continue
                tokens = flat_x[mask]
                flat_out[mask] = flat_out[mask] + w_k[mask] * self.experts[e](tokens)

        out = flat_out.view(B, T, C)
        return out, probs, top_i


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.moe = MoELayer(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        moe_out, probs, indices = self.moe(self.ln2(x))
        x = x + moe_out
        return x, probs, indices


class MoELanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        all_probs: list[torch.Tensor] = []
        all_idx: list[torch.Tensor] = []
        for block in self.blocks:
            x, probs, eidx = block(x)
            all_probs.append(probs)
            all_idx.append(eidx)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        aux: dict[str, Any] = {}
        if targets is not None:
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            bal = torch.zeros((), device=idx.device)
            for probs, eidx in zip(all_probs, all_idx):
                bal = bal + load_balancing_loss(probs, eidx, self.cfg.n_expert)
            bal = bal / max(len(all_probs), 1)
            aux["ce_loss"] = ce.detach()
            aux["balance_loss"] = bal.detach()
            # expert usage fraction over last layer
            last_idx = all_idx[-1][..., 0]  # top-1
            counts = torch.bincount(last_idx.reshape(-1), minlength=self.cfg.n_expert)
            aux["expert_counts"] = counts.detach()
            loss = ce  # balance coef applied in train loop
            aux["balance_raw"] = bal

        return logits, loss, aux

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.cfg)
