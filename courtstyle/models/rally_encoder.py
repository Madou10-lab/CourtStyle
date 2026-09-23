"""Sequence encoder for rally shot-transition records.

One rally is an ordered sequence of shot transitions. Each transition token
concatenates the current shot, the following shot and their difference, which
places shot-to-shot change directly in the input. The encoder therefore attends
over shots:

    linear projection  ->  + positional embedding  ->  prepend [CLS]
                       ->  n_layers x pre-norm transformer block
                       ->  [CLS] pooling as the rally embedding

``encoder_block="vanilla"`` is the configuration reported in the paper and uses
``torch.nn.TransformerEncoderLayer`` with ``norm_first=True``. ``"modern"``
selects RMSNorm, SwiGLU and gated attention instead and is retained for
completeness.

Positional information is carried by a learned embedding table of length
``max_len + 1``, not by a fixed sinusoidal encoding.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: W_o(silu(W_1 x) * (W_2 x))."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        return self.dropout(x)


class GatedMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional per-head sigmoid gating.

    Mask semantics:
        mask shape: (B, T), dtype bool
        True means padded position (must be masked out as attention key).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_head_gating: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.use_head_gating = bool(use_head_gating)

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.gate = nn.Linear(self.d_model, self.n_heads) if self.use_head_gating else None

        self.attn_drop = nn.Dropout(float(dropout))
        self.proj_drop = nn.Dropout(float(dropout))
        self.scale = self.head_dim**-0.5

    def _shape(self, t: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = t.shape
        t = t.view(bsz, seq_len, self.n_heads, self.head_dim)
        return t.transpose(1, 2)  # (B, H, T, Dh)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, T, T)

        if mask is not None:
            if mask.shape != (bsz, seq_len):
                raise ValueError(
                    f"mask must have shape {(bsz, seq_len)}, got {tuple(mask.shape)}"
                )
            key_pad = mask[:, None, None, :]  # True at padded keys
            scores = scores.masked_fill(key_pad, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, T, Dh)

        if self.gate is not None:
            gate = torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)  # (B,H,T,1)
            out = out * gate

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_proj(out)
        return self.proj_drop(out)

def build_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Return bool padding mask with True at padded positions."""
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


class _PlainMultiHeadAttention(nn.Module):
    """Standard self-attention wrapper with the same mask semantics."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.out_drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, key_padding_mask=mask, need_weights=False)
        return self.out_drop(out)


class _ModernTransformerBlock(nn.Module):
    """Pre-norm modern block for sequence modeling."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        ffn_mult: float = 4.0,
        use_gated_mha: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        if use_gated_mha:
            self.attn = GatedMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_head_gating=True,
            )
        else:
            self.attn = _PlainMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )
        self.resid_drop1 = nn.Dropout(float(dropout))

        hidden_dim = int(round(float(ffn_mult) * d_model))
        hidden_dim = max(hidden_dim, d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
        self.resid_drop2 = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.resid_drop1(self.attn(self.norm1(x), mask=pad_mask))
        x = x + self.resid_drop2(self.ffn(self.norm2(x)))
        return x


class RallyEncoder(nn.Module):
    """
    Transformer encoder for variable-length rally transition-token sequences.

    Pooling modes:
    - cls: prepend [CLS] token and return CLS embedding
    - mean: masked mean over non-padded tokens
    - attn: learned attention pooling over non-padded tokens
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "cls",
        max_len: int = 256,
        encoder_block: str = "vanilla",
        modern_use_gated_mha: bool = True,
        modern_ffn_mult: float = 4.0,
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean", "attn"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        if encoder_block not in {"vanilla", "modern"}:
            raise ValueError(
                f"Unsupported encoder_block: {encoder_block}. "
                "Choose from {'vanilla', 'modern'}."
            )

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.pooling = str(pooling)
        self.max_len = int(max_len)
        self.encoder_block = str(encoder_block)

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout),
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_len + 1, self.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.encoder_block == "vanilla":
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_model * 4,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)
            self.post_norm = nn.LayerNorm(self.d_model)
        else:
            self.encoder = nn.ModuleList(
                [
                    _ModernTransformerBlock(
                        d_model=self.d_model,
                        n_heads=self.n_heads,
                        dropout=self.dropout,
                        ffn_mult=modern_ffn_mult,
                        use_gated_mha=modern_use_gated_mha,
                    )
                    for _ in range(self.n_layers)
                ]
            )
            self.post_norm = RMSNorm(self.d_model)

        self.attn_pool = nn.Linear(self.d_model, 1, bias=False)

    def _add_positional(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.max_len + 1:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len+1={self.max_len + 1}"
            )
        return x + self.pos_embedding[:, :seq_len, :]

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be 3-D, got shape={tuple(tokens.shape)}")
        if lengths.ndim != 1:
            raise ValueError(f"lengths must be 1-D, got shape={tuple(lengths.shape)}")

        bsz, max_t, _ = tokens.shape
        x = self.input_proj(tokens)
        pad_mask = build_padding_mask(lengths, max_t)

        if self.pooling == "cls":
            cls = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_mask = torch.zeros(bsz, 1, dtype=torch.bool, device=x.device)
            pad_mask = torch.cat([cls_mask, pad_mask], dim=1)

        x = self._add_positional(x)

        if self.encoder_block == "vanilla":
            h = self.encoder(x, src_key_padding_mask=pad_mask)
        else:
            h = x
            for block in self.encoder:
                h = block(h, pad_mask=pad_mask)
        h = self.post_norm(h)

        if self.pooling == "cls":
            return h[:, 0, :]

        if self.pooling == "mean":
            valid = (~pad_mask).float().unsqueeze(-1)
            return (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        logits = self.attn_pool(h).squeeze(-1)
        logits = logits.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return (h * weights.unsqueeze(-1)).sum(dim=1)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive SSL."""

    def __init__(self, d_model: int, proj_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=1, eps=1e-8)


class RallySSLModel(nn.Module):
    """Rally encoder + projection head."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "cls",
        max_len: int = 256,
        proj_dim: int = 128,
        encoder_block: str = "vanilla",
        modern_use_gated_mha: bool = True,
        modern_ffn_mult: float = 4.0,
    ) -> None:
        super().__init__()
        self.encoder = RallyEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            pooling=pooling,
            max_len=max_len,
            encoder_block=encoder_block,
            modern_use_gated_mha=modern_use_gated_mha,
            modern_ffn_mult=modern_ffn_mult,
        )
        self.proj = ProjectionHead(d_model=d_model, proj_dim=proj_dim, dropout=dropout)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(tokens, lengths)
        z = self.proj(emb)
        return emb, z

