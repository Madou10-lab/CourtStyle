"""Tabular serve encoder with Manifold-Constrained Hyper-Connections (mHC).

One serve record is a fixed-length vector of speed, placement, variability and
context descriptors. The encoder tokenises the record by column, so attention
operates over serve variables and not over time:

    per-column MLP  ->  + learnable column identity  ->  prepend [CLS]
                    ->  depth x mHC transformer block
                    ->  final [CLS] state as the record embedding

The residual path is parameterised by manifold-constrained hyper-connections,
which maintain ``expansion_n`` parallel residual streams:

    x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l, W_l)

``H_res`` is made doubly stochastic by Sinkhorn-Knopp normalisation.

``encoder_block="prenorm"`` selects a standard pre-norm transformer with plain
multi-head attention and a single residual stream. That variant is the baseline
reported in the serve ablation.

Reference for mHC: Xie et al., Manifold-Constrained Hyper-Connections.
Reference for the MoCo v3 heads: Chen, Xie and He, ICCV 2021.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SinkhornKnopp(nn.Module):
    """
    Differentiable Sinkhorn-Knopp algorithm to produce doubly stochastic matrices.

    Given logits, computes M^(0) = exp(logits), then alternates row and column
    normalization for t_max iterations to produce a doubly stochastic matrix.
    """

    def __init__(self, t_max: int = 20, eps: float = 1e-8) -> None:
        super().__init__()
        self.t_max = t_max
        self.eps = eps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (*, n, n) matrix of logits

        Returns:
            (*, n, n) doubly stochastic matrix
        """
        # Numerical stability: subtract max to prevent exp overflow
        # This doesn't change the result since Sinkhorn normalizes anyway
        logits_stable = logits - logits.amax(dim=-1, keepdim=True)
        M = torch.exp(logits_stable)

        # Alternating row and column normalization
        for _ in range(self.t_max):
            # Row normalization
            M = M / (M.sum(dim=-1, keepdim=True) + self.eps)
            # Column normalization
            M = M / (M.sum(dim=-2, keepdim=True) + self.eps)

        return M


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: w3(silu(w1(x)) * w2(x))."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.act(self.w1(x)) * self.w2(x)
        out = self.w3(gated)
        return self.dropout(out)


class GatedMultiHeadAttention(nn.Module):
    """Multi-head attention with per-head sigmoid gating on the attention output."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # Gate produces one gate per head per token
        self.gate = nn.Linear(d_model, n_heads, bias=True)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
            mask: optional (batch, seq) attention mask (True for keep, False for mask)
        Returns:
            (batch, seq, d_model)
        """
        B, S, _ = x.shape

        def _shape(t: torch.Tensor) -> torch.Tensor:
            t = t.view(B, S, self.n_heads, self.head_dim)
            return t.transpose(1, 2)  # (B, heads, seq, head_dim)

        q = _shape(self.q_proj(x))
        k = _shape(self.k_proj(x))
        v = _shape(self.v_proj(x))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, seq, seq)
        if mask is not None:
            mask_exp = mask[:, None, None, :].to(dtype=attn_scores.dtype)
            attn_scores = attn_scores.masked_fill(mask_exp == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        attn_out = torch.matmul(attn_weights, v)  # (B, heads, seq, head_dim)

        # Gating per head
        gate_logits = self.gate(x)  # (B, seq, heads)
        gate = torch.sigmoid(gate_logits).transpose(1, 2).unsqueeze(-1)  # (B, heads, seq, 1)
        attn_out = attn_out * gate

        # Recombine heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.out_proj(attn_out)
        return self.proj_drop(out)


class MHCConnector(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Connector.

    Implements the mHC formulation:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l, W_l)

    The residual stream has shape (Batch, n, C) where n is the expansion factor.
    This module computes H_pre, H_post, and H_res from the current state.

    Args:
        d_model: Original width C
        expansion_n: Expansion factor n (default 4)
    """

    def __init__(self, d_model: int, expansion_n: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = expansion_n
        self.flat_dim = expansion_n * d_model

        # RMSNorm for the flattened state
        self.norm = RMSNorm(self.flat_dim)

        # Scalar gating factors, initialized small
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

        # Linear projections for computing hyper-connection coefficients
        # phi_pre: (n*C) -> n
        self.phi_pre = nn.Linear(self.flat_dim, expansion_n, bias=False)
        # phi_post: (n*C) -> n
        self.phi_post = nn.Linear(self.flat_dim, expansion_n, bias=False)
        # phi_res: (n*C) -> n*n
        self.phi_res = nn.Linear(self.flat_dim, expansion_n * expansion_n, bias=False)

        # Biases
        self.b_pre = nn.Parameter(torch.zeros(expansion_n))
        self.b_post = nn.Parameter(torch.zeros(expansion_n))
        self.b_res = nn.Parameter(torch.zeros(expansion_n, expansion_n))

        # Sinkhorn-Knopp for doubly stochastic constraint
        self.sinkhorn = SinkhornKnopp(t_max=20)

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Initialize projections with small weights
        nn.init.normal_(self.phi_pre.weight, std=0.01)
        nn.init.normal_(self.phi_post.weight, std=0.01)
        nn.init.normal_(self.phi_res.weight, std=0.01)
        # Initialize b_res close to identity
        with torch.no_grad():
            self.b_res.copy_(torch.eye(self.n) * 0.5)

    def forward(
        self, x_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute hyper-connection coefficients and aggregated input for sublayer.

        Args:
            x_state: Current expanded state (Batch, n, C)

        Returns:
            x_in: Aggregated input for sublayer (Batch, C)
            H_res: Residual mixing matrix (Batch, n, n)
            H_post: Post-sublayer expansion weights (Batch, n)
        """
        B = x_state.shape[0]

        # Flatten state: (B, n, C) -> (B, n*C)
        x_flat = x_state.view(B, self.flat_dim)

        # Normalize
        x_norm = self.norm(x_flat)

        # Compute raw coefficients
        # H_pre_tilde: (B, n)
        H_pre_tilde = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre
        # H_post_tilde: (B, n)
        H_post_tilde = self.alpha_post * self.phi_post(x_norm) + self.b_post
        # H_res_tilde: (B, n*n) -> (B, n, n)
        H_res_tilde = self.alpha_res * self.phi_res(x_norm).view(B, self.n, self.n) + self.b_res

        # Apply constraints
        # H_pre: sigmoid, shape (B, n) - will broadcast over C
        H_pre = torch.sigmoid(H_pre_tilde)
        # H_post: 2 * sigmoid for range [0, 2], shape (B, n)
        H_post = 2.0 * torch.sigmoid(H_post_tilde)
        # H_res: doubly stochastic via Sinkhorn, shape (B, n, n)
        H_res = self.sinkhorn(H_res_tilde)

        # Compute aggregated input: x_in = sum over n of (H_pre * x_state)
        # H_pre: (B, n) -> (B, n, 1) for broadcasting with (B, n, C)
        x_in = (H_pre.unsqueeze(-1) * x_state).sum(dim=1)  # (B, C)

        return x_in, H_res, H_post


class MHCTransformerBlock(nn.Module):
    """
    Transformer block with Manifold-Constrained Hyper-Connections.

    Replaces standard residual connections (x + F(x)) with:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

    Args:
        d_model: Model dimension C
        n_heads: Number of attention heads
        expansion_n: mHC expansion factor n (default 4)
        ff_multiplier: FFN hidden dimension multiplier
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_n: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n = expansion_n
        self.d_model = d_model

        # mHC connectors for attention and FFN sublayers
        self.mhc_attn = MHCConnector(d_model, expansion_n)
        self.mhc_ffn = MHCConnector(d_model, expansion_n)

        # Sublayers (preserved from original)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = GatedMultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.drop_attn = nn.Dropout(dropout)

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, ff_multiplier * d_model, dropout=dropout)
        self.drop_ffn = nn.Dropout(dropout)

    def forward(
        self, x_state: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass with mHC residual updates.

        Args:
            x_state: Expanded state (Batch, Seq, n, C)
            mask: Optional attention mask

        Returns:
            Updated expanded state (Batch, Seq, n, C)
        """
        B, S, n, C = x_state.shape

        # Reshape for processing: (B*S, n, C)
        x_flat = x_state.view(B * S, n, C)

        # === Attention sublayer with mHC ===
        # Get aggregated input and hyper-connection coefficients
        x_attn_in, H_res_attn, H_post_attn = self.mhc_attn(x_flat)

        # Reshape for attention: (B, S, C)
        x_attn_in = x_attn_in.view(B, S, C)

        # Apply attention sublayer: LayerNorm -> Attention -> Dropout
        F_attn = self.drop_attn(self.attn(self.ln_attn(x_attn_in), mask=mask))

        # Reshape F_attn for broadcasting: (B*S, C)
        F_attn_flat = F_attn.view(B * S, C)

        # mHC update: x_mid = H_res @ x_l + H_post^T @ F_attn
        # H_res @ x_l: (B*S, n, n) @ (B*S, n, C) -> (B*S, n, C)
        x_res = torch.bmm(H_res_attn, x_flat)
        # H_post^T @ F_attn: broadcast F_attn to n streams
        # H_post: (B*S, n) -> (B*S, n, 1), F_attn: (B*S, 1, C)
        x_post = H_post_attn.unsqueeze(-1) * F_attn_flat.unsqueeze(1)

        x_mid = x_res + x_post  # (B*S, n, C)

        # === FFN sublayer with mHC ===
        x_ffn_in, H_res_ffn, H_post_ffn = self.mhc_ffn(x_mid)

        # Reshape for FFN: (B, S, C)
        x_ffn_in = x_ffn_in.view(B, S, C)

        # Apply FFN sublayer: LayerNorm -> FFN -> Dropout
        F_ffn = self.drop_ffn(self.ffn(self.ln_ffn(x_ffn_in)))

        # Reshape F_ffn for broadcasting: (B*S, C)
        F_ffn_flat = F_ffn.view(B * S, C)

        # mHC update: x_out = H_res @ x_mid + H_post^T @ F_ffn
        x_res_ffn = torch.bmm(H_res_ffn, x_mid)
        x_post_ffn = H_post_ffn.unsqueeze(-1) * F_ffn_flat.unsqueeze(1)

        x_out = x_res_ffn + x_post_ffn  # (B*S, n, C)

        # Reshape back: (B, S, n, C)
        return x_out.view(B, S, n, C)


class PlainMultiHeadAttention(nn.Module):
    """Standard multi-head self-attention wrapper with batch-first inputs."""

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
        key_padding_mask = None
        if mask is not None:
            keep_mask = mask.to(dtype=torch.bool)
            key_padding_mask = ~keep_mask
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.out_drop(out)


class PreNormTransformerBlock(nn.Module):
    """Standard pre-norm Transformer block for controlled ablations."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(ff_multiplier * d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = PlainMultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.drop_path1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.drop_path2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.ln1(x), mask=mask))
        x = x + self.drop_path2(self.ffn(self.ln2(x)))
        return x


class ModernTabularEncoder(nn.Module):
    """
    Tabular encoder with Manifold-Constrained Hyper-Connections (mHC).

    Features:
      - Non-linear projections for continuous columns
      - Embeddings for categoricals
      - Additive learnable column identities
      - Expanded residual stream (B, S, n, C) through mHC blocks
      - CLS token pooled after collapsing streams
    """

    def __init__(
        self,
        continuous_cols: Sequence[str],
        categorical_cardinalities: Dict[str, int],
        d_model: int = 128,
        n_heads: int = 8,
        depth: int = 4,
        expansion_n: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        encoder_block: str = "mhc",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if encoder_block not in {"mhc", "prenorm"}:
            raise ValueError(
                f"Unsupported encoder_block: {encoder_block!r}. Choose from {{'mhc', 'prenorm'}}."
            )

        self.continuous_cols = list(continuous_cols)
        self.categorical_cols = list(categorical_cardinalities.keys())
        self.d_model = d_model
        self.n = expansion_n
        self.encoder_block = str(encoder_block)
        # Continuous features are already standardised by courtstyle.data.serve_dataset
        self.input_norm = nn.Identity()

        # Continuous projections
        self.cont_projs = nn.ModuleDict({
            col: nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            for col in self.continuous_cols
        })

        # Categorical embeddings
        # +1 for "unknown" token used in SCARF-style augmentation
        # The augmenter sets masked categories to cardinality index (unknown)
        self.cat_embs = nn.ModuleDict({
            col: nn.Embedding(cardinality + 1, d_model)  # +1 for unknown token
            for col, cardinality in categorical_cardinalities.items()
        })

        num_tokens = len(self.continuous_cols) + len(self.categorical_cols)
        self.column_embeddings = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        if self.encoder_block == "mhc":
            self.blocks = nn.ModuleList([
                MHCTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    expansion_n=expansion_n,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                PreNormTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(depth)
            ])
        self.final_norm = nn.LayerNorm(d_model)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.column_embeddings, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        for emb in self.cat_embs.values():
            nn.init.normal_(emb.weight, std=0.02)

    def _expand_to_residual_stream(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand input from (B, S, C) to (B, S, n, C) by padding with zeros.
        Initial state: (x^T, 0, ..., 0)^T
        """
        B, S, C = x.shape
        # Create expanded state with zeros
        x_expanded = torch.zeros(B, S, self.n, C, device=x.device, dtype=x.dtype)
        # Put original embeddings in the first stream
        x_expanded[:, :, 0, :] = x
        return x_expanded

    def _collapse_residual_stream(self, x_state: torch.Tensor) -> torch.Tensor:
        """
        Collapse expanded state from (B, S, n, C) to (B, S, C) via mean over streams.
        """
        return x_state.mean(dim=2)

    def forward(
        self,
        x_cont: Dict[str, torch.Tensor],
        x_cat: Dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_cont: dict of continuous column tensors, each (batch,) or (batch,1)
            x_cat: dict of categorical column tensors, each (batch,)
            mask: optional attention mask (batch, seq)
        Returns:
            CLS token embeddings, shape (batch, d_model)
        """
        batch_size = next(iter(x_cont.values())).shape[0] if x_cont else next(iter(x_cat.values())).shape[0]

        tokens: List[torch.Tensor] = []
        # Continuous tokens with shared batch norm
        if self.continuous_cols:
            cont_list = []
            for col in self.continuous_cols:
                t = x_cont[col]
                if t.dim() == 1:
                    t = t.unsqueeze(-1)
                cont_list.append(t)
            cont_mat = torch.cat(cont_list, dim=1)  # (batch, num_cont)
            cont_mat = self.input_norm(cont_mat)
            for col, t_norm in zip(self.continuous_cols, cont_mat.split(1, dim=1)):
                tokens.append(self.cont_projs[col](t_norm))
        # Categorical tokens
        for col in self.categorical_cols:
            tokens.append(self.cat_embs[col](x_cat[col]))

        x = torch.stack(tokens, dim=1)  # (batch, num_tokens, d_model)
        x = x + self.column_embeddings

        cls = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls, x], dim=1)  # prepend CLS: (B, S, C)

        if self.encoder_block == "mhc":
            # Expand to residual stream: (B, S, C) -> (B, S, n, C)
            x_state = self._expand_to_residual_stream(x)

            # Pass through mHC blocks
            for block in self.blocks:
                x_state = block(x_state, mask=mask)

            # Collapse streams: (B, S, n, C) -> (B, S, C)
            x = self._collapse_residual_stream(x_state)
        else:
            for block in self.blocks:
                x = block(x, mask=mask)

        x = self.final_norm(x)
        cls_out = x[:, 0]
        return cls_out


class ContrastiveHead(nn.Module):
    """
    MoCo v3 projection head: fc -> LN -> ReLU -> fc -> LN

    This head is used by both query and key encoders.

    NOTE: Uses LayerNorm instead of BatchNorm to avoid running buffer mismatch
    between query and key encoders in MoCo-style training. BatchNorm buffers
    (running_mean, running_var) are not parameters and don't get EMA updated.
    """

    def __init__(self, d_model: int, proj_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHead(nn.Module):
    """
    MoCo v3 prediction head: fc -> LN -> ReLU -> fc

    This head is ONLY used by the query encoder, not the key encoder.

    NOTE: Uses LayerNorm for consistency with ContrastiveHead and to avoid
    any BatchNorm buffer issues in contrastive learning setups.
    """

    def __init__(self, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

