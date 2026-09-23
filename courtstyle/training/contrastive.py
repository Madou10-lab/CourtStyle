"""Contrastive objectives for the serve and rally branches.

Both branches minimise InfoNCE over two augmented views of one record, with the
remaining records of the batch as negatives. They differ in instantiation.

Serve (MoCo v3, Chen, Xie and He, ICCV 2021)
    An online encoder, projector and predictor produce the query. A momentum
    encoder and projector produce the key. Similarities are scored as
    ``S = Q K^T / tau`` and the loss is applied in both directions. Negatives are
    drawn from the batch and no memory queue is maintained.

Rally (SimCLR NT-Xent, Chen et al., ICML 2020, plus order repulsion)
    One encoder and projection head process both views. ``order_repulsion_loss``
    embeds a shuffled copy of the same rally and penalises cosine similarity to
    the original above a fixed margin, which prevents the encoder from settling
    on order-invariant shot statistics.

    L_rally = L_NT-Xent + lambda * L_order
"""
from __future__ import annotations

import torch
from torch import nn


class MoCoV3Loss(nn.Module):
    """
    MoCo v3 contrastive loss with symmetric formulation.

    Based on: https://arxiv.org/abs/2104.02057

    The loss computes:
        loss = ctr(q1, k2) + ctr(q2, k1)

    Where ctr(q, k) is:
        logits = q @ k.T / temperature  # [N, N] pairs
        labels = range(N)  # positives are on diagonal
        loss = CrossEntropy(logits, labels)
        return 2 * temperature * loss  # scaling factor from paper

    This uses in-batch negatives (no queue).
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.ce = nn.CrossEntropyLoss()

    def _contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for one direction."""
        # Normalize embeddings
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        # Compute similarity matrix [N, N]
        logits = torch.mm(q, k.t()) / self.temperature

        # Labels: positives are on diagonal
        batch_size = q.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=q.device)

        # Cross-entropy loss with scaling factor
        loss = self.ce(logits, labels)
        return 2 * self.temperature * loss

    def forward(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetric MoCo v3 loss.

        Args:
            q1: query embeddings from view 1 (B, D)
            q2: query embeddings from view 2 (B, D)
            k1: key embeddings from view 1 (B, D)
            k2: key embeddings from view 2 (B, D)

        Returns:
            Symmetric contrastive loss: ctr(q1, k2) + ctr(q2, k1)
        """
        # Symmetric loss: each query matches the OTHER view's key
        loss = self._contrastive_loss(q1, k2) + self._contrastive_loss(q2, k1)
        return loss



def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    if z1.shape[0] <= 1:
        return z1.sum() * 0.0 + z2.sum() * 0.0
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / float(temperature)
    diag_mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag_mask, -1e9)
    target = torch.cat(
        [torch.arange(B, 2 * B, device=z.device), torch.arange(0, B, device=z.device)]
    )
    return nn.functional.cross_entropy(sim, target)


def order_repulsion_loss(
    e_orig: torch.Tensor,
    e_shuffled: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    cos_sim = nn.functional.cosine_similarity(e_orig, e_shuffled, dim=1)
    return torch.relu(cos_sim - margin).mean()

