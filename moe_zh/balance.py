"""Switch-Transformer style load balancing loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def load_balancing_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    n_expert: int,
) -> torch.Tensor:
    """Aux loss encouraging uniform expert usage.

    Args:
        router_probs: (B, T, E) softmax over experts
        expert_indices: (B, T, K) selected expert ids
        n_expert: number of experts E
    """
    # fraction of tokens routed to each expert (hard assignment, top-1 or multi)
    # one-hot over experts for each selected slot, average over batch*time*k
    flat = expert_indices.reshape(-1)
    me = F.one_hot(flat, num_classes=n_expert).float().mean(dim=0)  # (E,)
    # mean router probability per expert
    ce = router_probs.reshape(-1, n_expert).mean(dim=0)  # (E,)
    # scale as in Switch Transformer
    return n_expert * torch.sum(me * ce)
