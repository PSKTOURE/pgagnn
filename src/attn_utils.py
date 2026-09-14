import math

import torch
import torch.nn.functional as F

from .primitives import geometric_product, reverse


def normalize_inputs(
    Q_mv: torch.Tensor,
    K_mv: torch.Tensor,
    Q_sc: torch.Tensor | None = None,
    K_sc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
    """Ensure inputs have an explicit head dimension.
    Returns (Q_mv5, K_mv5, Q_sc5, K_sc5, is_multihead).
    """
    is_multihead = Q_mv.dim() == 5
    if not is_multihead:
        Q_mv = Q_mv.unsqueeze(2)
        K_mv = K_mv.unsqueeze(2)
        Q_sc = Q_sc.unsqueeze(2) if Q_sc is not None else None
        K_sc = K_sc.unsqueeze(2) if K_sc is not None else None
    return Q_mv, K_mv, Q_sc, K_sc, is_multihead


def maybe_expand_k_heads(t: torch.Tensor, H: int) -> torch.Tensor:
    """Broadcast key-head dimension to match query heads when K_H==1 and H>1."""
    if t.shape[2] == 1 and H > 1:
        shape = list(t.shape)
        shape[2] = H
        return t.expand(*shape)
    return t


def pairwise_geometric_product(Q_mv: torch.Tensor, K_mv: torch.Tensor) -> torch.Tensor:
    """Compute pairwise geometric product M_ij = Q_i * reverse(K_j) across heads.
    Expects Q_mv, K_mv shaped (B, N, H, C, 16) with the same H after broadcast.
    Returns M with shape (B, N, H, N, C, 16).
    """
    _B, _N, H, _C, _ = Q_mv.shape
    K_mv = maybe_expand_k_heads(K_mv, H)
    Qe = Q_mv.unsqueeze(3)  # (B, N, H, 1, C, 16)
    Ke = K_mv.unsqueeze(1).transpose(2, 3)  # (B, 1, H, N, C, 16)
    return geometric_product(Qe, reverse(Ke))


def mix_scalar(
    scores_mv: torch.Tensor,
    Q_sc: torch.Tensor | None,
    K_sc: torch.Tensor | None,
    out_sc: int | None,
    mix_logits: torch.Tensor | None,
) -> torch.Tensor:
    """Optionally mix multivector scores with scalar attention using given logits.
    scores_mv: (B, N, H, N)
    Q_sc: (B, N, H, out_sc) or None
    K_sc: (B, N, K_H, out_sc) or None (K_H may be 1 for broadcast)
    mix_logits: tensor of shape (2,) whose softmax gives mix weights
    """
    if Q_sc is None or K_sc is None or out_sc is None or mix_logits is None:
        return scores_mv
    _B, _N, H, _ = scores_mv.shape
    K_sc = maybe_expand_k_heads(K_sc, H)
    scores_sc = torch.einsum("bnhc,bmhc->bnhm", Q_sc, K_sc) / math.sqrt(out_sc)
    mix = F.softmax(mix_logits, dim=0)
    return mix[0] * scores_mv + mix[1] * scores_sc
