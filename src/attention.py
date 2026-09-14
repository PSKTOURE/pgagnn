import math

import torch
import torch.nn.functional as F
from torch import nn

from .attn_utils import mix_scalar, normalize_inputs, pairwise_geometric_product
from .utils import INNER_PRODUCT_INDICES, _build_dist_basis, _build_dist_vec


def get_attention_module(attention_type: str, out_mvc: int, out_sc: int | None = None) -> nn.Module:
    """Factory function to get attention module based on type."""
    attn_dict = {
        "gatr": GATrAttention,
        "gatr_sparse": GATrAttentionSparse,
        "similarity": GASimilarityAttention,
        "geometric": GAGeometricAttention,
        "dual_norm": GADualNormAttention,
    }
    if attention_type not in attn_dict:
        raise ValueError(f"Unknown attention type: {attention_type}")
    return attn_dict[attention_type](out_mvc, out_sc)


class GATrAttention(nn.Module):
    """Geometric Algebra Transformer Attention mechanism.

    Computes attention scores by combining three components:
    1. Multivector inner product attention
    2. Distance-based attention (from trivector components)
    3. Scalar dot product attention (optional)

    Each component is weighted and combined using learnable mixing parameters.

    Parameters
    ----------
    out_mvc : int
        Number of output multivector channels.
    out_sc : int or None
        Number of output scalar channels. If None, scalar attention is disabled.
    """

    def __init__(self, out_mvc: int, out_sc: int | None = None):
        super().__init__()
        self.out_mvc = out_mvc
        self.out_sc = out_sc

        # Learnable mixing weights for blending attention components
        num_components = 4 if out_sc is not None else 3
        self.attn_mix = nn.Parameter(torch.zeros(num_components))

        # Fixed normalization scales (not learnable)
        # Multivector: 8 elements used in inner product per channel
        self.scale_mv = 1.0 / math.sqrt(out_mvc * 8)
        # Distance: 5 elements in distance vector per channel
        self.scale_dist = 1.0 / math.sqrt(out_mvc * 5)
        # Scalars: standard scaling
        self.scale_sc = 1.0 / math.sqrt(out_sc) if out_sc else 1.0
        # Edges: 4 basis components for the inner product of grade-1 vectors
        self.scale_edge = 1.0 / math.sqrt(out_mvc * 4)

    def forward(
        self,
        Q_mv: torch.Tensor,
        K_mv: torch.Tensor,
        Q_sc: torch.Tensor | None = None,
        K_sc: torch.Tensor | None = None,
        Q_edge_agg: torch.Tensor | None = None,
        K_edge_agg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-component attention scores.

        Parameters
        ----------
        Q_mv : torch.Tensor
            Query multivectors. Shape: (B, N, C, 16) or (B, N, H, C, 16)
        K_mv : torch.Tensor
            Key multivectors. Shape: (B, N, C, 16) or (B, N, 1, C, 16)
        Q_sc : torch.Tensor or None
            Query scalars. Shape: (B, N, S) or (B, N, H, S)
        K_sc : torch.Tensor or None
            Key scalars. Shape: (B, N, S) or (B, N, 1, S)
        Q_edge_agg : torch.Tensor or None
            Aggregated outgoing edge multivectors. Shape: (B, N, E, 16) or (N, E, 16)
        K_edge_agg : torch.Tensor or None
            Aggregated incoming edge multivectors. Shape: (B, N, E, 16) or (N, E, 16)

        Returns
        -------
        torch.Tensor
            Unnormalized attention scores. Shape: (B, N, N, H)
            Apply softmax(scores, dim=-2) to get attention weights.
        """
        # Get concatenated Q/K features using shared feature preparation
        Q_all, K_all, is_multihead = self.compute_qk_features(Q_mv, K_mv, Q_sc, K_sc, Q_edge_agg, K_edge_agg)

        if is_multihead:
            # Q_all, K_all: (B, H, N, D)
            # Compute: (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
            scores = torch.matmul(Q_all, K_all.transpose(-2, -1))
            # Rearrange: (B, H, N, N) -> (B, N, N, H)
            scores = scores.permute(0, 2, 3, 1)
        else:
            # Q_all, K_all: (B, N, D)
            # Compute: (B, N, D) @ (B, D, N) -> (B, N, N)
            scores = torch.matmul(Q_all, K_all.transpose(-2, -1))

        return scores

    def compute_qk_features(
        self,
        Q_mv: torch.Tensor,
        K_mv: torch.Tensor,
        Q_sc: torch.Tensor | None = None,
        K_sc: torch.Tensor | None = None,
        Q_edge_agg: torch.Tensor | None = None,
        K_edge_agg: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Return concatenated Q/K features for edge-wise attention scoring.

        Returns:
          Q_all: (B, H, N, D) or (B, N, D) if not multihead
          K_all: (B, H, N, D) or (B, N, D) if not multihead
          is_multihead: bool
        """
        weights = self._get_mixing_weights()
        Q_mv, K_mv, Q_sc, K_sc, is_multihead = normalize_inputs(Q_mv, K_mv, Q_sc, K_sc)

        query_features = []
        key_features = []

        Q_mv_feat, K_mv_feat = self._prepare_multivector_features(Q_mv, K_mv, weights["mv"])
        query_features.append(Q_mv_feat)
        key_features.append(K_mv_feat)

        Q_dist_feat, K_dist_feat = self._prepare_distance_features(Q_mv, K_mv, weights["dist"])
        query_features.append(Q_dist_feat)
        key_features.append(K_dist_feat)

        if self.out_sc is not None and Q_sc is not None:
            Q_sc_feat, K_sc_feat = self._prepare_scalar_features(Q_sc, K_sc, weights["sc"])
            query_features.append(Q_sc_feat)
            key_features.append(K_sc_feat)

        if Q_edge_agg is not None and K_edge_agg is not None:
            if Q_edge_agg.dim() == 4:
                Q_edge_agg = Q_edge_agg.unsqueeze(2)
                K_edge_agg = K_edge_agg.unsqueeze(2)
            elif Q_edge_agg.dim() == 3:
                Q_edge_agg = Q_edge_agg.unsqueeze(1)
                K_edge_agg = K_edge_agg.unsqueeze(1)

            Q_edge_feat, K_edge_feat = self._prepare_edge_features(Q_edge_agg, K_edge_agg, weights["edge"])
            query_features.append(Q_edge_feat)
            key_features.append(K_edge_feat)

        Q_all = torch.cat(query_features, dim=-1)
        K_all = torch.cat(key_features, dim=-1)

        if not is_multihead:
            Q_all = Q_all.squeeze(1)
            K_all = K_all.squeeze(1)

        return Q_all, K_all, is_multihead

    def _get_mixing_weights(self) -> dict:
        """Compute softmax-normalized mixing weights for attention components."""
        weights_normalized = F.softmax(self.attn_mix, dim=0)

        result = {
            "mv": weights_normalized[0],
            "dist": weights_normalized[1],
        }

        if self.out_sc is not None:
            result["sc"] = weights_normalized[2]
            result["edge"] = weights_normalized[3]
        else:
            result["edge"] = weights_normalized[2]

        return result

    def _prepare_multivector_features(self, Q_mv, K_mv, weight):
        """Extract and scale multivector inner product features."""
        B, N, H, _C, _ = Q_mv.shape

        selector = INNER_PRODUCT_INDICES.to(Q_mv.device)
        Q_selected = Q_mv[..., selector]  # (B, N, H, C, 8)
        K_selected = K_mv[..., selector]  # (B, N, H_K, C, 8)

        Q_flat = Q_selected.reshape(B, N, H, -1)
        K_flat = K_selected.reshape(B, N, K_selected.shape[2], -1)

        Q_features = Q_flat.transpose(1, 2)
        K_features = K_flat.transpose(1, 2)

        Q_features = Q_features * (weight * self.scale_mv)
        return Q_features, K_features

    def _prepare_distance_features(self, Q_mv, K_mv, weight):
        """Extract and scale distance-based features from trivector components."""
        B, N, H, _C, _ = Q_mv.shape

        basis_q, basis_k = _build_dist_basis(Q_mv.device, Q_mv.dtype)
        Q_trivectors = Q_mv[..., 11:15]
        K_trivectors = K_mv[..., 11:15]

        Q_dist = _build_dist_vec(Q_trivectors, basis_q)
        K_dist = _build_dist_vec(K_trivectors, basis_k)

        Q_flat = Q_dist.reshape(B, N, H, -1)
        K_flat = K_dist.reshape(B, N, K_dist.shape[2], -1)

        Q_features = Q_flat.transpose(1, 2)
        K_features = K_flat.transpose(1, 2)

        Q_features = Q_features * (weight * self.scale_dist)
        return Q_features, K_features

    def _prepare_scalar_features(self, Q_sc, K_sc, weight):
        """Scale scalar features for attention."""
        Q_features = Q_sc.transpose(1, 2)
        K_features = K_sc.transpose(1, 2)

        Q_features = Q_features * (weight * self.scale_sc)
        return Q_features, K_features

    def _prepare_edge_features(self, Q_edge_agg, K_edge_agg, weight):
        """Extract and scale edge inner product features.

        Supports both dense and sparse shapes after head insertion:
        - Dense: (B, N, H, E, 16)
        - Sparse: (N, H, E, 16)
        """
        if Q_edge_agg.dim() == 5:
            B, N, H, _E, _ = Q_edge_agg.shape
            Q_edge_selected = Q_edge_agg[..., 1:5]
            K_edge_selected = K_edge_agg[..., 1:5]

            Q_flat = Q_edge_selected.reshape(B, N, H, -1)
            K_flat = K_edge_selected.reshape(B, N, K_edge_selected.shape[2], -1)

            Q_features = Q_flat.transpose(1, 2)
            K_features = K_flat.transpose(1, 2)
        else:
            N, H, _E, _ = Q_edge_agg.shape
            Q_edge_selected = Q_edge_agg[..., 1:5]
            K_edge_selected = K_edge_agg[..., 1:5]

            Q_features = Q_edge_selected.reshape(N, H, -1)
            K_features = K_edge_selected.reshape(N, K_edge_selected.shape[1], -1)

        Q_features = Q_features * (weight * self.scale_edge)
        return Q_features, K_features


class GATrAttentionSparse(nn.Module):
    """Sparse GATr attention for edge-list inputs.

    Works on packed node features and computes attention scores only
    for the provided edges, avoiding dense (N, N) matrices.
    """

    def __init__(self, out_mvc: int, out_sc: int | None = None):
        super().__init__()
        self.out_mvc = out_mvc
        self.out_sc = out_sc

        num_components = 4 if out_sc is not None else 3
        self.attn_mix = nn.Parameter(torch.zeros(num_components))

        self.scale_mv = 1.0 / math.sqrt(out_mvc * 8)
        self.scale_dist = 1.0 / math.sqrt(out_mvc * 5)
        self.scale_sc = 1.0 / math.sqrt(out_sc) if out_sc else 1.0
        self.scale_edge = 1.0 / math.sqrt(out_mvc * 4)

    def forward(
        self,
        Q_mv: torch.Tensor,
        K_mv: torch.Tensor,
        edge_index: torch.Tensor,
        Q_sc: torch.Tensor | None = None,
        K_sc: torch.Tensor | None = None,
        Q_edge_agg: torch.Tensor | None = None,
        K_edge_agg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention scores for a sparse edge list.

        Parameters
        ----------
        Q_mv, K_mv : torch.Tensor
            Multivectors shaped (N, C, 16) or (N, H, C, 16).
        edge_index : torch.Tensor
            Edge list shaped (2, E) with [src, dst] indices.
        Q_sc, K_sc : torch.Tensor, optional
            Scalars shaped (N, S) or (N, H, S).
        Q_edge_agg, K_edge_agg : torch.Tensor, optional
            Aggregated edge multivectors shaped (N, E, 16).

        Returns
        -------
        torch.Tensor
            Unnormalized attention scores shaped (E,) or (E, H).
        """
        Q_all, K_all, is_multihead = self.compute_qk_features(Q_mv, K_mv, Q_sc, K_sc, Q_edge_agg, K_edge_agg)

        src, dst = edge_index
        Q_edge = Q_all[dst]
        K_edge = K_all[src]

        scores = (Q_edge * K_edge).sum(dim=-1)
        if not is_multihead:
            scores = scores.squeeze(-1)
        return scores

    def compute_qk_features(
        self,
        Q_mv: torch.Tensor,
        K_mv: torch.Tensor,
        Q_sc: torch.Tensor | None = None,
        K_sc: torch.Tensor | None = None,
        Q_edge_agg: torch.Tensor | None = None,
        K_edge_agg: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Return concatenated Q/K features for edge-wise scoring.

        Returns:
          Q_all: (N, H, D)
          K_all: (N, H, D)
          is_multihead: bool
        """
        is_multihead = Q_mv.dim() == 4
        if not is_multihead:
            Q_mv = Q_mv.unsqueeze(1)
            K_mv = K_mv.unsqueeze(1)
            if Q_sc is not None:
                Q_sc = Q_sc.unsqueeze(1)
            if K_sc is not None:
                K_sc = K_sc.unsqueeze(1)
            if Q_edge_agg is not None:
                Q_edge_agg = Q_edge_agg.unsqueeze(1)
            if K_edge_agg is not None:
                K_edge_agg = K_edge_agg.unsqueeze(1)

        weights = self._get_mixing_weights()
        query_features = []
        key_features = []

        Q_mv_feat, K_mv_feat = self._prepare_multivector_features(Q_mv, K_mv, weights["mv"])
        query_features.append(Q_mv_feat)
        key_features.append(K_mv_feat)

        Q_dist_feat, K_dist_feat = self._prepare_distance_features(Q_mv, K_mv, weights["dist"])
        query_features.append(Q_dist_feat)
        key_features.append(K_dist_feat)

        if self.out_sc is not None and Q_sc is not None:
            Q_sc_feat, K_sc_feat = self._prepare_scalar_features(Q_sc, K_sc, weights["sc"])
            query_features.append(Q_sc_feat)
            key_features.append(K_sc_feat)

        # Handle edge aggregations if provided
        if Q_edge_agg is not None and K_edge_agg is not None:
            Q_edge_feat, K_edge_feat = self._prepare_edge_features(Q_edge_agg, K_edge_agg, weights["edge"])
            query_features.append(Q_edge_feat)
            key_features.append(K_edge_feat)

        Q_all = torch.cat(query_features, dim=-1)
        K_all = torch.cat(key_features, dim=-1)

        return Q_all, K_all, is_multihead

    def _get_mixing_weights(self) -> dict:
        weights_normalized = F.softmax(self.attn_mix, dim=0)
        result = {
            "mv": weights_normalized[0],
            "dist": weights_normalized[1],
        }
        if self.out_sc is not None:
            result["sc"] = weights_normalized[2]
            result["edge"] = weights_normalized[3]
        else:
            result["edge"] = weights_normalized[2]
        return result

    def _prepare_multivector_features(self, Q_mv, K_mv, weight):
        # Q_mv: (N, H, C, 16)
        selector = INNER_PRODUCT_INDICES.to(Q_mv.device)
        Q_selected = Q_mv[..., selector]
        K_selected = K_mv[..., selector]

        Q_flat = Q_selected.reshape(Q_mv.shape[0], Q_mv.shape[1], -1)
        K_flat = K_selected.reshape(K_mv.shape[0], K_mv.shape[1], -1)

        Q_features = Q_flat * (weight * self.scale_mv)
        K_features = K_flat
        return Q_features, K_features

    def _prepare_distance_features(self, Q_mv, K_mv, weight):
        # Q_mv: (N, H, C, 16)
        basis_q, basis_k = _build_dist_basis(Q_mv.device, Q_mv.dtype)
        Q_trivectors = Q_mv[..., 11:15]
        K_trivectors = K_mv[..., 11:15]

        Q_dist = _build_dist_vec(Q_trivectors, basis_q)
        K_dist = _build_dist_vec(K_trivectors, basis_k)

        Q_flat = Q_dist.reshape(Q_mv.shape[0], Q_mv.shape[1], -1)
        K_flat = K_dist.reshape(K_mv.shape[0], K_mv.shape[1], -1)

        Q_features = Q_flat * (weight * self.scale_dist)
        K_features = K_flat
        return Q_features, K_features

    def _prepare_scalar_features(self, Q_sc, K_sc, weight):
        Q_features = Q_sc * (weight * self.scale_sc)
        K_features = K_sc
        return Q_features, K_features

    def _prepare_edge_features(self, Q_edge_agg, K_edge_agg, weight):
        """Extract and scale edge inner product features for sparse attention.

        For sparse inputs, edges have shape (N, H, E, 16).
        """
        # Extract the 4 basis components for grade-1 vectors (indices 1-4)
        Q_edge_selected = Q_edge_agg[..., 1:5]  # (N, H, E, 4)
        K_edge_selected = K_edge_agg[..., 1:5]  # (N, H, E, 4)

        N, H, _E, _ = Q_edge_selected.shape

        # Flatten channels and elements: (N, H, E, 4) -> (N, H, E*4)
        Q_flat = Q_edge_selected.reshape(N, H, -1)
        K_flat = K_edge_selected.reshape(N, H, -1)

        # Apply scaling and mixing weight to queries
        Q_features = Q_flat * (weight * self.scale_edge)
        K_features = K_flat

        return Q_features, K_features


class GASimilarityAttention(nn.Module):
    """Shared similarity-based attention for GA-based models.
    Computes attention scores using grade-specific similarity measures.

    Implements the similarity metric from Section 3 of the mathematical description:
    Sim(X,Y) = w_0*s_0 + w_1*s_1 + w_2*s_2 + w_3*s_3 + w_4*s_4
    where each s_k is a grade-specific similarity term.

    Optimizations:
    - Direct grade extraction from Q_mv/K_mv using PGA basis indices instead of
      grade_project(), eliminating 2 large (B,N,C,5,16) tensor allocations.
    - Grade 3: Uses Lagrange Identity (||A ∨ B||² = ||A||²||B||² - (A·B)²) to avoid
      expensive join operation. This provides ~50-70% speedup on Grade 3 computation.
    - Grades 1 & 2: Pre-computes norms O(N) before pairwise operations O(N²).

    Note: Returns unnormalized scores. Apply softmax to get attention weights.
    Note: Assumes standard PGA basis ordering in Q_mv/K_mv.

    Parameters:
    out_mvc : int
        Number of output multivector channels.
    out_sc : int or None
        Number of output scalar channels.
    """

    def __init__(
        self,
        out_mvc: int,
        out_sc: int | None = None,
    ):
        super().__init__()
        self.out_mvc = out_mvc
        self.out_sc = out_sc

        # Weights for each grade similarity (w_0, w_1, w_2, w_3, w_4)
        # Plus weights for combining MV and scalar attention if scalars present
        if out_sc is not None:
            # 5 grades + 2 for MV/scalar mix
            self.weights = nn.Parameter(torch.zeros(7))
        else:
            self.weights = nn.Parameter(torch.zeros(5))  # 5 grades only

        # Alpha parameters for exponential similarities (log-space for stability)
        # Use log-space: alpha = softplus(log_alpha) to keep positive and bounded
        self.log_alpha_0 = nn.Parameter(torch.tensor(0.0))  # softplus(0) ≈ 0.69
        self.log_alpha_3 = nn.Parameter(torch.tensor(0.0))
        self.log_alpha_4 = nn.Parameter(torch.tensor(0.0))

        self.eps = 1e-6

    def forward(
        self,
        Q_mv: torch.Tensor,
        K_mv: torch.Tensor,
        Q_sc: torch.Tensor | None = None,
        K_sc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute similarity-based attention scores with clear, modular steps.

        Returns unnormalized logits shaped (B, N, N) or (B, N, N, H).
        """
        Q_mv, K_mv, Q_sc, K_sc, is_multihead = normalize_inputs(Q_mv, K_mv, Q_sc, K_sc)
        _B, _N, H, _C, _ = Q_mv.shape
        K_H = K_mv.shape[2]

        alpha_0 = torch.clamp(F.softplus(self.log_alpha_0), min=0.01, max=10.0)
        alpha_3 = torch.clamp(F.softplus(self.log_alpha_3), min=0.01, max=10.0)
        alpha_4 = torch.clamp(F.softplus(self.log_alpha_4), min=0.01, max=10.0)

        s0, s1, s2, s3, s4 = self._compute_grade_similarities(Q_mv, K_mv, H, K_H, alpha_0, alpha_3, alpha_4)

        similarities = torch.stack([s0, s1, s2, s3, s4], dim=-1).sum(dim=4)  # (B, N, H, N, 5)
        grade_weights = F.softmax(self.weights[:5] if self.out_sc is not None else self.weights, dim=0)
        scores_mv = torch.einsum("bnhik,k->bnhi", similarities, grade_weights)
        scores_mv = scores_mv / math.sqrt(self.out_mvc * 16)

        mix_logits = self.weights[5:] if (self.out_sc is not None) else None
        scores = mix_scalar(scores_mv, Q_sc, K_sc, self.out_sc, mix_logits)
        scores = scores.transpose(2, 3)
        if not is_multihead:
            scores = scores.squeeze(-1)
        return scores


    def _compute_grade_similarities(
        self,
        Q_mv: torch.Tensor,
        K_mv: torch.Tensor,
        H: int,
        K_H: int,
        alpha_0: torch.Tensor,
        alpha_3: torch.Tensor,
        alpha_4: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _, C, _ = Q_mv.shape

        # Grade 0
        Q0, K0 = Q_mv[..., 0], K_mv[..., 0]
        if K_H == 1 and H > 1:
            K0 = K0.expand(B, N, H, C)
        s0 = torch.exp(
            -torch.clamp(
                alpha_0 * (Q0.unsqueeze(3) - K0.unsqueeze(1).transpose(2, 3)).abs(),
                max=15.0,
            )
        )

        # Grade 1
        Q1, K1 = Q_mv[..., 1:5], K_mv[..., 1:5]
        Q1_norm = torch.clamp(torch.norm(Q1, dim=-1), min=self.eps)
        K1_norm = torch.clamp(torch.norm(K1, dim=-1), min=self.eps)
        if K_H == 1 and H > 1:
            K1 = K1.expand(B, N, H, C, 4)
            K1_norm = K1_norm.expand(B, N, H, C)
        s1_num = torch.einsum("bnhcd,bmhcd->bnhmc", Q1, K1)
        s1_den = Q1_norm.unsqueeze(3) * K1_norm.unsqueeze(1).transpose(2, 3) + self.eps
        s1 = torch.clamp(s1_num / s1_den, min=-1.0, max=1.0)

        # Grade 2
        Q2, K2 = Q_mv[..., 5:11], K_mv[..., 5:11]
        Q2_norm = torch.clamp(torch.norm(Q2, dim=-1), min=self.eps)
        K2_norm = torch.clamp(torch.norm(K2, dim=-1), min=self.eps)
        if K_H == 1 and H > 1:
            K2 = K2.expand(B, N, H, C, 6)
            K2_norm = K2_norm.expand(B, N, H, C)
        s2_num = torch.einsum("bnhcd,bmhcd->bnhmc", Q2, K2)
        s2_den = Q2_norm.unsqueeze(3) * K2_norm.unsqueeze(1).transpose(2, 3) + self.eps
        s2 = torch.clamp(s2_num / s2_den, min=-1.0, max=1.0)

        # Grade 3 (Lagrange identity)
        Q3, K3 = Q_mv[..., 11:15], K_mv[..., 11:15]
        Q3_ns = (Q3**2).sum(dim=-1)
        K3_ns = (K3**2).sum(dim=-1)
        if K_H == 1 and H > 1:
            K3 = K3.expand(B, N, H, C, 4)
            K3_ns = K3_ns.expand(B, N, H, C)
        dot3 = torch.einsum("bnhcd,bmhcd->bnhmc", Q3, K3)
        join_ns = torch.clamp(Q3_ns.unsqueeze(3) * K3_ns.unsqueeze(1).transpose(2, 3) - dot3**2, min=0.0)
        s3 = torch.exp(-torch.clamp(alpha_3 * join_ns, max=15.0))

        # Grade 4
        Q4, K4 = Q_mv[..., 15], K_mv[..., 15]
        if K_H == 1 and H > 1:
            K4 = K4.expand(B, N, H, C)
        s4 = torch.exp(
            -torch.clamp(
                alpha_4 * (Q4.unsqueeze(3) - K4.unsqueeze(1).transpose(2, 3)).abs(),
                max=15.0,
            )
        )

        return s0, s1, s2, s3, s4


class GAGeometricAttention(nn.Module):
    """Geometric attention using feature vectors from grade norms.

    This implements the approach from Section 4 of the mathematical description:
    Instead of projecting the geometric product to a scalar immediately, we extract
    the magnitude of its constituent grades to form an invariant feature vector.

    For each query-key pair (X_i, X_j):
    1. Compute geometric product: M_ij = X_i * X̃_j
    2. Extract grade components and compute their norms
    3. Form feature vector: f_ij = [⟨M_ij⟩_0, ||⟨M_ij⟩_2||, ||⟨M_ij⟩_4||, ...]
    4. Pass through MLP Φ to get scores α_ij = Φ(f_ij)

    This preserves E(3) equivariance because:
    - Scalar part is invariant under transformations
    - Grade norms (coefficient L2 norm) are invariant under transformations

    Optimizations:
    - Direct feature extraction from M using PGA basis indices [0], [1-4], [5-10],
      [11-14], [15] instead of full grade_project expansion to (B,N,N,C,5,16).
      This avoids large memory allocation.

    Note: Returns unnormalized scores. Apply softmax to get attention weights.
    Note: Assumes standard PGA basis ordering in M from geometric_product.

    Parameters
    ----------
    out_mvc : int
        Number of output multivector channels.
    out_sc : int or None
        Number of output scalar channels.
    num_grades : int
        Number of grade features to extract (default: 5 for all grades 0-4).
        Hidden dim is automatically set to 2 * (num_grades * out_mvc).
    """

    def __init__(
        self,
        out_mvc: int,
        out_sc: int | None = None,
        num_grades: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.out_mvc = out_mvc
        self.out_sc = out_sc
        self.num_grades = num_grades

        # MLP to process geometric invariant features
        # Input: num_grades features per channel

        self.feature_projection = nn.Sequential(
            nn.Linear(num_grades, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Normalize embeddings
            nn.GELU(),  # Smooth activation
        )

        self.scoring_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

        # Optional mixing with scalar attention
        if out_sc is not None:
            self.attn_mix = nn.Parameter(torch.zeros(2))

    def initialize_weights(self):
        # Initialize MLP weights for stable training
        modules = [self.feature_projection, self.scoring_mlp]
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        Q_mv: torch.Tensor,
        K_mv: torch.Tensor,
        Q_sc: torch.Tensor | None = None,
        K_sc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute geometric attention scores with modular helpers."""
        Q_mv, K_mv, Q_sc, K_sc, is_multihead = normalize_inputs(Q_mv, K_mv, Q_sc, K_sc)

        M = pairwise_geometric_product(Q_mv, K_mv)  # (B, N, H, N, C, 16)
        features = self._extract_features(M)  # (B, N, H, N, C, num_grades)

        embeddings = self.feature_projection(features)
        pooled = torch.max(embeddings, dim=4)[0]
        scores_mv = self.scoring_mlp(pooled).squeeze(-1)

        scores = mix_scalar(
            scores_mv,
            Q_sc,
            K_sc,
            self.out_sc,
            self.attn_mix if self.out_sc is not None else None,
        )
        scores = scores.transpose(2, 3)
        if not is_multihead:
            scores = scores.squeeze(-1)
        return scores

    def _extract_features(self, M: torch.Tensor) -> torch.Tensor:
        feats = [
            M[..., 0],
            torch.log1p(torch.sum(M[..., 1:5] ** 2, dim=-1)),
            torch.log1p(torch.sum(M[..., 5:11] ** 2, dim=-1)),
            torch.log1p(torch.sum(M[..., 11:15] ** 2, dim=-1)),
            M[..., 15],
        ]
        return torch.stack(feats, dim=-1)


class GADualNormAttention(nn.Module):
    """Dual-norm attention mechanism from Section 5 of the mathematical description.

    This implements the dual metric structure of PGA where a multivector M = M_E + e_0 M_I
    decomposes into Euclidean and Ideal parts. For each grade component, we compute:

    1. Standard Norm ||⟨M⟩_k|| = sqrt(|⟨M_E M̃_E⟩_0|) - measures Euclidean properties
       (angles, rotations, incidence)
    2. Infinity Norm ||⟨M⟩_k||_∞ = ||M_I||_Euclidean - measures distances and translations

    This dual-norm feature vector provides disentangled signals for:
    - Angles/Incidence (via standard norm)
    - Distances/Moments (via infinity norm)

    The feature vector for each query-key pair is:
    f_ij = [⟨M_ij⟩_0, ||⟨M_ij⟩_0||_∞, ||⟨M_ij⟩_1||, ||⟨M_ij⟩_1||_∞, ...]

    This formulation:
    - Is strictly E(3)-equivariant
    - Avoids metric collapse
    - Unifies treatment of points, lines, and planes
    - Separates angular and distance information

    Parameters
    ----------
    out_mvc : int
        Number of output multivector channels.
    out_sc : int or None
        Number of output scalar channels.
    num_grades : int
        Number of grade features to extract (default: 5 for all grades 0-4).
    hidden_dim : int
        Hidden dimension for the MLP that processes features.
    """

    def __init__(
        self,
        out_mvc: int,
        out_sc: int | None = None,
        num_grades: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.out_mvc = out_mvc
        self.out_sc = out_sc
        self.num_grades = num_grades
        self.eps = 1e-6

        # MLP to process dual-norm geometric invariant features
        # Input: 2 * num_grades features per channel (standard norm + infinity norm for each grade)
        input_dim = 2 * num_grades

        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Normalize embeddings
            nn.GELU(),  # Smooth activation
        )

        self.scoring_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

        # Optional mixing with scalar attention
        if out_sc is not None:
            self.attn_mix = nn.Parameter(torch.zeros(2))

    def initialize_weights(self):
        # Initialize MLP weights for stable training
        modules = [self.feature_projection, self.scoring_mlp]
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        Q_mv: torch.Tensor,
        K_mv: torch.Tensor,
        Q_sc: torch.Tensor | None = None,
        K_sc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dual-norm attention scores with modular helpers."""
        Q_mv, K_mv, Q_sc, K_sc, is_multihead = normalize_inputs(Q_mv, K_mv, Q_sc, K_sc)

        M = pairwise_geometric_product(Q_mv, K_mv)
        features = self._extract_dual_norm_features(M)
        embeddings = self.feature_projection(features)
        pooled = torch.max(embeddings, dim=4)[0]
        scores_mv = self.scoring_mlp(pooled).squeeze(-1)

        scores = mix_scalar(
            scores_mv,
            Q_sc,
            K_sc,
            self.out_sc,
            self.attn_mix if self.out_sc is not None else None,
        )
        scores = scores.transpose(2, 3)
        if not is_multihead:
            scores = scores.squeeze(-1)
        return scores

    def _extract_dual_norm_features(self, M: torch.Tensor) -> torch.Tensor:
        s_val = M[..., 0]
        vec_eucl, vec_ideal = M[..., 2:5], M[..., 1:2]
        biv_eucl, biv_ideal = M[..., 8:11], M[..., 5:8]
        tri_eucl, tri_ideal = M[..., 14:15], M[..., 11:14]
        ps_val = M[..., 15]

        feats = [
            s_val,
            torch.zeros_like(s_val),
            torch.log1p((vec_eucl**2).sum(dim=-1)),
            torch.log1p((vec_ideal**2).sum(dim=-1)),
            torch.log1p((biv_eucl**2).sum(dim=-1)),
            torch.log1p((biv_ideal**2).sum(dim=-1)),
            torch.log1p((tri_eucl**2).sum(dim=-1)),
            torch.log1p((tri_ideal**2).sum(dim=-1)),
            torch.zeros_like(ps_val),
            ps_val,
        ]
        return torch.stack(feats, dim=-1)

