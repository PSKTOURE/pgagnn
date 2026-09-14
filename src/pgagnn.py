import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_softmax

from .attention import get_attention_module
from .layers import (
    EquiLayerNorm,
    EquiLinear,
    GatedNonLinearity,
    GradeDropout,
    ResidualLayer,
)
from .primitives import geometric_product, reverse
from .utils import INNER_PRODUCT_INDICES


class EquiMLP(nn.Module):
    """Equivariant MLP strictly for multivectors."""

    def __init__(
        self,
        in_mvc: int,
        out_mvc: int,
        hidden_mvc: int | None = None,
        dropout_prob: float = 0.0,
        activation: str = "gelu",
        expansion_factor: float = 0.5,  # Bottleneck to save VRAM
        **kwargs,
    ):
        super().__init__()

        self.hidden_mvc = hidden_mvc if hidden_mvc is not None else max(int(in_mvc * expansion_factor), 1)

        self.equi_linear1 = EquiLinear(in_mvc, self.hidden_mvc, in_sc=None, out_sc=None, **kwargs)
        self.equi_linear2 = EquiLinear(self.hidden_mvc, out_mvc, in_sc=None, out_sc=None, **kwargs)

        self.act = GatedNonLinearity(activation=activation)
        self.dropout = GradeDropout(dropout_prob)
        self.norm = EquiLayerNorm(channel_dim=-2)

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        """
        mv: (B, N, in_mvc, 16) or (N, in_mvc, 16)
        Returns: (..., out_mvc, 16)
        """
        out_mv, _ = self.norm(mv, None)
        out_mv, _ = self.equi_linear1(out_mv, None)
        gates = out_mv[..., 0:1]
        out_mv, _ = self.act(out_mv, gates, None)
        out_mv, _ = self.dropout(out_mv, None)
        out_mv, _ = self.equi_linear2(out_mv, None)

        return out_mv


class ScalarMLP(nn.Module):
    """Standard MLP strictly for invariant scalar features."""

    def __init__(
        self,
        in_sc: int,
        out_sc: int,
        hidden_sc: int | None = None,
        dropout_prob: float = 0.0,
        activation: str = "gelu",
        expansion_factor: float = 2.0,
    ):
        super().__init__()

        self.hidden_sc = hidden_sc if hidden_sc is not None else max(int(in_sc * expansion_factor), 4)

        act_layer = nn.GELU() if activation == "gelu" else nn.SiLU()

        self.net = nn.Sequential(
            nn.LayerNorm(in_sc),
            nn.Linear(in_sc, self.hidden_sc),
            act_layer,
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_sc, out_sc),
        )

    def forward(self, sc: torch.Tensor) -> torch.Tensor:
        """
        sc: (B, N, in_sc) or (N, in_sc)
        Returns: (..., out_sc)
        """
        return self.net(sc)


class GA_MessageLayer(nn.Module):
    """Geometric Algebra Message Layer with multi-head attention.

    Geometric messages follow the same computation as before. Attention and
    aggregation are split across heads and merged back afterward.

    Parameters
    ----------
    in_mvc : int
        Number of input multivector channels.
    out_mvc : int
        Number of output multivector channels.
    in_sc : int or None
        Number of input scalar channels.
    out_sc : int or None
        Number of output scalar channels.
    dropout_prob : float
        Dropout probability.
    activation : str
        Activation function for the gated non-linearity.
    attention_type : str
        Type of attention to use: "gatr", "similarity", "geometric", or "dual_norm".
    """

    def __init__(
        self,
        in_mvc: int,
        out_mvc: int,
        in_sc: int | None = None,
        out_sc: int | None = None,
        edge_mvc: int | None = None,
        edge_sc: int | None = None,
        dropout_prob: float = 0.0,
        activation: str = "gelu",
        attention_type: str = "gatr",
        num_heads: int = 8,
        expansion_factor: float = 1.0,
        message: str = "geometric",
        **kwargs,
    ):
        super().__init__()
        self.in_mvc = in_mvc
        self.out_mvc = out_mvc
        self.h_mvc = in_mvc
        self.in_sc = in_sc
        self.out_sc = out_sc
        self.h_sc = self._round_up_to_heads(in_sc, num_heads) if in_sc is not None else None
        self.attention_type = attention_type
        self.dropout_prob = dropout_prob
        self.edge_mvc = edge_mvc
        self.edge_sc = edge_sc
        self.num_heads = num_heads
        self.message = message
        self._validate_head_compatibility()
        self._validate_message_product()

        self.attn_mvc = self.h_mvc
        self.attn_sc = self.h_sc if in_sc is not None else None
        self.mvc_per_head = self.h_mvc // self.num_heads
        self.attn_mvc_per_head = self.attn_mvc // self.num_heads
        self.sc_per_head = self.h_sc // self.num_heads if self.h_sc is not None else None
        self.attn_sc_per_head = self.attn_sc // self.num_heads if self.attn_sc is not None else None

        self.mlp_input_mvc = 2 * in_mvc
        self.mlp_input_sc = 2 * in_sc if in_sc is not None else None
        self.mlp_output_mvc = out_mvc
        self.mlp_output_sc = out_sc

        self.q_proj = EquiLinear(in_mvc, self.attn_mvc, in_sc, out_sc=self.attn_sc, **kwargs)
        self.k_proj = EquiLinear(in_mvc, self.attn_mvc, in_sc, out_sc=self.attn_sc, **kwargs)
        self.attention_scores = get_attention_module(
            attention_type=attention_type,
            out_mvc=self.attn_mvc_per_head,
            out_sc=self.attn_sc_per_head,
        )

        self.message_proj_src = EquiLinear(
            in_mvc=in_mvc,
            out_mvc=self.h_mvc,
            in_sc=in_sc if in_sc is not None else None,
            out_sc=self.h_sc if in_sc is not None else None,
            **kwargs,
        )
        self.message_proj_dst = EquiLinear(
            in_mvc=in_mvc,
            out_mvc=self.h_mvc,
            in_sc=in_sc if in_sc is not None else None,
            out_sc=None,
            **kwargs,
        )
        self.message_linear = (
            EquiLinear(
                in_mvc=2 * self.mvc_per_head, out_mvc=self.mvc_per_head, in_sc=None, out_sc=None, **kwargs
            )
            if message == "linear"
            else None
        )

        self.project_mvc = EquiLinear(
            in_mvc=self.h_mvc,
            out_mvc=in_mvc,
            in_sc=self.h_sc if in_sc is not None else None,
            out_sc=in_sc if in_sc is not None else None,
            **kwargs,
        )

        if edge_mvc is not None:
            self.edge_proj = EquiLinear(
                in_mvc=edge_mvc,
                out_mvc=self.h_mvc,
                in_sc=edge_sc if edge_sc is not None else None,
                out_sc=self.h_sc if edge_sc is not None else None,
                initialization="small",
                **kwargs,
            )
        else:
            self.edge_proj = None

        self.update_mlp_mv = EquiMLP(
            in_mvc=self.mlp_input_mvc,
            out_mvc=out_mvc,
            dropout_prob=dropout_prob,
            activation=activation,
            expansion_factor=expansion_factor,
            **kwargs,
        )

        if in_sc is not None:
            self.update_mlp_sc = ScalarMLP(
                in_sc=self.mlp_input_sc,
                out_sc=out_sc,
                dropout_prob=dropout_prob,
                activation=activation,
                expansion_factor=expansion_factor,
            )
        else:
            self.update_mlp_sc = None

        self.residual = ResidualLayer(in_mvc, out_mvc, in_sc, out_sc)
        self.norm = EquiLayerNorm(channel_dim=-2)

    @staticmethod
    def _round_up_to_heads(value: int | None, num_heads: int) -> int | None:
        if value is None:
            return None
        return math.ceil(value / num_heads) * num_heads

    def _validate_head_compatibility(self) -> None:
        if self.num_heads < 1:
            raise ValueError("num_heads must be >= 1.")
        if self.h_mvc % self.num_heads != 0:
            raise ValueError(f"in_mvc={self.h_mvc} must be divisible by num_heads={self.num_heads}.")

    def _validate_message_product(self) -> None:
        if self.message not in {"geometric", "sum", "linear"}:
            raise ValueError("message must be one of {'geometric', 'sum', 'linear'}.")

    def _combine_messages(self, dst_mv: torch.Tensor, src_mv: torch.Tensor) -> torch.Tensor:
        if self.message == "geometric":
            return geometric_product(dst_mv, src_mv)
        if self.message == "sum":
            return dst_mv + src_mv
        combined_mv = torch.cat([dst_mv, src_mv], dim=-2)
        out_mv, _ = self.message_linear(combined_mv, None)
        return out_mv

    def _reshape_mv_heads(self, mv: torch.Tensor | None, channels_per_head: int) -> torch.Tensor | None:
        if mv is None:
            return None
        return mv.view(*mv.shape[:-2], self.num_heads, channels_per_head, 16)

    def _reshape_sc_heads(
        self, sc: torch.Tensor | None, channels_per_head: int | None
    ) -> torch.Tensor | None:
        if sc is None or channels_per_head is None:
            return None
        return sc.view(*sc.shape[:-1], self.num_heads, channels_per_head)

    def _flatten_mv_heads(self, mv: torch.Tensor) -> torch.Tensor:
        return mv.reshape(*mv.shape[:-3], mv.shape[-3] * mv.shape[-2], 16)

    def _flatten_sc_heads(self, sc: torch.Tensor | None) -> torch.Tensor | None:
        if sc is None:
            return None
        return sc.reshape(*sc.shape[:-2], sc.shape[-2] * sc.shape[-1])

    def _compute_dense_edge_score_bias(
        self,
        Q_mv: torch.Tensor,
        Q_sc: torch.Tensor | None,
        E_mv: torch.Tensor | None,
        E_sc: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if E_mv is None and (E_sc is None or Q_sc is None):
            return None

        bias_query_parts = []
        bias_edge_parts = []

        if E_mv is not None:
            selector = INNER_PRODUCT_INDICES.to(Q_mv.device)
            scale_mv = max(self.attn_mvc_per_head * 8, 1) ** 0.5
            q_mv_bias = Q_mv[..., selector].reshape(*Q_mv.shape[:-2], -1) / scale_mv
            e_mv_bias = E_mv[..., selector].reshape(*E_mv.shape[:-2], -1)
            bias_query_parts.append(q_mv_bias)
            bias_edge_parts.append(e_mv_bias)

        if E_sc is not None and Q_sc is not None:
            scale_sc = max(self.attn_sc_per_head, 1) ** 0.5
            bias_query_parts.append(Q_sc / scale_sc)
            bias_edge_parts.append(E_sc)

        if not bias_query_parts:
            return None

        q_bias = torch.cat(bias_query_parts, dim=-1).unsqueeze(2)  # (B, N, 1, H, D)
        e_bias = torch.cat(bias_edge_parts, dim=-1)  # (B, N, N, H, D)
        return (q_bias * e_bias).sum(dim=-1).permute(0, 3, 1, 2)  # (B, H, N, N)

    def _compute_sparse_edge_score_bias(
        self,
        Q_mv_dst: torch.Tensor,
        Q_sc_dst: torch.Tensor | None,
        E_mv: torch.Tensor | None,
        E_sc: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if E_mv is None and (E_sc is None or Q_sc_dst is None):
            return None

        bias_query_parts = []
        bias_edge_parts = []

        if E_mv is not None:
            selector = INNER_PRODUCT_INDICES.to(Q_mv_dst.device)
            scale_mv = max(self.attn_mvc_per_head * 8, 1) ** 0.5
            q_mv_bias = Q_mv_dst[..., selector].reshape(*Q_mv_dst.shape[:-2], -1) / scale_mv
            e_mv_bias = E_mv[..., selector].reshape(*E_mv.shape[:-2], -1)
            bias_query_parts.append(q_mv_bias)
            bias_edge_parts.append(e_mv_bias)

        if E_sc is not None and Q_sc_dst is not None:
            scale_sc = max(self.attn_sc_per_head, 1) ** 0.5
            bias_query_parts.append(Q_sc_dst / scale_sc)
            bias_edge_parts.append(E_sc)

        if not bias_query_parts:
            return None

        q_bias = torch.cat(bias_query_parts, dim=-1)
        e_bias = torch.cat(bias_edge_parts, dim=-1)
        return (q_bias * e_bias).sum(dim=-1)

    def _validate_dense_edge_attributes(
        self,
        edge_attr_mv: torch.Tensor | None,
        edge_attr_sc: torch.Tensor | None,
        B: int,
        N: int,
    ):
        if self.edge_proj is None:
            raise ValueError("Dense edge attributes were provided, but edge_mvc/edge_sc are not configured.")

        if edge_attr_mv is None:
            raise ValueError("edge_attr_mv is required when dense edge attributes are enabled.")

        if edge_attr_mv.dim() != 5:
            raise ValueError("Dense edge_attr_mv must have shape (B, N, N, edge_mvc, 16).")

        if edge_attr_mv.shape[0] != B or edge_attr_mv.shape[1] != N or edge_attr_mv.shape[2] != N:
            raise ValueError("Dense edge_attr_mv leading dimensions must match (B, N, N).")
        if edge_attr_sc is not None and (
            edge_attr_sc.dim() != 4
            or edge_attr_sc.shape[0] != B
            or edge_attr_sc.shape[1] != N
            or edge_attr_sc.shape[2] != N
        ):
            raise ValueError("Dense edge_attr_sc must have shape (B, N, N, edge_sc).")

    def forward(
        self,
        mv: torch.Tensor,
        adj: torch.Tensor | None = None,
        sc: torch.Tensor | None = None,
        ref: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr_mv: torch.Tensor | None = None,
        edge_attr_sc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        mv : torch.Tensor
            Multivector features (B, N, in_mvc, 16)
        adj : torch.Tensor
            Adjacency matrix (B, N, N)
        sc : torch.Tensor or None
            Scalar features (B, N, in_sc) or None
        edge_index : torch.Tensor or None
            Edge indices (2, num_edges)
        edge_attr_mv : torch.Tensor or None
            Edge multivector attributes (num_edges, edge_mvc, 16)
        edge_attr_sc : torch.Tensor or None
            Edge scalar attributes (num_edges, edge_sc)

        Returns
        -------
        out_mv : torch.Tensor
            Output multivector features (B, N, out_mvc, 16)
        out_sc : torch.Tensor or None
            Output scalar features (B, N, out_sc) or None
        """
        # Sparse path: edge_index-driven attention (GATr variants only)
        if edge_index is not None and self.attention_type == "gatr_sparse":
            return self._forward_sparse(mv, sc, ref, edge_index, edge_attr_mv, edge_attr_sc)
        if edge_index is None and self.attention_type == "gatr_sparse":
            raise ValueError("gatr_sparse requires edge_index for sparse attention.")

        Q_mv, Q_sc = self.q_proj(mv, sc)  # (B, N, h_mvc, 16), (B, N, h_sc)
        K_mv, K_sc = self.k_proj(mv, sc)  # (B, N, h_mvc, 16), (B, N, h_sc)
        Q_mv = self._reshape_mv_heads(Q_mv, self.attn_mvc_per_head)
        K_mv = self._reshape_mv_heads(K_mv, self.attn_mvc_per_head)
        Q_sc = self._reshape_sc_heads(Q_sc, self.attn_sc_per_head)
        K_sc = self._reshape_sc_heads(K_sc, self.attn_sc_per_head)

        Q_all, K_all, _ = self.attention_scores.compute_qk_features(Q_mv, K_mv, Q_sc, K_sc)

        msg_dst, _ = self.norm(*self.message_proj_dst(mv, sc))  # (B, N, h_mvc, 16)
        msg_src, msg_sc_src = self.norm(*self.message_proj_src(mv, sc))  # (B, N, h_mvc, 16)
        msg_dst = self._reshape_mv_heads(msg_dst, self.mvc_per_head)
        msg_src = self._reshape_mv_heads(msg_src, self.mvc_per_head)
        msg_sc_src = self._reshape_sc_heads(msg_sc_src, self.sc_per_head)

        B, N = mv.shape[:2]
        V_mv = reverse(msg_src).permute(0, 2, 1, 3, 4).reshape(B, self.num_heads, N, self.mvc_per_head * 16)
        mv_val_dim = V_mv.shape[-1]
        if msg_sc_src is not None:
            V_sc = msg_sc_src.permute(0, 2, 1, 3)
            V_concat = torch.cat([V_mv, V_sc], dim=-1)
        else:
            V_concat = V_mv

        E_mv = None
        E_sc = None
        if (edge_attr_mv is not None or edge_attr_sc is not None) and self.edge_proj is not None:
            edge_mv_flat = edge_attr_mv.reshape(B * N * N, edge_attr_mv.size(-2), 16)
            edge_sc_flat = (
                edge_attr_sc.reshape(B * N * N, edge_attr_sc.size(-1)) if edge_attr_sc is not None else None
            )
            E_mv, E_sc = self.edge_proj(edge_mv_flat, edge_sc_flat)
            E_mv, E_sc = self.norm(E_mv, E_sc)
            E_mv = E_mv.view(B, N, N, self.h_mvc, 16)
            E_mv = self._reshape_mv_heads(E_mv, self.mvc_per_head)
            E_sc = E_sc.view(B, N, N, self.h_sc) if E_sc is not None else None
            E_sc = self._reshape_sc_heads(E_sc, self.sc_per_head) if E_sc is not None else None

        attn_mask = self._compute_dense_edge_score_bias(Q_mv, Q_sc, E_mv, E_sc)
        if attn_mask is None:
            attn_mask = torch.zeros((B, self.num_heads, N, N), device=mv.device, dtype=Q_all.dtype)

        mask = adj.bool()
        valid = mask.any(dim=2, keepdim=True)
        mask_to_inf = (~mask) & valid
        attn_mask = attn_mask.masked_fill(mask_to_inf.unsqueeze(1), float("-inf"))

        if E_mv is not None or E_sc is not None:
            attn_weights = torch.matmul(Q_all, K_all.transpose(-2, -1)) + attn_mask
            attn_weights = F.softmax(attn_weights.float(), dim=-1)
            attn_weights = torch.where(
                valid.unsqueeze(1),
                attn_weights,
                torch.zeros_like(attn_weights),
            ).to(Q_all.dtype)

            aggregated_concat = torch.matmul(attn_weights, V_concat)
            aggregated_mv = aggregated_concat[..., :mv_val_dim]
            aggregated_mv = aggregated_mv.reshape(B, self.num_heads, N, self.mvc_per_head, 16)
            aggregated_mv = aggregated_mv.permute(0, 2, 1, 3, 4)
            aggregated_mv = self._combine_messages(msg_dst, aggregated_mv)

            if msg_sc_src is not None:
                aggregated_sc = aggregated_concat[..., mv_val_dim:]
                aggregated_sc = aggregated_sc.permute(0, 2, 1, 3)
            else:
                aggregated_sc = None

            if E_mv is not None:
                edge_mv_sum = torch.einsum(
                    "bhij,bhijd->bhid",
                    attn_weights,
                    E_mv.reshape(B, N, N, self.num_heads, -1).permute(0, 3, 1, 2, 4),
                )
                edge_mv_sum = edge_mv_sum.reshape(B, self.num_heads, N, self.mvc_per_head, 16)
                edge_mv_sum = edge_mv_sum.permute(0, 2, 1, 3, 4)
                aggregated_mv = aggregated_mv + edge_mv_sum
            if E_sc is not None and aggregated_sc is not None:
                edge_sc_sum = torch.einsum(
                    "bhij,bhijd->bhid",
                    attn_weights,
                    E_sc.permute(0, 3, 1, 2, 4),
                ).permute(0, 2, 1, 3)
                aggregated_sc = aggregated_sc + edge_sc_sum
        else:
            aggregated_concat = F.scaled_dot_product_attention(
                Q_all,
                K_all,
                V_concat,
                attn_mask=attn_mask,
                scale=1.0,
            )

            aggregated_mv = aggregated_concat[..., :mv_val_dim]
            aggregated_mv = aggregated_mv.reshape(B, self.num_heads, N, self.mvc_per_head, 16)
            aggregated_mv = aggregated_mv.permute(0, 2, 1, 3, 4)
            aggregated_mv = self._combine_messages(msg_dst, aggregated_mv)

            if msg_sc_src is not None:
                aggregated_sc = aggregated_concat[..., mv_val_dim:]
                aggregated_sc = aggregated_sc.permute(0, 2, 1, 3)
            else:
                aggregated_sc = None

        aggregated_mv = self._flatten_mv_heads(aggregated_mv)  # (B, N, h_mvc, 16)
        aggregated_sc = self._flatten_sc_heads(aggregated_sc) if aggregated_sc is not None else None

        # Normalize aggregated messages
        aggregated_mv, aggregated_sc = self.norm(aggregated_mv, aggregated_sc)
        # Project aggregated messages
        combined_messages_mv, combined_messages_sc = self.project_mvc(aggregated_mv, aggregated_sc)
        # Concatenate original features with aggregated messages
        combined_mv = torch.cat([mv, combined_messages_mv], dim=-2)  # (B, N, 2*in_mvc, 16)
        if sc is not None and combined_messages_sc is not None:
            combined_sc = torch.cat([sc, combined_messages_sc], dim=-1)  # (B, N, 2*in_sc)
        else:
            combined_sc = None

        # Update node features
        h_mv = self.update_mlp_mv(combined_mv)
        h_sc = self.update_mlp_sc(combined_sc) if combined_sc is not None else None
        out_mv, out_sc = self.residual(h_mv, mv, h_sc, sc)
        out_mv, out_sc = self.norm(out_mv, out_sc)

        return out_mv, out_sc

    def _forward_sparse(
        self,
        mv: torch.Tensor,
        sc: torch.Tensor | None,
        ref: torch.Tensor | None,
        edge_index: torch.Tensor,
        edge_attr_mv: torch.Tensor | None = None,
        edge_attr_sc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sparse edge-wise attention path for GATr attention only."""
        if mv.dim() != 3:
            raise ValueError("Sparse GA_GNN expects packed node features with shape (N, C, 16).")

        num_nodes = mv.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        Q_mv, Q_sc = self.q_proj(mv, sc)
        K_mv, K_sc = self.k_proj(mv, sc)
        Q_mv = self._reshape_mv_heads(Q_mv, self.attn_mvc_per_head)
        K_mv = self._reshape_mv_heads(K_mv, self.attn_mvc_per_head)
        Q_sc = self._reshape_sc_heads(Q_sc, self.attn_sc_per_head)
        K_sc = self._reshape_sc_heads(K_sc, self.attn_sc_per_head)

        msg_dst, _ = self.norm(*self.message_proj_dst(mv, sc))  # (N, h_mvc, 16)
        msg_src, msg_sc_src = self.norm(*self.message_proj_src(mv, sc))  # (N, h_mvc, 16)

        msg_dst_heads = self._reshape_mv_heads(msg_dst, self.mvc_per_head)
        msg_src_heads = self._reshape_mv_heads(msg_src, self.mvc_per_head)

        # The value to be aggregated is the reversed source message
        V_mv = reverse(msg_src_heads)  # (N, H, mvc_per_head, 16)
        V_sc = self._reshape_sc_heads(msg_sc_src, self.sc_per_head) if msg_sc_src is not None else None

        E_mv = None
        E_sc = None
        if edge_attr_mv is not None or edge_attr_sc is not None:
            if self.edge_proj is None:
                raise ValueError("Sparse edge attributes provided, but edge_mvc/edge_sc are not configured.")
            E_mv, E_sc = self.edge_proj(edge_attr_mv, edge_attr_sc)  # (E, h_mvc, 16), (E, h_sc)
            E_mv, E_sc = self.norm(E_mv, E_sc)
            E_mv = self._reshape_mv_heads(E_mv, self.mvc_per_head)
            E_sc = self._reshape_sc_heads(E_sc, self.sc_per_head) if E_sc is not None else None

        scores = self.attention_scores(Q_mv, K_mv, edge_index, Q_sc, K_sc)  # (E, H)

        # Add edge-projected contributions to scores (Mask/Bias trick)
        edge_score_bias = self._compute_sparse_edge_score_bias(
            Q_mv[dst],
            Q_sc[dst] if Q_sc is not None else None,
            E_mv,
            E_sc,
        )
        if edge_score_bias is not None:
            scores = scores + edge_score_bias

        scores = scores.clamp(-30, 30)

        # Edge softmax per destination node
        attn = scatter_softmax(scores, dst, dim=0, dim_size=num_nodes)  # (E, H)

        # Expand attention weights for multivector broadcasts: (E, H, 1, 1)
        attn_mv = attn.unsqueeze(-1).unsqueeze(-1)

        # Aggregate V_mv
        weighted_V_mv = V_mv[src] * attn_mv
        agg_V_mv = scatter_add(src=weighted_V_mv, index=dst, dim=0, dim_size=num_nodes)

        # Combine destination and aggregated source messages once per node
        aggregated_mv = self._combine_messages(msg_dst_heads, agg_V_mv)

        # Aggregate V_sc
        if V_sc is not None:
            attn_sc = attn.unsqueeze(-1)  # (E, H, 1)
            weighted_V_sc = V_sc[src] * attn_sc
            aggregated_sc = scatter_add(src=weighted_V_sc, index=dst, dim=0, dim_size=num_nodes)
        else:
            aggregated_sc = None

        if E_mv is not None:
            weighted_E_mv = E_mv * attn_mv
            edge_mv_sum = scatter_add(src=weighted_E_mv, index=dst, dim=0, dim_size=num_nodes)
            aggregated_mv = aggregated_mv + edge_mv_sum

        if E_sc is not None and aggregated_sc is not None:
            weighted_E_sc = E_sc * attn_sc
            edge_sc_sum = scatter_add(src=weighted_E_sc, index=dst, dim=0, dim_size=num_nodes)
            aggregated_sc = aggregated_sc + edge_sc_sum

        aggregated_mv = self._flatten_mv_heads(aggregated_mv)
        aggregated_sc = self._flatten_sc_heads(aggregated_sc) if aggregated_sc is not None else None

        # Normalize aggregated messages (Delayed norm matching dense path)
        aggregated_mv, aggregated_sc = self.norm(aggregated_mv, aggregated_sc)

        # Project aggregated messages
        combined_messages_mv, combined_messages_sc = self.project_mvc(aggregated_mv, aggregated_sc)

        # Concatenate original features with aggregated messages
        combined_mv = torch.cat([mv, combined_messages_mv], dim=-2)
        if sc is not None and combined_messages_sc is not None:
            combined_sc = torch.cat([sc, combined_messages_sc], dim=-1)
        else:
            combined_sc = None

        # Update node features
        h_mv = self.update_mlp_mv(combined_mv)
        h_sc = self.update_mlp_sc(combined_sc) if combined_sc is not None else None

        # Residual connection + final norm
        out_mv, out_sc = self.residual(h_mv, mv, h_sc, sc)
        out_mv, out_sc = self.norm(out_mv, out_sc)

        return out_mv, out_sc


class PGA_GNN(nn.Module):
    """Geometric Algebra Graph Neural Network.

    Parameters
    ----------
    in_mvc : int
        Number of input multivector channels.
    hidden_mvc : int
        Number of hidden multivector channels.
    out_mvc : int
        Number of output multivector channels.
    in_sc : int or None
        Number of input scalar channels.
    hidden_sc : int or None
        Number of hidden scalar channels.
    out_sc : int or None
        Number of output scalar channels.
    num_layers : int
        Number of message passing layers.
    dropout_prob : float
        Dropout probability.
    activation : str
        Activation function for the gated non-linearity.
    attention_type : str
        Type of attention to use: "gatr", "similarity", "geometric", or "dual_norm".
    """

    def __init__(
        self,
        in_mvc: int,
        hidden_mvc: int,
        out_mvc: int = 1,
        in_sc: int | None = None,
        hidden_sc: int | None = None,
        out_sc: int | None = None,
        edge_mvc: int | None = None,
        edge_sc: int | None = None,
        num_layers: int = 1,
        dropout_prob: float = 0.0,
        activation: str = "gelu",
        attention_type: str = "gatr",
        num_heads: int = 8,
        expansion_factor: float = 1.0,
        num_spurions: int = 0,
        message: str = "geometric",
        **kwargs,
    ):
        super().__init__()
        self.in_mvc = in_mvc
        self.hidden_mvc = hidden_mvc
        self.out_mvc = out_mvc
        self.in_sc = in_sc
        self.hidden_sc = hidden_sc
        self.out_sc = out_sc
        self.edge_mvc = edge_mvc
        self.edge_sc = edge_sc
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.num_spurions = num_spurions
        self.use_spurions = num_spurions > 0
        self.expansion_factor = expansion_factor
        self.message = message

        if self.use_spurions:
            self.spurions = nn.Parameter(torch.empty(self.num_spurions, 16))
            nn.init.xavier_uniform_(self.spurions)

        # Adjust input dimension to account for spurions
        linear_in_mvc = in_mvc + num_spurions if self.use_spurions else in_mvc

        self.layers = nn.ModuleList()
        self.input_proj = EquiLinear(linear_in_mvc, hidden_mvc, in_sc, hidden_sc, **kwargs)
        for _ in range(num_layers):
            self.layers.append(
                GA_MessageLayer(
                    hidden_mvc,
                    hidden_mvc,
                    hidden_sc,
                    hidden_sc,
                    edge_mvc=edge_mvc,
                    edge_sc=edge_sc,
                    dropout_prob=dropout_prob,
                    activation=activation,
                    attention_type=attention_type,
                    num_heads=num_heads,
                    expansion_factor=expansion_factor,
                    message=message,
                    **kwargs,
                )
            )
        self.output_proj = EquiLinear(hidden_mvc, out_mvc, hidden_sc, out_sc, **kwargs)

    @staticmethod
    def _ensure_self_loops_dense(adj: torch.Tensor | None, mv: torch.Tensor) -> torch.Tensor:
        if adj is None:
            B, N = mv.shape[:2]
            return torch.ones((B, N, N), device=mv.device, dtype=mv.dtype)
        eye = torch.eye(adj.size(1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
        return torch.maximum(adj, eye)

    def forward(
        self,
        mv: torch.Tensor,
        adj: torch.Tensor | None = None,
        sc: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr_mv: torch.Tensor | None = None,
        edge_attr_sc: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        ref: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        mv : torch.Tensor
            Multivector features (B, N, in_mvc, 16)
        adj : torch.Tensor
            Adjacency matrix (B, N, N)
        sc : torch.Tensor or None
            Scalar features (B, N, in_sc) or None

        Returns
        -------
        out_mv : torch.Tensor
            Output multivector features (B, N, out_mvc, 16)
        out_sc : torch.Tensor or None
            Output scalar features (B, N, out_sc) or None
        """
        # Add learnable spurions for symmetry breaking if enabled
        if self.use_spurions:
            if mv.dim() == 4:
                B, N, _, _ = mv.shape
                spurions_expanded = self.spurions.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
                mv = torch.cat([mv, spurions_expanded], dim=-2)
            elif mv.dim() == 3:
                N = mv.size(0)
                spurions_expanded = self.spurions.unsqueeze(0).expand(N, -1, -1)
                mv = torch.cat([mv, spurions_expanded], dim=-2)
            else:
                raise ValueError("mv must have shape (B,N,C,16) or (N,C,16).")

        # Sparse edge_index path (packed nodes)
        if edge_index is not None and self.attention_type in ("gatr", "gatr_sparse"):
            if mv.dim() == 4:
                if mask is None:
                    raise ValueError("mask is required to pack dense inputs for sparse edge_index.")
                mv = mv[mask]
                sc = sc[mask] if sc is not None else None
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError("edge_index must have shape (2, E).")
            if batch is None:
                # Default to single-graph batch if not provided
                batch = torch.zeros(mv.size(0), device=mv.device, dtype=torch.long)

            mv, sc = self.input_proj(mv, sc)
            num_nodes = mv.size(0)
            ref_nodes = torch.ones((num_nodes, 1, 16), device=mv.device) if ref is None else ref[batch]

            for i in range(self.num_layers):
                layer = self.layers[i]
                mv, sc = layer(
                    mv=mv,
                    sc=sc,
                    ref=ref_nodes,
                    edge_index=edge_index,
                    edge_attr_mv=edge_attr_mv,
                    edge_attr_sc=edge_attr_sc,
                )
            mv, sc = self.output_proj(mv, sc)
            return mv, sc
        if edge_index is None and self.attention_type == "gatr_sparse":
            raise ValueError("gatr_sparse requires edge_index for sparse attention.")

        # Dense adjacency path
        adj = self._ensure_self_loops_dense(adj=adj, mv=mv)
        mv, sc = self.input_proj(mv, sc)
        for i in range(self.num_layers):
            layer = self.layers[i]
            mv, sc = layer(
                mv=mv,
                sc=sc,
                adj=adj,
                edge_attr_mv=edge_attr_mv,
                edge_attr_sc=edge_attr_sc,
                ref=ref,
            )
        mv, sc = self.output_proj(mv, sc)
        return mv, sc

    def readout(
        self,
        mv: torch.Tensor,
        sc: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        mv : torch.Tensor
            Multivector features (B, N, out_mvc, 16)
        sc : torch.Tensor or None
            Scalar features (B, N, out_sc) or None

        Returns
        -------
        out_mv : torch.Tensor
            Aggregated multivector features (B, out_mvc, 16)
        out_sc : torch.Tensor or None
            Aggregated scalar features (B, out_sc) or None
        """
        if mv.dim() == 4:
            out_mv = torch.sum(mv, dim=1)
            out_sc = torch.sum(sc, dim=1) if sc is not None else None
            return out_mv, out_sc

        # Packed node readout using batch vector
        if batch is None:
            raise ValueError("batch is required for packed node readout.")
        out_mv = scatter_add(mv, batch, dim=0)
        if sc is not None:
            out_sc = scatter_add(sc, batch, dim=0)
        else:
            out_sc = None
        return out_mv, out_sc
