import math

import torch
import torch.nn.functional as F
from torch import nn

from .primitives import (
    depthwise_equi_linear,
    equi_layer_norm,
    equi_linear,
    equivariant_join,
    gated_non_linearity,
    geometric_product,
    grade_dropout,
)

_GRADE_COMPONENT_INDEX = torch.tensor([0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4], dtype=torch.long)
_MV_COMP_FACTORS = torch.sqrt(torch.tensor([1.0, 4.0, 6.0, 2.0, 0.5, 0.5, 1.5, 1.5, 0.5]))


class EquiLinear(nn.Module):
    def __init__(
        self,
        in_mvc: int,
        out_mvc: int,
        in_sc: int | None = None,
        out_sc: int | None = None,
        bias: bool = True,
        initialization: str = "default",
        gain: float = 1.0,
        additional_factor: float = 1.0 / math.sqrt(3.0),
        use_mv_heuristics: bool = True,
        use_grade_modulation: bool = False,
        use_pseudoscalar: bool = False,
        factorize: bool = False,
    ):
        super().__init__()
        self.in_mvc, self.out_mvc = in_mvc, out_mvc
        self.in_sc, self.out_sc = in_sc, out_sc
        self.use_grade_modulation = use_grade_modulation
        self.use_pseudoscalar = use_pseudoscalar
        self.factorize = factorize
        self._build_modules(bias, in_sc, out_sc)
        self.register_buffer("_grade_idx", _GRADE_COMPONENT_INDEX)
        self._init_weights(initialization, gain, additional_factor, use_mv_heuristics)

    def _build_modules(self, bias: bool, in_sc, out_sc):
        in_mvc, out_mvc = self.in_mvc, self.out_mvc

        self.mv_weight = (
            nn.Parameter(torch.empty(out_mvc, in_mvc, 9))
            if not self.factorize
            else nn.Parameter(torch.empty(in_mvc, 9))
        )
        self.channel_mix = nn.Linear(in_mvc, out_mvc, bias=False) if self.factorize else None
        self.mv_bias = nn.Parameter(torch.zeros(out_mvc)) if (bias and in_sc is None) else None

        # SC → MV: output dim is out_mvc*2 when pseudoscalar is active, else out_mvc
        s2mvs_out = out_mvc * 2 if self.use_pseudoscalar else out_mvc
        self.s2mvs = nn.Linear(in_sc, s2mvs_out, bias=bias) if in_sc else None
        mvs2s_in = in_mvc * 2 if self.use_pseudoscalar else in_mvc
        self.mvs2s = nn.Linear(mvs2s_in, out_sc, bias=bias) if out_sc else None
        self.s2s = nn.Linear(in_sc, out_sc, bias=False) if (in_sc and out_sc) else None

        # Grade modulation: 5 scale factors per output channel
        self.grade_scale = (
            nn.Linear(in_sc, out_mvc * 5, bias=True) if (in_sc and self.use_grade_modulation) else None
        )

    def _init_weights(self, initialization, gain, additional_factor, use_mv_heuristics):
        mv_factor, s_factor, mvs_bias_shift, comp_factors = self._init_factors(
            initialization, gain, additional_factor, use_mv_heuristics
        )
        self._init_mv_path(mv_factor, mvs_bias_shift, comp_factors)
        self._init_scalar_path(s_factor)
        self._init_grade_scale_to_identity()

    @staticmethod
    def _init_factors(initialization, gain, additional_factor, use_mv_heuristics):
        schemes = {
            "default": (1.0, 0.0),
            "unit_scalar": (0.5, 1.0),
            "small": (0.1, 0.0),
        }
        if initialization not in schemes:
            raise ValueError(f"Unknown initialization: {initialization!r}")
        scale, mvs_bias_shift = schemes[initialization]
        mv_factor = scale * gain * additional_factor * math.sqrt(3.0)
        s_factor = scale * gain * math.sqrt(3.0)
        comp_factors = _MV_COMP_FACTORS if use_mv_heuristics else torch.ones(9)
        return mv_factor, s_factor, mvs_bias_shift, comp_factors

    def _init_mv_path(self, mv_factor, mvs_bias_shift, comp_factors):
        fan_in = max(self.in_mvc, 1)
        base_bnd = mv_factor if self.factorize else mv_factor / math.sqrt(fan_in)

        for i, factor in enumerate(comp_factors):
            nn.init.uniform_(self.mv_weight[..., i], -factor * base_bnd, factor * base_bnd)

        if self.s2mvs is not None:
            half_bnd = comp_factors[0] * mv_factor / math.sqrt(fan_in) / math.sqrt(2)
            nn.init.uniform_(self.mv_weight[..., [0]], -half_bnd, half_bnd)

            fan_s = max(nn.init._calculate_fan_in_and_fan_out(self.s2mvs.weight)[0], 1)
            s_bnd = comp_factors[0] * mv_factor / math.sqrt(fan_s) / math.sqrt(2)
            nn.init.uniform_(self.s2mvs.weight, -s_bnd, s_bnd)

            if self.s2mvs.bias is not None:
                total = fan_s + self.in_mvc
                b_bnd = comp_factors[0] / math.sqrt(total) if total else 0.0
                nn.init.uniform_(self.s2mvs.bias, mvs_bias_shift - b_bnd, mvs_bias_shift + b_bnd)

        if self.mv_bias is not None:
            nn.init.zeros_(self.mv_bias)

    def _init_scalar_path(self, s_factor):
        paths = [m for m in (self.s2s, self.mvs2s) if m is not None]
        for m in paths:
            fan_in = max(nn.init._calculate_fan_in_and_fan_out(m.weight)[0], 1)
            bnd = s_factor / math.sqrt(fan_in) / math.sqrt(len(paths))
            nn.init.uniform_(m.weight, -bnd, bnd)

        if self.mvs2s is not None and self.mvs2s.bias is not None:
            total = sum(nn.init._calculate_fan_in_and_fan_out(m.weight)[0] for m in paths)
            bnd = s_factor / math.sqrt(total) if total else 0.0
            nn.init.uniform_(self.mvs2s.bias, -bnd, bnd)

    def _init_grade_scale_to_identity(self):
        """Start grade modulation at all-ones (no effect at init)."""
        if self.grade_scale is not None:
            nn.init.zeros_(self.grade_scale.weight)
            nn.init.ones_(self.grade_scale.bias)

    def forward(
        self, mv: torch.Tensor, sc: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out_mv = self._mv_to_mv(mv)
        out_mv = self._apply_grade_modulation(out_mv, sc)
        out_mv = self._inject_scalars_into_mv(out_mv, sc)
        out_sc = self._mv_to_scalars(mv, sc)
        return out_mv, out_sc

    def _mv_to_mv(self, mv):
        out = depthwise_equi_linear(mv, self.mv_weight) if self.factorize else equi_linear(mv, self.mv_weight)
        if self.channel_mix is not None:
            out = self.channel_mix(out.transpose(-2, -1)).transpose(-2, -1)  # channel mixing
        if self.mv_bias is not None:
            out[..., 0] += self.mv_bias
        return out

    def _apply_grade_modulation(self, out_mv, sc):
        if self.grade_scale is None or sc is None:
            return out_mv
        # (..., out_mvc, 5) — one scale per grade per output channel
        scales = self.grade_scale(sc).view(*sc.shape[:-1], self.out_mvc, 5)
        return out_mv * scales[..., self._grade_idx]  # broadcast over 16 components

    def _inject_scalars_into_mv(self, out_mv, sc):
        if self.s2mvs is None or sc is None:
            return out_mv
        proj = self.s2mvs(sc)  # (..., out_mvc) or (..., out_mvc*2)
        out_mv[..., 0] += proj[..., : self.out_mvc]  # grade-0 always
        if self.use_pseudoscalar:
            out_mv[..., 15] += proj[..., self.out_mvc :]  # grade-4 (pseudoscalar)
        return out_mv

    def _mv_to_scalars(self, mv, sc):
        if self.mvs2s is None:
            return None
        mv_invariants = mv[..., 0]
        if self.use_pseudoscalar:
            mv_invariants = torch.cat([mv_invariants, mv[..., 15]], dim=-1)
        out_sc = self.mvs2s(mv_invariants)
        if self.s2s is not None and sc is not None:
            out_sc = out_sc + self.s2s(sc)
        return out_sc


class GeometricBilinear(nn.Module):
    """Equivariant bilinear layer for multivectors.

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
    """

    def __init__(
        self,
        in_mvc: int,
        out_mvc: int,
        in_sc: int | None = None,
        out_sc: int | None = None,
    ):
        super().__init__()
        self.in_mvc = in_mvc
        self.out_mvc = out_mvc
        self.in_sc = in_sc
        self.out_sc = out_sc

        hidden_dim = out_mvc // 2
        self.linear_left = EquiLinear(
            in_mvc=in_mvc,
            out_mvc=hidden_dim,
            in_sc=in_sc,
            out_sc=None,
            initialization="unit_scalar",
        )
        self.linear_right = EquiLinear(in_mvc=in_mvc, out_mvc=hidden_dim, in_sc=in_sc, out_sc=None)
        self.linear_join_left = EquiLinear(
            in_mvc=in_mvc,
            out_mvc=hidden_dim,
            in_sc=in_sc,
            out_sc=None,
        )
        self.linear_join_right = EquiLinear(in_mvc=in_mvc, out_mvc=hidden_dim, in_sc=in_sc, out_sc=None)
        self.output_linear = EquiLinear(in_mvc=2 * hidden_dim, out_mvc=out_mvc, in_sc=in_sc, out_sc=out_sc)

    def forward(
        self, x: torch.Tensor, ref: torch.Tensor, sc: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x: (B, N, in_mvc, 16)
        ref: (B, 1, 1, 16)
        sc: (B, N, in_sc) or None
        Returns: (B, N, out_mvc, 16), (B, N, out_sc) or None
        """
        x_gp, _ = self.linear_left(x, sc)
        y_gp, _ = self.linear_right(x, sc)
        gp = geometric_product(x_gp, y_gp)

        x_join, _ = self.linear_join_left(x, sc)
        y_join, _ = self.linear_join_right(x, sc)
        join = equivariant_join(x_join, y_join, ref)
        combined = torch.cat([gp, join], dim=-2)  # Concatenate along channel dimension

        # Output linear layer
        out_mv, out_sc = self.output_linear(combined, sc)
        return out_mv, out_sc


class GradeDropout(nn.Module):
    """Grade dropout for multivectors (and regular dropout for auxiliary scalars).

    Parameters
    ----------
    p : float
        Dropout probability.
    """

    def __init__(self, dropout_prob: float = 0.0):
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(self, mv: torch.Tensor, sc: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass. Applies dropout.

        Parameters
        ----------
        mv : torch.Tensor with shape (..., 16)
            Multivector inputs.
        sc : torch.Tensor with shape (..., num_scalars) or None
            Scalar inputs.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., 16)
            Multivector inputs with dropout applied.
        outputs_sc : torch.Tensor with shape (..., num_scalars) or None
            Scalar inputs with dropout applied.
        """

        out_mv = grade_dropout(mv, p=self.dropout_prob, training=self.training)
        out_sc = F.dropout(sc, p=self.dropout_prob, training=self.training) if sc is not None else None
        return out_mv, out_sc


class GatedNonLinearity(nn.Module):
    """Pin-equivariant gated nonlinearity.

    Given multivector input mv and scalar input gates (with matching batch dimensions), computes
    NonLinearity(gates) * mv.

    Parameters
    ----------
    None
    """

    def __init__(self, activation: str = "silu"):
        super().__init__()
        self.activation = activation
        self.dict = {
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
        }
        self.non_linearity = self.dict[activation]

    def forward(
        self, mv: torch.Tensor, gates: torch.Tensor, sc: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the GatedNonLinearity layer.

        Parameters
        ----------
        mv : torch.Tensor with shape (..., 16)
            Multivector input
        gates : torch.Tensor with shape (..., 1)
            Pin-invariant gates.
        sc : torch.Tensor with shape (..., num_scalars) or None
            Scalar input.

        Returns
        -------
        out_mv : torch.Tensor with shape (..., 16)
            Computes NonLinearity(gates) * mv, with broadcasting along the last dimension.
        out_sc : torch.Tensor with shape (..., num_scalars) or None
            Activated scalars.
        """
        out_mv = gated_non_linearity(mv, gates, self.non_linearity)
        out_sc = self.non_linearity(sc) if sc is not None else None
        return out_mv, out_sc


class EquiLayerNorm(nn.Module):
    """Equivariant LayerNorm for multivectors.

    Rescales input such that `mean_channels |inputs|^2 = 1`, where the norm is the GA norm and the
    mean goes over the channel dimensions.

    Using a factor `gain > 1` makes up for the fact that the GP norm overestimates the actual
    standard deviation of the input data.

    Parameters
    ----------
    channel_dim : int
        Channel dimension index. Defaults to the second-last entry (last are the multivector
        components).
    gain : float
        Target output scale.
    epsilon : float
        Small numerical factor to avoid instabilities. By default, we use a reasonably large number
        to balance issues that arise from some multivector components not contributing to the norm.
    """

    def __init__(self, channel_dim: int = -2, gain: float = 1.0, epsilon: float = 0.01):
        super().__init__()
        self.channel_dim = channel_dim
        self.gain = gain
        self.epsilon = epsilon

    def forward(self, mv: torch.Tensor, sc: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the EquiLayerNorm layer.

        Parameters
        ----------
        mv : torch.Tensor with shape `(batch_dim, *channel_dims, 16)`
            Input multivectors.
        sc : torch.Tensor with shape `(batch_dim, *channel_dims, num_scalars)` or None
            Input scalars.

        Returns
        -------
        out_mv : torch.Tensor with shape `(batch_dim, *channel_dims, 16)`
            Normalized inputs.
        out_sc : torch.Tensor with shape `(batch_dim, *channel_dims, num_scalars)` or None
            Normalized scalar inputs.
        """
        out_mv = equi_layer_norm(
            mv,
            channel_dim=self.channel_dim,
            gain=self.gain,
            epsilon=self.epsilon,
        )
        out_sc = F.layer_norm(sc, normalized_shape=sc.shape[-1:]) if sc is not None else None
        return out_mv, out_sc


class IdentityLayer(nn.Module):
    """Identity layer for multivectors and scalars."""

    def __init__(self):
        super().__init__()

    def forward(self, mv: torch.Tensor, sc: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        return mv, sc


class ResidualLayer(nn.Module):
    """Residual layer with equivariant linear transformation.

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
    """

    def __init__(
        self,
        in_mvc: int,
        out_mvc: int,
        in_sc: int | None = None,
        out_sc: int | None = None,
    ):
        super().__init__()
        self.in_mvc = in_mvc
        self.out_mvc = out_mvc
        self.in_sc = in_sc
        self.out_sc = out_sc

        if in_mvc == out_mvc and (in_sc == out_sc or in_sc is None):
            self.residual = IdentityLayer()
        else:
            self.residual = EquiLinear(in_mvc, out_mvc, in_sc, out_sc)

    def forward(
        self,
        mv_out: torch.Tensor,
        mv_in: torch.Tensor,
        sc_out: torch.Tensor | None,
        sc_in: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        mv_out: (..., out_mvc, 16)
        mv_in: (..., in_mvc, 16)
        sc_out: (..., out_sc) or None
        sc_in: (..., in_sc) or None

        Returns:
          out_mv: (..., out_mvc, 16)
          out_sc: (..., out_sc) or None
        """
        res_mv, res_sc = self.residual(mv_in, sc_in)
        out_mv = mv_out + res_mv
        if sc_out is not None and res_sc is not None:
            out_sc = sc_out + res_sc
        else:
            out_sc = None
        return out_mv, out_sc


class GaussianRadialBasisLayer(nn.Module):
    """Gaussian smearing with cosine cutoff envelope."""

    def __init__(self, num_bases: int = 64, cutoff: float = 5.0, mask: bool = True):
        super().__init__()
        self.num_bases = num_bases
        self.cutoff = cutoff
        self.mask = mask
        offset = torch.linspace(0.0, cutoff, num_bases + 1)[:-1]
        spacing = cutoff / num_bases
        coeff = -0.5 / (spacing**2)
        self.register_buffer("offset", offset)
        self.register_buffer("coeff", torch.tensor(coeff))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (..., 1)
        dist_expanded = dist - self.offset
        rbf = torch.exp(self.coeff * torch.pow(dist_expanded, 2))
        cutoff_envelope = 0.5 * (torch.cos(dist * math.pi / self.cutoff) + 1.0)
        if self.mask:
            cutoff_envelope = cutoff_envelope * (dist < self.cutoff).to(cutoff_envelope.dtype)
        return rbf * cutoff_envelope  # (..., num_bases)
