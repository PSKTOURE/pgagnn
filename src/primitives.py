import torch
import torch.nn.functional as F

from .utils import (
    INFINITE_NORM_INDICES,
    INNER_PRODUCT_INDICES,
    _compute_dualization,
    _compute_pin_equilinear_basis,
    _compute_reversal,
    _get_anti_dual_factors,
    _load_bilinear_basis,
)


def grade_project(x: torch.Tensor) -> torch.Tensor:
    """Projects an input multivector onto its grade components.

    Parameters
    ----------
    x : torch.Tensor
        The input multivector of shape (..., 16).

    Returns
    -------
    torch.Tensor
        The projected multivector of shape (..., 5, 16).
    """
    basis = _compute_pin_equilinear_basis(device=x.device, dtype=x.dtype, normalize=False)
    # Keep only the first 5 basis elements for grade projection
    # Shape: (5, 16, 16)
    basis = basis[:5]
    return torch.einsum("gij,...j->...gi", basis, x)


def invariants(x: torch.Tensor, *, centered: bool = False) -> torch.Tensor:
    """Compute five scalar features under the twisted versor action.

    The absolute grade-four coefficient removes its reflection sign. By default,
    metric grade norms exclude ideal blades and give E(3)-invariant features.
    With centered=True, full coefficient norms preserve geometry in a centered
    molecular frame. That mode is O(3)-invariant and requires per-graph centering
    upstream for translation invariance. Neither mode claims completeness.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.
    centered : bool
        Include ideal coefficients in the norms for centered representations.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 5)
        Scalar, absolute pseudoscalar coefficient, and three grade norms.
    """
    scalar = x[..., 0:1]
    pseudoscalar = x[..., 15:16].abs()

    # Add epsilon under the square root to avoid singular gradients at exactly zero norm.
    eps = 1e-12
    vector = x[..., 1:5] if centered else x[..., 2:5]
    bivector = x[..., 5:11] if centered else x[..., 8:11]
    trivector = x[..., 11:15] if centered else x[..., 14:15]
    v_norm = torch.sqrt(vector.square().sum(dim=-1, keepdim=True) + eps)
    bivector_norm = torch.sqrt(bivector.square().sum(dim=-1, keepdim=True) + eps)
    trivector_norm = torch.sqrt(trivector.square().sum(dim=-1, keepdim=True) + eps)
    outputs = torch.cat([scalar, pseudoscalar, v_norm, bivector_norm, trivector_norm], dim=-1)
    return outputs


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Computes the reversal of a multivector.

    Parameters
    ----------
    x : torch.Tensor
        Input multivector of shape (..., 16).

    Returns
    -------
    torch.Tensor
        Reversed multivector of shape (..., 16).
    """
    reversal_diag = _compute_reversal(device=x.device, dtype=x.dtype)
    return x * reversal_diag


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the geometric product f(x,y) = xy.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    # Select kernel on correct device
    gp = _load_bilinear_basis("gp", x.device, x.dtype)

    # Compute geometric product
    outputs = torch.einsum("i j k, ... j, ... k -> ... i", gp, x, y)

    return outputs


def outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the outer product `f(x,y) = x ^ y`.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x, y, and coeffs.
    """

    # Select kernel on correct device
    op = _load_bilinear_basis("op", x.device, x.dtype)

    # Compute geometric product
    outputs = torch.einsum("i j k, ... j, ... k -> ... i", op, x, y)

    return outputs.to(x.device)


def inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the inner product of multivectors f(x,y) = <x, y> = <~x y>_0.

    Sums over the 16 multivector dimensions.

    Equal to `geometric_product(reverse(x), y)[..., [0]]` (but faster).

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16) or (..., channels, 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16) or (..., channels, 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Result. Batch dimensions are result of broadcasting between x and y.
    """

    selector = INNER_PRODUCT_INDICES.to(x.device)
    x = x[..., selector]
    y = y[..., selector]

    outputs = torch.einsum("... i, ... i -> ...", x, y)

    # We want the output to have shape (..., 1)
    outputs = outputs.unsqueeze(-1)

    return outputs


def infinite_norm(x: torch.Tensor) -> torch.Tensor:
    """Computes the infinite norm inner product of multivectors f(x,y) = <x_ideal, y_ideal>.

    Sums over the 16 multivector dimensions.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16) or (..., channels, 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Result. Batch dimensions are result of broadcasting between x and y.
    """

    selector = INFINITE_NORM_INDICES.to(x.device)
    x = x[..., selector]

    outputs = torch.einsum("... i, ... i -> ...", x, x)
    outputs = outputs.unsqueeze(-1)

    return outputs


def norm_squared(x: torch.Tensor) -> torch.Tensor:
    """Computes the squared norm of a multivector f(x) = <x,reversal(x)>_0.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16) or (..., channels, 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Result.
    """
    return inner_product(x, x)


def dual(x: torch.Tensor) -> torch.Tensor:
    """Compute the right exterior complement in the fixed oriented basis.

    The PGA pseudoscalar is not invertible. This is not multiplication by
    I^{-1}, and the complement does not commute with translations.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result.
    """
    permutation, factors = _compute_dualization(device=x.device, dtype=x.dtype)
    x_permuted = x[..., permutation].to(device=x.device, dtype=x.dtype)
    outputs = x_permuted * factors

    return outputs


def anti_dual(x: torch.Tensor) -> torch.Tensor:
    """Compute the inverse exterior complement.

    Take the dual and multiply grades 1 and 3 by -1.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result.
    """
    dual_x = dual(x)
    # Multiply grades 1 and 3 by -1
    anti_dual_factors = _get_anti_dual_factors(device=x.device, dtype=x.dtype)
    outputs = dual_x * anti_dual_factors
    return outputs


def join(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes a join operation between two multivectors.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector.
    y : torch.Tensor with shape (..., 16)
        Second input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result.
    """
    return anti_dual(outer_product(dual(x), dual(y)))


def equivariant_join(x: torch.Tensor, y: torch.Tensor, ref: torch.Tensor = None) -> torch.Tensor:
    """Computes an equivariant join operation between two multivectors.
    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        First input multivector.
    y : torch.Tensor with shape (..., 16)
        Second input multivector.
    ref : torch.Tensor with shape (..., 16)
        Reference multivector transforming under the same versor action.
        Its e123 coefficient must change by det(R). Canonically re-embedded
        Cartesian points do not supply that sign automatically under reflection.
        Omitting ref uses a fixed canonical reference, with an SE(3) guarantee.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result.
    """
    if ref is None:
        ref = torch.ones_like(x)
    return ref[..., [14]] * join(x, y)


def equi_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Applies an equivariant linear map to a batch of multivectors.

    Parameters
    ----------
    x : torch.Tensor
        Input multivectors of shape (B, in_c, 16)
    weight : torch.Tensor
        Weight tensor of shape (out_c, in_c, 9)

    Returns
    -------
    torch.Tensor
        Output multivectors of shape (..., out_c, 16)
    """
    basis = _compute_pin_equilinear_basis(device=x.device, dtype=x.dtype, normalize=True)  # (9, 16, 16)
    return torch.einsum("yxa,aij,...xj->...yi", weight, basis, x)


def depthwise_equi_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Applies a depthwise equivariant linear map (no channel mixing).

    Parameters
    ----------
    x : torch.Tensor
        Input multivectors of shape (..., in_c, 16)
    weight : torch.Tensor
        Weight tensor of shape (in_c, 9)

    Returns
    -------
    torch.Tensor
        Output multivectors of shape (..., in_c, 16)
    """
    basis = _compute_pin_equilinear_basis(device=x.device, dtype=x.dtype, normalize=True)  # (9, 16, 16)

    return torch.einsum("xa,aij,...xj->...xi", weight, basis, x)


def gated_non_linearity(x: torch.Tensor, gates: torch.Tensor, f: callable) -> torch.Tensor:
    """Pin-equivariant gated non-linearity.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    NonLinearity(gates) * x.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Multivector input
    gates : torch.Tensor with shape (..., 1)
        Pin-invariant gates.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Computes NonLinearity(gates) * x, with broadcasting along the last dimension.
    """
    weights = f(gates)  # (..., 1)
    return weights * x  # (..., 16)


def grade_dropout(x: torch.Tensor, p: float = 0.5, training: bool = True):
    """Multivector dropout, dropping out grades independently.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input data.
    p : float
        Dropout probability (assumed the same for each grade).
    training : bool
        Switches between train-time and test-time behaviour.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Inputs with dropout applied.
    """

    # Project to grades
    x = grade_project(x)

    # Apply standard 1D dropout
    # For whatever reason, that only works with a single batch dimension, so let's reshape a bit
    h = x.view(-1, 5, 16)
    h = F.dropout1d(h, p=p, training=training, inplace=False)
    h = h.view(x.shape)

    # Combine grades again
    h = torch.sum(h, dim=-2)

    return h


def equi_layer_norm(
    x: torch.Tensor, channel_dim: int = -2, gain: float = 1.0, epsilon: float = 0.01
) -> torch.Tensor:
    """Equivariant LayerNorm for multivectors.

    Rescales input such that `mean_channels |inputs|^2 = 1`, where the norm is the GA norm and the
    mean goes over the channel dimensions.

    Using a factor `gain > 1` makes up for the fact that the GP norm overestimates the actual
    standard deviation of the input data.

    Parameters
    ----------
    x : torch.Tensor with shape `(batch_dim, *channel_dims, 16)`
        Input multivectors.
    channel_dim : int
        Channel dimension index. Defaults to the second-last entry (last are the multivector
        components).
    gain : float
        Target output scale.
    epsilon : float
        Small numerical factor to avoid instabilities. By default, we use a reasonably large number
        to balance issues that arise from some multivector components not contributing to the norm.

    Returns
    -------
    outputs : torch.Tensor with shape `(batch_dim, *channel_dims, 16)`
        Normalized inputs.
    """

    # Compute mean_channels |inputs|^2
    squared_norms = inner_product(x, x)
    squared_norms = torch.mean(squared_norms, dim=channel_dim, keepdim=True)

    # Insure against low-norm tensors (which can arise even when `x.var(dim=-1)` is high b/c some
    # entries don't contribute to the inner product / GP norm!)
    squared_norms = torch.clamp(squared_norms, epsilon)

    # Rescale inputs
    outputs = gain * x / torch.sqrt(squared_norms)

    return outputs


def embed_point(x: torch.Tensor, type: str = "trivector") -> torch.Tensor:
    """
    Embed 3D coordinates.
    Embeds as Grade-3 (Dual PGA Points). The type argument is retained for compatibility.
    x : (B, N, 3)
    returns : (B, N, 16)
    """
    mv = torch.zeros(*x.shape[:-1], 16, device=x.device, dtype=x.dtype)
    mv[..., 14] = 1.0  # homogeneous point weight
    mv[..., 13] = -x[..., 0]  # x-coordinate embedded in x_023
    mv[..., 12] = x[..., 1]  # y-coordinate embedded in x_013
    mv[..., 11] = -x[..., 2]  # z-coordinate embedded in x_012

    return mv


def extract_point(multivector: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
    """Given a multivector, extract any potential 3D point from the trivector components.

    Exact projective division is equivariant when |w| > threshold. The existing
    near-zero denominator clamp is retained; the equivariance guarantee does
    not apply where that clamp changes the homogeneous weight.
    """

    coordinates = torch.cat(
        [-multivector[..., [13]], multivector[..., [12]], -multivector[..., [11]]],
        dim=-1,
    )

    # Divide by embedding dim
    # Embedding dimension / scale of homogeneous coordinates
    embedding_dim = multivector[..., [14]]
    embedding_dim = torch.where(torch.abs(embedding_dim) > threshold, embedding_dim, threshold)
    coordinates = coordinates / embedding_dim

    return coordinates


def embed_translation(translation_vector: torch.Tensor) -> torch.Tensor:
    """Embeds a 3D translation in multivectors.

    In our convention, a translation vector is embedded into a combination of the scalar and
    bivector components.

    We have (in agreement with Eq. (82) of the reference below) that
    ```
    T(t) = 1 - e_0 / 2 (t_1 e_1 + t_2 e_2 + t_3 e_3) .
    ```

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
    https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    translation_vector : torch.Tensor with shape (..., 3)
        Vectorial amount of translation.

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = translation_vector.shape[:-1]
    multivector = torch.zeros(
        *batch_shape,
        16,
        dtype=translation_vector.dtype,
        device=translation_vector.device,
    )

    # Embedding into trivectors
    multivector[..., 0] = 1.0  # scalar
    # Translation vector embedded in x_0i with i = 1, 2, 3
    multivector[..., 5:8] = -0.5 * translation_vector[..., :]

    return multivector


def embed_scalar(scalars: torch.Tensor, mode="scalar") -> torch.Tensor:
    """Embeds a scalar tensor into multivectors.

    Parameters
    ----------
    scalars: torch.Tensor with shape (..., 1)
        Scalar inputs.

    Returns
    -------
    multivectors: torch.Tensor with shape (..., 16)
        Multivector outputs. `multivectors[..., [0]]` is the same as `scalars`. The other components
        are zero.
    """

    non_scalar_shape = list(scalars.shape[:-1]) + [15]
    non_scalar_components = torch.zeros(non_scalar_shape, device=scalars.device, dtype=scalars.dtype)
    if mode == "scalar":
        embedding = torch.cat((scalars, non_scalar_components), dim=-1)
    elif mode == "pseudoscalar":
        # pseudoscalar is last component
        embedding = torch.cat((non_scalar_components, scalars), dim=-1)

    return embedding


def embed_vector(x: torch.Tensor) -> torch.Tensor:
    """
    Embed 3D vectors into 16-D GA multivectors as pure grade-1 elements.
    x : (..., 3)
    returns : (..., 16)
    """
    mv = torch.zeros(*x.shape[:-1], 16, device=x.device, dtype=x.dtype)
    # Grade-1 Vector components
    mv[..., 1] = 1.0  # e0 component (homogeneous coordinate)
    mv[..., 2] = x[..., 0]  # e1 component (x-coord)
    mv[..., 3] = x[..., 1]  # e2 component (y-coord)
    mv[..., 4] = x[..., 2]  # e3 component (z-coord)
    return mv


def embed_pluecker_ray(pluecker_ray: torch.Tensor) -> torch.Tensor:
    """Embed ray in Plücker coordinates as a multivector.

    Plücker coords are (v, o x v) for ray through o in direction v.

    Args:
        pluecker_ray (Tensor): of shape [..., 6]

    Returns:
        Tensor: of shape [..., 16]
    """
    mv = torch.zeros(*pluecker_ray.shape[:-1], 16, device=pluecker_ray.device, dtype=pluecker_ray.dtype)
    mv[..., 5:8] = pluecker_ray[..., 3:6]
    mv[..., 8] = pluecker_ray[..., 2]
    mv[..., 9] = -pluecker_ray[..., 1]
    mv[..., 10] = pluecker_ray[..., 0]
    return mv


def embed_oriented_plane(normal: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Embeds an (oriented plane) in the PGA.

    Following L. Dorst, the plane is represent as PGA vectors.

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
        https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    normal : torch.Tensor with shape (..., 3)
        Normal to the plane.
    position : torch.Tensor with shape (..., 3)
        One position on the plane.

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = normal.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=normal.dtype, device=normal.device)

    # Embedding a plane through origin into vectors
    multivector[..., 2:5] = normal[..., :]

    # Shift away from origin by translating
    translation = embed_translation(position)
    inverse_translation = embed_translation(-position)
    multivector = geometric_product(
        geometric_product(translation, multivector), inverse_translation
    )

    return multivector
