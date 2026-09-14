import torch

from src.primitives import (
    depthwise_equi_linear,
    embed_point,
    embed_scalar,
    embed_vector,
    equi_linear,
    extract_point,
    geometric_product,
    grade_project,
    invariants,
    outer_product,
    reverse,
)


def test_embed_and_extract_point_roundtrip():
    torch.manual_seed(42)
    pos = torch.randn(4, 10, 3, dtype=torch.float64)
    mv = embed_point(pos)
    assert mv.shape == (4, 10, 16)
    recovered = extract_point(mv)
    assert torch.allclose(recovered, pos, atol=1e-7)


def test_embed_scalar():
    sc = torch.tensor([[1.5], [2.5], [-3.0]])
    mv = embed_scalar(sc)
    assert mv.shape == (3, 16)
    assert torch.allclose(mv[:, 0:1], sc)
    assert torch.allclose(mv[:, 1:], torch.zeros(3, 15))


def test_embed_vector():
    vec = torch.tensor([[1.0, 2.0, 3.0]])
    mv = embed_vector(vec)
    assert mv.shape == (1, 16)
    assert mv[0, 1] == 1.0  # homogeneous e0
    assert torch.allclose(mv[0, 2:5], vec[0])


def test_geometric_product_identity():
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)
    one = torch.zeros(2, 4, 16)
    one[..., 0] = 1.0  # scalar 1
    prod = geometric_product(one, x)
    assert torch.allclose(prod, x, atol=1e-6)


def test_geometric_product_associativity():
    torch.manual_seed(42)
    x = torch.randn(2, 3, 16, dtype=torch.float64)
    y = torch.randn(2, 3, 16, dtype=torch.float64)
    z = torch.randn(2, 3, 16, dtype=torch.float64)
    xy_z = geometric_product(geometric_product(x, y), z)
    x_yz = geometric_product(x, geometric_product(y, z))
    assert torch.allclose(xy_z, x_yz, atol=1e-6)


def test_outer_product_antisymmetry_for_vectors():
    torch.manual_seed(42)
    u_vec = torch.randn(5, 3)
    v_vec = torch.randn(5, 3)
    u = embed_vector(u_vec)
    v = embed_vector(v_vec)
    uv = outer_product(u, v)
    vu = outer_product(v, u)
    assert torch.allclose(uv, -vu, atol=1e-6)


def test_outer_product_self_is_zero_for_vectors():
    torch.manual_seed(42)
    u_vec = torch.randn(5, 3)
    u = embed_vector(u_vec)
    uu = outer_product(u, u)
    assert torch.allclose(uu, torch.zeros_like(uu), atol=1e-6)


def test_reversal_properties():
    torch.manual_seed(42)
    x = torch.randn(3, 16, dtype=torch.float64)
    y = torch.randn(3, 16, dtype=torch.float64)
    # Double reversal is identity
    assert torch.allclose(reverse(reverse(x)), x, atol=1e-7)
    # Reversal of product is product of reversals reversed
    rev_xy = reverse(geometric_product(x, y))
    ry_rx = geometric_product(reverse(y), reverse(x))
    assert torch.allclose(rev_xy, ry_rx, atol=1e-6)


def test_grade_projection_reconstruction():
    torch.manual_seed(42)
    x = torch.randn(4, 8, 16)
    proj = grade_project(x)
    assert proj.shape == (4, 8, 5, 16)
    reconstructed = torch.sum(proj, dim=-2)
    assert torch.allclose(reconstructed, x, atol=1e-6)


def test_invariants_shape():
    torch.manual_seed(42)
    x = torch.randn(2, 6, 16)
    inv = invariants(x)
    assert inv.shape == (2, 6, 5)


def test_equi_linear_shape():
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)  # (B, in_c, 16)
    weight = torch.randn(8, 4, 9)  # (out_c, in_c, 9)
    out = equi_linear(x, weight)
    assert out.shape == (2, 8, 16)


def test_depthwise_equi_linear_shape():
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)  # (B, in_c, 16)
    weight = torch.randn(4, 9)  # (in_c, 9)
    out = depthwise_equi_linear(x, weight)
    assert out.shape == (2, 4, 16)
