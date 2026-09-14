import torch

from src.layers import GaussianRadialBasisLayer
from src.pgagnn import PGA_GNN
from src.primitives import (
    embed_point,
    embed_scalar,
    embed_translation,
    extract_point,
    outer_product,
)

EDGE_SC_DIST_CHANNELS = 4


def random_orthogonal_matrix(det_sign: int = 1) -> torch.Tensor:
    """Return a random O(3) matrix with the requested determinant sign (+1 or -1)."""
    a = torch.randn(3, 3, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    d = torch.diagonal(r).sign()
    q = q * d
    if torch.det(q).sign().item() != det_sign:
        q[:, 0] *= -1
    assert torch.allclose(torch.det(q), torch.tensor(float(det_sign), dtype=torch.float64), atol=1e-6)
    return q


def build_model(edge_sc=None, attention_type="gatr"):
    model = PGA_GNN(
        in_mvc=1,
        hidden_mvc=8,
        out_mvc=1,
        in_sc=1,
        hidden_sc=8,
        out_sc=1,
        edge_mvc=1 if edge_sc is not None else None,
        edge_sc=edge_sc,
        num_layers=2,
        num_heads=2,
        dropout_prob=0.0,
        attention_type=attention_type,
    )
    return model.double().eval()


def random_physical_inputs(B=2, N=6, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    mass = torch.rand(B, N, 1, dtype=torch.float64) + 0.1
    pos = torch.randn(B, N, 3, dtype=torch.float64)
    vel = torch.randn(B, N, 3, dtype=torch.float64)
    adj = torch.ones(B, N, N, dtype=torch.float64)
    return mass, pos, vel, adj


def apply_rotation(x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    return x @ Q.T


def predict_points(model, mass, pos, vel, adj, use_edge_attr, dist_embedder=None):
    m_embedded = embed_scalar(mass)
    x_embedded = embed_point(pos).squeeze()
    v_embedded = embed_translation(vel)
    inputs_emb = (m_embedded + x_embedded + v_embedded).unsqueeze(2)

    B, N = mass.shape[:2]
    sc = torch.zeros(B, N, 1, dtype=mass.dtype)
    ref = inputs_emb.mean(dim=(1, 2), keepdim=True)

    edge_attr_mv, edge_attr_sc = None, None
    if use_edge_attr:
        pos_embedded = embed_point(pos).squeeze()
        src = pos.unsqueeze(2).expand(-1, -1, N, -1)
        dst = pos.unsqueeze(1).expand(-1, N, -1, -1)
        u = dst - src
        src_embedded = pos_embedded.unsqueeze(2).expand(-1, -1, N, -1)
        dst_embedded = pos_embedded.unsqueeze(1).expand(-1, N, -1, -1)
        edge_attr_mv = outer_product(src_embedded, dst_embedded).unsqueeze(-2)
        rel_norm = torch.linalg.norm(u, dim=-1, keepdim=True) + 1e-8
        rel_norm = dist_embedder(rel_norm)
        edge_attr_sc = rel_norm

    mv, _ = model(inputs_emb, ref=ref, sc=sc, adj=adj, edge_attr_mv=edge_attr_mv, edge_attr_sc=edge_attr_sc)
    return extract_point(mv[:, :, 0, :])


def _make_dist_embedder(use_edge_attr):
    if not use_edge_attr:
        return None
    return GaussianRadialBasisLayer(num_bases=EDGE_SC_DIST_CHANNELS).double()


@torch.no_grad()
def test_rotation_equivariance_no_edge_attr():
    torch.manual_seed(42)
    model = build_model(edge_sc=None)
    dist_embedder = None
    mass, pos, vel, adj = random_physical_inputs(seed=42)
    Q = random_orthogonal_matrix(det_sign=1)

    pts_A = predict_points(
        model, mass, apply_rotation(pos, Q), apply_rotation(vel, Q), adj, False, dist_embedder
    )
    pts_B = predict_points(model, mass, pos, vel, adj, False, dist_embedder)
    pts_B_rot = apply_rotation(pts_B, Q)

    assert torch.allclose(pts_A, pts_B_rot, atol=1e-8, rtol=1e-6)


@torch.no_grad()
def test_reflection_equivariance_no_edge_attr():
    torch.manual_seed(42)
    model = build_model(edge_sc=None)
    dist_embedder = None
    mass, pos, vel, adj = random_physical_inputs(seed=42)
    Q = random_orthogonal_matrix(det_sign=-1)

    pts_A = predict_points(
        model, mass, apply_rotation(pos, Q), apply_rotation(vel, Q), adj, False, dist_embedder
    )
    pts_B = predict_points(model, mass, pos, vel, adj, False, dist_embedder)
    pts_B_rot = apply_rotation(pts_B, Q)

    assert torch.allclose(pts_A, pts_B_rot, atol=1e-8, rtol=1e-6)


@torch.no_grad()
def test_translation_equivariance_no_edge_attr():
    torch.manual_seed(7)
    model = build_model(edge_sc=None)
    dist_embedder = None
    mass, pos, vel, adj = random_physical_inputs(seed=7)
    t = torch.randn(3, dtype=torch.float64) * 10.0

    pts_shifted = predict_points(model, mass, pos + t, vel, adj, False, dist_embedder)
    pts_base = predict_points(model, mass, pos, vel, adj, False, dist_embedder)

    assert torch.allclose(pts_shifted, pts_base + t, atol=1e-8, rtol=1e-6)


@torch.no_grad()
def test_rotation_equivariance_with_edge_attr():
    torch.manual_seed(42)
    edge_sc = EDGE_SC_DIST_CHANNELS
    model = build_model(edge_sc=edge_sc)
    dist_embedder = _make_dist_embedder(True)
    mass, pos, vel, adj = random_physical_inputs(seed=42)
    Q = random_orthogonal_matrix(det_sign=1)

    pts_A = predict_points(
        model, mass, apply_rotation(pos, Q), apply_rotation(vel, Q), adj, True, dist_embedder
    )
    pts_B = predict_points(model, mass, pos, vel, adj, True, dist_embedder)
    pts_B_rot = apply_rotation(pts_B, Q)

    assert torch.allclose(pts_A, pts_B_rot, atol=1e-8, rtol=1e-6)


@torch.no_grad()
def test_translation_equivariance_with_edge_attr():
    torch.manual_seed(7)
    edge_sc = EDGE_SC_DIST_CHANNELS
    model = build_model(edge_sc=edge_sc)
    dist_embedder = _make_dist_embedder(True)
    mass, pos, vel, adj = random_physical_inputs(seed=7)
    t = torch.randn(3, dtype=torch.float64) * 10.0

    pts_shifted = predict_points(model, mass, pos + t, vel, adj, True, dist_embedder)
    pts_base = predict_points(model, mass, pos, vel, adj, True, dist_embedder)

    assert torch.allclose(pts_shifted, pts_base + t, atol=1e-8, rtol=1e-6)


@torch.no_grad()
def test_negative_control():
    model = build_model(edge_sc=None)
    mass, pos, vel, adj = random_physical_inputs(seed=42)
    Q = random_orthogonal_matrix(det_sign=1)

    pts_A = predict_points(model, mass, apply_rotation(pos, Q), apply_rotation(vel, Q), adj, False, None)
    pts_B = predict_points(model, mass, pos, vel, adj, False, None)

    # Comparing rotated output with unrotated output without correction must fail
    assert not torch.allclose(pts_A, pts_B, atol=1e-9, rtol=1e-7)
