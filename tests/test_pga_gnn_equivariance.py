"""
Numerical E(3) equivariance tests for PGA_GNN.

Unlike GGNN, which operates directly on 3D vectors/scalars (so rotations act
on those tensors in the obvious way), PGA_GNN operates on 16-dim projective
geometric algebra (PGA) multivectors. We don't attempt to test equivariance
of those internal multivectors directly -- that would mean reimplementing how
PGA represents rotations/reflections just to check it. Instead we test the
whole pipeline actually used at training time -- embed raw (mass, position,
velocity) -> PGA_GNN -> extract predicted point -- as one black box, in plain
R^3 space, and check that rotating/translating the raw inputs commutes with
running that pipeline. `predict_points` below is a direct copy of the dense
GA forward pass in `_forward_ga_dense` from the training code, scoped down to
PGA_GNN, so this tests exactly what training runs.

Checks:
  1. Rotation equivariance   (proper orthogonal Q, det = +1)
  2. Reflection equivariance (improper orthogonal Q, det = -1)
     -> Evaluated both without edge attributes (core GA message-passing layers)
        and with algebraic bivector edge attributes constructed via the PGA
        outer product: edge_attr_mv = outer_product(pos_i, pos_j). Exact
        reflection equivariance is preserved across both modes.
  3. Translation equivariance: PGA_GNN predicts *absolute* future positions
     (not a relative displacement added back on afterwards, unlike GGNN), so
     shifting every input position by a constant t should shift every
     predicted output point by that same t.
  4. A negative control, as in the GGNN tests: compare rotated-input output
     against un-rotated-output without correcting for the rotation, and
     confirm that mismatches (i.e. the harness itself is discriminating).

All tests run in float64 and in eval() mode (dropout off) to keep numerical
noise well below the equivariance-violation scale you'd get from a real bug.
"""

import torch

from src.layers import (
    GaussianRadialBasisLayer,
)
from src.pgagnn import PGA_GNN
from src.primitives import (
    embed_point,
    embed_scalar,
    embed_translation,
    equivariant_join,
    extract_point,
)

torch.manual_seed(0)

EDGE_SC_DIST_CHANNELS = 4  # channels the radial basis embedder produces


def random_orthogonal_matrix(det_sign: int = 1) -> torch.Tensor:
    """Return a random O(3) matrix with the requested determinant sign (+1 or -1)."""
    a = torch.randn(3, 3, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # Fix sign ambiguity from QR so Q is uniformly distributed over O(3)
    d = torch.diagonal(r).sign()
    q = q * d
    if torch.det(q).sign().item() != det_sign:
        # Flip one column to change the determinant's sign without changing orthogonality
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
        factorize=True,
        use_grade_modulation=False,
        use_pseudoscalar=False,
    )
    return model.double().eval()


def random_physical_inputs(B=2, N=6, seed=None):
    """Mass, position, velocity as raw (B, N, ...) tensors -- the same layout
    NBodyDataset produces, before `embed_inputs` packs them into multivectors.
    """
    if seed is not None:
        torch.manual_seed(seed)
    mass = torch.rand(B, N, 1, dtype=torch.float64) + 0.1
    pos = torch.randn(B, N, 3, dtype=torch.float64)
    vel = torch.randn(B, N, 3, dtype=torch.float64)
    adj = torch.ones(B, N, N, dtype=torch.float64)
    return mass, pos, vel, adj


def apply_rotation(x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """x has shape (..., 3) in the last dim. Rotate every 3-vector by Q."""
    return x @ Q.T


def predict_points(model, mass, pos, vel, adj, use_edge_attr, dist_embedder=None):
    """Runs the same dense GA forward pass as `_forward_ga_dense` in the
    training code, scoped to PGA_GNN, and returns predicted points in raw
    R^3 space -- this is the function whose E(3) equivariance we're testing.
    """
    m_embedded = embed_scalar(mass)
    x_embedded = embed_point(pos).squeeze()
    v_embedded = embed_translation(vel)
    inputs_emb = (m_embedded + x_embedded + v_embedded).unsqueeze(2)  # (B, N, 1, 16)

    B, N = mass.shape[:2]
    sc = torch.zeros(B, N, 1, dtype=mass.dtype)
    ref = inputs_emb.mean(dim=(1, 2), keepdim=True)

    edge_attr_mv, edge_attr_sc = None, None
    if use_edge_attr:
        pos_embedded = embed_point(pos).squeeze()  # (B, N, 16)
        src = pos.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, 3)
        dst = pos.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, 3)
        u = dst - src
        src_embedded = pos_embedded.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, 16)
        dst_embedded = pos_embedded.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, 16)
        edge_attr_mv = equivariant_join(src_embedded, dst_embedded, ref).unsqueeze(-2)  # (B, N, N, 1, 16)
        rel_norm = torch.linalg.norm(u, dim=-1, keepdim=True) + 1e-8
        rel_norm = dist_embedder(rel_norm)  # (B, N, N, edge_sc)
        edge_attr_sc = rel_norm

    mv, _ = model(inputs_emb, ref=ref, sc=sc, adj=adj, edge_attr_mv=edge_attr_mv, edge_attr_sc=edge_attr_sc)
    return extract_point(mv[:, :, 0, :])


def _make_dist_embedder(use_edge_attr):
    if not use_edge_attr:
        return None
    return GaussianRadialBasisLayer(num_bases=EDGE_SC_DIST_CHANNELS).double()


@torch.no_grad()
def check_rotation_equivariance(use_edge_attr: bool, det_sign: int, atol: float = 1e-8, rtol: float = 1e-6):
    label = f"{'ROTATION' if det_sign == 1 else 'REFLECTION'} (edge_attr={use_edge_attr})"
    edge_sc = EDGE_SC_DIST_CHANNELS if use_edge_attr else None
    model = build_model(edge_sc=edge_sc)
    dist_embedder = _make_dist_embedder(use_edge_attr)

    mass, pos, vel, adj = random_physical_inputs(seed=42)
    Q = random_orthogonal_matrix(det_sign=det_sign)

    # --- Path A: transform inputs, then run the pipeline ---
    pts_A = predict_points(
        model, mass, apply_rotation(pos, Q), apply_rotation(vel, Q), adj, use_edge_attr, dist_embedder
    )

    # --- Path B: run the pipeline, then transform the output ---
    pts_B = predict_points(model, mass, pos, vel, adj, use_edge_attr, dist_embedder)
    pts_B_rot = apply_rotation(pts_B, Q)

    ok = torch.allclose(pts_A, pts_B_rot, atol=atol, rtol=rtol)
    err = (pts_A - pts_B_rot).abs().max().item()
    print(f"[{label}] predicted points equivariant: {ok}  (max abs err = {err:.3e})")
    return ok, err


@torch.no_grad()
def check_translation_equivariance(use_edge_attr: bool, atol: float = 1e-8, rtol: float = 1e-6):
    """PGA_GNN predicts absolute positions, so shifting all inputs by a
    constant t should shift the predicted points by that same t (contrast
    with the GGNN test, which checks translation *invariance* because that
    model predicts a relative displacement added to the position outside
    the network).
    """
    edge_sc = EDGE_SC_DIST_CHANNELS if use_edge_attr else None
    model = build_model(edge_sc=edge_sc)
    dist_embedder = _make_dist_embedder(use_edge_attr)

    mass, pos, vel, adj = random_physical_inputs(seed=7)
    t = torch.randn(3, dtype=torch.float64) * 10.0  # arbitrary translation

    pts_shifted = predict_points(model, mass, pos + t, vel, adj, use_edge_attr, dist_embedder)
    pts_base = predict_points(model, mass, pos, vel, adj, use_edge_attr, dist_embedder)

    ok = torch.allclose(pts_shifted, pts_base + t, atol=atol, rtol=rtol)
    err = (pts_shifted - (pts_base + t)).abs().max().item()
    print(
        f"[TRANSLATION] (edge_attr={use_edge_attr}) predicted points equivariant: {ok}  (max abs err = {err:.3e})"
    )
    return ok, err


@torch.no_grad()
def negative_control(use_edge_attr: bool = False):
    """Sanity check on the test harness itself: rotate the inputs but DO NOT
    rotate the pipeline's output before comparing. This should almost always
    fail. If it doesn't, the tolerances are too loose to mean anything.
    """
    edge_sc = EDGE_SC_DIST_CHANNELS if use_edge_attr else None
    model = build_model(edge_sc=edge_sc)
    dist_embedder = _make_dist_embedder(use_edge_attr)

    mass, pos, vel, adj = random_physical_inputs(seed=42)
    Q = random_orthogonal_matrix(det_sign=1)

    pts_A = predict_points(
        model, mass, apply_rotation(pos, Q), apply_rotation(vel, Q), adj, use_edge_attr, dist_embedder
    )
    pts_B = predict_points(model, mass, pos, vel, adj, use_edge_attr, dist_embedder)

    # Compare WITHOUT rotating pts_B -- expect this to differ
    mismatched = not torch.allclose(pts_A, pts_B, atol=1e-9, rtol=1e-7)
    print(f"[NEGATIVE CONTROL] un-rotated comparison correctly differs: {mismatched}")
    assert mismatched, "Negative control did not trigger -- test harness may be vacuous!"


def test_rotation_equivariance_without_edge_attributes():
    ok, _ = check_rotation_equivariance(use_edge_attr=False, det_sign=1)
    assert ok


def test_reflection_equivariance_without_edge_attributes():
    ok, _ = check_rotation_equivariance(use_edge_attr=False, det_sign=-1)
    assert ok


def test_translation_equivariance_without_edge_attributes():
    ok, _ = check_translation_equivariance(use_edge_attr=False)
    assert ok


def test_rotation_equivariance_with_edge_attributes():
    ok, _ = check_rotation_equivariance(use_edge_attr=True, det_sign=1)
    assert ok


def test_translation_equivariance_with_edge_attributes():
    ok, _ = check_translation_equivariance(use_edge_attr=True)
    assert ok


def test_negative_control():
    negative_control(use_edge_attr=False)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
