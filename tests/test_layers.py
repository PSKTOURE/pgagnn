import torch

from src.layers import (
    BesselRadialBasisLayer,
    CentralityEncoding,
    EquiLayerNorm,
    EquiLinear,
    GatedNonLinearity,
    GaussianRadialBasisLayer,
    GeometricBilinear,
    GradeDropout,
    ResidualLayer,
)


def test_equilinear_dense_forward():
    torch.manual_seed(42)
    layer = EquiLinear(in_mvc=4, out_mvc=8, in_sc=2, out_sc=3, factorize=False)
    mv = torch.randn(2, 5, 4, 16)
    sc = torch.randn(2, 5, 2)
    out_mv, out_sc = layer(mv, sc)
    assert out_mv.shape == (2, 5, 8, 16)
    assert out_sc.shape == (2, 5, 3)


def test_equilinear_factorized_forward():
    torch.manual_seed(42)
    layer = EquiLinear(in_mvc=4, out_mvc=8, in_sc=2, out_sc=3, factorize=True)
    mv = torch.randn(2, 5, 4, 16)
    sc = torch.randn(2, 5, 2)
    out_mv, out_sc = layer(mv, sc)
    assert out_mv.shape == (2, 5, 8, 16)
    assert out_sc.shape == (2, 5, 3)


def test_equilinear_grade_modulation():
    torch.manual_seed(42)
    layer = EquiLinear(in_mvc=4, out_mvc=4, in_sc=2, out_sc=2, use_grade_modulation=True)
    mv = torch.randn(2, 5, 4, 16)
    sc = torch.randn(2, 5, 2)
    out_mv, out_sc = layer(mv, sc)
    assert out_mv.shape == (2, 5, 4, 16)
    assert out_sc.shape == (2, 5, 2)


def test_geometric_bilinear_forward():
    torch.manual_seed(42)
    layer = GeometricBilinear(in_mvc=4, out_mvc=8, in_sc=2, out_sc=4)
    mv = torch.randn(2, 5, 4, 16)
    sc = torch.randn(2, 5, 2)
    ref = torch.ones(2, 1, 1, 16)
    out_mv, out_sc = layer(mv, ref, sc)
    assert out_mv.shape == (2, 5, 8, 16)
    assert out_sc.shape == (2, 5, 4)


def test_equi_layer_norm():
    torch.manual_seed(42)
    norm = EquiLayerNorm(channel_dim=-2)
    mv = torch.randn(2, 5, 4, 16) * 10.0
    sc = torch.randn(2, 5, 8) * 10.0
    out_mv, out_sc = norm(mv, sc)
    assert out_mv.shape == (2, 5, 4, 16)
    assert out_sc.shape == (2, 5, 8)


def test_gated_non_linearity():
    torch.manual_seed(42)
    for act in ["gelu", "silu", "relu", "sigmoid"]:
        layer = GatedNonLinearity(activation=act)
        mv = torch.randn(2, 5, 4, 16)
        gates = torch.randn(2, 5, 4, 1)
        sc = torch.randn(2, 5, 2)
        out_mv, out_sc = layer(mv, gates, sc)
        assert out_mv.shape == (2, 5, 4, 16)
        assert out_sc.shape == (2, 5, 2)


def test_grade_dropout():
    torch.manual_seed(42)
    dropout = GradeDropout(dropout_prob=0.5)
    dropout.eval()
    mv = torch.randn(2, 5, 16)
    sc = torch.randn(2, 5, 2)
    out_mv, out_sc = dropout(mv, sc)
    assert torch.allclose(out_mv, mv)
    assert torch.allclose(out_sc, sc)


def test_gaussian_radial_basis_layer():
    torch.manual_seed(42)
    rbf = GaussianRadialBasisLayer(num_bases=32, cutoff=5.0)
    dist = torch.tensor([[0.5], [2.5], [4.9], [5.5]])
    out = rbf(dist)
    assert out.shape == (4, 32)
    # Values beyond cutoff should be 0 due to masking
    assert torch.allclose(out[3], torch.zeros(32))


def test_bessel_radial_basis_layer():
    torch.manual_seed(42)
    rbf = BesselRadialBasisLayer(num_bases=32, cutoff=5.0)
    dist = torch.tensor([[0.5], [2.5], [4.9], [5.5]])
    out = rbf(dist)
    assert out.shape == (4, 32)
    # Values beyond cutoff should be 0 due to envelope
    assert torch.allclose(out[3], torch.zeros(32))
    assert torch.isfinite(out).all()



def test_residual_layer():
    torch.manual_seed(42)
    res_same = ResidualLayer(in_mvc=4, out_mvc=4, in_sc=2, out_sc=2)
    mv_in = torch.randn(2, 5, 4, 16)
    mv_out = torch.randn(2, 5, 4, 16)
    sc_in = torch.randn(2, 5, 2)
    sc_out = torch.randn(2, 5, 2)
    combined_mv, combined_sc = res_same(mv_out, mv_in, sc_out, sc_in)
    assert torch.allclose(combined_mv, mv_out + mv_in)
    assert torch.allclose(combined_sc, sc_out + sc_in)


def test_centrality_encoding():
    torch.manual_seed(42)
    enc = CentralityEncoding(num_rbf=16, sc_dim=32, use_chemical_context=True)

    # Sparse
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    rbf = torch.randn(4, 16, requires_grad=True)
    sc = torch.randn(3, 32, requires_grad=True)
    out_sp = enc.forward_sparse(edge_index, rbf, num_nodes=3, sc=sc)
    assert out_sp.shape == (3, 32)
    out_sp.sum().backward()
    assert rbf.grad is not None and sc.grad is not None

    # Dense
    enc.zero_grad()
    rbf_d = torch.randn(2, 4, 4, 16, requires_grad=True)
    mask_2d = torch.ones(2, 4, 4, 1, dtype=torch.bool)
    sc_d = torch.randn(2, 4, 32, requires_grad=True)
    out_d = enc.forward_dense(rbf_d, mask_2d, sc=sc_d)
    assert out_d.shape == (2, 4, 32)
    out_d.sum().backward()
    assert rbf_d.grad is not None and sc_d.grad is not None

