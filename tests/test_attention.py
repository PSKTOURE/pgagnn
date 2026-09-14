import torch

from src.attention import (
    GADualNormAttention,
    GAGeometricAttention,
    GASimilarityAttention,
    GATrAttention,
    GATrAttentionSparse,
    get_attention_module,
)


def test_gatr_attention_shapes():
    attn = GATrAttention(out_mvc=4, out_sc=2)
    Q_mv = torch.randn(2, 5, 4, 16)
    K_mv = torch.randn(2, 5, 4, 16)
    Q_sc = torch.randn(2, 5, 2)
    K_sc = torch.randn(2, 5, 2)
    scores = attn(Q_mv, K_mv, Q_sc, K_sc)
    assert scores.shape == (2, 5, 5)


def test_gatr_attention_multihead():
    attn = GATrAttention(out_mvc=4, out_sc=2)
    Q_mv_mh = torch.randn(2, 5, 4, 4, 16)
    K_mv_mh = torch.randn(2, 5, 1, 4, 16)
    Q_sc_mh = torch.randn(2, 5, 4, 2)
    K_sc_mh = torch.randn(2, 5, 1, 2)
    scores = attn(Q_mv_mh, K_mv_mh, Q_sc_mh, K_sc_mh)
    assert scores.shape == (2, 5, 5, 4)


def test_gatr_sparse_attention_shapes():
    attn = GATrAttentionSparse(out_mvc=4, out_sc=2)
    num_nodes = 10
    num_edges = 25
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    Q_mv = torch.randn(num_nodes, 4, 16)
    K_mv = torch.randn(num_nodes, 4, 16)
    Q_sc = torch.randn(num_nodes, 2)
    K_sc = torch.randn(num_nodes, 2)
    scores = attn(Q_mv, K_mv, edge_index=edge_index, Q_sc=Q_sc, K_sc=K_sc)
    assert scores.shape == (num_edges,)


def test_similarity_attention():
    attn = GASimilarityAttention(out_mvc=4, out_sc=2)
    Q_mv = torch.randn(2, 5, 4, 16)
    K_mv = torch.randn(2, 5, 4, 16)
    Q_sc = torch.randn(2, 5, 2)
    K_sc = torch.randn(2, 5, 2)
    scores = attn(Q_mv, K_mv, Q_sc, K_sc)
    assert scores.shape == (2, 5, 5)

    # Multi-head
    Q_mv_mh = torch.randn(2, 5, 4, 4, 16)
    K_mv_mh = torch.randn(2, 5, 1, 4, 16)
    Q_sc_mh = torch.randn(2, 5, 4, 2)
    K_sc_mh = torch.randn(2, 5, 1, 2)
    scores_mh = attn(Q_mv_mh, K_mv_mh, Q_sc_mh, K_sc_mh)
    assert scores_mh.shape == (2, 5, 5, 4)


def test_geometric_attention():
    attn = GAGeometricAttention(out_mvc=4, out_sc=2, num_grades=5)
    Q_mv = torch.randn(2, 5, 4, 16)
    K_mv = torch.randn(2, 5, 4, 16)
    Q_sc = torch.randn(2, 5, 2)
    K_sc = torch.randn(2, 5, 2)
    scores = attn(Q_mv, K_mv, Q_sc, K_sc)
    assert scores.shape == (2, 5, 5)

    # Without scalars
    attn_no_sc = GAGeometricAttention(out_mvc=4, out_sc=None, num_grades=5)
    scores_no_sc = attn_no_sc(Q_mv, K_mv)
    assert scores_no_sc.shape == (2, 5, 5)


def test_dual_norm_attention():
    attn = GADualNormAttention(out_mvc=4, out_sc=2, num_grades=5, hidden_dim=64)
    Q_mv = torch.randn(2, 5, 4, 16)
    K_mv = torch.randn(2, 5, 4, 16)
    Q_sc = torch.randn(2, 5, 2)
    K_sc = torch.randn(2, 5, 2)
    scores = attn(Q_mv, K_mv, Q_sc, K_sc)
    assert scores.shape == (2, 5, 5)

    # Multi-head
    Q_mv_mh = torch.randn(2, 5, 4, 4, 16)
    K_mv_mh = torch.randn(2, 5, 1, 4, 16)
    Q_sc_mh = torch.randn(2, 5, 4, 2)
    K_sc_mh = torch.randn(2, 5, 1, 2)
    scores_mh = attn(Q_mv_mh, K_mv_mh, Q_sc_mh, K_sc_mh)
    assert scores_mh.shape == (2, 5, 5, 4)


def test_get_attention_module_factory():
    types = ["gatr", "gatr_sparse", "similarity", "geometric", "dual_norm"]
    for t in types:
        module = get_attention_module(t, out_mvc=4, out_sc=2)
        assert module is not None
