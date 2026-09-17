import torch

from src.ggnn import GGNN
from src.pgagnn import PGA_GNN


def test_pga_gnn_dense_forward_and_backward():
    torch.manual_seed(42)
    model = PGA_GNN(
        in_mvc=1,
        out_mvc=1,
        hidden_mvc=8,
        in_sc=2,
        out_sc=2,
        hidden_sc=16,
        edge_mvc=1,
        edge_sc=8,
        num_layers=2,
        num_heads=2,
        attention_type="gatr",
    )

    B, N = 2, 5
    mv = torch.randn(B, N, 1, 16, requires_grad=True)
    sc = torch.randn(B, N, 2, requires_grad=True)
    adj = torch.ones(B, N, N)
    edge_attr_mv = torch.randn(B, N, N, 1, 16)
    edge_attr_sc = torch.randn(B, N, N, 8)

    out_mv, out_sc = model(
        mv=mv,
        sc=sc,
        adj=adj,
        edge_attr_mv=edge_attr_mv,
        edge_attr_sc=edge_attr_sc,
    )

    assert out_mv.shape == (B, N, 1, 16)
    assert out_sc.shape == (B, N, 2)

    loss = out_mv.sum() + out_sc.sum()
    loss.backward()
    assert mv.grad is not None
    assert sc.grad is not None


def test_pga_gnn_sparse_forward_and_backward():
    torch.manual_seed(42)
    model = PGA_GNN(
        in_mvc=1,
        out_mvc=1,
        hidden_mvc=8,
        in_sc=2,
        out_sc=2,
        hidden_sc=16,
        edge_mvc=1,
        edge_sc=8,
        num_layers=2,
        num_heads=2,
        attention_type="gatr_sparse",
    )

    num_nodes = 12
    num_edges = 30
    mv = torch.randn(num_nodes, 1, 16, requires_grad=True)
    sc = torch.randn(num_nodes, 2, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr_mv = torch.randn(num_edges, 1, 16)
    edge_attr_sc = torch.randn(num_edges, 8)
    batch = torch.tensor([0] * 6 + [1] * 6)

    out_mv, out_sc = model(
        mv=mv,
        sc=sc,
        edge_index=edge_index,
        edge_attr_mv=edge_attr_mv,
        edge_attr_sc=edge_attr_sc,
        batch=batch,
    )

    assert out_mv.shape == (num_nodes, 1, 16)
    assert out_sc.shape == (num_nodes, 2)

    loss = out_mv.sum() + out_sc.sum()
    loss.backward()
    assert mv.grad is not None
    assert sc.grad is not None


def test_ggnn_forward_and_backward():
    torch.manual_seed(42)
    model = GGNN(
        in_vc=1,
        in_sc=2,
        hidden_vc=8,
        hidden_sc=16,
        out_vc=1,
        out_sc=2,
        edge_vc=1,
        edge_sc=8,
        num_layers=2,
        num_heads=2,
    )

    B, N = 2, 6
    vectors = torch.randn(B, N, 1, 3, requires_grad=True)
    scalars = torch.randn(B, N, 2, requires_grad=True)
    edge_vectors = torch.randn(B, N, N, 1, 3)
    edge_scalars = torch.randn(B, N, N, 8)
    adj = torch.ones(B, N, N)

    out_v, out_s = model(
        vectors=vectors,
        scalars=scalars,
        edge_vectors=edge_vectors,
        edge_scalars=edge_scalars,
        adj=adj,
    )

    assert out_v.shape == (B, N, 1, 3)
    assert out_s.shape == (B, N, 2)

    loss = out_v.sum() + out_s.sum()
    loss.backward()
    assert vectors.grad is not None
    assert scalars.grad is not None


