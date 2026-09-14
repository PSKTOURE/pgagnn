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


def test_md17_gamodel_energy_and_forces():
    from torch_geometric.data import Batch, Data

    from experiments.md17.train_md17_energy_based import (
        EnergyPredictor,
        GAModel,
        get_model,
        predict_energy_and_forces,
    )

    torch.manual_seed(42)
    device = torch.device("cpu")

    base_model = get_model(
        model_name="pgagnn",
        in_mvc=1,
        out_mvc=4,
        hidden_mvc=8,
        in_sc=16,
        out_sc=16,
        hidden_sc=32,
        edge_mvc=1,
        edge_sc=16,
        num_layers=2,
        num_heads=2,
    ).to(device)

    head = EnergyPredictor(
        mv_channels=4,
        sc_channels=16,
        hidden=32,
        num_elements=10,
        use_atomic_energy=True,
    ).to(device)

    model = GAModel(
        base_model=base_model,
        head=head,
        sc_dim=16,
        edge_rbf_dim=16,
        edge_sc_hidden=16,
        cutoff=5.0,
        rbf_type="bessel",
        use_edge_chemical_context=True,
    ).to(device)

    # Create dummy molecular batch (2 molecules with 5 and 4 atoms)
    data1 = Data(
        pos=torch.randn(5, 3),
        z=torch.tensor([6, 1, 1, 1, 8], dtype=torch.long),
        energy=torch.tensor([-150.0]),
        force=torch.randn(5, 3),
        edge_index=torch.tensor([[0, 1, 1, 2, 0, 4], [1, 0, 2, 1, 4, 0]], dtype=torch.long),
    )
    data2 = Data(
        pos=torch.randn(4, 3),
        z=torch.tensor([6, 1, 1, 1], dtype=torch.long),
        energy=torch.tensor([-120.0]),
        force=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 1, 2, 0, 3], [1, 0, 2, 1, 3, 0]], dtype=torch.long),
    )
    batch = Batch.from_data_list([data1, data2])

    mu_E = torch.tensor(-20.0)
    sigma_F = torch.tensor(1.5)

    (
        pred_energy_norm,
        pred_energy_denorm,
        target_energy,
        pred_forces_norm,
        pred_forces_physical,
        target_forces,
        mask,
    ) = predict_energy_and_forces(
        model_name="pgagnn",
        model=model,
        batch=batch,
        device=device,
        mu_E=mu_E,
        sigma_F=sigma_F,
        use_sparse=False,
    )

    assert pred_energy_norm.shape == (2,)
    assert pred_forces_norm.shape == (2, 5, 3)
    assert torch.isfinite(pred_energy_norm).all()
    assert torch.isfinite(pred_forces_norm).all()

    # Test backward pass
    loss = pred_energy_norm.sum() + pred_forces_norm.sum()
    loss.backward()

    # Verify gradients flowed into base model and atomic energy
    assert head.atomic_energy.weight.grad is not None
    assert any(p.grad is not None for p in model.parameters())


def test_md17_gamodel_sparse_energy_and_forces():
    from torch_geometric.data import Batch, Data

    from experiments.md17.train_md17_energy_based import (
        EnergyPredictor,
        GAModel,
        get_model,
        predict_energy_and_forces,
    )

    torch.manual_seed(42)
    device = torch.device("cpu")

    base_model = get_model(
        model_name="pgagnn",
        in_mvc=1,
        out_mvc=4,
        hidden_mvc=8,
        in_sc=16,
        out_sc=16,
        hidden_sc=32,
        edge_mvc=1,
        edge_sc=16,
        num_layers=2,
        num_heads=2,
        attention_type="gatr_sparse",
    ).to(device)
    base_model.use_sparse = True

    head = EnergyPredictor(
        mv_channels=4,
        sc_channels=16,
        hidden=32,
        num_elements=10,
        use_atomic_energy=True,
    ).to(device)

    model = GAModel(
        base_model=base_model,
        head=head,
        sc_dim=16,
        edge_rbf_dim=16,
        edge_sc_hidden=16,
        cutoff=5.0,
        rbf_type="bessel",
        use_edge_chemical_context=True,
    ).to(device)

    data1 = Data(
        pos=torch.randn(5, 3),
        z=torch.tensor([6, 1, 1, 1, 8], dtype=torch.long),
        energy=torch.tensor([-150.0]),
        force=torch.randn(5, 3),
        edge_index=torch.tensor([[0, 1, 1, 2, 0, 4], [1, 0, 2, 1, 4, 0]], dtype=torch.long),
    )
    data2 = Data(
        pos=torch.randn(4, 3),
        z=torch.tensor([6, 1, 1, 1], dtype=torch.long),
        energy=torch.tensor([-120.0]),
        force=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 1, 2, 0, 3], [1, 0, 2, 1, 3, 0]], dtype=torch.long),
    )
    batch = Batch.from_data_list([data1, data2])

    mu_E = torch.tensor(-20.0)
    sigma_F = torch.tensor(1.5)

    (
        pred_energy_norm,
        pred_energy_denorm,
        target_energy,
        pred_forces_norm,
        pred_forces_physical,
        target_forces,
        mask,
    ) = predict_energy_and_forces(
        model_name="pgagnn",
        model=model,
        batch=batch,
        device=device,
        mu_E=mu_E,
        sigma_F=sigma_F,
        use_sparse=True,
    )

    assert pred_energy_norm.shape == (2,)
    assert pred_forces_norm.shape == (2, 5, 3)
    assert torch.isfinite(pred_energy_norm).all()
    assert torch.isfinite(pred_forces_norm).all()

    loss = pred_energy_norm.sum() + pred_forces_norm.sum()
    loss.backward()

    assert head.atomic_energy.weight.grad is not None
    assert any(p.grad is not None for p in model.parameters())


def test_energy_predictors_dense_and_sparse():
    from experiments.md17.train_md17_energy_based import (
        EnergyPredictor,
        GGNNEnergyPredictor,
        ScalarEnergyPredictor,
        create_energy_head,
    )

    torch.manual_seed(42)

    # 1. ScalarEnergyPredictor (e.g. EGNN / SEGNN)
    scalar_head = ScalarEnergyPredictor(in_channels=16, hidden=32, num_elements=10, use_atomic_energy=True)
    # Dense test
    B, N = 2, 4
    x_dense = torch.randn(B, N, 16, requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    z_dense = torch.tensor([[6, 1, 1, 0], [6, 1, 1, 8]])
    e_dense = scalar_head(x_dense, mask=mask, z_int=z_dense)
    assert e_dense.shape == (B,)
    e_dense.sum().backward()
    assert x_dense.grad is not None
    assert scalar_head.atomic_energy.weight.grad is not None

    # Sparse test
    total_nodes = 7
    batch_vec = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    x_sparse = torch.randn(total_nodes, 16, requires_grad=True)
    z_sparse = torch.tensor([6, 1, 1, 6, 1, 1, 8])
    e_sparse = scalar_head(x_sparse, batch_vec=batch_vec, z_int=z_sparse)
    assert e_sparse.shape == (2,)
    e_sparse.sum().backward()
    assert x_sparse.grad is not None

    # 2. GGNNEnergyPredictor (e.g. GGNN)
    ggnn_head = GGNNEnergyPredictor(vc_channels=4, sc_channels=8, hidden=32, num_elements=10, use_atomic_energy=True)
    v_dense = torch.randn(B, N, 4, 3, requires_grad=True)
    s_dense = torch.randn(B, N, 8, requires_grad=True)
    e_ggnn = ggnn_head(v_dense, scalars=s_dense, mask=mask, z_int=z_dense)
    assert e_ggnn.shape == (B,)
    e_ggnn.sum().backward()
    assert v_dense.grad is not None
    assert s_dense.grad is not None

    # Sparse test for GGNN
    v_sparse = torch.randn(total_nodes, 4, 3, requires_grad=True)
    s_sparse = torch.randn(total_nodes, 8, requires_grad=True)
    e_ggnn_sp = ggnn_head(v_sparse, scalars=s_sparse, batch_vec=batch_vec, z_int=z_sparse)
    assert e_ggnn_sp.shape == (2,)
    e_ggnn_sp.sum().backward()
    assert v_sparse.grad is not None

    # 3. EnergyPredictor (multivectors)
    mv_head = EnergyPredictor(mv_channels=4, sc_channels=8, hidden=32, num_elements=10, use_atomic_energy=True)
    mv_dense = torch.randn(B, N, 4, 16, requires_grad=True)
    e_mv = mv_head(mv_dense, sc=s_dense, mask=mask, z_int=z_dense)
    assert e_mv.shape == (B,)
    e_mv.sum().backward()
    assert mv_dense.grad is not None

    # Sparse test for MV
    mv_sparse = torch.randn(total_nodes, 4, 16, requires_grad=True)
    e_mv_sp = mv_head(mv_sparse, sc=s_sparse, batch_vec=batch_vec, z_int=z_sparse)
    assert e_mv_sp.shape == (2,)
    e_mv_sp.sum().backward()
    assert mv_sparse.grad is not None

    # 4. Factory create_energy_head
    h_egnn = create_energy_head("egnn", out_mvc=4, out_sc=8, hidden_sc=16)
    assert isinstance(h_egnn, ScalarEnergyPredictor)
    h_segnn = create_energy_head("segnn", out_mvc=4, out_sc=8, hidden_sc=16)
    assert isinstance(h_segnn, ScalarEnergyPredictor)
    h_ggnn = create_energy_head("ggnn", out_mvc=4, out_sc=8, hidden_sc=16)
    assert isinstance(h_ggnn, GGNNEnergyPredictor)
    h_pga = create_energy_head("pgagnn", out_mvc=4, out_sc=8, hidden_sc=16)
    assert isinstance(h_pga, EnergyPredictor)


