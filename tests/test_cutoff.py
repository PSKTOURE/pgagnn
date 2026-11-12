"""Cutoff tests exercise full messages, neighbor removal and force derivatives."""
import pytest
import torch
from torch_geometric.data import Batch, Data

from src.layers import BesselRadialBasisLayer, PolynomialCutoff
from src.pgagnn import PGA_GNN, _cutoff_softmax
from experiments.md17.train_md17_energy_based import GAModel, EnergyPredictor, embed_inputs


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_polynomial_cutoff_value_and_two_derivatives(dtype):
    r = torch.tensor([0., 1., 1.999, 2., 2.001], dtype=dtype, requires_grad=True)
    g = PolynomialCutoff(2.)(r)
    first = torch.autograd.grad(g.sum(), r, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), r)[0]
    assert torch.isfinite(g).all() and (g >= 0).all() and (g <= 1).all()
    torch.testing.assert_close(g[[0, 1]], torch.tensor([1., .7734375], dtype=dtype))
    assert (g[3:] == 0).all() and (first[3:] == 0).all() and (second[3:] == 0).all()
    assert 0 < g[2] < 1e-7  # stable near cutoff even in float32


@pytest.mark.parametrize('sparse', [False, True])
def test_weighted_softmax_extreme_logits_has_finite_second_derivatives(sparse):
    r = torch.tensor(2., requires_grad=True)
    logits = torch.tensor([[-30., 30.]], requires_grad=True)
    weights = torch.stack([r * 0 + 1, PolynomialCutoff(2.)(r)])[None]
    if sparse:
        attention = _cutoff_softmax(logits.T, weights.T, dst=torch.zeros(2, dtype=torch.long), num_nodes=1).T
    else:
        attention = _cutoff_softmax(logits, weights)
    torch.testing.assert_close(attention, torch.tensor([[1., 0.]]))
    loss = (attention * torch.tensor([[.3, .8]])).sum()
    grads = torch.autograd.grad(loss, (r, logits), create_graph=True)
    seconds = torch.autograd.grad(sum(g.sum() for g in grads), (r, logits))
    assert all(torch.isfinite(g).all() for g in (*grads, *seconds))
    assert grads[0] == 0 and seconds[0] == 0


def model_for(sparse, smooth=True, dtype=torch.float64, radial_basis='gaussian'):
    torch.manual_seed(104)
    base = PGA_GNN(1, 8, out_mvc=2, in_sc=8, hidden_sc=8, out_sc=4,
                   edge_mvc=1, edge_sc=8, num_layers=2, num_heads=2,
                   attention_type='gatr_sparse' if sparse else 'gatr', factorize=True)
    return GAModel(base, EnergyPredictor(2, 4, hidden=8), sc_dim=8,
                   edge_rbf_dim=4, edge_sc_hidden=8, cutoff=2., smooth_cutoff=smooth,
                   radial_basis=radial_basis).to(dtype).eval()


def energy(model, distance, sparse, include_edge, explicit_self=False):
    zero = distance * 0
    pos = torch.stack([torch.stack([zero, zero, zero]), torch.stack([distance, zero, zero])])
    edges = torch.tensor([[0, 1], [1, 0]]) if include_edge else torch.empty(2, 0, dtype=torch.long)
    if explicit_self:
        edges = torch.cat([edges, torch.tensor([[0, 1], [0, 1]])], dim=1)
    batch = Batch.from_data_list([Data(pos=pos, z=torch.tensor([1, 6]), edge_index=edges)])
    return model(embed_inputs(batch, torch.device('cpu'), sparse)).sum()


def edge_effect(model, value, sparse, explicit_self=False):
    r = torch.tensor(value, dtype=next(model.parameters()).dtype, requires_grad=True)
    effect = energy(model, r, sparse, True, explicit_self) - energy(model, r, sparse, False, explicit_self)
    first = torch.autograd.grad(effect, r, create_graph=True)[0]
    second = torch.autograd.grad(first, r)[0]
    return torch.stack([effect, first, second]).detach()


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_whole_energy_force_and_hessian_match_removed_neighbor(sparse, dtype):
    model = model_for(sparse, dtype=dtype)
    # Nonzero encoded edge at zero RBF exposes biased encoders and normalization:
    # smoothness must come from the final attention/value gate, not RBF zeros.
    assert model.edge_encoder(torch.zeros(1, 4, dtype=dtype)).abs().max() > .01
    assert edge_effect(model, 1., sparse)[0].abs() > 1e-6
    for r in [2., 2.01]:
        torch.testing.assert_close(edge_effect(model, r, sparse), torch.zeros(3, dtype=dtype),
                                   atol=2e-11 if dtype == torch.float64 else 2e-5, rtol=0)
    near = edge_effect(model, 2. - 1e-5, sparse)
    farther = edge_effect(model, 2. - 1e-3, sparse)
    assert torch.isfinite(near).all()
    assert near[0].abs() < (1e-10 if dtype == torch.float64 else 2e-5)
    assert near[1].abs() < (1e-7 if dtype == torch.float64 else 2e-5)
    assert near[2].abs() < farther[2].abs() * .1


def test_dense_sparse_tapered_aggregation_agrees_for_identical_features():
    dense, sparse = model_for(False).base_model, model_for(True).base_model
    sparse.load_state_dict(dense.state_dict())
    torch.manual_seed(29)
    mv = torch.randn(1, 3, 1, 16, dtype=torch.float64)
    sc = torch.randn(1, 3, 8, dtype=torch.float64)
    emv = torch.randn(1, 3, 3, 1, 16, dtype=torch.float64)
    esc = torch.randn(1, 3, 3, 8, dtype=torch.float64)
    weight = torch.tensor([[[1., .4, 0.], [.2, 1., .7], [.3, .6, 1.]]], dtype=torch.float64)
    dst, src = torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij')
    a = dense(mv, sc=sc, adj=torch.ones(1, 3, 3), edge_attr_mv=emv,
              edge_attr_sc=esc, edge_weight=weight)
    b = sparse(mv[0], sc=sc[0], edge_index=torch.stack([src.flatten(), dst.flatten()]),
               edge_attr_mv=emv.reshape(9, 1, 16), edge_attr_sc=esc.reshape(9, 8),
               edge_weight=weight.reshape(9))
    for x, y in zip(a, b):
        torch.testing.assert_close(x[0], y, atol=2e-10, rtol=2e-10)


def test_untapered_model_exposes_the_boundary_discontinuity():
    # Regression control: this would fail to detect the bug if it tested only RBFs.
    assert edge_effect(model_for(True, smooth=False), 2., True, explicit_self=True)[0].abs() > 1e-5


def test_bessel_origin_and_cutoff_have_finite_force_derivatives():
    basis = BesselRadialBasisLayer(8, cutoff=2.).double()
    r = torch.tensor([[0.], [1e-8], [1.], [2.], [2.01]], dtype=torch.float64, requires_grad=True)
    values = basis(r)
    torch.testing.assert_close(values[0], basis.frequencies, atol=1e-12, rtol=1e-12)
    first = torch.autograd.grad(values.sum(), r, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), r)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert (values[3:] == 0).all() and (first[3:] == 0).all() and (second[3:] == 0).all()


@pytest.mark.parametrize('sparse', [False, True])
def test_bessel_model_boundary_and_force_loss_gradients(sparse):
    model = model_for(sparse, radial_basis='bessel')
    torch.testing.assert_close(edge_effect(model, 2., sparse), torch.zeros(3, dtype=torch.float64),
                               atol=2e-11, rtol=0)
    r = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    e = energy(model, r, sparse, True)
    force = torch.autograd.grad(e, r, create_graph=True)[0]
    (e.square() + force.square()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_skip_test_saves_selected_checkpoint_without_reading_test_frames(tmp_path):
    from torch_geometric.loader import DataLoader
    from experiments.md17.train_md17_energy_based import train_model
    class UnreadableTest:
        def __iter__(self):
            raise AssertionError('Test set must remain untouched during selection')
    graph = Data(pos=torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float64),
                 z=torch.tensor([1, 6]), edge_index=torch.tensor([[0, 1], [1, 0]]),
                 energy=torch.tensor([-1.], dtype=torch.float64), force=torch.zeros(2, 3, dtype=torch.float64))
    loader = DataLoader([graph], batch_size=1, generator=torch.Generator().manual_seed(42))
    model = model_for(True, radial_basis='bessel')
    model.base_model.use_sparse = True
    result = train_model('pgagnn', model, loader, loader, UnreadableTest(), torch.device('cpu'),
        num_epochs=1, mu_E=torch.tensor(0.), sigma_F=torch.tensor(1.), force_weight=1000.,
        energy_weight=1., lr=.001, weight_decay=5e-7, patience=200, ema_decay=.999,
        swa_start_epoch=3000, swa_lr=5e-5, sdpa_mode='math',
        checkpoint_path=str(tmp_path/'best_model.pth'),
        last_checkpoint_path=str(tmp_path/'last_model.pt'), no_tqdm=True, evaluate_test=False)
    assert result['test_loss'] is None and (tmp_path/'selected_model.pth').is_file()
    saved = torch.load(tmp_path/'last_model.pt', weights_only=False)
    assert saved['epoch'] == 1 and torch.equal(saved['train_generator_state'], loader.generator.get_state())
