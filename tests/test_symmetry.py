"""Independent exterior-action and actual Cartesian-pipeline regression checks."""
from itertools import combinations

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.pgagnn import PGA_GNN
from src.primitives import (anti_dual, dual, embed_point, equivariant_join,
                            extract_point, geometric_product, invariants, join,
                            outer_product, reverse)

DTYPE = torch.float64
BASIS = [tuple(c) for k in range(5) for c in combinations(range(4), k)]


def transform_matrix(det=1, translate=True, dtype=DTYPE):
    generator = torch.Generator().manual_seed(73)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=dtype))
    if torch.linalg.det(q).sign().item() != det:
        q[:, 0] *= -1
    t = torch.tensor([.7, -.4, 1.2], dtype=dtype) if translate else torch.zeros(3, dtype=dtype)
    L = torch.zeros(4, 4, dtype=dtype)
    L[0, 0] = 1
    L[1:, 1:] = q
    L[0, 1:] = -t @ q
    # Independently build the full exterior power, with minors rather than PGA kernels.
    action = torch.zeros(16, 16, dtype=dtype)
    for i, a in enumerate(BASIS):
        for j, b in enumerate(BASIS):
            if len(a) == len(b):
                action[i, j] = torch.linalg.det(L[list(a)][:, list(b)]) if a else 1
    return q, t, action


def act(x, action):
    return x @ action.T


@pytest.mark.parametrize('det', [1, -1])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_primitives_under_full_exterior_action(det, dtype):
    torch.manual_seed(19)
    _, _, action = transform_matrix(det, dtype=dtype)
    x, y, z = [torch.randn(5, 2, 16, dtype=dtype) for _ in range(3)]
    tol = dict(atol=2e-10, rtol=2e-10) if dtype == DTYPE else dict(atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(invariants(act(x, action)), invariants(x), **tol)
    for product in (geometric_product, outer_product):
        torch.testing.assert_close(product(act(x, action), act(y, action)), act(product(x, y), action), **tol)
    torch.testing.assert_close(join(act(x, action), act(y, action)), det * act(join(x, y), action), **tol)
    torch.testing.assert_close(equivariant_join(act(x, action), act(y, action), act(z, action)),
                               act(equivariant_join(x, y, z), action), **tol)
    torch.testing.assert_close(anti_dual(dual(x)), x, **tol)
    # The invalid proof's identity must continue to fail, even after repair.
    assert not torch.allclose(dual(act(x, action)), det * act(dual(x), action))


def test_invariant_features_and_point_join():
    p = embed_point(torch.zeros(1, 3, dtype=DTYPE))
    shifted = embed_point(torch.tensor([[2., 0, 0]], dtype=DTYPE))
    torch.testing.assert_close(invariants(p), invariants(shifted))
    x = torch.randn(3, 16, dtype=DTYPE)
    x_mirror = x.clone()
    x_mirror[..., 15] *= -1
    torch.testing.assert_close(invariants(x), invariants(x_mirror))
    assert outer_product(p, shifted).count_nonzero() == 0
    assert join(p, shifted).count_nonzero() > 0


def test_projective_readout_with_signed_finite_scale():
    pos = torch.tensor([[.4, -.7, 1.1]], dtype=DTYPE)
    x = -2 * embed_point(pos)
    _, t, action = transform_matrix(-1)
    q, _, _ = transform_matrix(-1)
    torch.testing.assert_close(extract_point(act(x, action)), pos @ q.T + t)


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('det', [1, -1])
def test_hidden_model_covariance_with_nonzero_edges(sparse, det):
    torch.manual_seed(41)
    model = PGA_GNN(2, 8, out_mvc=2, in_sc=3, hidden_sc=8, out_sc=3,
                    edge_mvc=1, edge_sc=4, num_layers=3, num_heads=2,
                    attention_type='gatr_sparse' if sparse else 'gatr').double().eval()
    _, _, action = transform_matrix(det)
    mv, sc = torch.randn(1, 4, 2, 16, dtype=DTYPE), torch.randn(1, 4, 3, dtype=DTYPE)
    edges = torch.randn(1, 4, 4, 1, 16, dtype=DTYPE)
    es = torch.randn(1, 4, 4, 4, dtype=DTYPE)
    kwargs = dict(adj=torch.ones(1, 4, 4, dtype=DTYPE))
    if sparse:
        # Packed source,destination convention; dense stores destination,source.
        dst, src = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
        kwargs = dict(edge_index=torch.stack([src.flatten(), dst.flatten()]))
        mv, sc, edges, es = mv[0], sc[0], edges.reshape(16, 1, 16), es.reshape(16, 4)
    a, a_sc = model(mv, sc=sc, edge_attr_mv=edges, edge_attr_sc=es, **kwargs)
    b, b_sc = model(act(mv, action), sc=sc, edge_attr_mv=act(edges, action), edge_attr_sc=es, **kwargs)
    torch.testing.assert_close(b, act(a, action), atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(b_sc, a_sc, atol=2e-8, rtol=2e-8)


def test_geometric_aggregation_factorization_forward_and_gradients():
    torch.manual_seed(9)
    dst = torch.randn(4, 3, 16, dtype=DTYPE, requires_grad=True)
    src = torch.randn(4, 5, 3, 16, dtype=DTYPE, requires_grad=True)
    logits = torch.randn(4, 5, dtype=DTYPE, requires_grad=True)
    weights = logits.softmax(dim=1)[..., None, None]
    direct = (weights * geometric_product(dst[:, None], reverse(src))).sum(dim=1)
    factorized = geometric_product(dst, (weights * reverse(src)).sum(dim=1))
    torch.testing.assert_close(direct, factorized)
    upstream = torch.randn_like(direct)
    ga = torch.autograd.grad((direct * upstream).sum(), (dst, src, logits), retain_graph=True)
    gb = torch.autograd.grad((factorized * upstream).sum(), (dst, src, logits))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)


def molecule_batch(dtype=DTYPE):
    torch.manual_seed(17)
    graphs = []
    for n in (4, 3):
        nodes = torch.arange(n)
        src, dst = torch.meshgrid(nodes, nodes, indexing='ij')
        graphs.append(Data(pos=torch.randn(n, 3, dtype=dtype), z=torch.tensor([1, 6, 7, 8][:n]),
                           x=torch.randn(n, 11, dtype=dtype), edge_index=torch.stack([src.flatten(), dst.flatten()]),
                           force=torch.zeros(n, 3, dtype=dtype), y=torch.zeros(1, dtype=dtype)))
    return Batch.from_data_list(graphs)


def make_base(sparse, layers=3):
    return PGA_GNN(1, 8, out_mvc=2, in_sc=8, hidden_sc=8, out_sc=4,
                   edge_mvc=1, edge_sc=8, num_layers=layers, num_heads=2,
                   attention_type='gatr_sparse' if sparse else 'gatr').double().eval()


@pytest.mark.parametrize('sparse', [False, True])
def test_molecular_embedding_translation_padding_and_distances(sparse):
    from experiments.md17.train_md17_energy_based import embed_inputs as md_embed
    from experiments.qm9.train_qm9 import embed_inputs as qm_embed
    b = molecule_batch()
    shifted = b.clone()
    shifted.pos = b.pos + torch.tensor([3., -4., 2.], dtype=DTYPE)
    for embed in (lambda x: md_embed(x, 'cpu', sparse),
                  lambda x: qm_embed(x, 'trivector', torch.device('cpu'), sparse)):
        g, gt = embed(b), embed(shifted)
        torch.testing.assert_close(g.mv, gt.mv)
        torch.testing.assert_close(g.edge_attr_mv, gt.edge_attr_mv)
        assert g.edge_attr_mv.abs().max() > .1
        # Distance channels contain lengths, not squared lengths.
        vector = getattr(g, 'vect_src_dst', None)
        if vector is None:
            vector = g.edge_vect
        distance, expected = g.edge_attr_sc, vector.norm(dim=-1, keepdim=True)
        if not sparse:
            valid = g.mask[..., None] & g.mask[:, None, :]
            assert not g.adj[~valid].any()
            distance, expected = distance[valid], expected[valid]
        torch.testing.assert_close(distance, expected, atol=2e-12, rtol=1e-12)


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('smooth_cutoff', [False, True])
def test_md17_energy_force_proper_rotations_and_translations(sparse, smooth_cutoff):
    from experiments.md17.train_md17_energy_based import GAModel, EnergyPredictor, embed_inputs
    torch.manual_seed(31)
    model = GAModel(make_base(sparse), EnergyPredictor(2, 4, hidden=8),
                    sc_dim=8, edge_rbf_dim=4, edge_sc_hidden=8, smooth_cutoff=smooth_cutoff).double().eval()
    q, t, _ = transform_matrix(1)
    def evaluate(batch):
        batch.pos.requires_grad_(True)
        energy = model(embed_inputs(batch, torch.device('cpu'), sparse))
        force = -torch.autograd.grad(energy.sum(), batch.pos, create_graph=True)[0]
        return energy, force
    b = molecule_batch()
    transformed = b.clone()
    transformed.pos = b.pos @ q.T + t
    e, f = evaluate(b)
    et, ft = evaluate(transformed)
    torch.testing.assert_close(et, e, atol=3e-9, rtol=3e-9)
    torch.testing.assert_close(ft, f @ q.T, atol=3e-9, rtol=3e-9)
    # Force training requires second derivatives through the graph.
    (e.square().sum() + f.square().sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Translation invariance also implies zero total force for each molecule.
    for k in (0, 1):
        torch.testing.assert_close(f[b.batch == k].sum(dim=0), torch.zeros(3, dtype=DTYPE), atol=1e-9, rtol=0)


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('mode', ['atomic_sum', 'pooled_mlp', 'dipole_vector'])
def test_qm9_readouts_proper_rotations_and_translations(sparse, mode):
    from experiments.qm9.train_qm9 import ModelWrapper, GraphRegressor, embed_inputs
    torch.manual_seed(7)
    model = ModelWrapper('pgagnn', make_base(sparse), GraphRegressor(2, 4, hidden=8, readout_mode=mode),
                         sc_dim=8, edge_rbf_dim=4, edge_sc_hidden=8).double().eval()
    b = molecule_batch()
    q, t, _ = transform_matrix(1)
    transformed = b.clone()
    transformed.pos = b.pos @ q.T + t
    def predict(batch):
        return model(embed_inputs(batch, 'trivector', torch.device('cpu'), sparse))
    torch.testing.assert_close(predict(transformed), predict(b), atol=2e-9, rtol=2e-9)
    # Variable-size padding must not change a molecule's prediction.
    alone = Batch.from_data_list([b.to_data_list()[0]])
    torch.testing.assert_close(predict(alone)[0], predict(b)[0], atol=2e-9, rtol=2e-9)




def test_canonical_point_parity_is_not_hidden_covariance():
    pos = torch.tensor([[.3, -.2, .7]], dtype=DTYPE)
    q, t, action = transform_matrix(-1)
    torch.testing.assert_close(embed_point(pos @ q.T + t), -act(embed_point(pos), action))
    assert not torch.allclose(embed_point(pos @ q.T + t), act(embed_point(pos), action))


@pytest.mark.parametrize('sparse', [False, True])
def test_float32_force_training_smoke(sparse):
    from experiments.md17.train_md17_energy_based import GAModel, EnergyPredictor, embed_inputs
    torch.manual_seed(13)
    model = GAModel(make_base(sparse), EnergyPredictor(2, 4, hidden=8),
                    sc_dim=8, edge_rbf_dim=4, edge_sc_hidden=8).float().eval()
    b = molecule_batch(torch.float32)
    q, t, _ = transform_matrix(1, dtype=torch.float32)
    bt = b.clone()
    bt.pos = b.pos @ q.T + t
    def evaluate(batch):
        batch.pos.requires_grad_(True)
        e = model(embed_inputs(batch, torch.device('cpu'), sparse))
        f = -torch.autograd.grad(e.sum(), batch.pos, create_graph=True)[0]
        return e, f
    e, f = evaluate(b)
    et, ft = evaluate(bt)
    torch.testing.assert_close(et, e, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(ft, f @ q.T, atol=2e-5, rtol=2e-4)
    f.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('det', [1, -1])
def test_centered_readout_features_under_orthogonal_action(det):
    torch.manual_seed(29)
    _, _, action = transform_matrix(det, translate=False)
    x = torch.randn(5, 2, 16, dtype=DTYPE)
    torch.testing.assert_close(invariants(act(x, action), centered=True), invariants(x, centered=True))
    torch.testing.assert_close(invariants(x, centered=True)[..., 1], x[..., 15].abs())


def test_centered_readout_retains_point_radius():
    pos = torch.tensor([[0., 0., 0.], [2., 0., 0.]], dtype=DTYPE)
    centered = pos - pos.mean(dim=0, keepdim=True)
    points = embed_point(centered)
    features = invariants(points, centered=True)
    expected = torch.sqrt(1 + centered.square().sum(dim=-1) + 1e-12)
    torch.testing.assert_close(features[..., 4], expected)
    shifted = pos + pos.new_tensor([4., -2., 7.])
    shifted_centered = shifted - shifted.mean(dim=0, keepdim=True)
    torch.testing.assert_close(invariants(embed_point(shifted_centered), centered=True), features)
    # abs has finite autograd behavior at zero as used by the QM9 scalar head.
    zero = torch.zeros(2, 16, dtype=DTYPE, requires_grad=True)
    invariants(zero, centered=True).sum().backward()
    assert torch.isfinite(zero.grad).all()
