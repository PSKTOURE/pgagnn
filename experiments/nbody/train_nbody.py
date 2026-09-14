import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
gatr_dir = ROOT_DIR / "geometric-algebra-transformer"
if gatr_dir.exists() and str(gatr_dir) not in sys.path:
    sys.path.insert(0, str(gatr_dir))

from src.xformers_stub import ensure_xformers_stub

ensure_xformers_stub()

import matplotlib.pyplot as plt
import numpy as np
import torch
from balanced_irreps import WeightBalancedIrreps
from e3nn.o3 import Irreps, spherical_harmonics

try:
    from gatr import GATr, MLPConfig, SelfAttentionConfig
except ImportError:

    class GATr:  # type: ignore
        """Placeholder for GATr when not available."""

    MLPConfig = None  # type: ignore
    SelfAttentionConfig = None  # type: ignore
from segnn.segnn import SEGNN
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as SparseDataLoader
from torch_geometric.nn import knn_graph
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, to_dense_batch
from torch_scatter import scatter, scatter_mean
from tqdm import tqdm

from egnn.models.egnn_clean.egnn_clean import get_edges_batch
from egnn.n_body_system.model import EGNN_vel
from experiments.nbody.dataset import NBodyDataset, SpringNBodyDataset, SpringNBodyDatasetSparse
from src.ggnn import GGNN
from src.layers import GaussianRadialBasisLayer
from src.pgagnn import PGA_GNN
from src.primitives import (
    embed_point,
    embed_scalar,
    embed_translation,
    equivariant_join,
    extract_point,
)
from src.utils import seed_everything

ROOT_DIR = Path(__file__).resolve().parents[2]


def get_model(
    model_name: str,
    in_mvc: int = 1,
    out_mvc: int = 16,
    hidden_mvc: int = 16,
    in_sc: int | None = None,
    out_sc: int | None = None,
    hidden_sc: int | None = 32,
    edge_mvc: int | None = None,
    edge_sc: int | None = None,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout_prob: float = 0.1,
    activation: str = "gelu",
    attention_type: str = "gatr",
    message: str = "geometric",
):
    def _create_gatr():
        if MLPConfig is None or SelfAttentionConfig is None:
            raise ImportError(
                "GATr is not available. Please ensure the geometric-algebra-transformer repository is available."
            )
        return GATr(
            in_mv_channels=in_mvc,
            out_mv_channels=out_mvc,
            hidden_mv_channels=hidden_mvc,
            in_s_channels=in_sc,
            out_s_channels=out_sc,
            hidden_s_channels=hidden_sc,
            num_blocks=num_layers,
            dropout_prob=dropout_prob,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )

    builders = {
        "pgagnn": lambda: PGA_GNN(
            in_mvc=in_mvc,
            out_mvc=out_mvc,
            hidden_mvc=hidden_mvc,
            in_sc=in_sc,
            out_sc=out_sc,
            hidden_sc=hidden_sc,
            edge_mvc=edge_mvc,
            edge_sc=edge_sc,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            activation=activation,
            attention_type=attention_type,
            expansion_factor=1.0,
            message=message,
            factorize=True,
            use_grade_modulation=False,
            use_pseudoscalar=False,
        ),
        "gatr": _create_gatr,
        "ggnn": lambda: GGNN(
            in_vc=in_mvc,
            in_sc=in_sc,
            hidden_vc=hidden_mvc,
            hidden_sc=hidden_sc,
            out_vc=out_mvc,
            out_sc=out_sc,
            edge_vc=edge_mvc,
            edge_sc=edge_sc,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            num_heads=num_heads,
        ),
        "egnn": lambda: EGNN_vel(
            in_node_nf=1,
            in_edge_nf=2,
            hidden_nf=64,
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_layers=4,
            recurrent=True,
            norm_diff=False,
            tanh=False,
        ),
        "segnn": lambda: SEGNN(
            input_irreps=Irreps("2x1o + 1x0e"),
            output_irreps=Irreps("1x1o"),
            edge_attr_irreps=Irreps.spherical_harmonics(1),
            node_attr_irreps=Irreps.spherical_harmonics(1),
            additional_message_irreps=Irreps("2x0e"),
            hidden_irreps=WeightBalancedIrreps(
                Irreps(f"{64}x0e"), Irreps.spherical_harmonics(1), sh=True, lmax=1
            ),
            num_layers=4,
            task="node",
            pool=None,
            norm=None,
        ),
    }

    try:
        return builders[model_name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model name: {model_name}") from exc


def get_dataset(
    filename: str,
    subsample: float | None = None,
    keep_trajectories: bool = False,
    use_spring: bool = False,
    group_size: int | None = None,
    group_shuffle: bool = False,
    seed: int = 0,
):
    """Helper function to create NBodyDataset."""
    dataset_class = SpringNBodyDataset if use_spring else NBodyDataset
    return dataset_class(
        filename,
        subsample=subsample,
        keep_trajectories=keep_trajectories,
        group_size=group_size,
        group_shuffle=group_shuffle,
        seed=seed,
    )


def create_run_directory(base_dir, args):
    """Create a unique directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    run_name = (
        f"{args.model_name}_bs{args.batch_size}_"
        f"hl{args.num_layers}_h{args.hidden_mvc}_lr{args.learning_rate}_"
        f"sub{args.subsample}_"
        f"{timestamp}"
    )
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    return run_dir, run_name


def save_metrics_to_json(results, run_dir, args):
    """Save training metrics and summary to JSON file."""
    metrics_data = {
        "config": vars(args),
        "summary": {
            "best_val_loss": results["best_val_loss"],
            "best_train_loss": min(results["train"]),
            "best_step": results.get("best_step", None),
            "test_loss": results.get("test_loss", None),
            "test_ood_loss": results.get("test_ood_loss", None),
            "total_steps": results["step"][-1] if results["step"] else 0,
            "num_eval_checkpoints": len(results["train"]),
        },
        "metrics": {
            "steps": results["step"],
            "train_losses": results["train"],
            "val_losses": results["val"],
        },
    }

    json_path = os.path.join(run_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Metrics JSON saved to: {json_path}")
    return json_path


def save_training_plots(results, args, run_dir):
    """Save training and validation loss plots to the run directory."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    train_losses = np.array(results["train"])
    val_losses = np.array(results["val"])
    max_train = np.percentile(train_losses, 99)
    max_val = np.percentile(val_losses, 99)
    train_losses_clipped = np.clip(train_losses, a_min=1e-6, a_max=max_train)
    val_losses_clipped = np.clip(val_losses, a_min=1e-6, a_max=max_val)

    steps = results.get("step", list(range(len(train_losses))))
    ax.plot(steps, train_losses_clipped, label="Train", linewidth=2, color="#2E86AB")
    ax.plot(steps, val_losses_clipped, label="Validation", linewidth=2, color="#D62246")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if steps:
        ax.set_xlim(0, steps[-1])

    fig.suptitle(
        f"{args.model_name.upper()} - Best Val Loss: {results['best_val_loss']:.6f} | "
        f"Objects: {args.num_objects} | Batch: {args.batch_size} | Layers: {args.num_layers}",
        fontsize=12,
        y=1.02,
    )

    plt.tight_layout()

    plot_path = os.path.join(run_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Training plot saved to: {plot_path}")
    plt.close()


class O3Transform:
    def __init__(self, lmax_attr):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)

    def __call__(self, graph):
        pos = graph.pos
        vel = graph.vel
        mass = graph.mass
        spring_k = getattr(graph, "spring_k", None)

        prod_mass = mass[graph.edge_index[0]] * mass[graph.edge_index[1]]
        rel_pos = pos[graph.edge_index[0]] - pos[graph.edge_index[1]]
        edge_dist = torch.sqrt(rel_pos.pow(2).sum(1, keepdims=True))

        graph.edge_attr = spherical_harmonics(
            self.attr_irreps, rel_pos, normalize=True, normalization="integral"
        )
        vel_embedding = spherical_harmonics(self.attr_irreps, vel, normalize=True, normalization="integral")
        graph.node_attr = scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce="mean") + vel_embedding

        if pos.ndim == 3:
            mean_pos = pos.mean(dim=1, keepdim=True)
        elif pos.ndim == 2:
            mean_pos = pos.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unexpected pos shape: {pos.shape}")

        pos_centered = pos - mean_pos
        vel_norm = vel.norm(dim=-1, keepdim=True)
        graph.x = torch.cat([pos_centered, vel, vel_norm], dim=-1)
        if spring_k is not None:
            graph.additional_message_features = torch.cat((edge_dist, prod_mass, spring_k), dim=-1)
        else:
            graph.additional_message_features = torch.cat((edge_dist, prod_mass), dim=-1)
        return graph


def embed_inputs(inputs: torch.Tensor) -> torch.Tensor:
    """Embeds the input tensor using appropriate embedding functions.

    Parameters
    ----------
    inputs : torch.Tensor with shape (batch_size, num_objects, 7)
        Input tensor where each object has 7 features: mass, position (x,y,z), velocity (vx,vy,vz).

    Returns
    -------
    embedded_inputs : torch.Tensor with shape (batch_size, num_objects, embedded_dim)
        Embedded input tensor.
    """
    m = inputs[..., 0:1]  # (batch_size, num_objects, 1)
    x = inputs[..., 1:4]  # (batch_size, num_objects, 3)
    v = inputs[..., 4:7]  # (batch_size, num_objects, 3)

    m_embedded = embed_scalar(m)  # (batch_size, num_objects, 16)
    x_embedded = embed_point(x)  # (batch_size, num_objects, 16)
    v_embedded = embed_translation(v)  # (batch_size, num_objects, 16)

    # (batch_size, num_objects, 16)
    embedded_inputs = m_embedded + x_embedded + v_embedded
    embedded_inputs = embedded_inputs.unsqueeze(2)  # (batch_size, num_objects, 1, 16)
    return embedded_inputs


def embed_inputs_packed(inputs: torch.Tensor) -> torch.Tensor:
    """Embeds packed input tensor (N, 7) for sparse PyG-style batches."""
    m = inputs[..., 0:1]  # (N, 1)
    x = inputs[..., 1:4]  # (N, 3)
    v = inputs[..., 4:7]  # (N, 3)

    m_embedded = embed_scalar(m)  # (N, 16)
    x_embedded = embed_point(x.unsqueeze(0)).squeeze()  # (N, 16)
    v_embedded = embed_translation(v)  # (N, 16)

    embedded_inputs = m_embedded + x_embedded + v_embedded
    return embedded_inputs.unsqueeze(1)  # (N, 1, 16)


def parse_batch(batch, device, use_spring=False):
    """
    Returns:
        inputs, targets, edge_attr, adj, is_sparse
    """
    if isinstance(batch, (tuple, list)):
        if use_spring:
            inputs, targets, adj, spring_k = batch
            adj = adj.to(device)
            edge_attr = spring_k.to(device)
        else:
            inputs, targets = batch
            adj = None
            edge_attr = None
        is_sparse = False
    else:
        # PyG sparse batch
        inputs = batch.x
        targets = batch.y
        edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None
        adj = None
        is_sparse = True

    return (
        inputs.to(device),
        targets.to(device),
        edge_attr.to(device) if edge_attr is not None else None,
        adj,
        is_sparse,
    )


def _forward_segnn_dense(model, batch, inputs, targets, edge_attr, adj, device, **kwargs):
    B, N = inputs.shape[:2]
    m = inputs[..., 0:1].reshape(B * N, 1)
    x = inputs[..., 1:4].reshape(B * N, 3)
    v = inputs[..., 4:7].reshape(B * N, 3)
    y = targets.reshape(B * N, 3)

    graph = Data(pos=x, mass=m, vel=v, y=y)
    graph.batch = torch.arange(B, device=device).repeat_interleave(N)
    graph.edge_index = knn_graph(graph.pos, k=6, batch=graph.batch)
    graph = O3Transform(lmax_attr=1)(graph).to(device)

    displacement = model(graph)
    points_pred = graph.pos + displacement
    return points_pred, graph.y, B, 0


def _forward_segnn_sparse(model, batch, inputs, targets, edge_attr, adj, device, **kwargs):
    m = inputs[:, 0:1]
    x = inputs[:, 1:4]
    v = inputs[:, 4:7]
    spring_k = getattr(batch, "spring_k", None)

    graph = Data(pos=x, mass=m, vel=v, y=targets)
    graph.batch = batch.batch.to(device)
    graph.edge_index = batch.edge_index.to(device)
    if spring_k is not None:
        graph.spring_k = spring_k.to(device)
    graph = O3Transform(lmax_attr=1)(graph).to(device)

    displacement = model(graph)
    points_pred = graph.pos + displacement
    B = int(batch.num_graphs) if hasattr(batch, "num_graphs") else inputs.size(0)
    return points_pred, graph.y, B, 0


def _forward_egnn_dense(model, batch, inputs, targets, edge_attr, adj, device, use_spring=False, **kwargs):
    batch_size, n_nodes = inputs.shape[:2]
    loc = inputs[..., 1:4].reshape(batch_size * n_nodes, 3)
    vel = inputs[..., 4:7].reshape(batch_size * n_nodes, 3)
    loc_end = targets.reshape(batch_size * n_nodes, 3)

    edge_index, _ = get_edges_batch(n_nodes=n_nodes, batch_size=batch_size)
    edge_index = [edge_index[0].to(device), edge_index[1].to(device)]
    rows, cols = edge_index

    # Match EGNN source: node feature is speed magnitude.
    nodes = torch.sqrt(torch.sum(vel**2, dim=1, keepdim=True)).detach()
    loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, dim=1, keepdim=True)

    # Edge features = [base_edge_feature, pairwise_distance].
    if use_spring and edge_attr is not None:
        spring_dense = edge_attr.to(device)
        graph_idx = torch.div(rows, n_nodes, rounding_mode="floor")
        src_local = rows % n_nodes
        dst_local = cols % n_nodes
        base_edge_attr = spring_dense[graph_idx, src_local, dst_local].unsqueeze(1)
    else:
        base_edge_attr = torch.ones((rows.size(0), 1), device=device)

    egnn_edge_attr = torch.cat([base_edge_attr, loc_dist], dim=1).detach()
    points_pred = model(nodes, loc.detach(), edge_index, vel, egnn_edge_attr)
    return points_pred, loc_end, batch_size, 0


def _forward_egnn_sparse(
    model, batch, inputs, targets, edge_attr, adj, device, use_edge_attr=False, **kwargs
):
    loc = inputs[:, 1:4]
    vel = inputs[:, 4:7]
    loc_end = targets

    rows, cols = batch.edge_index.to(device)
    loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, dim=1, keepdim=True)
    if use_edge_attr and edge_attr is not None:
        base_edge_attr = edge_attr if edge_attr.dim() > 1 else edge_attr.unsqueeze(1)
    else:
        base_edge_attr = torch.ones((rows.size(0), 1), device=device)

    egnn_edge_attr = torch.cat([base_edge_attr, loc_dist], dim=1).detach()
    nodes = torch.sqrt(torch.sum(vel**2, dim=1, keepdim=True)).detach()
    points_pred = model(nodes, loc.detach(), [rows, cols], vel, egnn_edge_attr)
    B = int(batch.num_graphs) if hasattr(batch, "num_graphs") else inputs.size(0)
    return points_pred, loc_end, B, 0


def _forward_ggnn_dense(model, batch, inputs, targets, edge_attr, adj, device, dist_embedder=None, **kwargs):
    B, N = inputs.shape[:2]
    scalars = inputs[..., 0:1]  # (B, N, 1)
    pos = inputs[..., 1:4]  # (B, N, 3) - absolute positions
    vel = inputs[..., 4:7]  # (B, N, 3) - invariant velocities
    vectors = vel.unsqueeze(-2)  # (B, N, 1, 3)
    rel = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 3)
    dist = torch.linalg.norm(rel, dim=-1, keepdim=True) + 1e-8  # (B, N, N, 1)
    edge_attr_v = rel.unsqueeze(-2)  # (B, N, N, 1, 3)
    edge_attr_sc = dist_embedder(dist)  # (B, N, N, edge_sc)

    if adj is None:
        adj = torch.ones((B, N, N), device=device)

    mv, sc = model(vectors, scalars, edge_attr_v, edge_attr_sc, adj)
    displacement_pred = mv[:, :, 0, :] * sc[:, :, 0:1]  # (B, N, 3)
    points_pred = pos + displacement_pred  # (B, N, 3)
    return points_pred, targets, B, 0


def _forward_ggnn_sparse(model, batch, inputs, targets, edge_attr, adj, device, dist_embedder=None, **kwargs):
    # Convert the sparse PyG batch to a dense representation for GGNN.
    batch = batch.to(device)
    batch_vec = batch.batch.to(device)
    scalars = inputs[:, 0:1]  # (N_total, 1)
    pos = inputs[:, 1:4]  # (N_total, 3)
    vel = inputs[:, 4:7]  # (N_total, 3)
    num_graphs = int(batch_vec.max()) + 1 if batch_vec.numel() > 0 else 0

    pos_dense, mask = to_dense_batch(pos, batch_vec)  # (B, N, 3)
    scalars_dense, _ = to_dense_batch(scalars, batch_vec)  # (B, N, 1)
    vel_dense, _ = to_dense_batch(vel, batch_vec)  # (B, N, 3)
    vectors_dense = vel_dense.unsqueeze(-2)  # (B, N, 1, 3)
    rel_dense = pos_dense.unsqueeze(2) - pos_dense.unsqueeze(1)  # (B, N, N, 3)
    dist_dense = torch.linalg.norm(rel_dense, dim=-1, keepdim=True) + 1e-8  # (B, N, N, 1)
    edge_attr_v = rel_dense.unsqueeze(-2)  # (B, N, N, 1, 3)
    edge_attr_sc = dist_embedder(dist_dense)  # (B, N, N, edge_sc)

    adj_dense = to_dense_adj(batch.edge_index, batch_vec, max_num_nodes=None)  # (B, N, N)
    if edge_attr is not None:
        edge_attr_dense = to_dense_adj(batch.edge_index, batch_vec, edge_attr)
        edge_attr_sc = torch.cat([edge_attr_sc, edge_attr_dense], dim=-1)

    mv, sc = model(vectors_dense, scalars_dense, edge_attr_v, edge_attr_sc, adj_dense)
    displacement_pred = mv[:, :, 0, :] * sc[:, :, 0:1]  # (B, N, 3)
    points_pred = pos_dense + displacement_pred  # (B, N, 3)
    return points_pred[mask], targets, num_graphs, 0  # Only compute loss on valid nodes


def _forward_ga_dense(
    model, batch, inputs, targets, edge_attr, adj, device, use_edge_attr=False, dist_embedder=None, **kwargs
):
    B, N = inputs.shape[:2]
    scalars = torch.zeros((B, N, 1), device=device)
    reg = 0.0
    inputs_emb = embed_inputs(inputs)
    if adj is None:
        adj = torch.ones((B, N, N), device=device)
    ref = inputs_emb.mean(dim=(1, 2), keepdim=True)

    pos = inputs[..., 1:4]  # (B, N, 3)
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

    if isinstance(model, GATr):
        adj_mask = adj.unsqueeze(1) if adj is not None else None
        mv, sc = model(inputs_emb, scalars=scalars, join_reference=ref, attention_mask=adj_mask)
        reg = (torch.abs(mv[:, :, 0, 14:15]) - 1.0 + torch.sum(sc**2, dim=(1, 2), keepdim=True)).mean()
    else:
        mv, _ = model(
            inputs_emb, ref=ref, sc=scalars, adj=adj, edge_attr_mv=edge_attr_mv, edge_attr_sc=edge_attr_sc
        )
    points_pred = extract_point(mv[:, :, 0, :])
    return points_pred, targets, B, reg


def _forward_ga_sparse(
    model, batch, inputs, targets, edge_attr, adj, device, use_edge_attr=False, dist_embedder=None, **kwargs
):
    if getattr(model, "attention_type", None) != "gatr_sparse":
        raise ValueError("Sparse GA models require attention_type='gatr_sparse'")

    batch_vec = batch.batch.to(device)
    reg = 0.0
    scalars = torch.zeros((inputs.size(0), 1), device=device)
    inputs_emb = embed_inputs_packed(inputs)  # (N, 1, 16)
    edge_index = batch.edge_index.to(device)

    base_edge_attr_sc = edge_attr.to(device) if (use_edge_attr and edge_attr is not None) else None
    edge_index, edge_attr_sc = add_remaining_self_loops(
        edge_index=edge_index,
        edge_attr=base_edge_attr_sc,
        fill_value=0.0,
        num_nodes=inputs.size(0),
    )

    num_graphs = int(batch_vec.max()) + 1 if batch_vec.numel() > 0 else 0
    ref = scatter_mean(inputs_emb, batch_vec, dim=0, dim_size=num_graphs)  # (Num_graph, 1, 16)
    edge_attr_mv = None
    if use_edge_attr:
        # Compute (x_i - x_j) for each edge to use as edge MV attributes.
        src, dst = edge_index
        x_i = inputs[src][:, 1:4]  # (num_edges, 3)
        x_j = inputs[dst][:, 1:4]  # (num_edges, 3)
        u = x_j - x_i  # (num_edges, 3)
        src_embedded = embed_point(x_i).squeeze()  # (num_edges, 16)
        dst_embedded = embed_point(x_j).squeeze()  # (num_edges, 16)
        ref_edge = ref[batch_vec[src]].squeeze(1)  # (E, 16)
        edge_attr_mv = equivariant_join(src_embedded, dst_embedded, ref_edge).unsqueeze(
            -2
        )  # (num_edges, 1, 16)
        rel_norm = torch.linalg.norm(u, dim=-1, keepdim=True) + 1e-8
        rel_norm = dist_embedder(rel_norm).to(device)  # (num_edges, edge_sc)
        edge_attr_sc = torch.cat([rel_norm, edge_attr_sc], dim=-1)

    mv, _ = model(
        inputs_emb,
        ref=ref,
        sc=scalars,
        edge_index=edge_index,
        edge_attr_mv=edge_attr_mv,
        edge_attr_sc=edge_attr_sc,
        batch=batch_vec,
    )
    points_pred = extract_point(mv[:, 0, :].float())
    B = num_graphs
    return points_pred, targets, B, reg


_DENSE_HANDLERS = {
    SEGNN: _forward_segnn_dense,
    EGNN_vel: _forward_egnn_dense,
    GGNN: _forward_ggnn_dense,
}
_SPARSE_HANDLERS = {
    SEGNN: _forward_segnn_sparse,
    EGNN_vel: _forward_egnn_sparse,
    GGNN: _forward_ggnn_sparse,
}


def forward_and_loss(
    model,
    batch,
    inputs,
    targets,
    edge_attr,
    adj,
    criterion,
    device,
    reg_scale=0.0,
    training=True,
    use_spring=False,
    use_edge_attr=False,
    dist_embedder=None,
):
    """Runs a forward pass and computes the loss for any (model, batch layout)
    combination this project supports, then applies the (currently unused)
    regularization scale.

    Dispatches on the batch layout (dense tuple vs. sparse PyG Data) and the
    model class; GA-attention models (GATr, PGA_GNN) are the default when the
    model class isn't SEGNN, EGNN_vel, or GGNN. See the handler functions
    above for what each combination actually does.

    Returns:
        loss, batch_size
    """
    is_dense = isinstance(batch, (tuple, list))
    handlers = _DENSE_HANDLERS if is_dense else _SPARSE_HANDLERS
    default_handler = _forward_ga_dense if is_dense else _forward_ga_sparse
    handler = handlers.get(type(model), default_handler)

    points_pred, target_vals, B, reg = handler(
        model,
        batch,
        inputs,
        targets,
        edge_attr,
        adj,
        device,
        use_spring=use_spring,
        use_edge_attr=use_edge_attr,
        dist_embedder=dist_embedder,
    )

    loss = criterion(points_pred, target_vals)
    if training:
        loss = loss + reg_scale * reg
    return loss, B


@torch.no_grad()
def evaluate(
    model,
    dist_embedder,
    dataloader,
    criterion,
    device,
    use_spring=False,
    use_edge_attr=False,
):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    for batch in dataloader:
        inputs, targets, edge_attr, adj, _ = parse_batch(batch, device, use_spring)

        loss, B = forward_and_loss(
            model=model,
            batch=batch,
            inputs=inputs,
            targets=targets,
            edge_attr=edge_attr,
            adj=adj,
            criterion=criterion,
            device=device,
            reg_scale=0.0,  # no regularization for evaluation
            training=False,
            use_spring=use_spring,
            use_edge_attr=use_edge_attr,
            dist_embedder=dist_embedder,
        )

        total_loss += loss.item() * B
        num_samples += B

    return total_loss / num_samples


def train_model(
    model,
    dist_embedder,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    test_ood_dataloader,
    optimizer,
    criterion,
    reg_scale,
    device,
    num_steps: int = 50_000,
    eval_every: int = 500,
    early_stopping_patience: int = 20,
    checkpoint_path: str | None = None,
    scheduler=None,
    use_spring=False,
    use_edge_attr=False,
    no_tqdm=False,
):
    """Step-based training loop (GATr-style): trains for `num_steps` optimizer
    steps total, cycling through `train_dataloader` as many times as needed,
    evaluating on `val_dataloader` every `eval_every` steps. Early stopping is
    triggered after `early_stopping_patience` *evaluation checkpoints*
    (i.e. patience * eval_every steps) without an improvement in val loss.
    """
    best_val_loss = float("inf")
    best_step = 0
    losses = {"train": [], "val": [], "step": []}
    patience_counter = 0
    step = 0
    running_loss = 0.0
    running_samples = 0
    import time

    start = time.time()
    use_tqdm = not no_tqdm

    train_iter = iter(train_dataloader)
    model.train()

    pbar = tqdm(total=num_steps, desc="Training", disable=not use_tqdm)
    try:
        while step < num_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Cycle back through the dataset (like GATr: no notion of "epoch").
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            inputs, targets, edge_attr, adj, _ = parse_batch(batch, device, use_spring)

            optimizer.zero_grad()

            loss, B = forward_and_loss(
                model=model,
                batch=batch,
                inputs=inputs,
                targets=targets,
                edge_attr=edge_attr,
                adj=adj,
                criterion=criterion,
                device=device,
                reg_scale=reg_scale,
                training=True,
                use_spring=use_spring,
                use_edge_attr=use_edge_attr,
                dist_embedder=dist_embedder,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            # Learning rate scheduling is per-step, not per-epoch.
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * B
            running_samples += B
            step += 1
            pbar.update(1)

            is_last_step = step == num_steps
            if step % eval_every == 0 or is_last_step:
                train_loss = running_loss / running_samples
                running_loss = 0.0
                running_samples = 0

                val_loss = evaluate(
                    model,
                    dist_embedder,
                    val_dataloader,
                    criterion,
                    device,
                    use_spring=use_spring,
                    use_edge_attr=use_edge_attr,
                )
                model.train()

                losses["train"].append(train_loss)
                losses["val"].append(val_loss)
                losses["step"].append(step)

                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {step:6d}/{num_steps}: Train Loss={train_loss:.6f}, "
                    f"Val Loss={val_loss:.6f}, LR={lr:.6f}"
                )

                # Early stopping and checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    patience_counter = 0

                    # Save best model
                    if checkpoint_path:
                        model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                        torch.save(
                            {
                                "step": step,
                                "model_state_dict": model_to_save.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "val_loss": val_loss,
                                "train_loss": train_loss,
                            },
                            checkpoint_path,
                        )
                else:
                    patience_counter += 1

                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    print(
                        f"\nEarly stopping at step {step} "
                        f"({patience_counter} eval checkpoints without improvement). "
                        f"Best val loss: {best_val_loss:.6f}"
                    )
                    break
    finally:
        pbar.close()

    end = time.time()
    print(f"Total training time: {end - start:.2f} seconds")

    losses["best_val_loss"] = best_val_loss
    losses["best_step"] = best_step

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss = evaluate(
        model,
        dist_embedder,
        test_dataloader,
        criterion,
        device,
        use_spring=use_spring,
        use_edge_attr=use_edge_attr,
    )
    print("\nEvaluating on OOD test set...")
    test_ood_loss = evaluate(
        model,
        dist_embedder,
        test_ood_dataloader,
        criterion,
        device,
        use_spring=use_spring,
        use_edge_attr=use_edge_attr,
    )
    losses["test_loss"] = test_loss
    losses["test_ood_loss"] = test_ood_loss
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test OOD Loss: {test_ood_loss:.6f}")

    return losses


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Train PGA-GNN, or GATr models on N-body dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model parameters
    parser.add_argument("--model_name", type=str, default="pgagnn")
    parser.add_argument("--in_mvc", type=int, default=1, help="Number of input multivector channels")
    parser.add_argument("--out_mvc", type=int, default=1, help="Number of output multivector channels")
    parser.add_argument("--hidden_mvc", type=int, default=16)
    parser.add_argument("--in_sc", type=int, default=1, help="Number of input scalar channels")
    parser.add_argument("--out_sc", type=int, default=1, help="Number of output scalar channels")
    parser.add_argument("--hidden_sc", type=int, default=128)
    parser.add_argument("--edge_sc", type=int, default=64, help="Number of scalar edge attribute channels")
    parser.add_argument(
        "--edge_mv", type=int, default=1, help="Number of multivector edge attribute channels"
    )
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attention_type", type=str, default="gatr")
    parser.add_argument("--message", type=str, default="geometric", choices=["geometric", "sum", "linear"])
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GA layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    # Data parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_objects", type=int, default=15)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default=str(ROOT_DIR / "datasets" / "nbody"))
    parser.add_argument("--use_spring", action="store_true")
    parser.add_argument("--use_edge_attr", action="store_true")
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--group_shuffle", action="store_true", default=False)
    parser.add_argument("--use_sparse", action="store_true", default=False)
    parser.add_argument("--no_scalar", action="store_true", help="Disable scalar channels in the GA model.")

    # Training parameters (step-based, GATr-style)
    parser.add_argument("--num_steps", type=int, default=50_000, help="Total number of training steps")
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--reg_scale", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--scheduler_factor", type=float, default=0.8)
    parser.add_argument("--warmup_steps", type=int, default=2_000, help="Number of LR warmup steps")
    parser.add_argument("--scheduler_type", type=str, default="cosine_warmup")
    parser.add_argument("--lr_final", type=float, default=3e-6)
    parser.add_argument("--gpu", type=int, default=None)

    # Device and reproducibility
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4)

    # Output parameters
    parser.add_argument("--runs_dir", type=str, default=str(ROOT_DIR / "runs" / "nbody"))
    parser.add_argument("--no_save_plots", action="store_true", help="Disable saving training plots")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def compute_dataset_statistics(dataset):
    """Compute and print statistics about the dataset."""
    masses = []
    positions = []
    velocities = []
    distances = []
    for batch in dataset:
        # Support both dense dataset batches (tuples) and PyG sparse batches (Data/Batch)
        if hasattr(batch, "x"):
            # PyG Batch: node features in batch.x and graph assignment in batch.batch
            x_nodes = batch.x
            batch_vec = batch.batch
            x_dense, mask = to_dense_batch(x_nodes, batch_vec)
            # x_dense: (batch_size, num_nodes, feat)
            m = x_dense[..., 0]
            x = x_dense[..., 1:4]
            v = x_dense[..., 4:7]

            masses.append(m.cpu().numpy())
            positions.append(x.cpu().numpy())
            velocities.append(v.cpu().numpy())

            # Compute pairwise distances and mask padded nodes
            with torch.no_grad():
                rel = x.unsqueeze(2) - x.unsqueeze(1)
                dist = torch.linalg.norm(rel, dim=-1).cpu().numpy()
                mask_np = mask.cpu().numpy()
                # mask_np: (batch_size, num_nodes) -> valid pairs where both nodes exist
                pair_mask = mask_np[:, :, None] & mask_np[:, None, :]
                dist[~pair_mask] = np.nan
                distances.append(dist)
        else:
            inputs, _ = batch[:2]
            m = inputs[..., 0]  # (batch_size, num_objects)
            x = inputs[..., 1:4]  # (batch_size, num_objects, 3)
            v = inputs[..., 4:7]  # (batch_size, num_objects, 3)

            masses.append(m.cpu().numpy())
            positions.append(x.cpu().numpy())
            velocities.append(v.cpu().numpy())

            # Compute pairwise distances
            with torch.no_grad():
                rel = x.unsqueeze(2) - x.unsqueeze(1)
                dist = torch.linalg.norm(rel, dim=-1)
                distances.append(dist.cpu().numpy())
    masses = np.concatenate(masses, axis=0)
    positions = np.concatenate(positions, axis=0)
    velocities = np.concatenate(velocities, axis=0)
    distances = np.concatenate(distances, axis=0)

    return {
        "mass": {"mean": np.nanmean(masses), "std": np.nanstd(masses)},
        "position": {"mean": np.nanmean(positions), "std": np.nanstd(positions)},
        "velocity": {"mean": np.nanmean(velocities), "std": np.nanstd(velocities)},
        "distance": {
            "mean": np.nanmean(distances),
            "std": np.nanstd(distances),
            "min": np.nanmin(distances),
            "max": np.nanmax(distances),
        },
    }


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    seed_everything(args.seed)
    if args.gpu is not None:
        if not torch.cuda.is_available():
            print("Warning: --gpu specified but CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print configuration
    print("=" * 70)
    print("Training Configuration:")
    print("=" * 70)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:30s}: {value}")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    suffix = "_augmented" if args.group_size is not None else ""
    train_file_name = f"spring_train{suffix}.npz" if args.use_spring else "nbody_train.npz"
    val_file_name = "spring_val.npz" if args.use_spring else "nbody_val.npz"
    test_file_name = "spring_test.npz" if args.use_spring else "nbody_test.npz"
    test_ood_file_name = "spring_test_ood.npz" if args.use_spring else "nbody_test_ood.npz"
    train_file = os.path.join(args.data_dir, train_file_name)
    val_file = os.path.join(args.data_dir, val_file_name)
    test_file = os.path.join(args.data_dir, test_file_name)
    test_ood_file = os.path.join(args.data_dir, test_ood_file_name)

    # Determine DataLoader class and dataset class
    use_sparse_loader = args.use_sparse
    DataLoaderClass = SparseDataLoader if use_sparse_loader else DataLoader

    if args.use_sparse:
        if not args.use_spring:
            raise ValueError("Sparse mode currently supports SpringNBodyDataset only.")
        dataset_class = SpringNBodyDatasetSparse
        train_dataset = dataset_class(
            train_file,
            subsample=args.subsample,
            group_size=args.group_size,
            group_shuffle=args.group_shuffle,
            seed=args.seed,
        )
        val_dataset = dataset_class(val_file, seed=args.seed)
        test_dataset = dataset_class(test_file, seed=args.seed)
        test_ood_dataset = dataset_class(test_ood_file, seed=args.seed)
    else:
        train_dataset = get_dataset(
            train_file,
            subsample=args.subsample,
            use_spring=args.use_spring,
            group_size=args.group_size,
            group_shuffle=args.group_shuffle,
            seed=args.seed,
        )
        val_dataset = get_dataset(val_file, use_spring=args.use_spring)
        test_dataset = get_dataset(test_file, use_spring=args.use_spring)
        test_ood_dataset = get_dataset(test_ood_file, use_spring=args.use_spring)

    # Create loaders with common configuration
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "generator": g,
    }
    train_loader = DataLoaderClass(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoaderClass(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoaderClass(test_dataset, shuffle=False, **loader_kwargs)
    test_ood_loader = DataLoaderClass(test_ood_dataset, shuffle=False, **loader_kwargs)

    print(f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
    print(f"Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
    print(f"Test samples: {len(test_dataset)}, batches: {len(test_loader)}")
    print(f"Test OOD samples: {len(test_ood_dataset)}, batches: {len(test_ood_loader)}")
    # Create model
    print(f"\nCreating {args.model_name.upper()} model...")
    if args.use_edge_attr:
        args.edge_mvc = 1
        args.edge_sc = 65 if args.use_spring else 64
    else:
        args.edge_mvc = None
        args.edge_sc = None
    if args.no_scalar:
        args.in_sc = None
        args.edge_sc = None
        args.hidden_sc = None
        args.out_sc = None
    stats = compute_dataset_statistics(train_dataset)
    max_dist = stats["distance"]["max"]
    dist_embedder = GaussianRadialBasisLayer(cutoff=max_dist, num_bases=64, mask=False).to(args.device)
    model = get_model(
        model_name=args.model_name,
        in_mvc=args.in_mvc,
        out_mvc=args.out_mvc,
        hidden_mvc=args.hidden_mvc,
        in_sc=args.in_sc,
        out_sc=args.out_sc,
        hidden_sc=args.hidden_sc,
        edge_mvc=args.edge_mvc,
        edge_sc=args.edge_sc,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_prob=args.dropout,
        activation=args.activation,
        attention_type=args.attention_type,
        message=args.message,
    )
    model = model.to(device)
    model.use_sparse = args.use_sparse

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    args.total_params = total_params
    args.trainable_params = trainable_params

    # Create run directory
    print("\nCreating run directory...")
    run_dir, run_name = create_run_directory(args.runs_dir, args)
    print(f"Run directory: {run_dir}")
    print(f"Run name: {run_name}\n")

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Create learning rate scheduler based on type
    if args.scheduler_type == "cosine_warmup":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps - args.warmup_steps, eta_min=args.lr_final
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_steps],
        )
        print(
            f"Using CosineAnnealingLR with warmup (warmup: {args.warmup_steps} steps, "
            f"lr: {args.learning_rate:.6f} -> {args.lr_final:.6f})"
        )
    elif args.scheduler_type == "exponential":
        gamma = (args.lr_final / args.learning_rate) ** (1 / args.num_steps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        print(
            f"Using ExponentialLR scheduler (lr: {args.learning_rate:.6f} -> {args.lr_final:.6f}, gamma={gamma:.6f})"
        )
    else:
        scheduler = None
        print("No learning rate scheduler")

    # Set checkpoint path
    checkpoint_path = os.path.join(run_dir, "best_model.pt")

    # Define loss criterion
    criterion = torch.nn.MSELoss()

    # Train model
    print(f"\nStarting training for up to {args.num_steps} steps (eval every {args.eval_every} steps)...")
    print(f"Device: {device}")
    print(f"Checkpoint will be saved to: {checkpoint_path}\n")
    results = train_model(
        model=model,
        dist_embedder=dist_embedder,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        test_ood_dataloader=test_ood_loader,
        optimizer=optimizer,
        criterion=criterion,
        reg_scale=args.reg_scale,
        device=device,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        use_spring=args.use_spring,
        use_edge_attr=args.use_edge_attr,
        no_tqdm=args.no_tqdm,
    )

    # Save metrics to CSV and JSON
    print("\nSaving training metrics...")
    save_metrics_to_json(results, run_dir, args)

    # Save training plots
    if not args.no_save_plots:
        print("\nGenerating and saving training plots...")
        save_training_plots(results, args, run_dir)

    # Print final summary
    print("\n" + "=" * 70)
    print("Training Summary:")
    print("=" * 70)
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Final train loss: {results['train'][-1]:.6f}")
    print(f"Final val loss: {results['val'][-1]:.6f}")
    print(f"Test loss: {results['test_loss']:.6f}")
    print(f"Total steps: {results['step'][-1] if results['step'] else 0}")
    print(f"Best step: {results.get('best_step', 'N/A')}")
    print(f"Run directory: {run_dir}")
    print(f"Model saved to: {checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
