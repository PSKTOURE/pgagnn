import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
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

try:
    from gatr import GATr, MLPConfig, SelfAttentionConfig
except ImportError:

    class GATr:  # type: ignore
        """Placeholder for GATr when not available."""

    MLPConfig = None  # type: ignore
    SelfAttentionConfig = None  # type: ignore

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph
from torch_geometric.utils import add_self_loops, to_dense_adj, to_dense_batch
from torch_scatter import scatter_mean, scatter_sum
from tqdm import tqdm

from src.checkpointing import StopSignalHandler, load_checkpoint, parse_signals, save_checkpoint
from src.layers import GaussianRadialBasisLayer
from src.pgagnn import PGA_GNN
from src.primitives import embed_point, equivariant_join, invariants
from src.utils import seed_everything

torch.set_printoptions(profile="short", linewidth=1000, sci_mode=True)
ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "datasets"
TARGET_NAMES: list[str] = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "U0_atom",
    "U_atom",
    "H_atom",
    "G_atom",
    "A",
    "B",
    "C",
]


def get_target_readout_config(target_name: str) -> tuple[str, str, bool]:
    """Select a readout style that matches the target's physical structure."""
    if target_name == "mu":
        return "dipole_vector", "sum", False

    atomic_sum_targets = {"zpve", "U0", "U", "H", "G"}
    pooled_targets = {
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "Cv",
        "U0_atom",
        "U_atom",
        "H_atom",
        "G_atom",
        "A",
        "B",
        "C",
    }

    if target_name in atomic_sum_targets:
        return "atomic_sum", "sum", False
    if target_name in pooled_targets:
        return "pooled_mlp", "mean", True

    return "atomic_sum", "sum", False


def get_target_index(target: str) -> int:
    if target.isdigit():
        idx = int(target)
        if idx < 0 or idx >= len(TARGET_NAMES):
            raise ValueError(f"Target index must be in [0, {len(TARGET_NAMES) - 1}]")
        return idx
    if target not in TARGET_NAMES:
        raise ValueError(f"Unknown target '{target}'. Valid options: {TARGET_NAMES}")
    return TARGET_NAMES.index(target)


def create_run_directory(base_dir, args):
    """Create a unique directory for this run."""
    if not args.no_time_suffix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        run_name = f"{args.model_name}_bs{args.batch_size}_hl{args.num_layers}_h{args.hidden_mvc}_lr{args.lr}_{timestamp}"
    else:
        run_name = f"{args.model_name}"
    target_dir = os.path.join(base_dir, args.target)
    os.makedirs(target_dir, exist_ok=True)
    run_dir = os.path.join(target_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    return run_dir, run_name


def resolve_run_directory(base_dir, args):
    """Reuse the existing run directory when resuming from a checkpoint."""
    if args.resume_from is not None:
        resume_path = Path(args.resume_from).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        run_dir = str(resume_path.parent)
        run_name = resume_path.parent.name

        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)

        return run_dir, run_name

    return create_run_directory(base_dir, args)


def save_metrics_to_json(results, run_dir, args):
    """Save training metrics and summary to JSON file with clear naming."""
    metrics_data = {
        "config": vars(args),
        "summary": {
            "best_val_loss": results["best_val_loss"],
            "test_loss": results["test_loss"],
            "best_train_loss": min(results["train_loss"]),
            "total_epochs": len(results["train_loss"]),
        },
        "metrics": {
            "train_loss": results["train_loss"],
            "val_loss": results["val_loss"],
        },
    }

    json_path = os.path.join(run_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Metrics JSON saved to: {json_path}")
    return json_path


def init_tensorboard_writer(args, run_dir: str) -> SummaryWriter | None:
    if getattr(args, "no_tensorboard", False):
        return None
    tb_dir = os.path.join(run_dir, "tensorboard")
    return SummaryWriter(log_dir=tb_dir)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, AveragedModel) else model


def get_model(
    model_name: str,
    in_mvc: int = 1,
    out_mvc: int = 4,
    hidden_mvc: int = 32,
    in_sc: int | None = None,
    out_sc: int | None = None,
    hidden_sc: int | None = 64,
    edge_mvc: int | None = None,
    edge_sc: int | None = None,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout_prob: float = 0.1,
    activation: str = "gelu",
    attention_type: str = "gatr",
    message_product: str = "geometric",
):
    if model_name == "pgagnn":
        return PGA_GNN(
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
            message=message_product,
            expansion_factor=1.0,
            factorize=True,
            use_grade_modulation=False,
            use_pseudoscalar=False,
        )
    if model_name == "gatr":
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

    raise ValueError(f"Unknown model name: {model_name}")


def get_atomref_tensor(dataset, target_idx: int, device: torch.device) -> torch.Tensor | None:
    """Get the atom reference energies for the target.

    Returns a tensor of shape (100, 1) where index corresponds to atomic number Z.
    For example, index 1 = H, 6 = C, 7 = N, 8 = O, 9 = F.
    Returns None for non-energy targets.
    """
    # Only use atomref for energy-related targets
    energy_targets = ["U0", "U", "H", "G"]
    target_name = TARGET_NAMES[target_idx]

    if target_name not in energy_targets:
        return None

    # Get atomref from dataset
    if not hasattr(dataset, "atomref") or dataset.atomref is None:
        return None

    atomref = dataset.atomref(target_idx)
    if atomref is None:
        return None

    return atomref.to(device)


@torch.no_grad()
def compute_target_stats(
    dataset,
    target_idx: int,
    ref_table: torch.Tensor | None = None,
    batch_size: int = 512,
) -> tuple[float, float]:
    """
    Compute mean and MAD of the learning target:
        y            (non-energy targets)
        y - atomref  (energy targets)

    Must be called on TRAIN SET ONLY.
    """

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_sum = 0.0
    total_count = 0

    # ---- First pass: mean ----
    for batch in loader:
        y = batch.y[:, target_idx]
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        y = y[mask]
        if ref_table is not None:
            z = batch.z.long()
            atom_energies = ref_table[z]  # (num_nodes, 1)
            # (num_graphs,)
            graph_ref = scatter_sum(atom_energies, batch.batch, dim=0).squeeze(-1)
            graph_ref = graph_ref[mask]
            y = y - graph_ref
        total_sum += y.sum().item()
        total_count += y.numel()
    mean = total_sum / max(total_count, 1)

    # ---- Second pass: MAD ----
    total_abs_dev = 0.0
    for batch in loader:
        y = batch.y[:, target_idx]
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        y = y[mask]
        if ref_table is not None:
            z = batch.z.long()
            atom_energies = ref_table[z]
            graph_ref = scatter_sum(atom_energies, batch.batch, dim=0).squeeze(-1)
            graph_ref = graph_ref[mask]
            y = y - graph_ref
        total_abs_dev += (y - mean).abs().sum().item()
    mad = total_abs_dev / max(total_count, 1)
    return mean, mad


def load_dataset(
    target: str,
    cutoff: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    data_dir: str | None = None,
):
    torch.manual_seed(seed)
    target_idx = get_target_index(target)
    transform = RadiusGraph(r=cutoff)
    dataset_root = data_dir if data_dir else str(DATASET_DIR / "qm9")
    dataset = QM9(root=dataset_root, transform=transform)
    ref_table = get_atomref_tensor(dataset, target_idx, torch.device("cpu"))
    print(f"Original dataset size: {len(dataset)} samples.")
    # Filter out samples with NaN target values
    target_values = dataset.y[:, target_idx]
    valid_indices = torch.where(~torch.isnan(target_values))[0]
    dataset = dataset[valid_indices]
    num_node_features = (
        dataset.num_node_features if hasattr(dataset, "num_node_features") else dataset.num_features
    )

    # Cormorant split: 100k train, 10% test, rest validation
    np.random.seed(0)
    data_perm = np.random.permutation(len(dataset))
    train_num = 100_000
    test_num = int(0.1 * len(dataset))

    train_idx, test_idx, val_idx = np.split(data_perm, [train_num, train_num + test_num])

    train_set = dataset[train_idx]
    val_set = dataset[val_idx]
    test_set = dataset[test_idx]

    print(f"Dataset loaded with {len(dataset)} samples.")
    print(f"Training set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    mean, mad = compute_target_stats(train_set, target_idx, ref_table)
    print(f"Target mean: {mean}, Target MAD: {mad}")

    return (
        train_loader,
        val_loader,
        test_loader,
        target_idx,
        mean,
        mad,
        num_node_features,
        ref_table,
    )


@dataclass
class GraphBatch:
    mv: torch.Tensor
    sc: torch.Tensor
    adj: torch.Tensor | None
    ref: torch.Tensor | None
    mask: torch.Tensor | None
    pos: torch.Tensor | None = None
    z: torch.Tensor | None = None
    edge_index: torch.Tensor | None = None
    edge_attr_mv: torch.Tensor | None = None
    edge_attr_sc: torch.Tensor | None = None
    edge_vect: torch.Tensor | None = None
    batch: torch.Tensor | None = None
    packed: bool = False


def _compute_edge_attributes_sparse(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    batch_vec: torch.Tensor,
    ref: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute centered positions and edge attributes for sparse batches."""
    num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0

    center = scatter_mean(pos, batch_vec, dim=0, dim_size=num_graphs)
    pos_centered = pos - center[batch_vec]

    src = pos_centered[edge_index[0]]  # (E, 3)
    dst = pos_centered[edge_index[1]]  # (E, 3)
    edge_vect = dst - src  # (E, 3)
    src_embedded = embed_point(src)  # (E, 16)
    dst_embedded = embed_point(dst)  # (E, 16)
    edge_attr_mv = equivariant_join(src_embedded, dst_embedded, ref).unsqueeze(-2)  # (E, 1, 16)
    edge_attr_sc = torch.linalg.norm(edge_vect, dim=-1, keepdim=True)  # (E, 1)
    return pos_centered, edge_vect, edge_attr_mv, edge_attr_sc


def _compute_edge_attributes_dense(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    batch_vec: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute edge attributes for dense (padded) batches."""
    _, N = pos.shape[:2]
    src = pos.unsqueeze(2)  # (B, N, 1, 3)
    dst = pos.unsqueeze(1)  # (B, 1, N, 3)
    edge_vect = src - dst  # (B, N, N, 3)
    ref_edge = ref[batch_vec[src]].squeeze(1)  # (E, 16)
    src_embedded = embed_point(pos).unsqueeze(2)  # (B, N, 1, 16)
    dst_embedded = embed_point(pos).unsqueeze(1)  # (B, 1, N, 16)
    edge_attr_mv = equivariant_join(src_embedded, dst_embedded, ref_edge).unsqueeze(-2)  # (B, N, N, 1, 16)
    edge_attr_sc = torch.linalg.norm(edge_vect, dim=-1, keepdim=True)

    adj = to_dense_adj(edge_index, batch=batch_vec, max_num_nodes=N).to(device)
    eye = torch.eye(N, device=device).unsqueeze(0)
    adj = torch.maximum(adj, eye)

    mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    off_diagonal_mask = mask_2d + eye
    adj = adj * off_diagonal_mask
    edge_vect = edge_vect * mask_2d.unsqueeze(-1)

    return edge_vect, edge_attr_mv, edge_attr_sc, adj


def embed_inputs(batch, embedding: str, device: torch.device, use_sparse: bool = False) -> GraphBatch:
    """Embed QM9 batch into multivectors and scalars.

    QM9 node features (11 dims):
    - [0-4]: One-hot atom type (H, C, N, O, F)
    - [5]: Atomic number
    - [6]: Aromatic (0/1)
    - [7]: sp
    - [8]: sp2
    - [9]: sp3 hybridization (0/1)
    - [10]: number of hydrogens

    The mask tensor indicates real nodes (1) vs padding (0) since molecules
    in a batch have different sizes and are padded to max_num_nodes.
    Shape: (B, N) where B=batch_size, N=max_num_nodes in batch
    """
    batch = batch.to(device)
    pos = batch.pos.to(device)
    feats = batch.x[:, 6:].to(device)
    z = batch.z.long().to(device)
    batch_vec = batch.batch.to(device)

    if use_sparse:
        edge_index, _ = add_self_loops(batch.edge_index, num_nodes=pos.size(0))
        edge_index = edge_index.to(device)

        center = scatter_mean(pos, batch_vec, dim=0)
        pos_centered = pos - center[batch_vec]
        mv = embed_point(pos_centered, type=embedding).unsqueeze(1)  # (num_nodes, 1, 16)

        ref = scatter_mean(mv, batch_vec, dim=0)
        _, edge_vect, edge_attr_mv, edge_attr_sc = _compute_edge_attributes_sparse(
            pos, edge_index, batch_vec, ref, device
        )

        return GraphBatch(
            mv=mv,
            sc=feats,
            adj=None,
            ref=ref,
            mask=None,
            pos=pos_centered,
            z=z,
            edge_index=edge_index,
            batch=batch_vec,
            edge_attr_mv=edge_attr_mv,
            edge_attr_sc=edge_attr_sc,
            edge_vect=edge_vect,
            packed=True,
        )

    pos, mask = to_dense_batch(pos, batch=batch_vec)
    num_nodes = mask.sum(dim=1, keepdim=True).clamp(min=1)
    center = (pos * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / num_nodes.unsqueeze(-1)
    pos = (pos - center) * mask.unsqueeze(-1)

    feats, _ = to_dense_batch(batch.x[:, 6:], batch=batch_vec)
    z, _ = to_dense_batch(z, batch=batch_vec)
    mask = mask.to(device)

    mv = embed_point(pos, type=embedding).unsqueeze(-2)  # (B, N, 1, 16)
    ref = torch.mean(mv, dim=(1, 2), keepdim=True)

    edge_vect, edge_attr_mv, edge_attr_sc, adj = _compute_edge_attributes_dense(
        pos, batch.edge_index, batch_vec, ref, mask, device
    )

    return GraphBatch(
        mv=mv,
        sc=feats,
        adj=adj,
        ref=ref,
        mask=mask,
        pos=pos,
        edge_attr_mv=edge_attr_mv,
        edge_attr_sc=edge_attr_sc,
        edge_vect=edge_vect,
        z=z,
    )


class EdgeEncoder(nn.Module):
    def __init__(self, rbf_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rbf):
        return self.mlp(rbf)


class ScalarFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sc):
        return self.mlp(sc)


class ModelWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        base_model: nn.Module,
        head: nn.Module,
        sc_dim: int = 64,
        edge_rbf_dim: int = 64,
        edge_sc_hidden: int = 128,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.base_model = base_model
        self.head = head
        self.edge_encoder = EdgeEncoder(rbf_dim=edge_rbf_dim, hidden_dim=edge_sc_hidden)

        self.distance_embedder = GaussianRadialBasisLayer(num_bases=edge_rbf_dim, cutoff=cutoff)
        self.sc_embed = ScalarFeatureEncoder(in_dim=5, hidden_dim=sc_dim)
        self.z_embed = nn.Embedding(10, sc_dim)
        self.combine = nn.Linear(2 * sc_dim, sc_dim)

    def _forward_base(
        self,
        mv: torch.Tensor,
        sc: torch.Tensor | None,
        batch: GraphBatch,
        edge_attr_sc: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.model_name == "pgagnn":
            return self.base_model(
                mv=mv,
                ref=batch.ref,
                adj=batch.adj,
                sc=sc,
                edge_index=batch.edge_index,
                edge_attr_mv=batch.edge_attr_mv,
                edge_attr_sc=edge_attr_sc,
                batch=batch.batch,
                mask=batch.mask,
            )
        if self.model_name == "gatr":
            if batch.adj is None:
                raise ValueError("GATr requires a dense attention mask (adjacency).")
            attention_mask = (batch.adj > 0).float().unsqueeze(1)
            return self.base_model(mv, scalars=sc, attention_mask=attention_mask)
        raise ValueError(f"Unknown model name: {self.model_name}")

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        sc = torch.cat(
            [
                self.z_embed(batch.z),
                self.sc_embed(batch.sc),
            ],
            dim=-1,
        )
        edge_attr_sc = batch.edge_attr_sc
        sc = self.combine(sc)

        if self.edge_encoder is not None and edge_attr_sc is not None:
            rbf = self.distance_embedder(edge_attr_sc)
            edge_attr_sc = self.edge_encoder(rbf)

        out_mv, out_sc = self._forward_base(batch.mv, sc, batch, edge_attr_sc)
        return self.head(out_mv, out_sc, batch.mask, batch.batch, batch.pos)


class GraphRegressor(nn.Module):
    def __init__(
        self,
        mv_channels: int,
        sc_channels: int,
        hidden: int = 128,
        pool_type: str = "sum",
        late_pooling: bool = False,
        readout_mode: str = "atomic_sum",
    ):
        super().__init__()
        if pool_type not in {"sum", "mean"}:
            raise ValueError(f"Unsupported pool_type: {pool_type}")
        if readout_mode not in {"atomic_sum", "pooled_mlp", "dipole_vector"}:
            raise ValueError(f"Unsupported readout_mode: {readout_mode}")

        self.pool_type = pool_type
        self.late_pooling = late_pooling
        self.readout_mode = readout_mode
        self.mv_channels = mv_channels

        in_dim = mv_channels * 5
        if sc_channels is not None:
            in_dim += sc_channels

        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.node_gate = nn.Linear(hidden, hidden)

        if self.readout_mode == "atomic_sum":
            self.atom_out = nn.Linear(hidden, 1)
        elif self.readout_mode == "pooled_mlp":
            self.final_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.charge_head = nn.Linear(hidden, 1)

    def forward(
        self,
        mv: torch.Tensor,
        sc: torch.Tensor,
        mask: torch.Tensor = None,
        batch_vec: torch.Tensor = None,
        pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict graph-level properties with target-specific readout modes."""
        inv = invariants(mv).flatten(start_dim=-2)

        if sc is not None:
            node_feats = torch.cat([inv, sc], dim=-1)
        else:
            node_feats = inv
        node_hidden = self.node_encoder(node_feats)
        node_hidden = node_hidden * torch.sigmoid(self.node_gate(node_hidden))

        if self.readout_mode == "atomic_sum":
            atom_values = self.atom_out(node_hidden)
            graph_values = self._pool_nodes(atom_values, mask, batch_vec)
            return graph_values.squeeze(-1)

        if self.readout_mode == "pooled_mlp":
            graph_rep = self._pool_nodes(node_hidden, mask, batch_vec)
            graph_values = self.final_proj(graph_rep)
            return graph_values.squeeze(-1)

        if pos is None:
            raise ValueError("Dipole-vector readout requires centered positions.")

        charges = self.charge_head(node_hidden).squeeze(-1)

        if mask is not None and batch_vec is None:
            charges = charges * mask
            pos = pos * mask.unsqueeze(-1)

        if batch_vec is not None:
            charge_correction = scatter_mean(charges, batch_vec, dim=0)[batch_vec]
        else:
            if mask is not None:
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(charges.dtype)
                charge_correction = charges.sum(dim=1, keepdim=True) / denom
                charges = charges - charge_correction
            else:
                charge_correction = charges.mean(dim=1, keepdim=True)
                charges = charges - charge_correction.squeeze(-1)

        dipole_contrib = charges.unsqueeze(-1) * pos
        graph_dipole = self._pool_nodes(dipole_contrib, mask, batch_vec)
        return torch.linalg.norm(graph_dipole + 1e-8, dim=-1)

    def _pool_nodes(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        batch_vec: torch.Tensor | None,
    ) -> torch.Tensor:
        if batch_vec is not None:
            if self.pool_type == "sum":
                return scatter_sum(x, batch_vec, dim=0)
            return scatter_mean(x, batch_vec, dim=0)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        pooled = x.sum(dim=1)
        if self.pool_type == "sum":
            return pooled

        if mask is not None:
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=x.dtype)
        else:
            denom = torch.full(
                (x.size(0), 1),
                x.size(1),
                device=x.device,
                dtype=x.dtype,
            )
        return pooled / denom


def train_one_epoch(
    model: ModelWrapper,
    loader: DataLoader,
    criterion: callable,
    optimizer: torch.optim.Optimizer,
    target_idx: int,
    device: torch.device,
    target_mean: float,
    target_mad: float,
    ref_table: torch.Tensor | None = None,
    embedding: str = "trivector",
    ema_model: AveragedModel | None = None,
    progress_bar=None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    model_parameters = list(model.parameters())

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        base_model = unwrap_model(model)
        use_sparse = (
            base_model.model_name == "pgagnn"
            and getattr(base_model.base_model, "attention_type", None) == "gatr_sparse"
            and getattr(base_model.base_model, "use_sparse", False)
        )

        graph_batch = embed_inputs(batch, embedding, device, use_sparse=use_sparse)
        pred = model(graph_batch)

        target = batch.y[:, target_idx].to(device)
        mask = ~torch.isnan(target)
        count = mask.sum().item()
        if count == 0:
            continue

        if ref_table is not None:
            z = batch.z.long().to(device)
            atom_energies = ref_table[z]
            graph_ref = scatter_sum(atom_energies, batch.batch.to(device), dim=0).squeeze(-1)
            graph_ref = graph_ref[mask]
            target_learning = target[mask] - graph_ref
        else:
            target_learning = target[mask]

        pred = pred[mask]
        if TARGET_NAMES[target_idx] == "mu":
            target_norm = target_learning / max(target_mad, 1e-8)
        else:
            target_norm = (target_learning - target_mean) / max(target_mad, 1e-8)

        loss = criterion(pred, target_norm)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=5.0)
        optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(model)

        total_loss += loss.item() * count
        total_samples += count

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    target_idx: int,
    device: torch.device,
    target_mean: float,
    target_mad: float,
    ref_table: torch.Tensor | None = None,
    embedding: str = "trivector",
):
    model.eval()

    total_abs_error = 0.0
    total_samples = 0

    for batch in loader:
        base_model = unwrap_model(model)
        use_sparse = (
            base_model.model_name == "pgagnn"
            and getattr(base_model.base_model, "attention_type", None) == "gatr_sparse"
            and getattr(base_model.base_model, "use_sparse", False)
        )

        graph_batch = embed_inputs(batch, embedding, device, use_sparse=use_sparse)
        pred_norm = model(graph_batch)

        target = batch.y[:, target_idx].to(device)
        mask = ~torch.isnan(target)
        count = mask.sum().item()
        if count == 0:
            continue

        if ref_table is not None:
            z = batch.z.long().to(device)
            atom_energies = ref_table[z]
            graph_ref = scatter_sum(atom_energies, batch.batch.to(device), dim=0).squeeze(-1)
            graph_ref = graph_ref[mask]
            target = target[mask]
            pred_phys = pred_norm[mask] * target_mad + target_mean + graph_ref
        else:
            target = target[mask]
            if TARGET_NAMES[target_idx] == "mu":
                pred_phys = pred_norm[mask] * target_mad
            else:
                pred_phys = pred_norm[mask] * target_mad + target_mean

        total_abs_error += torch.sum(torch.abs(pred_phys - target)).item()
        total_samples += count

    return total_abs_error / max(total_samples, 1)


def train_model(
    model: ModelWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: callable,
    target_idx: int,
    device: torch.device,
    num_epochs: int,
    warmup_epochs: int,
    target_mean: float,
    target_mad: float,
    lr: float,
    weight_decay: float,
    patience: int,
    ema_decay: float,
    ref_table: torch.Tensor | None = None,
    embedding: str = "trivector",
    tbw: SummaryWriter | None = None,
    checkpoint_path: str | None = None,
    last_checkpoint_path: str | None = None,
    resume_from: str | None = None,
    stop_handler: StopSignalHandler | None = None,
    signals_to_handle: list | None = None,
    no_tqdm: bool = False,
):
    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    ema_model: AveragedModel | None = None
    if ema_decay > 0.0:
        ema_avg_fn = get_ema_multi_avg_fn(ema_decay)
        ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)

    anneal_epochs = max(num_epochs - warmup_epochs, 1)
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=anneal_epochs,
        eta_min=lr * 0.001,
    )

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    # Resume from checkpoint if provided
    start_epoch = 1
    best_val = float("inf")
    losses = {
        "train_loss": [],
        "val_loss": [],
    }
    best_state = None
    epochs_without_improve = 0
    last_completed_epoch = 0

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = load_checkpoint(
            path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            ema_model=ema_model,
        )
        saved_epoch = ckpt["epoch"]
        if saved_epoch <= 0:
            inferred_epoch = len(ckpt["losses"].get("train_loss", []))
            if inferred_epoch > 0:
                print(
                    f"Checkpoint epoch metadata is stale; inferring resume epoch from loss history ({inferred_epoch})."
                )
                saved_epoch = inferred_epoch
        last_completed_epoch = saved_epoch
        start_epoch = saved_epoch + 1
        best_val = ckpt["best_val_loss"]
        losses = ckpt["losses"]
        if ckpt["target_mean"] is not None:
            target_mean = ckpt["target_mean"]
        if ckpt["target_mad"] is not None:
            target_mad = ckpt["target_mad"]
        print(f"Resumed at epoch {start_epoch}, best_val={best_val:.6f}")

    # Setup signal handler for graceful shutdown
    if stop_handler is None:
        stop_handler = StopSignalHandler()

    def build_checkpoint_payload(
        epoch: int,
        interrupted: bool = False,
        signal_name: str | None = None,
    ) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val,
            "losses": losses,
            "target_mean": target_mean,
            "target_mad": target_mad,
            "args": None,
            "interrupted": interrupted,
            "signal": signal_name,
            "rng_state": torch.get_rng_state().cpu(),
            "cuda_rng_state": (torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None),
            "np_rng_state": np.random.get_state(),
        }

    if signals_to_handle and last_checkpoint_path:
        stop_handler.install(
            save_fn=lambda interrupted, signal_name=None: save_checkpoint(
                path=last_checkpoint_path,
                payload=build_checkpoint_payload(
                    epoch=last_completed_epoch,
                    interrupted=interrupted,
                    signal_name=signal_name,
                ),
            ),
            signals_to_handle=signals_to_handle,
        )
    use_tqdm = not no_tqdm
    for epoch in range(start_epoch, num_epochs + 1):
        pbar_context = (
            tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch}/{num_epochs} [Train]",
                leave=False,
            )
            if use_tqdm
            else nullcontext()
        )
        with pbar_context as pbar:
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                target_idx,
                device,
                target_mean,
                target_mad,
                ref_table=ref_table,
                embedding=embedding,
                ema_model=ema_model,
                progress_bar=pbar,
            )

        eval_model = ema_model if ema_model is not None else model

        val_loss = evaluate(
            eval_model,
            val_loader,
            target_idx,
            device,
            target_mean,
            target_mad,
            ref_table=ref_table,
            embedding=embedding,
        )

        losses["train_loss"].append(train_loss)
        losses["val_loss"].append(val_loss)

        if tbw is not None:
            tbw.add_scalar("train/loss", train_loss, epoch)
            tbw.add_scalar("val/loss", val_loss, epoch)
            tbw.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step()

        if epoch % 10 == 0 or epoch == num_epochs:
            print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.6f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            best_state = {
                "model": unwrap_model(eval_model).state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improve += 1

        # Save last checkpoint periodically
        if last_checkpoint_path and (epoch % 10 == 0 or epoch == num_epochs):
            last_completed_epoch = epoch
            save_checkpoint(
                path=last_checkpoint_path,
                payload=build_checkpoint_payload(epoch=epoch),
            )

        # Check for stop signal
        if stop_handler is not None and stop_handler.stop_requested:
            print(f"Stop requested ({stop_handler.last_signal}). Saved checkpoint and stopping training.")
            break

        if patience > 0 and epochs_without_improve >= patience:
            print("Early stopping triggered")
            break

    # If interrupted, don't evaluate on test - just return
    if stop_handler is not None and stop_handler.stop_requested:
        losses["best_val_loss"] = best_val
        losses["test_loss"] = None
        return losses

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Evaluate best validation model on test
    test_loss = evaluate(
        model,
        test_loader,
        target_idx,
        device,
        target_mean,
        target_mad,
        ref_table=ref_table,
        embedding=embedding,
    )

    losses["test_loss"] = test_loss
    losses["best_val_loss"] = best_val

    print(f"Best val loss {best_val:.4f}")
    print(f"Test loss {test_loss:.4f}")
    if tbw is not None:
        tbw.add_scalar("test/loss", test_loss, len(losses.get("train_loss", [])))
    return losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train GA models on QM9")
    parser.add_argument("--model_name", choices=["pgagnn", "gatr"], default="pgagnn")
    parser.add_argument("--target", default="mu", help="Target name or index")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--in_mvc", type=int, default=1)
    parser.add_argument("--in_sc", type=int, default=10)
    parser.add_argument("--hidden_mvc", type=int, default=32)
    parser.add_argument("--hidden_sc", type=int, default=64)
    parser.add_argument("--out_mvc", type=int, default=16)
    parser.add_argument("--out_sc", type=int, default=32)
    parser.add_argument("--edge_mvc", type=int, default=1)
    parser.add_argument("--edge_sc", type=int, default=128)
    parser.add_argument("--embedding", type=str, default="trivector", choices=["vector", "trivector"])
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--attention_type", type=str, default="gatr_sparse")
    parser.add_argument(
        "--message_product",
        type=str,
        default="geometric",
        choices=["geometric", "sum", "linear"],
        help="GA_GNN ablation for replacing the geometric product in message combination.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_sparse", action="store_true")
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--runs_dir", type=str, default=str(ROOT_DIR / "runs" / "qm9"))
    parser.add_argument("--data_dir", type=str, default=str(DATASET_DIR / "qm9"))
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--handle_signals",
        type=str,
        default=None,
        help="Comma-separated signal names to handle (e.g., 'TERM,USR1').",
    )
    parser.add_argument(
        "--no_time_suffix", action="store_true", help="Disable time suffix for run directory."
    )
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ema_decay < 0.0 or args.ema_decay >= 1.0:
        raise ValueError("--ema_decay must be in [0, 1).")

    if args.gpu is not None:
        if not torch.cuda.is_available():
            print("Warning: --gpu specified but CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    seed_everything(args.seed)

    signals_to_handle = parse_signals(args.handle_signals) if args.handle_signals else None

    print("=" * 70)
    print("Training configuration:")
    print("=" * 70)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:20s}: {value}")
    print("=" * 70)

    print("\nLoading dataset...")
    (
        train_loader,
        val_loader,
        test_loader,
        target_idx,
        target_mean,
        target_mad,
        _,
        ref_table,
    ) = load_dataset(
        target=args.target,
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    target_name = TARGET_NAMES[target_idx]
    readout_mode, pool_type, late_pooling = get_target_readout_config(target_name)
    args.resolved_target = target_name
    args.readout_mode = readout_mode
    args.pool_type = pool_type
    args.late_pooling = late_pooling

    model = get_model(
        model_name=args.model_name,
        in_mvc=args.in_mvc,
        out_mvc=args.out_mvc,
        hidden_mvc=args.hidden_mvc,
        in_sc=args.in_sc,
        out_sc=args.out_sc,
        hidden_sc=args.hidden_sc,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_prob=args.dropout,
        activation=args.activation,
        attention_type=args.attention_type,
        edge_mvc=args.edge_mvc,
        edge_sc=args.edge_sc,
        message_product=args.message_product,
    )
    model.use_sparse = args.use_sparse

    head = GraphRegressor(
        mv_channels=args.out_mvc,
        sc_channels=args.out_sc,
        pool_type=pool_type,
        late_pooling=late_pooling,
        readout_mode=readout_mode,
    )
    model = ModelWrapper(
        model_name=args.model_name,
        base_model=model,
        head=head,
        cutoff=args.cutoff,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{args.model_name}' has {total_params} trainable parameters.")
    print(
        f"Readout for target '{target_name}': mode={readout_mode}, pool={pool_type}, late_pooling={late_pooling}"
    )
    args.total_params = total_params

    # Create or recover the run directory.
    run_dir, run_name = resolve_run_directory(
        base_dir=args.runs_dir,
        args=args,
    )
    print(f"\nRun directory: {run_dir}\n")
    print(f"Run name: {run_name}\n")

    checkpoint_path = os.path.join(run_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(run_dir, "last_model.pt")
    tb_run_dir = Path(run_dir).resolve().parent / "tensorboard" / run_name
    tbw = init_tensorboard_writer(args, tb_run_dir) if not getattr(args, "no_tensorboard", False) else None
    if ref_table is not None:
        ref_table = ref_table.to(device)
        print(f"Using atom reference energies for target '{args.target}'")
    else:
        print(f"Not using atom reference energies for target '{args.target}'")

    criterion = nn.SmoothL1Loss(beta=0.5) if args.target == "mu" else nn.L1Loss()

    stop_handler = StopSignalHandler()

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        target_idx=target_idx,
        device=device,
        num_epochs=args.epochs,
        target_mean=target_mean,
        target_mad=target_mad,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        ema_decay=args.ema_decay,
        ref_table=ref_table,
        embedding=args.embedding,
        tbw=tbw,
        checkpoint_path=checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        resume_from=args.resume_from,
        stop_handler=stop_handler,
        signals_to_handle=signals_to_handle,
        no_tqdm=args.no_tqdm,
    )

    save_metrics_to_json(
        results=results,
        run_dir=run_dir,
        args=args,
    )

    if tbw is not None:
        tbw.close()

    if stop_handler.stop_requested:
        print(f"Exiting with code 3 to signal SLURM requeue (interrupted by {stop_handler.last_signal}).")
        sys.exit(3)


if __name__ == "__main__":
    main()
