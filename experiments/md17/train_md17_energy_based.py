import argparse
import json
import os
import random
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from e3nn.o3 import Irreps, spherical_harmonics
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import SWALR, AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph
from torch_geometric.utils import scatter, to_dense_adj, to_dense_batch
from torch_scatter import scatter_mean
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[2]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
for sub_path in [
    ROOT_DIR / "geometric-algebra-transformer",
    ROOT_DIR / "egnn",
    ROOT_DIR / "Steerable-E3-GNN" / "models",
]:
    if str(sub_path) not in sys.path:
        sys.path.append(str(sub_path))

from balanced_irreps import WeightBalancedIrreps
from gatr import GATr, MLPConfig, SelfAttentionConfig
from md17_data import MD17WithFallback
from segnn.segnn import SEGNN

from egnn.models.egnn_clean.egnn_clean import EGNN
from src.checkpointing import StopSignalHandler, load_checkpoint, parse_signals, save_checkpoint
from src.ggnn import GGNN
from src.layers import GaussianRadialBasisLayer
from src.pgagnn import PGA_GNN
from src.primitives import embed_point, outer_product
from src.utils import seed_everything

# Extend pytorch printing options for debugging (disable line breaks)
torch.set_printoptions(profile="short", linewidth=1000, sci_mode=True)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "datasets"
KCALMOL_TO_MEV = 43.3641
_VALID_SDPA_MODES = {"auto", "math", "default"}


def _kcal_to_mev(value: float) -> float:
    return value * KCALMOL_TO_MEV


def _resolve_force_math_sdpa(sdpa_mode: str, training: bool) -> bool:
    if sdpa_mode == "auto":
        return training
    if sdpa_mode == "math":
        return True
    if sdpa_mode == "default":
        return False
    raise ValueError(f"Unknown --sdpa_mode '{sdpa_mode}'. Expected one of {sorted(_VALID_SDPA_MODES)}.")


@contextmanager
def _temporary_attention_math_sdpa(enabled: bool):
    """Temporarily force math SDPA backend at runtime for all SDPA calls."""
    if not enabled or not torch.cuda.is_available():
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        kernel_ctx = sdpa_kernel(backends=[SDPBackend.MATH])
    except (ImportError, AttributeError):
        kernel_ctx = torch.nn.attention.sdpa_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )

    with kernel_ctx:
        yield


def _safe_l2_norm(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Numerically stable L2 norm for higher-order gradient paths."""
    return torch.sum(x * x, dim=dim, keepdim=keepdim)


@dataclass
class GraphBatch:
    mv: torch.Tensor
    sc: torch.Tensor
    adj: torch.Tensor | None
    ref: torch.Tensor | None
    mask: torch.Tensor | None
    pos_ref: torch.Tensor | None = None
    vect_src_dst: torch.Tensor | None = None
    edge_index: torch.Tensor | None = None
    edge_attr_mv: torch.Tensor | None = None
    edge_attr_sc: torch.Tensor | None = None
    batch: torch.Tensor | None = None
    packed: bool = False


class EdgeEncoder(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, rbf):
        return self.mlp(rbf)


class GAModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        head: nn.Module,
        num_elements: int = 10,
        sc_dim: int = 64,
        edge_rbf_dim: int = 64,
        edge_sc_hidden: int = 128,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.head = head
        self.edge_encoder = EdgeEncoder(in_dim=edge_rbf_dim, out_dim=edge_sc_hidden)
        self.distance_embedder = GaussianRadialBasisLayer(num_bases=edge_rbf_dim, cutoff=cutoff)
        self.z_embed = nn.Embedding(num_elements, sc_dim)
        # self.vect_encoder = nn.Linear(3, edge_sc_hidden)

    def forward(self, graph_batch: GraphBatch):
        sc_embedded = self.z_embed(graph_batch.sc)

        if isinstance(self.base_model, EGNN):
            rbf = self.distance_embedder(graph_batch.edge_attr_sc)
            edge_attr_sc_encoded = self.edge_encoder(rbf)
            node_features, _ = self.base_model(
                h=sc_embedded,
                x=graph_batch.pos_ref,
                edges=graph_batch.edge_index,
                edge_attr=edge_attr_sc_encoded,
            )
            return self.head(node_features, batch_vec=graph_batch.batch)

        if isinstance(self.base_model, SEGNN):
            segnn_graph = Data(
                pos=graph_batch.pos_ref,
                sc=graph_batch.sc,
                edge_index=graph_batch.edge_index,
                batch=graph_batch.batch,
            )
            segnn_graph = O3Transform(lmax_attr=1)(segnn_graph).to(graph_batch.pos_ref.device)
            node_features = self.base_model(segnn_graph)
            return self.head(node_features, batch_vec=graph_batch.batch)

        # embed distances with internal distance embedder then encode
        rbf = self.distance_embedder(graph_batch.edge_attr_sc)
        edge_attr_sc_encoded = self.edge_encoder(rbf)
        if isinstance(self.base_model, GATr):
            out_mv, out_sc = self.base_model(
                graph_batch.mv,
                scalars=sc_embedded,
                attention_mask=(graph_batch.adj > 0).float().unsqueeze(1),
            )
        elif isinstance(self.base_model, GGNN):
            rbf = self.distance_embedder(graph_batch.edge_attr_sc)
            edge_attr_sc_encoded = self.edge_encoder(rbf)
            edge_vectors = graph_batch.vect_src_dst.unsqueeze(-2)
            node_vectors = graph_batch.pos_ref.unsqueeze(-2)
            out_v, out_s = self.base_model(
                vectors=node_vectors,
                scalars=sc_embedded,
                edge_vectors=edge_vectors,
                edge_scalars=edge_attr_sc_encoded,
                adj=graph_batch.adj,
            )
            return self.head(out_v, out_s, mask=graph_batch.mask, batch_vec=graph_batch.batch)
        else:
            out_mv, out_sc = self.base_model(
                mv=graph_batch.mv,
                ref=graph_batch.ref,
                adj=graph_batch.adj,
                sc=sc_embedded,
                edge_index=graph_batch.edge_index,
                edge_attr_mv=graph_batch.edge_attr_mv,
                edge_attr_sc=edge_attr_sc_encoded,
                batch=graph_batch.batch,
                mask=graph_batch.mask,
            )
        return self.head(out_mv, out_sc, graph_batch.mask, graph_batch.batch)


def create_run_directory(base_dir, args, group: str | None = None):
    """Create a unique directory for this run."""
    if getattr(args, "no_time_suffix", False):
        run_name = f"{args.model_name}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        run_name = f"{args.model_name}_bs{args.batch_size}_hl{args.num_layers}_h{args.hidden_mvc}_lr{args.lr}_{timestamp}"
    target_name = group or getattr(args, "target", None) or getattr(args, "dataset", "default")
    target_name = target_name.replace(" ", "_").lower()
    # target_dir = os.path.join(base_dir, target_name)
    target_dir = base_dir
    os.makedirs(target_dir, exist_ok=True)
    run_dir = os.path.join(target_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    return run_dir, run_name


def resolve_run_directory(base_dir, args, group: str | None = None):
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

    return create_run_directory(base_dir, args, group=group)


def init_tensorboard_writer(args, run_dir: str) -> SummaryWriter | None:
    if getattr(args, "no_tensorboard", False):
        return None
    tb_dir = os.path.join(run_dir, "tensorboard")
    return SummaryWriter(log_dir=tb_dir)


def clone_state_dict_to_cpu(state_dict: dict) -> dict:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def save_metrics_to_json(results, run_dir, args):
    """Save training metrics and summary to JSON file."""
    metrics_data = {
        "config": vars(args),
        "summary": {
            "best_val_loss": results["best_val_loss"],
            "best_train_loss": min(results["train"]),
            "test_loss": results.get("test_loss", None),
            "best_val_force_mae_mev_A": min(results["val_force_mae_mev_A"]),
            "best_val_energy_mae_mev": min(results["val_energy_mae_mev"]),
            "test_force_mae_mev_A": results.get("test_force_mae_mev_A", None),
            "test_energy_mae_mev": results.get("test_energy_mae_mev", None),
            "total_epochs": len(results["train"]),
        },
        "units": {
            "energy_mae": "meV",
            "force_mae": "meV/Å",
        },
        "metrics": {
            "train_losses": results["train"],
            "val_losses": results["val"],
            "val_force_mae_mev_A": results.get("val_force_mae_mev_A", []),
            "val_energy_mae_mev": results.get("val_energy_mae_mev", []),
        },
    }

    json_path = os.path.join(run_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Metrics JSON saved to: {json_path}")
    return json_path


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
    message: str = "geometric",
):
    if model_name not in {"pgagnn", "gatr", "egnn", "segnn"}:
        raise ValueError(f"Unknown model name: {model_name}")

    # All parameters for all models
    params = {
        "in_mvc": in_mvc,
        "out_mvc": out_mvc,
        "hidden_mvc": hidden_mvc,
        "in_sc": in_sc,
        "out_sc": out_sc,
        "hidden_sc": hidden_sc,
        "edge_mvc": edge_mvc,
        "edge_sc": edge_sc,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout_prob": dropout_prob,
        "activation": activation,
        "attention_type": attention_type,
    }
    if model_name == "pgagnn":
        # params.pop("num_heads")  # PGA_GNN does not use num_heads
        return PGA_GNN(
            **params, message=message, factorize=True, use_grade_modulation=False, use_pseudoscalar=False
        )
    elif model_name == "gatr":  # gatr
        # If GATr expects different parameter names, adjust here
        return GATr(
            in_mv_channels=params["in_mvc"],
            out_mv_channels=params["out_mvc"],
            hidden_mv_channels=params["hidden_mvc"],
            in_s_channels=params["in_sc"],
            out_s_channels=params["out_sc"],
            hidden_s_channels=params["hidden_sc"],
            num_blocks=params["num_layers"],
            dropout_prob=params["dropout_prob"],
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )
    elif model_name == "ggnn":
        return GGNN(
            in_vc=params["in_mvc"],
            out_vc=params["out_mvc"],
            hidden_vc=params["hidden_mvc"],
            in_sc=params["in_sc"],
            out_sc=params["out_sc"],
            hidden_sc=params["hidden_sc"],
            edge_vc=params["edge_mvc"] if params["edge_mvc"] is not None else 1,
            edge_sc=params["edge_sc"],
            num_layers=params["num_layers"],
            num_heads=params["num_heads"],
            dropout_prob=params["dropout_prob"],
        )
    elif model_name == "egnn":
        return EGNN(
            in_node_nf=in_sc,
            hidden_nf=hidden_sc,
            out_node_nf=hidden_sc,
            in_edge_nf=edge_sc,
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_layers=num_layers,
            residual=True,
            attention=False,
            normalize=False,
            tanh=False,
        )
    elif model_name == "segnn":
        return SEGNN(
            input_irreps=Irreps("1x1o + 1x0e"),
            output_irreps=Irreps(f"{hidden_sc}x0e"),
            edge_attr_irreps=Irreps.spherical_harmonics(1),
            node_attr_irreps=Irreps.spherical_harmonics(1),
            additional_message_irreps=Irreps("1x0e"),
            hidden_irreps=WeightBalancedIrreps(
                Irreps(f"{hidden_sc}x0e"),
                Irreps.spherical_harmonics(1),
                sh=True,
                lmax=1,
            ),
            num_layers=num_layers,
            task="node",
            pool=None,
            norm=None,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class O3Transform:
    def __init__(self, lmax_attr):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)

    def __call__(self, graph):
        pos = graph.pos
        z = graph.sc.float().unsqueeze(-1)
        rel_pos = pos[graph.edge_index[0]] - pos[graph.edge_index[1]]
        edge_dist = torch.sqrt(rel_pos.pow(2).sum(1, keepdims=True))

        graph.edge_attr = spherical_harmonics(
            self.attr_irreps, rel_pos, normalize=True, normalization="integral"
        )
        graph.node_attr = scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce="mean")

        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        graph.x = torch.cat((pos, z), dim=-1)
        graph.additional_message_features = edge_dist
        return graph


class ScalarEnergyPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 * hidden),
            nn.LayerNorm(4 * hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        x_dense, mask = to_dense_batch(x, batch=batch_vec)
        e_atom = self.mlp(x_dense) * mask.unsqueeze(-1).float()
        e_total = e_atom.sum(dim=1)
        return e_total.squeeze(-1)


def load_dataset(
    name: str,
    cutoff: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    split_idx: int = 1,
    data_dir: str | None = None,
):
    """Load MD17 dataset and return train/val/test loaders.

    For revised MD17 datasets, uses official splits from the dataset.
    For non-revised datasets, uses random split.

    Args:
        split_idx: Which split to use (1-5) for revised datasets
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = RadiusGraph(r=cutoff)
    root_dir = str(DATASET_DIR / "md17") if data_dir is None else data_dir
    os.makedirs(root_dir, exist_ok=True)
    dataset = MD17WithFallback(root=root_dir, name=name, transform=transform)

    print(f"MD17 '{name}' dataset loaded with {len(dataset)} samples.")


    # Try to load official splits for revised datasets
    splits_dir = Path(root_dir) / "raw" / "rmd17" / "splits"
    if name.startswith("revised") and splits_dir.exists():
        # Load official train/test split
        train_idx_file = splits_dir / f"index_train_{split_idx:02d}.csv"
        test_idx_file = splits_dir / f"index_test_{split_idx:02d}.csv"

        if train_idx_file.exists() and test_idx_file.exists():
            print(f"Using official split {split_idx} from {splits_dir}")
            train_indices = np.loadtxt(train_idx_file, dtype=int)
            test_indices = np.loadtxt(test_idx_file, dtype=int)
            rng = np.random.default_rng(seed)
            rng.shuffle(train_indices)

            # Create train/val/test split from indices
            # Use 90% of official train for training, 10% for validation
            num_train = len(train_indices)
            val_size = 50
            train_size = num_train - val_size

            train_idx_split = train_indices[:train_size]
            val_idx_split = train_indices[train_size:]

            train_set = dataset[train_idx_split.tolist()]
            val_set = dataset[val_idx_split.tolist()]
            test_set = dataset[test_indices.tolist()]

            print(
                f"Dataset '{name}' splits: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}"
            )
        else:
            print(f"Warning: Split files not found at {splits_dir}, using random split")
            # Fall back to random split
            total_len = len(dataset)
            train_len = 1000
            val_len = 50
            test_len = total_len - train_len - val_len

            generator = torch.Generator().manual_seed(seed)
            train_set, val_set, test_set = random_split(
                dataset, [train_len, val_len, test_len], generator=generator
            )
    else:
        # We only use the revised datasets with official splits
        raise ValueError("Only revised MD17 datasets with official splits are allowed.")

    # MACE-style normalization statistics on training split.
    per_atom_energies = torch.cat([d.energy.view(-1) / d.z.numel() for d in train_set])
    mu_E = per_atom_energies.mean()

    all_forces = torch.cat([d.force.view(-1, 3) for d in train_set], dim=0)
    sigma_F = torch.sqrt((all_forces**2).sum() / (3 * all_forces.shape[0]))
    sigma_F = sigma_F.clamp_min(1e-8)

    print(f"Per-atom energy shift mu_E: {mu_E.item():.6f}")
    print(f"Force RMS scale sigma_F: {sigma_F.item():.6f}")

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
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        mu_E,
        sigma_F,
    )


def embed_inputs(
    batch,
    device,
    use_sparse: bool = False,
) -> GraphBatch:
    if use_sparse:
        pos = batch.pos.to(device)
        batch_vec = batch.batch.to(device)
        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0

        # Center positions per graph
        pos_sum = torch.zeros((num_graphs, 3), device=device, dtype=pos.dtype)
        pos_sum.index_add_(0, batch_vec, pos)
        counts = torch.bincount(batch_vec, minlength=num_graphs).clamp_min(1).to(pos.dtype)
        mean_pos = pos_sum / counts[:, None]
        pos_centered = pos - mean_pos[batch_vec]

        mv = embed_point(pos_centered).unsqueeze(1)  # (N, 1, 16)

        z_int = batch.z.long().to(device)
        sc = z_int

        edge_index = batch.edge_index.to(device)
        src, dst = edge_index[0], edge_index[1]
        u = pos_centered[dst] - pos_centered[src]  # (E, 3)
        src_embedded = embed_point(pos_centered[src])  # (E, 16)
        dst_embedded = embed_point(pos_centered[dst]) # (E, 16)
        edge_attr_mv = outer_product(src_embedded, dst_embedded).unsqueeze(-2)  # (E, 1, 16)
        d = _safe_l2_norm(u, dim=-1, keepdim=True)
        edge_attr_sc = d

        # Reference (Mean of MV) per graph (Num_grahs, 1, 16)
        ref = scatter_mean(mv, batch_vec, dim=0, dim_size=num_graphs).to(device)

        mv = mv.to(device)
        sc = sc.to(device)
        edge_attr_mv = edge_attr_mv.to(device)
        edge_attr_sc = edge_attr_sc.to(device)

        return GraphBatch(
            mv=mv,
            sc=sc,
            adj=None,
            ref=ref,
            mask=None,
            pos_ref=pos_centered,
            vect_src_dst=u,
            edge_index=edge_index,
            edge_attr_mv=edge_attr_mv,
            edge_attr_sc=edge_attr_sc,
            batch=batch_vec,
            packed=True,
        )

    pos, mask = to_dense_batch(batch.pos, batch=batch.batch)
    mean_pos = torch.mean(pos, dim=1, keepdim=True)
    pos_centered = pos - mean_pos  # (B, N, 3)
    mv = embed_point(pos_centered).unsqueeze(-2)  # (B, N, 1, 16)
    pos_i = pos_centered.unsqueeze(2)  # (B, N, 1, 3)
    pos_j = pos_centered.unsqueeze(1)  # (B, 1, N, 3)
    u = pos_j - pos_i
    pos_i_embedded = embed_point(pos_i).unsqueeze(2)  # (B, N, 1, 16)
    pos_j_embedded = embed_point(pos_j).unsqueeze(1)  # (B, 1, N, 16)
    edge_attr_mv = outer_product(pos_i_embedded, pos_j_embedded).unsqueeze(-2)  # (B, N, N, 1, 16)
    distances = _safe_l2_norm(u, dim=-1, keepdim=True)
    edge_attr_sc = distances

    z_int = batch.z.long()
    sc = z_int
    sc, _ = to_dense_batch(sc, batch=batch.batch)

    mask = mask.to(device)
    mv = mv.to(device)
    sc = sc.to(device)
    edge_attr_mv = edge_attr_mv.to(device)
    edge_attr_sc = edge_attr_sc.to(device)
    u = u.to(device)

    adj = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=pos.size(1)).to(device)
    eye = torch.eye(adj.size(1), device=device).unsqueeze(0)
    mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    adj = torch.maximum(adj, eye) * (mask_2d + eye)
    adj = adj.to(device)

    ref = torch.mean(mv, dim=(1, 2), keepdim=True).to(device)  # (B, 1, 16)

    return GraphBatch(
        mv=mv,
        sc=sc,
        adj=adj,
        ref=ref,
        mask=mask,
        pos_ref=pos_centered,
        vect_src_dst=u,
        edge_index=None,
        edge_attr_mv=edge_attr_mv,
        edge_attr_sc=edge_attr_sc,
        batch=None,
        packed=False,
    )


class EnergyPredictor(nn.Module):
    def __init__(self, mv_channels: int, sc_channels: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(mv_channels + sc_channels, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 * hidden),
            nn.LayerNorm(4 * hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, mv, sc, mask=None, batch_vec=None):
        if batch_vec is not None and mv.dim() == 3:
            mv_flat = mv.reshape(mv.size(0), -1)
            mv_dense, mask = to_dense_batch(mv_flat, batch=batch_vec)
            mv = mv_dense.view(mv_dense.size(0), mv_dense.size(1), -1, 16)
            if sc is not None:
                sc, _ = to_dense_batch(sc, batch=batch_vec)
        if mask is None:
            mask = torch.ones(mv.shape[:2], device=mv.device, dtype=torch.bool)
        invariant = mv[..., 0]  # (B, N, mv_channels)
        x = torch.cat([invariant, sc], dim=-1)
        e_atom = self.mlp(x) * mask.unsqueeze(-1).float()  # (B, N, 1)
        e_total = e_atom.sum(dim=1)  # (B, 1)
        return e_total.squeeze(-1)  # (B,)


class GGNNEnergyPredictor(nn.Module):
    def __init__(self, vc_channels: int, sc_channels: int, hidden: int = 128):
        super().__init__()
        in_channels = sc_channels + vc_channels

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 * hidden),
            nn.LayerNorm(4 * hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, vectors, scalars, mask=None, batch_vec=None):
        if mask is None:
            mask = torch.ones(scalars.shape[:2], device=scalars.device, dtype=torch.bool)
        v_norm = torch.sqrt(vectors.square().sum(dim=-1) + 1e-8)  # Shape: (B, N, vc_channels)
        invariant_features = torch.cat([scalars, v_norm], dim=-1)  # Shape: (B, N, sc_channels + vc_channels)
        e_atom = self.mlp(invariant_features) * mask.unsqueeze(-1).float()  # (B, N, 1)
        e_total = e_atom.sum(dim=1)  # (B, 1)
        return e_total.squeeze(-1)  # (B,)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, AveragedModel) else model


def should_use_sparse_path(model_name: str, model: torch.nn.Module) -> bool:
    base_model = unwrap_model(model).base_model
    if model_name in {"egnn", "segnn"}:
        return True
    return (
        model_name in {"gagat", "pgagnn"}
        and getattr(base_model, "attention_type", None) == "gatr_sparse"
        and getattr(base_model, "use_sparse", False)
    )


def _get_target_energy(batch, device: torch.device) -> torch.Tensor:
    target_energy = batch.energy.to(device)
    if target_energy.dim() > 1:
        target_energy = target_energy.view(target_energy.size(0), -1).squeeze(-1)
    return target_energy.view(-1)


def predict_energy_and_forces(
    model_name: str,
    model: torch.nn.Module,
    batch,
    device: torch.device,
    mu_E: torch.Tensor,
    sigma_F: torch.Tensor,
    use_sparse: bool | None = None,
    sdpa_mode: str = "auto",
):
    batch = batch.to(device)
    pos = batch.pos.to(device)
    pos.requires_grad_(True)
    batch.pos = pos

    if use_sparse is None:
        use_sparse = should_use_sparse_path(model_name, model)

    graph_batch = embed_inputs(batch, device, use_sparse=use_sparse)
    force_math_sdpa = _resolve_force_math_sdpa(sdpa_mode=sdpa_mode, training=model.training)
    with _temporary_attention_math_sdpa(force_math_sdpa):
        pred_energy_norm = model(graph_batch)

    mu_E_dev = mu_E.to(device=device, dtype=pred_energy_norm.dtype)
    sigma_F_dev = sigma_F.to(device=device, dtype=pred_energy_norm.dtype)

    pred_forces_norm_grad = -torch.autograd.grad(
        outputs=pred_energy_norm,
        inputs=pos,
        grad_outputs=torch.ones_like(pred_energy_norm),
        create_graph=model.training,
        retain_graph=model.training,
        allow_unused=True,
    )[0]

    pred_forces_physical = pred_forces_norm_grad * sigma_F_dev

    batch_vec = batch.batch.to(device)
    pred_forces_norm_dense, mask = to_dense_batch(pred_forces_norm_grad, batch=batch_vec)
    pred_forces_physical_dense, _ = to_dense_batch(pred_forces_physical, batch=batch_vec)
    target_forces_dense, _ = to_dense_batch(batch.force.to(device), batch=batch_vec)
    target_energy = _get_target_energy(batch, device)

    num_atoms_per_graph = mask.sum(dim=1).to(pred_energy_norm.dtype)
    pred_energy_denorm = pred_energy_norm * sigma_F_dev + num_atoms_per_graph * mu_E_dev

    return (
        pred_energy_norm,
        pred_energy_denorm,
        target_energy,
        pred_forces_norm_dense,
        pred_forces_physical_dense,
        target_forces_dense,
        mask.to(device),
    )


def batch_force_mse_sum_and_count(
    pred_forces: torch.Tensor,
    target_forces_dense: torch.Tensor,
    mask: torch.Tensor,
    sigma_F: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    sigma_F = sigma_F.to(device=pred_forces.device, dtype=pred_forces.dtype)
    target_norm = target_forces_dense / sigma_F
    pred_forces_norm = pred_forces
    mse_per_component = (pred_forces_norm - target_norm) ** 2
    mse_sum = (mse_per_component * mask.unsqueeze(-1).float()).sum()
    num_atoms = mask.sum().item()
    return mse_sum, num_atoms


def batch_energy_mse_sum_and_count(
    pred_energy: torch.Tensor,
    target_energy: torch.Tensor,
    n_atoms_per_graph: torch.Tensor,
    mu_E: torch.Tensor,
    sigma_F: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    mu_E = mu_E.to(device=pred_energy.device, dtype=pred_energy.dtype)
    sigma_F = sigma_F.to(device=pred_energy.device, dtype=pred_energy.dtype)
    target_norm = (target_energy - n_atoms_per_graph.to(pred_energy.dtype) * mu_E) / sigma_F
    pred_norm = pred_energy
    mse_sum = ((pred_norm - target_norm) ** 2).sum()
    return mse_sum, target_energy.numel()


def train_one_epoch(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mu_E: torch.Tensor,
    sigma_F: torch.Tensor,
    force_weight: float,
    energy_weight: float,
    sdpa_mode: str = "auto",
    ema_model: AveragedModel | None = None,
    progress_bar=None,
) -> tuple[float, float, float]:
    model.train()

    total_force_mse_sum = 0.0
    total_energy_mse_sum = 0.0
    total_weighted_loss_sum = 0.0
    total_atom_count = 0
    total_graph_count = 0
    model_parameters = list(model.parameters())
    use_sparse = should_use_sparse_path(model_name, model)

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        (
            pred_energy_norm,
            _,
            target_energy,
            pred_forces_norm,
            _,
            target_forces_dense,
            mask,
        ) = predict_energy_and_forces(
            model_name=model_name,
            model=model,
            batch=batch,
            device=device,
            mu_E=mu_E,
            sigma_F=sigma_F,
            use_sparse=use_sparse,
            sdpa_mode=sdpa_mode,
        )
        force_mse_sum, num_atoms = batch_force_mse_sum_and_count(
            pred_forces=pred_forces_norm,
            target_forces_dense=target_forces_dense,
            mask=mask,
            sigma_F=sigma_F,
        )
        n_atoms_per_graph = mask.sum(dim=1).to(target_energy.dtype)
        energy_mse_sum, num_graphs = batch_energy_mse_sum_and_count(
            pred_energy=pred_energy_norm,
            target_energy=target_energy,
            n_atoms_per_graph=n_atoms_per_graph,
            mu_E=mu_E,
            sigma_F=sigma_F,
        )

        force_loss = force_mse_sum / max(num_atoms * 3, 1)
        energy_loss = energy_mse_sum / max(num_graphs, 1)
        loss = force_weight * force_loss + energy_weight * energy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=10.0)
        optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(model)

        total_force_mse_sum += force_mse_sum.item()
        total_energy_mse_sum += energy_mse_sum.item()
        total_weighted_loss_sum += loss.item()
        total_atom_count += num_atoms * 3
        total_graph_count += num_graphs

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "force": f"{force_loss.item():.4f}",
                    "energy": f"{energy_loss.item():.4f}",
                }
            )

    avg_force = total_force_mse_sum / max(total_atom_count, 1)
    avg_energy = total_energy_mse_sum / max(total_graph_count, 1)
    avg_total = force_weight * avg_force + energy_weight * avg_energy
    return avg_total, avg_force, avg_energy


def evaluate(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mu_E: torch.Tensor,
    sigma_F: torch.Tensor,
    force_weight: float,
    energy_weight: float,
    sdpa_mode: str = "auto",
) -> dict[str, float]:
    model.eval()

    # Freeze model and head parameters to avoid building gradients for weights
    # while still allowing gradients w.r.t. `pos` for force computation.
    for param in model.parameters():
        param.requires_grad_(False)

    total_norm_mse = 0.0
    total_force_abs_error = 0.0
    total_energy_norm_mse = 0.0
    total_energy_abs_error = 0.0
    total_atom_count = 0
    total_graph_count = 0
    use_sparse = should_use_sparse_path(model_name, model)
    sigma_F = sigma_F.to(device)
    mu_E = mu_E.to(device)

    for batch in loader:
        (
            pred_energy_norm,
            pred_energy,
            target_energy,
            pred_forces_norm,
            pred_forces,
            target_forces_dense,
            mask,
        ) = predict_energy_and_forces(
            model_name=model_name,
            model=model,
            batch=batch,
            device=device,
            mu_E=mu_E,
            sigma_F=sigma_F,
            use_sparse=use_sparse,
            sdpa_mode=sdpa_mode,
        )

        target_norm = target_forces_dense / sigma_F
        mask_expanded = mask.unsqueeze(-1).float()

        norm_error_sq = (pred_forces_norm - target_norm) ** 2
        total_norm_mse += (norm_error_sq * mask_expanded).sum().item()

        force_abs_error = torch.abs(pred_forces - target_forces_dense)
        total_force_abs_error += (force_abs_error * mask_expanded).sum().item()

        num_atoms = mask.sum().item()
        total_atom_count += num_atoms * 3
        num_graphs = target_energy.numel()
        total_graph_count += num_graphs

        n_atoms_per_graph = mask.sum(dim=1).to(target_energy.dtype)
        target_energy_norm = (target_energy - n_atoms_per_graph * mu_E) / sigma_F
        total_energy_norm_mse += ((pred_energy_norm - target_energy_norm) ** 2).sum().item()
        total_energy_abs_error += torch.abs(pred_energy - target_energy).sum().item()

    force_divider = max(total_atom_count, 1)
    energy_divider = max(total_graph_count, 1)

    avg_force_norm_mse = total_norm_mse / force_divider
    avg_force_mae_kcal = total_force_abs_error / force_divider
    avg_energy_norm_mse = total_energy_norm_mse / energy_divider
    avg_energy_mae_kcal = total_energy_abs_error / energy_divider
    avg_total_loss = force_weight * avg_force_norm_mse + energy_weight * avg_energy_norm_mse

    # Unfreeze parameters before returning (so training can continue normally)
    for param in model.parameters():
        param.requires_grad_(True)

    return {
        "total_loss": avg_total_loss,
        "force_norm_mse": avg_force_norm_mse,
        "force_mae_mev_A": _kcal_to_mev(avg_force_mae_kcal),
        "energy_norm_mse": avg_energy_norm_mse,
        "energy_mae_mev": _kcal_to_mev(avg_energy_mae_kcal),
    }


def train_model(
    model_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    mu_E: torch.Tensor,
    sigma_F: torch.Tensor,
    force_weight: float,
    energy_weight: float,
    lr: float,
    weight_decay: float,
    patience: int,
    ema_decay: float,
    swa_start_epoch: int,
    swa_lr: float,
    sdpa_mode: str = "auto",
    checkpoint_path: str | None = None,
    last_checkpoint_path: str | None = None,
    resume_from: str | None = None,
    stop_handler: StopSignalHandler | None = None,
    signals_to_handle: list | None = None,
    tbw: SummaryWriter | None = None,
    no_tqdm: bool = False,
):
    params = list(model.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=True,
    )
    ema_model: AveragedModel | None = None
    if ema_decay > 0.0:
        ema_avg_fn = get_ema_multi_avg_fn(ema_decay)
        ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)

    # SWA uses uniform averaging and is activated only after a configured epoch.
    swa_model: AveragedModel | None = AveragedModel(model)
    swa_activated = False

    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=30)
    warmup_epochs = 0
    # warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=swa_start_epoch - warmup_epochs, eta_min=1e-6)
    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, cosine_scheduler],
    #     milestones=[warmup_epochs],
    # )
    scheduler = cosine_scheduler
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    best_val = float("inf")
    best_force_mae = float("inf")
    best_energy_mae = float("inf")
    best_swa_val = float("inf")
    best_swa_force_mae = float("inf")
    best_swa_energy_mae = float("inf")
    losses = {
        "train": [],
        "val": [],
        "val_force_mae_mev_A": [],
        "val_energy_mae_mev": [],
    }
    best_state = None
    best_swa_state = None
    epochs_without_improve = 0
    start_epoch = 1
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
            swa_model=swa_model,
            swa_scheduler=swa_scheduler,
        )
        saved_epoch = ckpt.get("epoch", 0)
        if saved_epoch <= 0:
            inferred_epoch = len(ckpt.get("losses", {}).get("train", []))
            if inferred_epoch > 0:
                print(
                    f"Checkpoint epoch metadata is stale; inferring resume epoch from loss history ({inferred_epoch})."
                )
                saved_epoch = inferred_epoch
        start_epoch = saved_epoch + 1
        last_completed_epoch = saved_epoch
        best_val = ckpt.get("best_val_loss", best_val)
        best_force_mae = ckpt.get("best_force_mae", best_force_mae)
        best_energy_mae = ckpt.get("best_energy_mae", best_energy_mae)
        best_swa_val = ckpt.get("best_swa_val", best_swa_val)
        best_swa_force_mae = ckpt.get("best_swa_force_mae", best_swa_force_mae)
        best_swa_energy_mae = ckpt.get("best_swa_energy_mae", best_swa_energy_mae)
        losses = ckpt.get("losses", losses)
        best_state = ckpt.get("best_state")
        best_swa_state = ckpt.get("best_swa_state")
        epochs_without_improve = ckpt.get("epochs_without_improve", epochs_without_improve)
        swa_activated = ckpt.get("swa_activated", swa_activated)
        print(f"Resumed at epoch {start_epoch}, best_force_mae={best_force_mae:.6f}")

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
            "swa_state_dict": swa_model.state_dict() if swa_model is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "swa_scheduler_state_dict": (swa_scheduler.state_dict() if swa_scheduler is not None else None),
            "best_val_loss": best_val,
            "best_force_mae": best_force_mae,
            "best_energy_mae": best_energy_mae,
            "best_swa_val": best_swa_val,
            "best_swa_force_mae": best_swa_force_mae,
            "best_swa_energy_mae": best_swa_energy_mae,
            "best_state": best_state,
            "best_swa_state": best_swa_state,
            "epochs_without_improve": epochs_without_improve,
            "swa_activated": swa_activated,
            "losses": losses,
            "mu_E": mu_E.detach().cpu(),
            "sigma_F": sigma_F.detach().cpu(),
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
            train_loss, train_force_loss, train_energy_loss = train_one_epoch(
                model_name,
                model,
                train_loader,
                optimizer,
                device,
                mu_E,
                sigma_F,
                force_weight,
                energy_weight,
                sdpa_mode=sdpa_mode,
                ema_model=ema_model,
                progress_bar=pbar if use_tqdm else None,
            )

        # Update SWA from raw model (avoid EMA+SWA double-smoothing) once threshold is reached.
        if epoch >= swa_start_epoch:
            if not swa_activated:
                swa_activated = True
            swa_model.update_parameters(model)

        # Use EMA (if enabled) as the primary online validation model during training.
        eval_model = ema_model if ema_model is not None else model

        val_metrics = evaluate(
            model_name,
            eval_model,
            val_loader,
            device,
            mu_E,
            sigma_F,
            force_weight,
            energy_weight,
            sdpa_mode=sdpa_mode,
        )

        losses["train"].append(train_loss)
        losses["val"].append(val_metrics["total_loss"])
        losses["val_force_mae_mev_A"].append(val_metrics["force_mae_mev_A"])
        losses["val_energy_mae_mev"].append(val_metrics["energy_mae_mev"])

        if epoch >= swa_start_epoch:
            swa_scheduler.step()
        else:
            scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
                f"train force {train_force_loss:.4f} | train energy {train_energy_loss:.4f} | "
                f"val loss {val_metrics['total_loss']:.4f} | "
                f"val F-MAE {val_metrics['force_mae_mev_A']:.2f} meV/Å | "
                f"val E-MAE {val_metrics['energy_mae_mev']:.2f} meV | "
                f"lr {optimizer.param_groups[0]['lr']:.6f}"
            )

        if tbw is not None:
            tbw.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            tbw.add_scalar("train/loss", train_loss, epoch)
            tbw.add_scalar("train/force_loss", train_force_loss, epoch)
            tbw.add_scalar("train/energy_loss", train_energy_loss, epoch)
            tbw.add_scalar("val/loss", val_metrics["total_loss"], epoch)
            tbw.add_scalar("val/force_loss", val_metrics["force_norm_mse"], epoch)
            tbw.add_scalar("val/energy_loss", val_metrics["energy_norm_mse"], epoch)
            tbw.add_scalar("val/force_mae_mev_A", val_metrics["force_mae_mev_A"], epoch)
            tbw.add_scalar("val/energy_mae_mev", val_metrics["energy_mae_mev"], epoch)

        # Track the best EMA/raw checkpoint based on validation Force MAE (meV/Å).
        if val_metrics["force_mae_mev_A"] < best_force_mae:
            best_force_mae = val_metrics["force_mae_mev_A"]
            best_val = val_metrics["total_loss"]
            best_energy_mae = val_metrics["energy_mae_mev"]
            epochs_without_improve = 0
            eval_state_dict = (
                eval_model.module.state_dict()
                if isinstance(eval_model, AveragedModel)
                else eval_model.state_dict()
            )
            best_state = {
                "model": clone_state_dict_to_cpu(eval_state_dict),
                "epoch": epoch,
                "val_force_mae_mev_A": val_metrics["force_mae_mev_A"],
            }

            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improve += 1

        # Track the best SWA checkpoint on validation separately, then compare
        # EMA/raw vs SWA using validation only before the final test evaluation.
        if swa_activated:
            swa_val_metrics = evaluate(
                model_name,
                swa_model,
                val_loader,
                device,
                mu_E,
                sigma_F,
                force_weight,
                energy_weight,
                sdpa_mode=sdpa_mode,
            )

            if swa_val_metrics["force_mae_mev_A"] < best_swa_force_mae:
                best_swa_force_mae = swa_val_metrics["force_mae_mev_A"]
                best_swa_val = swa_val_metrics["total_loss"]
                best_swa_energy_mae = swa_val_metrics["energy_mae_mev"]
                best_swa_state = {
                    "model": clone_state_dict_to_cpu(swa_model.module.state_dict()),
                    "epoch": epoch,
                    "val_force_mae_mev_A": swa_val_metrics["force_mae_mev_A"],
                }

        last_completed_epoch = epoch
        if last_checkpoint_path and (epoch % 10 == 0 or epoch == num_epochs):
            save_checkpoint(
                path=last_checkpoint_path,
                payload=build_checkpoint_payload(epoch=epoch),
            )

        if stop_handler is not None and stop_handler.stop_requested:
            print(f"Stop requested ({stop_handler.last_signal}). Saved checkpoint and stopping training.")
            break

        if patience > 0 and epochs_without_improve >= patience:
            print("Early stopping triggered")
            break

    if stop_handler is not None and stop_handler.stop_requested:
        losses["test_force_mae_mev_A"] = None
        losses["test_energy_mae_mev"] = None
        losses["test_loss"] = None
        losses["best_val_loss"] = best_val
        return losses

    selected_model_name = "ema" if ema_model is not None else "raw"
    selected_state = best_state

    if best_swa_state is not None and best_swa_force_mae < best_force_mae:
        selected_model_name = "swa"
        selected_state = best_swa_state
        best_val = best_swa_val
        best_force_mae = best_swa_force_mae
        best_energy_mae = best_swa_energy_mae

    if selected_state is None:
        selected_model_name = "raw"
        selected_state = {
            "model": model.state_dict(),
            "epoch": last_completed_epoch,
            "val_force_mae_mev_A": best_force_mae,
        }

    model.load_state_dict(selected_state["model"])
    print(
        f"Selected {selected_model_name.upper()} checkpoint from epoch {selected_state['epoch']} for final test."
    )

    test_metrics = evaluate(
        model_name,
        model,
        test_loader,
        device,
        mu_E,
        sigma_F,
        force_weight,
        energy_weight,
        sdpa_mode=sdpa_mode,
    )

    losses["test_force_mae_mev_A"] = test_metrics["force_mae_mev_A"]
    losses["test_energy_mae_mev"] = test_metrics["energy_mae_mev"]
    losses["test_loss"] = test_metrics["total_loss"]
    losses["best_val_loss"] = best_val
    print(f"Best val loss {best_val:.4f}")
    print(
        f"Test total loss {test_metrics['total_loss']:.4f} | "
        f"Test F-MAE {test_metrics['force_mae_mev_A']:.2f} meV/Å | "
        f"Test E-MAE {test_metrics['energy_mae_mev']:.2f} meV"
    )

    return losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train GA models on MD17")
    parser.add_argument(
        "--model_name",
        choices=["pgagnn", "gatr", "egnn", "segnn", "ggnn"],
        default="pgagnn",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "revised aspirin",
            "revised benzene",
            "revised azobenzene",
            "revised malonaldehyde",
            "revised naphthalene",
            "revised salicylic acid",
            "revised toluene",
            "revised uracil",
            "revised ethanol",
            "revised paracetamol",
        ],
        default="revised aspirin",
        help="Revised MD17 dataset name",
    )
    parser.add_argument("--data_dir", type=str, default=str(ROOT_DIR / "datasets" / "md17"))
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-7)
    parser.add_argument("--hidden_mvc", type=int, default=16)
    parser.add_argument("--hidden_sc", type=int, default=128)
    parser.add_argument("--out_mvc", type=int, default=16)
    parser.add_argument("--out_sc", type=int, default=64)
    parser.add_argument("--edge_mvc", type=int, default=None)
    parser.add_argument("--edge_sc", type=int, default=None)
    parser.add_argument("--num_rbf", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--force_weight", type=float, default=1000.0)
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--attention_type", type=str, default="gatr")
    parser.add_argument(
        "--message",
        type=str,
        default="geometric",
        choices=["geometric", "sum", "linear"],
        help="GA_GNN ablation for replacing the geometric product in message combination.",
    )
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_sparse", action="store_true")
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--swa_start_epoch", type=int, default=3000)
    parser.add_argument("--swa_lr", type=float, default=5e-5)
    parser.add_argument(
        "--sdpa_mode",
        type=str,
        choices=["auto", "math", "default"],
        default="auto",
        help=(
            "SDPA backend policy for attention modules. "
            "'auto': force math only during training-time higher-order gradients, "
            "'math': always force math in predict_energy_and_forces, "
            "'default': never force math (use PyTorch default kernel dispatch)."
        ),
    )
    parser.add_argument("--split_idx", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--runs_dir", type=str, default=str(ROOT_DIR / "runs" / "md17"))
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--handle_signals", type=str, default=None)
    parser.add_argument(
        "--no_time_suffix", action="store_true", help="Disable time suffix for run directory."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ema_decay < 0.0 or args.ema_decay >= 1.0:
        raise ValueError("--ema_decay must be in [0, 1).")
    if args.swa_start_epoch < 0:
        raise ValueError("--swa_start_epoch must be >= 0.")
    if args.swa_lr <= 0.0:
        raise ValueError("--swa_lr must be > 0.")

    signals_to_handle = parse_signals(args.handle_signals) if args.handle_signals else None

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

    # Set random seeds for reproducibility
    seed_everything(args.seed)

    # print configuration
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
        mu_E,
        sigma_F,
    ) = load_dataset(
        name=args.dataset,
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        split_idx=args.split_idx,
        data_dir=args.data_dir,
    )

    edge_sc = args.edge_sc
    if args.model_name =="pgagnn" and edge_sc is None:
        edge_sc = args.num_rbf
        print(f"Using edge_sc={edge_sc} to match --num_rbf.")

    args.edge_sc = edge_sc
    edge_hidden_dim = edge_sc if edge_sc is not None else args.hidden_sc

    base_model = get_model(
        model_name=args.model_name,
        in_mvc=1,
        out_mvc=args.out_mvc,
        hidden_mvc=args.hidden_mvc,
        in_sc=args.embed_dim,
        out_sc=args.out_sc,
        hidden_sc=args.hidden_sc,
        edge_mvc=args.edge_mvc,
        edge_sc=edge_sc,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_prob=args.dropout,
        activation=args.activation,
        attention_type=args.attention_type,
        message=args.message,
    ).to(device)
    base_model.use_sparse = args.use_sparse or args.model_name in {"egnn", "segnn"}
    if args.model_name in {"egnn", "segnn"}:
        head = ScalarEnergyPredictor(in_channels=args.hidden_sc).to(device)
    elif args.model_name == "ggnn":
        head = GGNNEnergyPredictor(vc_channels=args.out_mvc, sc_channels=args.out_sc).to(device)
    else:
        head = EnergyPredictor(mv_channels=args.out_mvc, sc_channels=args.out_sc).to(device)
    model = GAModel(
        base_model=base_model,
        head=head,
        sc_dim=args.embed_dim,
        edge_rbf_dim=args.num_rbf,
        edge_sc_hidden=edge_hidden_dim,
        cutoff=args.cutoff,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{args.model_name}' has {total_params} trainable parameters.")
    args.total_params = total_params

    run_dir, run_name = resolve_run_directory(args.runs_dir, args, group=args.dataset)

    print(f"\nRun directory: {run_dir}\n")
    print(f"Run name: {run_name}\n")

    checkpoint_path = os.path.join(run_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(run_dir, "last_model.pt")

    tb_run_dir = Path(run_dir).resolve().parent / "tensorboard" / run_name
    tbw = init_tensorboard_writer(args=args, run_dir=tb_run_dir) if not args.no_tensorboard else None
    if tbw is not None:
        print(
            f"TensorBoard logging enabled. Launch with: tensorboard --logdir {tb_run_dir.parent} --port 6006"
        )

    stop_handler = StopSignalHandler()

    results = train_model(
        model_name=args.model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=args.epochs,
        mu_E=mu_E,
        sigma_F=sigma_F,
        force_weight=args.force_weight,
        energy_weight=args.energy_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        ema_decay=args.ema_decay,
        swa_start_epoch=args.swa_start_epoch,
        swa_lr=args.swa_lr,
        sdpa_mode=args.sdpa_mode,
        checkpoint_path=checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        resume_from=args.resume_from,
        stop_handler=stop_handler,
        signals_to_handle=signals_to_handle,
        tbw=tbw,
        no_tqdm=args.no_tqdm,
    )

    json_path = save_metrics_to_json(results, run_dir, args)

    if tbw is not None:
        tbw.flush()
        tbw.close()

    if stop_handler.stop_requested:
        print(f"Exiting with code 3 to signal SLURM requeue (interrupted by {stop_handler.last_signal}).")
        sys.exit(3)

    # Print final summary
    print("\n" + "=" * 70)
    print("Training Summary:")
    print("=" * 70)
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Final train loss: {results['train'][-1]:.6f}")
    print(f"Final val loss: {results['val'][-1]:.6f}")
    print(f"Test loss: {results['test_loss']:.6f}")
    print(f"Test Force MAE: {results['test_force_mae_mev_A']:.4f} meV/Å")
    print(f"Test Energy MAE: {results['test_energy_mae_mev']:.4f} meV")
    print(f"Total epochs: {len(results['train'])}")
    print(f"Run directory: {run_dir}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"Metrics JSON saved to: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
