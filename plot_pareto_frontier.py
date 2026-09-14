import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
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
from e3nn.o3 import Irreps

try:
    from gatr import GATr, MLPConfig, SelfAttentionConfig
except ImportError:

    class GATr:  # type: ignore
        """Placeholder for GATr when not available."""

    MLPConfig = None  # type: ignore
    SelfAttentionConfig = None  # type: ignore
from segnn.segnn import SEGNN
from torch.utils.flop_counter import FlopCounterMode
from torch_geometric.data import Data as PyGData

from egnn.n_body_system.model import EGNN_vel
from src.pgagnn import PGA_GNN


def load_nbody_data(base_dir: str = "runs/nbody_spring", subsample: str = "sub0.001", seeds=(0, 1, 2)):
    """Load test_ood_loss across seeds and compute mean and std."""
    data = {}
    base_path = Path(base_dir)

    for seed in seeds:
        seed_path = base_path / str(seed)
        if not seed_path.exists():
            continue
        for d in seed_path.iterdir():
            if d.is_dir() and subsample in d.name:
                m_file = d / "metrics.json"
                if m_file.exists():
                    with open(m_file) as f:
                        m_data = json.load(f)
                    cfg = m_data.get("config", {})
                    sumry = m_data.get("summary", {})
                    m_name = cfg.get("model_name", d.name.split("_")[0]).lower()
                    ood_loss = sumry.get("test_ood_loss", None)
                    if ood_loss is not None:
                        data.setdefault(m_name, {}).setdefault("all_ood_losses", []).append(ood_loss)
                        data[m_name].setdefault("seeds", []).append(seed)
                        if "total_params" not in data[m_name] and "total_params" in cfg:
                            data[m_name]["total_params"] = cfg.get("total_params")

    for m_name, m_info in data.items():
        all_losses = m_info.get("all_ood_losses", [])
        if all_losses:
            m_info["mean_ood_loss"] = float(np.mean(all_losses))
            m_info["std_ood_loss"] = float(np.std(all_losses)) if len(all_losses) > 1 else 0.0

    return data


def build_models(device: str = "cuda"):
    models = {}

    # PGA-GNN
    models["PGA-GNN"] = (
        PGA_GNN(
            in_mvc=1,
            out_mvc=1,
            hidden_mvc=64,
            in_sc=1,
            out_sc=1,
            hidden_sc=128,
            edge_mvc=1,
            edge_sc=64,
            num_layers=2,
            num_heads=8,
            expansion_factor=1.0,
            factorize=True,
            use_grade_modulation=False,
            use_pseudoscalar=False,
            attention_type="gatr_sparse",
        ).to(device),
        "pgagnn",
    )

    # GATr
    models["GATr"] = (
        GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=16,
            in_s_channels=1,
            out_s_channels=1,
            hidden_s_channels=128,
            num_blocks=10,
            attention=SelfAttentionConfig(num_heads=8),
            mlp=MLPConfig(),
        ).to(device),
        "gatr",
    )

    # EGNN (EGNN_vel)
    models["EGNN"] = (
        EGNN_vel(
            in_node_nf=1,
            in_edge_nf=2,
            hidden_nf=64,
            n_layers=4,
            recurrent=True,
            norm_diff=False,
            tanh=False,
        ).to(device),
        "egnn",
    )

    # SEGNN
    _lmax = 1
    _hidden_irreps = WeightBalancedIrreps(
        Irreps("64x0e"), Irreps.spherical_harmonics(_lmax), sh=True, lmax=_lmax
    )
    models["SEGNN"] = (
        SEGNN(
            input_irreps=Irreps("2x1o + 1x0e"),
            output_irreps=Irreps("1x1o"),
            edge_attr_irreps=Irreps.spherical_harmonics(_lmax),
            node_attr_irreps=Irreps.spherical_harmonics(_lmax),
            additional_message_irreps=Irreps("2x0e"),
            hidden_irreps=_hidden_irreps,
            num_layers=4,
            task="node",
            pool=None,
            norm=None,
        ).to(device),
        "segnn",
    )

    return models


def prepare_inputs(B: int = 64, N: int = 15, device: str = "cuda"):
    torch.manual_seed(42)
    # Dense GA / GATr inputs
    mv = torch.randn(B, N, 1, 16, device=device)
    sc = torch.randn(B, N, 1, device=device)
    edge_mv = torch.randn(B, N, N, 1, 16, device=device)
    edge_sc = torch.randn(B, N, N, 64, device=device)
    adj = torch.ones((B, N, N), dtype=torch.bool, device=device)
    ref = torch.mean(mv, dim=(1, 2), keepdim=True)

    # Sparse PGA-GNN inputs
    adj_sl = torch.maximum(
        adj,
        torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0),
    )
    b_idx, src_idx, dst_idx = adj_sl.nonzero(as_tuple=True)
    edge_index_sp = torch.stack([b_idx * N + src_idx, b_idx * N + dst_idx], dim=0)
    mv_sp = mv.reshape(B * N, 1, 16)
    sc_sp = sc.reshape(B * N, 1)
    edge_mv_sp = edge_mv[b_idx, src_idx, dst_idx]
    edge_sc_sp = edge_sc[b_idx, src_idx, dst_idx]

    # EGNN inputs
    nodes_egnn = torch.randn(B * N, 1, device=device)
    loc_egnn = torch.randn(B * N, 3, device=device)
    vel_egnn = torch.randn(B * N, 3, device=device)
    edges_egnn = [b_idx * N + src_idx, b_idx * N + dst_idx]
    edge_attr_egnn = torch.randn(len(b_idx), 2, device=device)

    # SEGNN inputs
    total_nodes = B * N
    n_edges = edge_index_sp.shape[1]
    segnn_graph = PyGData(
        x=torch.randn(total_nodes, 7, device=device),
        pos=torch.randn(total_nodes, 3, device=device),
        edge_index=edge_index_sp,
        edge_attr=torch.cat(
            [torch.ones(n_edges, 1, device=device), torch.randn(n_edges, 3, device=device)], dim=1
        ),
        node_attr=torch.cat(
            [torch.ones(total_nodes, 1, device=device), torch.randn(total_nodes, 3, device=device)], dim=1
        ),
        additional_message_features=torch.randn(n_edges, 2, device=device),
        batch=torch.arange(B, device=device).repeat_interleave(N),
    )

    forward_fns = {
        "PGA-GNN": lambda m: m(
            mv=mv_sp, sc=sc_sp, edge_index=edge_index_sp, edge_attr_mv=edge_mv_sp, edge_attr_sc=edge_sc_sp
        ),
        "GATr": lambda m: m(mv, scalars=sc, join_reference=ref),
        "EGNN": lambda m: m(nodes_egnn, loc_egnn, edges_egnn, vel_egnn, edge_attr_egnn),
        "SEGNN": lambda m: m(segnn_graph),
    }

    return forward_fns


def measure_flops_and_latency(models, forward_fns):
    results = {}

    for name, (model, key) in models.items():
        fn = forward_fns[name]
        model.eval()

        # FLOPs counting on CPU/no_grad for accuracy
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            fn(model)
        flops = counter.get_total_flops()

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[name] = {
            "key": key,
            "params": params,
            "flops": flops,
            "gflops": flops / 1e9,
        }

    return results


def compute_pareto_frontier(x_vals, y_vals):
    """
    Compute 2D Pareto frontier minimizing both x (e.g. GFLOPs/Latency) and y (Test Loss).
    Returns indices of non-dominated points sorted by x.
    """
    sorted_indices = sorted(range(len(x_vals)), key=lambda i: x_vals[i])
    pareto_indices = []
    min_y = float("inf")

    for idx in sorted_indices:
        if y_vals[idx] < min_y:
            pareto_indices.append(idx)
            min_y = y_vals[idx]

    return pareto_indices


def generate_pareto_plots(benchmark_results, nbody_data, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(benchmark_results.keys())
    gflops = [benchmark_results[m]["gflops"] for m in model_names]

    losses = []
    loss_stds = []
    num_seeds_list = []
    for m in model_names:
        key = benchmark_results[m]["key"]
        m_info = nbody_data.get(key, {})
        all_vals = m_info.get("all_ood_losses", [])
        if all_vals:
            mean_val = float(np.mean(all_vals))
            std_val = float(np.std(all_vals)) if len(all_vals) > 1 else 0.0
            num_seeds_list.append(len(all_vals))
        else:
            mean_val = float("nan")
            std_val = 0.0
        losses.append(mean_val)
        loss_stds.append(std_val)

    # Colors and markers styling
    style_map = {
        "PGA-GNN": {"color": "#1B9E77", "marker": "o", "label": "PGA-GNN"},
        "GATr": {"color": "#D95F02", "marker": "^", "label": "GATr"},
        "GGNN": {"color": "#7570B3", "marker": "D", "label": "GGNN"},
        "EGNN": {"color": "#E7298A", "marker": "p", "label": "EGNN"},
        "SEGNN": {"color": "#E6AB02", "marker": "X", "label": "SEGNN"},
    }

    # Set up publication-quality plot style
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
        }
    )

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # -------------------------------------------------------------
    # Panel 1: Test OOD Loss vs GFLOPs (Compute Efficiency)
    # -------------------------------------------------------------
    pareto_idx_flops = compute_pareto_frontier(gflops, losses)

    # Plot Pareto line
    pareto_x_f = [gflops[i] for i in pareto_idx_flops]
    pareto_y_f = [losses[i] for i in pareto_idx_flops]
    ax1.step(
        pareto_x_f,
        pareto_y_f,
        where="post",
        color="#333333",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label="Pareto Frontier",
    )
    ax1.plot(pareto_x_f, pareto_y_f, color="#333333", alpha=0.3, linewidth=1.0)

    # Custom annotation offsets for (dx, dy) in data/display coordinates
    anno_offsets_flops = {
        "EGNN": (0.4, 0.00007),
        "PGA-GNN": (0.5, 0.0000),
        "GATr": (-2.8, 0.00008),
        "SEGNN": (0.4, 0.00006),
    }

    anno_offsets_params = {
        "EGNN": (20, 0.0008),
        "PGA-GNN": (30, 0.0000005),
        "GATr": (-320, 0.0004),
        "SEGNN": (20, 0.0001),
    }

    params_k = [benchmark_results[m]["params"] / 1e3 for m in model_names]

    for i, name in enumerate(model_names):
        st = style_map.get(name, {"color": "#333", "marker": "o", "label": name})
        pk = params_k[i]

        ax1.errorbar(
            gflops[i],
            losses[i],
            yerr=loss_stds[i],
            fmt=st["marker"],
            color=st["color"],
            markersize=9,
            capsize=3,
            elinewidth=1.5,
            markeredgewidth=1.2,
            markeredgecolor="black",
            label=f"{name} ({pk:.0f}k params)",
        )
        dx, dy = anno_offsets_flops.get(name, (0.3, 0.00005))
        ax1.annotate(
            name,
            (gflops[i], losses[i]),
            xytext=(gflops[i] + dx, losses[i] + dy),
            fontsize=9.5,
            fontweight="bold" if "PGA" in name else "normal",
            color=st["color"],
        )

    ax1.set_xlabel("Compute per Forward Pass [GFLOPs] (B=64, N=15)")
    ax1.set_ylabel("Test OOD Loss (MSE)")
    ax1.set_yscale("log")
    ax1.set_title("(a) Compute Efficiency: Accuracy vs. FLOPs")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=True, framealpha=0.9, loc="best")

    # -------------------------------------------------------------
    # Panel 2: Test OOD Loss vs Parameters (Parameter Efficiency)
    # -------------------------------------------------------------
    pareto_idx_params = compute_pareto_frontier(params_k, losses)
    pareto_x_p = [params_k[i] for i in pareto_idx_params]
    pareto_y_p = [losses[i] for i in pareto_idx_params]
    ax2.step(
        pareto_x_p,
        pareto_y_p,
        where="post",
        color="#333333",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label="Pareto Frontier",
    )
    ax2.plot(pareto_x_p, pareto_y_p, color="#333333", alpha=0.3, linewidth=1.0)

    for i, name in enumerate(model_names):
        st = style_map.get(name, {"color": "#333", "marker": "o", "label": name})
        ax2.errorbar(
            params_k[i],
            losses[i],
            yerr=loss_stds[i],
            fmt=st["marker"],
            color=st["color"],
            markersize=9,
            capsize=3,
            elinewidth=1.5,
            markeredgewidth=1.2,
            markeredgecolor="black",
            label=name,
        )
        dx, dy = anno_offsets_params.get(name, (20, 0.00005))
        ax2.annotate(
            name,
            (params_k[i], losses[i]),
            xytext=(params_k[i] + dx, losses[i] + dy),
            fontsize=9.5,
            fontweight="bold" if "PGA" in name else "normal",
            color=st["color"],
        )

    ax2.set_xlabel("Model Parameters [kParams]")
    ax2.set_ylabel("Test OOD Loss (MSE)")
    ax2.set_yscale("log")
    ax2.set_title("(b) Parameter Efficiency: Accuracy vs. Model Size")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=True, framealpha=0.9, loc="best")

    num_seeds = max(num_seeds_list) if num_seeds_list else 1
    seed_str = f"{num_seeds} Seeds" if num_seeds > 1 else "Seed 0"
    plt.suptitle(f"Pareto Efficiency on N-body Benchmark (Subsample 0.001, {seed_str})", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = output_dir / "pareto_frontier.png"
    pdf_path = output_dir / "pareto_frontier.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Pareto plots saved to:\n  - {png_path}\n  - {pdf_path}")
    return png_path, pdf_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Pareto benchmark on device: {device}...")

    nbody_data = load_nbody_data()
    models = build_models(device)
    forward_fns = prepare_inputs(B=64, N=15, device=device)

    print("\nMeasuring FLOPs and Latency...")
    bench_results = measure_flops_and_latency(models, forward_fns)

    print("\nGenerating Pareto plots...")
    plots_nbody_dir = Path("plots/nbody")
    generate_pareto_plots(bench_results, nbody_data, plots_nbody_dir)

    print("\n" + "=" * 75)
    print(f"{'Model':<20} {'Params':>10} {'FLOPs':>14} {'Test OOD Loss (mean ± std)':>28}")
    print("-" * 75)
    for name, r in bench_results.items():
        k = r["key"]
        m_info = nbody_data.get(k, {})
        mean_loss = m_info.get("mean_ood_loss", None)
        std_loss = m_info.get("std_ood_loss", 0.0)
        if mean_loss is not None:
            loss_str = (
                f"{mean_loss:.6f} ± {std_loss:.6f}"
                if mean_loss >= 1e-4
                else f"{mean_loss:.2e} ± {std_loss:.2e}"
            )
        else:
            loss_str = "N/A"
        print(f"{name:<20} {r['params']:>10,d} {r['gflops']:>11.3f} G {loss_str:>28}")
    print("=" * 75)


if __name__ == "__main__":
    main()
