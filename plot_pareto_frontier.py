import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from balanced_irreps import WeightBalancedIrreps
from e3nn.o3 import Irreps
from gatr import GATr, MLPConfig, SelfAttentionConfig
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


def measure_flops(models, forward_fns):
    """FLOPs + parameter counts (hardware-independent)."""
    results = {}

    for name, (model, key) in models.items():
        fn = forward_fns[name]
        model.eval()

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


def measure_memory(models, forward_fns, device: str = "cuda"):
    """Peak CUDA memory allocated during a single forward pass, in MB.

    NOTE: this is hardware/driver/PyTorch-version dependent (allocator
    behaviour, cuDNN algorithm choice, fragmentation, etc.) and only
    meaningful on CUDA. On CPU it falls back to NaN since there is no
    reliable, portable equivalent of `max_memory_allocated`.
    """
    results = {}

    for name, (model, key) in models.items():
        fn = forward_fns[name]
        model.eval()

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                fn(model)
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_mb = float("nan")

        results[name] = {"peak_mem_mb": peak_mb}

    return results


def measure_latency(models, forward_fns, device: str = "cuda", n_warmup: int = 10, n_repeats: int = 50):
    """Wall-clock inference latency per forward pass, in milliseconds.

    NOTE: this is entirely hardware-dependent (GPU/CPU model, clocks,
    thermal state, other processes on the machine, PyTorch/cuDNN version,
    etc.) and should only be compared *within* a single benchmarking run
    on the same machine — never across machines or papers.
    """
    results = {}

    for name, (model, key) in models.items():
        fn = forward_fns[name]
        model.eval()

        with torch.no_grad():
            # Warmup: let cuDNN autotune, JIT/caches warm up, allocator settle.
            for _ in range(n_warmup):
                fn(model)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

            times_ms = []
            for _ in range(n_repeats):
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                fn(model)
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - start) * 1000.0)

        results[name] = {
            "mean_latency_ms": float(np.mean(times_ms)),
            "std_latency_ms": float(np.std(times_ms)),
            "n_repeats": n_repeats,
        }

    return results


def measure_efficiency_metrics(
    models,
    forward_fns,
    device: str = "cuda",
    n_warmup: int = 10,
    n_repeats: int = 50,
    measure_mem: bool = True,
    measure_time: bool = True,
):
    """Combine FLOPs/params (hardware-independent) with optional
    memory and latency (hardware-dependent) measurements into one dict.

    Kept as three separate passes over the models rather than one fused
    loop: the FlopCounterMode context and CUDA memory-stat resets can
    otherwise interfere with each other's readings.
    """
    results = measure_flops(models, forward_fns)

    if measure_mem:
        mem_results = measure_memory(models, forward_fns, device=device)
        for name, r in mem_results.items():
            results[name].update(r)

    if measure_time:
        lat_results = measure_latency(
            models, forward_fns, device=device, n_warmup=n_warmup, n_repeats=n_repeats
        )
        for name, r in lat_results.items():
            results[name].update(r)

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


def _plot_pareto_panel(ax, x_vals, y_vals, y_stds, model_names, benchmark_results, style_map, anno_offsets, xlabel, title, label_params=False):
    pareto_idx = compute_pareto_frontier(x_vals, y_vals)
    pareto_x = [x_vals[i] for i in pareto_idx]
    pareto_y = [y_vals[i] for i in pareto_idx]
    ax.step(pareto_x, pareto_y, where="post", color="#333333", linestyle="--", linewidth=1.8, alpha=0.85, label="Pareto Frontier")
    ax.plot(pareto_x, pareto_y, color="#333333", alpha=0.3, linewidth=1.0)

    for i, name in enumerate(model_names):
        st = style_map.get(name, {"color": "#333", "marker": "o", "label": name})
        if label_params:
            pk = benchmark_results[name]["params"] / 1e3
            lbl = f"{name} ({pk:.0f}k params)"
        else:
            lbl = name

        ax.errorbar(
            x_vals[i],
            y_vals[i],
            yerr=y_stds[i],
            fmt=st["marker"],
            color=st["color"],
            markersize=9,
            capsize=3,
            elinewidth=1.5,
            markeredgewidth=1.2,
            markeredgecolor="black",
            label=lbl,
        )
        dx, dy = anno_offsets.get(name, (0.3, 0.00005))
        ax.annotate(
            name,
            (x_vals[i], y_vals[i]),
            xytext=(x_vals[i] + dx, y_vals[i] + dy),
            fontsize=9.5,
            fontweight="bold" if "PGA" in name else "normal",
            color=st["color"],
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test OOD Loss (MSE)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, framealpha=0.9, loc="best")


def generate_pareto_plots(benchmark_results, nbody_data, output_dir: Path, device: str = "cuda"):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(benchmark_results.keys())
    gflops = [benchmark_results[m]["gflops"] for m in model_names]
    params_k = [benchmark_results[m]["params"] / 1e3 for m in model_names]

    has_latency = all("mean_latency_ms" in benchmark_results[m] for m in model_names)
    has_memory = all(
        "peak_mem_mb" in benchmark_results[m] and not np.isnan(benchmark_results[m]["peak_mem_mb"])
        for m in model_names
    )

    latency_ms = [benchmark_results[m].get("mean_latency_ms", float("nan")) for m in model_names]
    latency_std = [benchmark_results[m].get("std_latency_ms", 0.0) for m in model_names]
    mem_mb = [benchmark_results[m].get("peak_mem_mb", float("nan")) for m in model_names]

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

    style_map = {
        "PGA-GNN": {"color": "#1B9E77", "marker": "o", "label": "PGA-GNN"},
        "GATr": {"color": "#D95F02", "marker": "^", "label": "GATr"},
        "GGNN": {"color": "#7570B3", "marker": "D", "label": "GGNN"},
        "EGNN": {"color": "#E7298A", "marker": "p", "label": "EGNN"},
        "SEGNN": {"color": "#E6AB02", "marker": "X", "label": "SEGNN"},
    }

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

    # Decide layout: always show FLOPs + Params; add latency/memory panels if available.
    n_panels = 2 + int(has_latency) + int(has_memory)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    _fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows), dpi=300)
    axes = np.array(axes).reshape(-1)

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
    # Generic offset fallback for the new panels (auto-scaled from data range).
    def _auto_offsets(x_vals):
        span = (max(x_vals) - min(x_vals)) or 1.0
        return {name: (span * 0.02, 0.0) for name in model_names}

    panel_idx = 0

    _plot_pareto_panel(
        axes[panel_idx], gflops, losses, loss_stds, model_names, benchmark_results, style_map,
        anno_offsets_flops, "Compute per Forward Pass [GFLOPs] (B=64, N=15)",
        "(a) Compute Efficiency: Accuracy vs. FLOPs", label_params=True,
    )
    panel_idx += 1

    _plot_pareto_panel(
        axes[panel_idx], params_k, losses, loss_stds, model_names, benchmark_results, style_map,
        anno_offsets_params, "Model Parameters [kParams]",
        "(b) Parameter Efficiency: Accuracy vs. Model Size", label_params=False,
    )
    panel_idx += 1

    panel_letter = ord("c")
    if has_latency:
        _plot_pareto_panel(
            axes[panel_idx], latency_ms, losses, loss_stds, model_names, benchmark_results, style_map,
            _auto_offsets(latency_ms), "Inference Latency [ms/forward] (hardware-dependent)",
            f"({chr(panel_letter)}) Latency Efficiency: Accuracy vs. Wall-clock Time", label_params=False,
        )
        panel_idx += 1
        panel_letter += 1

    if has_memory:
        _plot_pareto_panel(
            axes[panel_idx], mem_mb, losses, loss_stds, model_names, benchmark_results, style_map,
            _auto_offsets(mem_mb), "Peak GPU Memory [MB] (hardware-dependent)",
            f"({chr(panel_letter)}) Memory Efficiency: Accuracy vs. Peak Memory", label_params=False,
        )
        panel_idx += 1

    # Hide any unused axes (e.g. odd panel count in a 2-col grid).
    for j in range(panel_idx, len(axes)):
        axes[j].axis("off")

    num_seeds = max(num_seeds_list) if num_seeds_list else 1
    seed_str = f"{num_seeds} Seeds" if num_seeds > 1 else "Seed 0"
    plt.suptitle(f"Pareto Efficiency on N-body Benchmark (100 examples, {seed_str})", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

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
    if device == "cpu":
        print("  (No CUDA device found: memory measurement will be skipped, and "
              "latency numbers are CPU wall-clock time only.)")

    nbody_data = load_nbody_data()
    models = build_models(device)
    forward_fns = prepare_inputs(B=64, N=15, device=device)

    print("\nMeasuring FLOPs, memory and latency...")
    bench_results = measure_efficiency_metrics(
        models,
        forward_fns,
        device=device,
        n_warmup=10,
        n_repeats=50,
        measure_mem=(device == "cuda"),
        measure_time=True,
    )

    print("\nGenerating Pareto plots...")
    plots_nbody_dir = Path("plots/nbody")
    generate_pareto_plots(bench_results, nbody_data, plots_nbody_dir, device=device)

    print("\n" + "=" * 100)
    print(
        f"{'Model':<12} {'Params':>10} {'FLOPs':>13} {'Latency (ms)':>16} {'Peak Mem (MB)':>14} {'Test OOD Loss (mean ± std)':>28}"
    )
    print("-" * 100)
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

        lat_str = (
            f"{r['mean_latency_ms']:.3f} ± {r['std_latency_ms']:.3f}"
            if "mean_latency_ms" in r
            else "N/A"
        )
        mem_val = r.get("peak_mem_mb", float("nan"))
        mem_str = f"{mem_val:.1f}" if not np.isnan(mem_val) else "N/A"

        print(
            f"{name:<12} {r['params']:>10,d} {r['gflops']:>10.3f} G {lat_str:>16} {mem_str:>14} {loss_str:>28}"
        )
    print("=" * 100)
    print("\nNote: latency and memory figures are hardware/software-stack dependent")
    print("(GPU model, drivers, cuDNN, PyTorch version, thermal/load state) and are")
    print("only meaningful for comparisons made within this same run/machine.")


if __name__ == "__main__":
    main()