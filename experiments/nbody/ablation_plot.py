import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT_DIR / "plots" / "nbody"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
AB_RUN_DIR = ROOT_DIR / "runs" / "nbody" / "ablations"
ablation_dict = {
    "pgagnn_baseline": "Baseline",
    "geometric_product_no_edge_attr": "No Edge Attr",
    "no_geometric_product_with_edge_attr": "No Geo Prod",
    "no_geometric_product_no_edge_attr": "No Geo Prod & No Edge Attr",
    "no_scalar": "No Scalar",
}


def collect_ablation_results(layers_only=True):
    # Collect results from all ablation runs and save to a single CSV
    records = {}
    for run_dir in AB_RUN_DIR.glob("*"):
        if not run_dir.is_dir():
            continue
        if layers_only and "layers" not in run_dir.name:
            continue  # skip non-ablation runs
        if layers_only:
            layers = run_dir.name.split("_")[-1]
            # Each layer has multiple seed runs, we collect them
            layer_run_dir = AB_RUN_DIR / f"layers_{layers}"
        elif not layers_only and "layers" in run_dir.name:
            continue  # skip layer ablation runs
        else:
            ablation_name = run_dir.name
            layer_run_dir = AB_RUN_DIR / ablation_name
        for seed_dir in layer_run_dir.glob("*"):
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name.split("_")[-1]
            # Each seed dir contains 3 models train on different sample sizes
            for model_dir in seed_dir.glob("*"):
                if not model_dir.is_dir():
                    continue
                sample_size = model_dir.name.split("_")[-3]
                sample_size = float(sample_size.replace("sub", ""))
                # Load metrics from metrics.json
                metrics_file = model_dir / "metrics.json"
                if not metrics_file.exists():
                    print(f"Warning: Missing metrics.json in {model_dir}")
                    continue
                with open(metrics_file) as f:
                    metrics = json.load(f)
                summary = metrics.get("summary", {})
                val_loss = round(summary.get("best_val_loss", np.nan), 4)
                test_loss = round(summary.get("test_loss", np.nan), 4)
                ood_loss = round(summary.get("test_ood_loss", np.nan), 4)
                # Store record
                records.setdefault("config", []).append(
                    f"layers_{layers}"
                ) if layers_only else records.setdefault("config", []).append(
                    f"{ablation_dict.get(ablation_name, ablation_name)}"
                )
                records.setdefault("val_loss", []).append(val_loss)
                records.setdefault("test_loss", []).append(test_loss)
                records.setdefault("ood_loss", []).append(ood_loss)
                records.setdefault("sample_size", []).append(sample_size)
                records.setdefault("seed", []).append(seed)
    # Save to CSV
    df = pd.DataFrame(records)
    csv_path = (
        ROOT_DIR / "runs" / "nbody" / "ablations_layers.csv"
        if layers_only
        else ROOT_DIR / "runs" / "nbody" / "ablations.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"Collected ablation results saved to {csv_path}")


def plot_layers_scaling():
    # Load collected results
    df = pd.read_csv(ROOT_DIR / "runs" / "nbody" / "ablations_layers.csv")
    # Compute mean and std for each config and sample size
    df = (
        df.groupby(["config", "sample_size"])
        .agg(
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            ood_loss_mean=("ood_loss", "mean"),
            ood_loss_std=("ood_loss", "std"),
        )
        .reset_index()
    )
    # Extract number of layers
    df["num_layers"] = df["config"].str.extract(r"(\d+)").astype(int)

    # Legend labels
    size_map = {0.001: "100 examples", 0.010: "1000 examples", 0.100: "10000 examples"}
    df["Data Fraction"] = df["sample_size"].map(size_map)

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("muted", 3)

    # Metrics to plot: (mean column, std column, y-label, title)
    metrics = [
        ("val_loss_mean", "val_loss_std", "Validation MSE", "Validation Loss"),
        ("test_loss_mean", "test_loss_std", "Test MSE", "Test Loss"),
        ("ood_loss_mean", "ood_loss_std", "OOD MSE", "OOD Test Loss"),
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
        sharex=True,
    )

    for ax, (mean_col, std_col, ylabel, subtitle) in zip(axes, metrics):
        for (fraction, label), color in zip(size_map.items(), colors):
            subset = df[df["sample_size"] == fraction].sort_values("num_layers")

            # Mean curve
            ax.plot(
                subset["num_layers"], subset[mean_col], marker="o", linewidth=2.5, label=label, color=color
            )

            # Std band
            ax.fill_between(
                subset["num_layers"],
                subset[mean_col] - subset[std_col],
                subset[mean_col] + subset[std_col],
                color=color,
                alpha=0.15,
            )
            ax.set_xlabel("Number of Layers", fontsize=12)

        ax.set_title(subtitle, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.6)

    # Shared x-axis formatting
    axes[-1].set_xticks([1, 2, 3, 4, 5, 6])

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Data Regime",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),  # Move legend below the subplots
        bbox_transform=fig.transFigure,
    )

    # fig.suptitle(
    #     "PGA-GNN Depth Architecture Analysis",
    #     fontsize=16,
    #     fontweight='bold',
    #     y=1.05
    # )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ablation_scaling_layers.pdf", bbox_inches="tight")
    plt.show()


def ablation_components():
    df = pd.read_csv(ROOT_DIR / "runs" / "nbody" / "ablations.csv")

    stats_df = (
        df.groupby(["config", "sample_size"])
        .agg(
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            ood_loss_mean=("ood_loss", "mean"),
            ood_loss_std=("ood_loss", "std"),
        )
        .reset_index()
    )

    print("--- Tabular Results (Mean & Std) ---")
    print(stats_df.to_string(index=False))

    configs = stats_df["config"].unique()
    cmap = plt.get_cmap("tab10", len(configs))
    COLOR = {c: cmap(i) for i, c in enumerate(configs)}
    MARKER = {c: m for c, m in zip(configs, ["o", "s", "^", "D", "v", "P", "X", "*"])}

    metrics = ["val_loss", "test_loss", "ood_loss"]
    titles = ["Validation Loss", "Test Loss", "OOD Loss"]

    _fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, metric, title in zip(axes, metrics, titles):
        for config, grp in stats_df.groupby("config"):
            grp = grp.sort_values("sample_size")
            x = grp["sample_size"].to_numpy()
            mu = grp[f"{metric}_mean"].to_numpy()
            sig = grp[f"{metric}_std"].to_numpy()

            # ── log-space symmetric band ──────────────────────────────────────────
            # In log space: log(mu) ± sig/mu  (first-order propagation of relative error)
            # → lower = mu * exp(-sig/mu),  upper = mu * exp(+sig/mu)
            rel = sig / mu  # coefficient of variation
            lower = mu * np.exp(-rel)
            upper = mu * np.exp(+rel)

            color = COLOR[config]
            ax.plot(
                x,
                mu,
                marker=MARKER[config],
                label=config,
                color=color,
                linewidth=2.5,
                markersize=8,
                alpha=0.8,
            )
            ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)

        ax.set_xscale("log")
        ax.set_yscale("log")
        # ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Sample Size", fontsize=12)
        ax.set_ylabel("Mean Squared Error", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].legend(title="Configuration", fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ablation_components.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    collect_ablation_results(layers_only=True)
    collect_ablation_results(layers_only=False)

    ablation_components()
    plot_layers_scaling()
