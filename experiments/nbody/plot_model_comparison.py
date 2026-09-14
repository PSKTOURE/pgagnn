"""
Plot validation, test, and test OOD losses for different models across data subsamples.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration
ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_RUN_IDS = {
    "nbody": [0, 1, 2],
    "nbody_spring": [0, 1, 2],
}
DEFAULT_DATASET = "nbody_spring"
SUBSAMPLES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
MODELS = ["gatr", "pgagnn", "egnn", "segnn", "ggnn"]
COLORS = {"gatr": "#1f77b4", "pgagnn": "#ff7f0e", "egnn": "#2ca02c", "segnn": "#d62728", "ggnn": "#9467bd"}
MARKERS = {"gatr": "o", "pgagnn": "s", "egnn": "^", "segnn": "D", "ggnn": "v"}


def get_plot_dirs(dataset_name):
    """Return the output directories for a dataset and ensure they exist."""
    plot_dir = ROOT_DIR / "plots" / dataset_name
    compare_dir = plot_dir / "compare_runs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir, compare_dir


def extract_subsample_and_model_from_path(path_str, range_idx=4):
    """Extract subsample and model from directory name."""
    parts = path_str.split("_")

    model_name = None
    for part in parts:
        if part in MODELS[:range_idx]:
            model_name = part
            break

    subsample = 1.0  # default
    for i, part in enumerate(parts):
        if part.startswith("sub"):
            try:
                subsample_str = part[3:]
                subsample = float(subsample_str)
                break
            except (ValueError, IndexError):
                pass

    return model_name, subsample


def find_best_run_for_model(runs_dir, model_name, subsample):
    """Find the best run for a model at a given subsample."""
    best_val_loss = float("inf")
    best_metrics = None
    best_path = None

    # Recursively search all directories for metrics.json files
    for root, dirs, files in os.walk(runs_dir):
        if "metrics.json" not in files:
            continue

        metrics_file = Path(root) / "metrics.json"

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            config = metrics.get("config", {})

            # Check if this is the right model and subsample
            if config.get("model_name") != model_name:
                continue
            if config.get("subsample") != subsample:
                continue

            val_loss = metrics["summary"]["best_val_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics
                best_path = Path(root)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return best_metrics, best_path


def collect_data(dataset_name=DEFAULT_DATASET, run_id=10, range_idx=4):
    """Collect loss data for all models and subsamples."""
    runs_dir = ROOT_DIR / "runs" / dataset_name / str(run_id)
    data = {
        "val_loss": defaultdict(dict),
        "test_loss": defaultdict(dict),
        "test_ood_loss": defaultdict(dict),
    }

    for subsample in SUBSAMPLES:
        for model in MODELS[:range_idx]:
            metrics, _path = find_best_run_for_model(runs_dir, model, subsample)

            if metrics is None:
                print(f"No metrics found for {model} at subsample {subsample}")
                continue

            summary = metrics["summary"]
            data["val_loss"][model][subsample] = summary["best_val_loss"]
            data["test_loss"][model][subsample] = summary["test_loss"]
            data["test_ood_loss"][model][subsample] = summary["test_ood_loss"]

            print(
                f"{model:6s} @ {subsample:5.3f}: val={summary['best_val_loss']:.6f}, "
                f"test={summary['test_loss']:.6f}, test_ood={summary['test_ood_loss']:.6f}"
            )

    return data


def collect_mean_std_data(dataset_name, run_ids, models=MODELS, subsamples=SUBSAMPLES):
    """Collect mean and standard deviation across multiple runs for each model/subsample."""
    data = {
        "val_loss": defaultdict(dict),
        "test_loss": defaultdict(dict),
        "test_ood_loss": defaultdict(dict),
    }

    for model_name in models:
        for subsample in subsamples:
            values = {
                "val_loss": [],
                "test_loss": [],
                "test_ood_loss": [],
            }

            for run_id in run_ids:
                runs_dir = ROOT_DIR / "runs" / dataset_name / str(run_id)
                metrics, _ = find_best_run_for_model(runs_dir, model_name, subsample)

                if metrics is None:
                    print(
                        f"No metrics found for {model_name} in {dataset_name} "
                        f"run {run_id} at subsample {subsample}"
                    )
                    continue

                summary = metrics["summary"]
                values["val_loss"].append(summary["best_val_loss"])
                values["test_loss"].append(summary["test_loss"])
                values["test_ood_loss"].append(summary["test_ood_loss"])

            for loss_key, loss_values in values.items():
                if not loss_values:
                    continue

                loss_array = np.asarray(loss_values, dtype=float)
                mean = float(loss_array.mean())
                std = float(loss_array.std()) if loss_array.size > 1 else 0.0

                data[loss_key][model_name][subsample] = {
                    "mean": mean,
                    "std": std,
                    "values": loss_array.tolist(),
                }

            print(
                f"{dataset_name} {model_name:6s} @ {subsample:5.3f}: "
                f"val={data['val_loss'][model_name][subsample]['mean']:.6f} ± "
                f"{data['val_loss'][model_name][subsample]['std']:.6f}, "
                f"test={data['test_loss'][model_name][subsample]['mean']:.6f} ± "
                f"{data['test_loss'][model_name][subsample]['std']:.6f}, "
                f"test_ood={data['test_ood_loss'][model_name][subsample]['mean']:.6f} ± "
                f"{data['test_ood_loss'][model_name][subsample]['std']:.6f}"
            )

    return data


def collect_single_model_across_runs(run_ids, model_name, subsamples=SUBSAMPLES):
    """Collect loss data for one model across multiple runs."""
    data = {}
    for run_id in run_ids:
        runs_dir = ROOT_DIR / "runs" / DEFAULT_DATASET / str(run_id)
        run_data = {
            "val_loss": {},
            "test_loss": {},
            "test_ood_loss": {},
        }
        for subsample in subsamples:
            metrics, _ = find_best_run_for_model(runs_dir, model_name, subsample)

            if metrics is None:
                print(f"No metrics found for {model_name} in run {run_id} at subsample {subsample}")
                continue

            summary = metrics["summary"]
            run_data["val_loss"][subsample] = summary["best_val_loss"]
            run_data["test_loss"][subsample] = summary["test_loss"]
            run_data["test_ood_loss"][subsample] = summary["test_ood_loss"]

            print(
                f"run {run_id} {model_name:6s} @ {subsample:5.3f}: val={summary['best_val_loss']:.6f}, "
                f"test={summary['test_loss']:.6f}, test_ood={summary['test_ood_loss']:.6f}"
            )

        data[run_id] = run_data

    return data


def collect_all_models_across_runs(run_ids, models=MODELS, subsamples=SUBSAMPLES):
    """Collect loss data for all models across multiple runs."""
    data = {}

    for run_id in run_ids:
        runs_dir = ROOT_DIR / "runs" / DEFAULT_DATASET / str(run_id)
        run_data = {
            "val_loss": defaultdict(dict),
            "test_loss": defaultdict(dict),
            "test_ood_loss": defaultdict(dict),
        }

        for model_name in models:
            for subsample in subsamples:
                metrics, _ = find_best_run_for_model(runs_dir, model_name, subsample)

                if metrics is None:
                    print(f"No metrics found for {model_name} in run {run_id} at subsample {subsample}")
                    continue

                summary = metrics["summary"]
                run_data["val_loss"][model_name][subsample] = summary["best_val_loss"]
                run_data["test_loss"][model_name][subsample] = summary["test_loss"]
                run_data["test_ood_loss"][model_name][subsample] = summary["test_ood_loss"]

                print(
                    f"run {run_id} {model_name:6s} @ {subsample:5.3f}: "
                    f"val={summary['best_val_loss']:.6f}, "
                    f"test={summary['test_loss']:.6f}, "
                    f"test_ood={summary['test_ood_loss']:.6f}"
                )

        data[run_id] = run_data

    return data


def create_plots(
    data,
    remove_outliers=False,
    range_idx=4,
    dataset_name=DEFAULT_DATASET,
    run_ids=None,
    axes=None,
    row_idx=0,
):
    """Create three plots for validation, test, and test OOD losses."""
    _, _plot_compare_dir = get_plot_dirs(dataset_name)

    loss_types = [
        ("val_loss", "Validation Loss"),
        ("test_loss", "Token number generalization"),
        ("test_ood_loss", "Test OOD Loss"),
    ]

    for i, (loss_key, title) in enumerate(loss_types):
        ax = axes[row_idx, i]  # Use the provided axes
        for model in MODELS[:range_idx]:
            series = data[loss_key][model]
            subsamples_sorted = sorted(series.keys())
            if not subsamples_sorted:
                continue

            means = []
            stds = []
            for subsample in subsamples_sorted:
                entry = series[subsample]
                if isinstance(entry, dict) and "mean" in entry:
                    means.append(entry["mean"])
                    stds.append(entry.get("std", 0.0))
                else:
                    means.append(entry)
                    stds.append(0.0)

            linestyle = "-" if model == "pgagnn" else "--"

            if remove_outliers:
                median = np.median(means)
                threshold = median * 1.5
                means = [min(loss, threshold) for loss in means]
                stds = [min(std, threshold) for std in stds]

            means_array = np.asarray(means, dtype=float)
            stds_array = np.asarray(stds, dtype=float)
            lower = means_array * np.exp(-stds_array / means_array)
            upper = means_array * np.exp(+stds_array / means_array)
            subsamples_sorted = np.array(subsamples_sorted) * 1e5
            label = model.upper() if model != "pgagnn" else "PGA-GNN(ours)"

            ax.plot(
                subsamples_sorted,
                means_array,
                marker=MARKERS[model],
                label=f"{label}",
                color=COLORS[model],
                linestyle=linestyle,
                linewidth=2.5,
                markersize=8,
                alpha=0.8,
            )
            if np.any(stds_array > 0):
                ax.fill_between(
                    subsamples_sorted,
                    lower,
                    upper,
                    color=COLORS[model],
                    alpha=0.18,
                    linewidth=0,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training Samples", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax.set_title(f"{dataset_name} - {title}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc="best")


def plot_model_across_runs(model_name, run_ids, remove_outliers=False, subsamples=SUBSAMPLES):
    """Plot a single model's losses across multiple runs."""
    data = collect_single_model_across_runs(run_ids, model_name, subsamples=subsamples)
    _, plot_compare_dir = get_plot_dirs(DEFAULT_DATASET)

    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    loss_types = [
        ("val_loss", "Validation Loss", axes[0]),
        ("test_loss", "Token number generalization", axes[1]),
        ("test_ood_loss", "Test OOD Loss", axes[2]),
    ]

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for loss_key, title, ax in loss_types:
        for idx, run_id in enumerate(run_ids):
            subsamples_sorted = sorted(data.get(run_id, {}).get(loss_key, {}).keys())
            if not subsamples_sorted:
                continue
            losses = [data[run_id][loss_key][s] for s in subsamples_sorted]
            subsamples_sorted = np.array(subsamples_sorted) * 1e5

            if remove_outliers:
                median = np.median(losses)
                threshold = median * 1.5
                losses = [min(loss, threshold) for loss in losses]

            ax.plot(
                subsamples_sorted,
                losses,
                marker=markers[idx % len(markers)],
                label=f"Run {run_id}",
                color=colors[idx % len(colors)],
                linewidth=2.5,
                markersize=8,
                alpha=0.8,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Data Subsample Fraction", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc="best")

    plt.tight_layout()

    suffix = "_outliers_clipped" if remove_outliers else ""
    run_str = "_".join(str(r) for r in run_ids)
    filename = plot_compare_dir / f"{model_name}_runs_{run_str}{suffix}.pdf"
    plt.savefig(filename, bbox_inches="tight")
    print(f"\nPlot saved to {filename}")
    plt.close()


def plot_all_models_across_runs(run_ids, models=MODELS, remove_outliers=False, subsamples=SUBSAMPLES):
    """Plot all models across multiple runs in a single comparison figure."""
    data = collect_all_models_across_runs(run_ids, models=models, subsamples=subsamples)
    _, plot_compare_dir = get_plot_dirs(DEFAULT_DATASET)

    _fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    loss_types = [
        ("val_loss", "Validation Loss", axes[0]),
        ("test_loss", "Test Loss", axes[1]),
        ("test_ood_loss", "Test OOD Loss", axes[2]),
    ]

    run_line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    for loss_key, title, ax in loss_types:
        for model_name in models:
            for run_idx, run_id in enumerate(run_ids):
                series = data.get(run_id, {}).get(loss_key, {}).get(model_name, {})
                subsamples_sorted = sorted(series.keys())

                if not subsamples_sorted:
                    continue

                losses = [series[s] for s in subsamples_sorted]
                x_values = np.array(subsamples_sorted) * 1e5

                if remove_outliers:
                    median = np.median(losses)
                    threshold = median * 1.5
                    losses = [min(loss, threshold) for loss in losses]

                ax.plot(
                    x_values,
                    losses,
                    marker=MARKERS.get(model_name, "o"),
                    linestyle=run_line_styles[run_idx % len(run_line_styles)],
                    label=f"{model_name.upper()} - Run {run_id}",
                    color=COLORS.get(model_name, None),
                    linewidth=2.2,
                    markersize=7,
                    alpha=0.85,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Data Subsample Fraction", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best", ncol=2)

    plt.tight_layout()

    suffix = "_outliers_clipped" if remove_outliers else ""
    run_str = "_".join(str(r) for r in run_ids)
    filename = plot_compare_dir / f"all_models_runs_{run_str}{suffix}.pdf"
    plt.savefig(filename, bbox_inches="tight")
    print(f"\nPlot saved to {filename}")
    plt.close()


def collect_train_curves_across_runs(run_ids, model_name, subsample=1.0):
    """Collect training loss histories for one model across multiple runs at a given subsample."""
    data = {}

    for run_id in run_ids:
        runs_dir = ROOT_DIR / "runs" / DEFAULT_DATASET / str(run_id)
        metrics, _ = find_best_run_for_model(runs_dir, model_name, subsample)

        if metrics is None:
            print(f"No metrics found for {model_name} in run {run_id} at subsample {subsample}")
            continue

        train_losses = metrics.get("metrics", {}).get("train_losses")
        if train_losses is None:
            print(f"No training-history found for {model_name} in run {run_id} at subsample {subsample}")
            continue

        data[run_id] = np.array(train_losses)

    return data


def plot_training_curves(model_name, run_ids, subsample=1.0, use_symlog=True):
    """Plot a single model's training-loss curves across multiple runs.

    Args:
        model_name: Name of the model (e.g. 'gatr')
        run_ids: List of top-level run ids (integers or strings)
        subsample: Data subsample fraction to pick the run for
        use_symlog: If True use symmetric-log scale on y to handle negative values
    """
    data = collect_train_curves_across_runs(run_ids, model_name, subsample=subsample)
    _, plot_compare_dir = get_plot_dirs(DEFAULT_DATASET)

    if not data:
        print(f"No training curves found for {model_name} at subsample {subsample}")
        return

    _fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for idx, (run_id, train_arr) in enumerate(data.items()):
        epochs = np.arange(1, len(train_arr) + 1)
        ax.plot(
            epochs,
            train_arr,
            label=f"Run {run_id}",
            color=colors[idx % len(colors)],
            linewidth=2.2,
            alpha=0.9,
        )

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Loss", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name.upper()} Training Loss (sub={subsample})", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if use_symlog:
        ax.set_yscale("symlog", linthresh=1e-6)
    else:
        ax.set_yscale("log")

    ax.legend(fontsize=11, loc="best")
    plt.tight_layout()

    run_str = "_".join(str(r) for r in run_ids)
    filename = plot_compare_dir / f"{model_name}_train_runs_{run_str}_sub{subsample}.pdf"
    plt.savefig(filename, bbox_inches="tight")
    print(f"\nPlot saved to {filename}")
    plt.close()


def plot_training_curves_all_models(run_ids, models=MODELS, subsample=1.0, use_symlog=True):
    """Plot training-loss curves for all models across multiple runs in one figure."""
    _fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    _, plot_compare_dir = get_plot_dirs(DEFAULT_DATASET)

    run_line_styles = ["-", "--", "-.", ":"]
    colors = plt.cm.tab10.colors
    plotted_any = False
    for model_idx, model_name in enumerate(models):
        for run_idx, run_id in enumerate(run_ids):
            runs_dir = ROOT_DIR / "runs" / DEFAULT_DATASET / str(run_id)
            metrics, _ = find_best_run_for_model(runs_dir, model_name, subsample)

            if metrics is None:
                print(f"No metrics found for {model_name} in run {run_id} at subsample {subsample}")
                continue

            train_losses = metrics.get("metrics", {}).get("train_losses")
            if not train_losses:
                continue

            epochs = np.arange(1, len(train_losses) + 1)
            ax.plot(
                epochs,
                train_losses,
                label=f"{model_name.upper()} - Run {run_id}",
                color=COLORS.get(model_name, colors[model_idx % len(colors)]),
                linestyle=run_line_styles[run_idx % len(run_line_styles)],
                linewidth=1.8,
                alpha=0.9,
            )
            plotted_any = True

    if not plotted_any:
        print("No training curves found for given models/runs.")
        return

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Loss", fontsize=12, fontweight="bold")
    ax.set_title(f"Training Loss Comparison (sub={subsample})", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if use_symlog:
        ax.set_yscale("symlog", linthresh=1e-6)
    else:
        ax.set_yscale("log")

    ax.legend(fontsize=9, loc="best", ncol=2)
    plt.tight_layout()

    run_str = "_".join(str(r) for r in run_ids)
    filename = plot_compare_dir / f"all_models_train_runs_{run_str}_sub{subsample}.pdf"
    plt.savefig(filename, bbox_inches="tight")
    print(f"\nPlot saved to {filename}")
    plt.close()


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, (dataset_name, run_ids) in enumerate(DATASET_RUN_IDS.items()):
        print("=" * 70)
        print(f"Collecting metrics from runs for {dataset_name}...")
        print("=" * 70)

        data = collect_mean_std_data(dataset_name, run_ids)

        print("\n" + "=" * 70)
        print(f"Creating plots for {dataset_name}...")
        print("=" * 70)

        create_plots(
            data,
            remove_outliers=False,
            range_idx=4,
            dataset_name=dataset_name,
            run_ids=run_ids,
            axes=axes,
            row_idx=row_idx,
        )

    plt.tight_layout()
    combined_path = ROOT_DIR / "plots" / "nbody" / "compare_runs" / "model_comparison_losses_combined.pdf"
    plt.savefig(combined_path, bbox_inches="tight")
    print(f"\nCombined plot saved to {combined_path}")
    plt.close()
