import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect_data(data_dir: Path, output_file: Path) -> None:
    """Collect test metrics from benchmark run directories and save to CSV."""
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    print(f"Collecting data from {data_dir}...")
    data = {}
    data_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    data_dirs.sort(key=lambda x: x.name)
    for dir_entry in data_dirs:
        for model_dir in dir_entry.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name.split("_")[0]
            for file in model_dir.iterdir():
                if file.name != "metrics.json":
                    continue
                with open(file, "r") as f:
                    metrics = json.load(f).get("summary", {})
                force = round(metrics.get("test_force_mae_mev_A", np.nan), 3)
                energy = round(metrics.get("test_energy_mae_mev", np.nan), 3)
                data.setdefault("model", []).append(model_name)
                data.setdefault("target", []).append(dir_entry.name)
                data.setdefault("force", []).append(force)
                data.setdefault("energy", []).append(energy)

    if not data:
        print("No metrics found to collect.")
        return

    df = pd.DataFrame(data)
    mean_df = df.groupby("model", as_index=False).agg({"force": "mean", "energy": "mean"})
    mean_df["target"] = "mean"
    df = pd.concat([df, mean_df], ignore_index=True)
    df = df.sort_values(["model", "target"], kind="stable").reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"Data collected and saved to {output_file}")


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Collect experiment metrics into a summary CSV.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=f"{ROOT_DIR}/runs/md17/all",
        help="Path to directory containing run folders (default: runs/md17/all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"{ROOT_DIR}/collected_data.csv",
        help="Output CSV path (default: collected_data.csv)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    in_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else (root / args.data_dir)
    out_path = Path(args.output) if Path(args.output).is_absolute() else (root / args.output)

    collect_data(in_dir, out_path)
