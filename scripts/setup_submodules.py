"""Cross-platform script to link baseline model submodules into the active Python environment.

Links:
  - Repository root
  - geometric-algebra-transformer
  - egnn
  - Steerable-E3-GNN/models
"""

import sysconfig
from pathlib import Path


def setup_submodules() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    paths = [
        root_dir,
        root_dir / "geometric-algebra-transformer",
        root_dir / "egnn",
        root_dir / "Steerable-E3-GNN" / "models",
    ]

    # Verify submodule directories exist
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("Warning: The following submodule paths were not found:")
        for p in missing:
            print(f"  - {p}")
        print("Make sure you cloned with submodules: git clone --recursive <URL>")
        print("or run: git submodule update --init --recursive\n")

    purelib = Path(sysconfig.get_paths()["purelib"])
    purelib.mkdir(parents=True, exist_ok=True)
    pth_file = purelib / "vendor_models.pth"

    content = "\n".join(str(p) for p in paths) + "\n"
    pth_file.write_text(content)

    print("Successfully linked baseline submodules in:")
    print(f"  {pth_file}")
    for p in paths:
        status = "found" if p.exists() else "MISSING"
        print(f"  [{status}] {p.name}")


if __name__ == "__main__":
    setup_submodules()
