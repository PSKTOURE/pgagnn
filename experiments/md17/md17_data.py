import os
from pathlib import Path

from torch_geometric.data import download_url, extract_tar
from torch_geometric.datasets import MD17


# Create a custom MD17 class with fallback download handling
class MD17WithFallback(MD17):
    def download(self) -> None:
        """Download with fallback: try revised URL first, then provide helpful error."""
        try:
            # Try the revised URL first
            print("Attempting to download revised MD17 dataset from archive.materialscloud.org...")
            super().download()
            print("Successfully downloaded from revised URL")
            return
        except Exception as e:
            print(f"✗ Revised URL failed: {e}")

        # Fallback: direct link to mirrored rMD17 archive (materialscloud record pfffs-fff86)
        if self.revised:
            fallback_url = (
                "https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1"
            )
            try:
                print("Attempting fallback download from materialscloud (record pfffs-fff86)...")
                path = download_url(fallback_url, self.raw_dir)
                extract_tar(path, self.raw_dir, mode="r:bz2")
                os.unlink(path)
                print("Fallback download and extraction succeeded")
                return
            except Exception as e2:
                print(f"✗ Fallback download failed: {e2}")

            print("\n" + "=" * 80)
            print("ERROR: The revised MD17 dataset could not be downloaded automatically.")
            print(f"Tried revised URL and fallback: {fallback_url}")
            print("\nPlease manually download and extract to:")
            print(f"  {self.raw_dir}/rmd17/npz_data/")
            print("Files should be named like rmd17_aspirin.npz, rmd17_benzene.npz, etc.")
            print("=" * 80 + "\n")

            # Check if processed data already exists locally
            if Path(self.processed_dir).exists():
                files = list(Path(self.processed_dir).glob("*.pt"))
                if files:
                    print(f"Found existing processed data at {self.processed_dir}")
                    print("  Using cached data instead.")
                    return

            raise RuntimeError(
                f"Could not download revised MD17 dataset for '{self.name}'. "
                f"Tried primary and fallback URLs. "
                f"Please manually download from {fallback_url} and extract to: {self.raw_dir}/rmd17/npz_data/"
            )

        # Non-revised: re-raise original exception to follow base behavior
        raise
