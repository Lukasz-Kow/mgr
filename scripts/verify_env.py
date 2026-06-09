#!/usr/bin/env python3
"""Verify Windows conda environment and project prerequisites."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main() -> int:
    print("=" * 60)
    print("Environment verification (Windows / mgr)")
    print("=" * 60)

    all_ok = True
    all_ok &= check("Python", sys.version_info >= (3, 10), sys.version.split()[0])

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        all_ok &= check("PyTorch", True, f"{torch.__version__}")
        all_ok &= check("CUDA", cuda_ok, gpu)
    except ImportError:
        all_ok &= check("PyTorch", False, "not installed")

    for pkg in ("nibabel", "sklearn", "pandas", "yaml", "tqdm"):
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
            all_ok &= check(pkg, True)
        except ImportError:
            all_ok &= check(pkg, False, "not installed")

    data_dir = PROJECT_ROOT / "Data baseline" / "ADNI"
    csv_labels = PROJECT_ROOT / "Data_baseline_2_23_2026.csv"
    metadata = PROJECT_ROOT / "data_metadata_adni.csv"

    all_ok &= check("ADNI data dir", data_dir.exists(), str(data_dir))
    all_ok &= check("ADNI labels CSV", csv_labels.exists(), str(csv_labels))
    all_ok &= check("Metadata CSV", metadata.exists(), str(metadata))

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from src.data import MCIDataModule
        all_ok &= check("Project imports", True, "MCIDataModule")
    except Exception as e:
        all_ok &= check("Project imports", False, str(e))

    print("=" * 60)
    if all_ok:
        print("All checks passed.")
        return 0
    print("Some checks failed — fix issues above before training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
