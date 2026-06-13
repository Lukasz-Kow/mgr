"""Archive production checkpoints and results before overwrite."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent

PRODUCTION_MODELS = [
    ("baseline", "best_model.pth"),
    ("selective_net", "best_model.pt"),
    ("evidential", "best_model.pt"),
    ("hybrid", "best_model.pt"),
]

PRODUCTION_RESULT_FILES = [
    "results/final_comparison.csv",
    "results/comparison_with_abstention.csv",
    "results/fp_coverage.csv",
    "results/imbalance_strategy_decision.json",
]


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def archive_production(
    root: Optional[Path] = None,
    label: Optional[str] = None,
    dry_run: bool = False,
) -> Path:
    """
    Copy current production checkpoints + key results to archives/production/{timestamp}/.

    Returns path to archive directory (even in dry-run).
    """
    root = root or ROOT
    ts = label or _timestamp_label()
    archive_dir = root / "archives" / "production" / ts
    manifest = {
        "type": "production",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": ts,
        "files": [],
    }

    if dry_run:
        print(f"[DRY-RUN] Would archive production to {archive_dir}")
    else:
        archive_dir.mkdir(parents=True, exist_ok=True)

    for model_dir, ckpt_name in PRODUCTION_MODELS:
        src = root / "checkpoints" / model_dir / ckpt_name
        if not src.exists():
            continue
        dst = archive_dir / "checkpoints" / model_dir / ckpt_name
        if dry_run:
            print(f"  [DRY-RUN] {src} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        manifest["files"].append(str(src.relative_to(root)))

    for rel in PRODUCTION_RESULT_FILES:
        src = root / rel
        if not src.exists():
            continue
        dst = archive_dir / rel
        if dry_run:
            print(f"  [DRY-RUN] {src} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        manifest["files"].append(str(src.relative_to(root)))

    if not dry_run:
        (archive_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Archived {len(manifest['files'])} files to {archive_dir}")

    return archive_dir


def archive_ablation_experiment(
    exp_root: Path,
    root: Optional[Path] = None,
    label: Optional[str] = None,
    dry_run: bool = False,
) -> Path:
    """Zip or copy entire ablation experiment folder to archives/ablation/."""
    root = root or ROOT
    ts = label or _timestamp_label()
    archive_dir = root / "archives" / "ablation" / ts

    if dry_run:
        print(f"[DRY-RUN] Would archive ablation experiment to {archive_dir}")
        return archive_dir

    if not exp_root.exists():
        print(f"[WARN] Ablation root not found: {exp_root}")
        return archive_dir

    archive_dir.mkdir(parents=True, exist_ok=True)
    zip_base = archive_dir / "imbalance_ablation"
    shutil.make_archive(str(zip_base), "zip", exp_root)
    manifest = {
        "type": "ablation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": ts,
        "source": str(exp_root.relative_to(root)),
        "zip": str(zip_base.with_suffix(".zip").relative_to(root)),
    }
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Archived ablation to {zip_base.with_suffix('.zip')}")
    return archive_dir


def list_archives(root: Optional[Path] = None) -> dict:
    """Return available archive timestamps under archives/production and archives/ablation."""
    root = root or ROOT
    out = {"production": [], "ablation": []}
    for kind in out:
        base = root / "archives" / kind
        if base.exists():
            out[kind] = sorted(p.name for p in base.iterdir() if p.is_dir())
    return out
