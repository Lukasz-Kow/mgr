#!/usr/bin/env python3
"""Archive production checkpoints/results or ablation experiment before overwrite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.archive_utils import (  # noqa: E402
    archive_ablation_experiment,
    archive_production,
    list_archives,
)

EXP_ROOT = ROOT / "experiments" / "imbalance_ablation"


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive snapshots before training or promote")
    parser.add_argument(
        "--target",
        choices=["production", "ablation", "both"],
        default="production",
        help="What to archive (default: production checkpoints + results)",
    )
    parser.add_argument("--label", type=str, default=None, help="Optional archive folder name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="List existing archives and exit")
    args = parser.parse_args()

    if args.list:
        archives = list_archives(ROOT)
        print(json.dumps(archives, indent=2))
        return 0

    if args.target in ("production", "both"):
        archive_production(ROOT, label=args.label, dry_run=args.dry_run)
    if args.target in ("ablation", "both"):
        ab_label = f"{args.label}_ablation" if args.label else None
        archive_ablation_experiment(EXP_ROOT, ROOT, label=ab_label, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
