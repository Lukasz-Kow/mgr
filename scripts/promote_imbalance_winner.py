#!/usr/bin/env python3
"""Promote winning imbalance strategy to production and optionally cleanup ablation artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT / "experiments" / "imbalance_ablation"
EXP_CONFIG = EXP_ROOT / "config.yaml"

from src.training.imbalance_strategy import normalize_strategy, STRATEGIES  # noqa: E402
from src.training.archive_utils import archive_production, archive_ablation_experiment  # noqa: E402


def load_exp_config() -> dict:
    with open(EXP_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_arm_for_strategy(cfg: dict, strategy: str) -> str:
    for arm, arm_cfg in cfg["arms"].items():
        if arm_cfg["strategy"] == strategy:
            return arm
    raise ValueError(f"No arm configured for strategy '{strategy}'")


def update_data_config(strategy: str, dry_run: bool) -> None:
    path = ROOT / "configs" / "data_config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    settings = STRATEGIES[strategy]
    data_cfg["imbalance_strategy"] = strategy
    data_cfg.setdefault("dataloader", {})["balance_classes"] = settings["balance_classes"]

    if dry_run:
        print(f"[DRY-RUN] Would set data_config imbalance_strategy={strategy}")
        return
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Updated {path}")


def copy_checkpoints(cfg: dict, arm: str, dry_run: bool, archive_first: bool = True) -> None:
    if archive_first:
        archive_production(ROOT, dry_run=dry_run)
    for model in cfg["models"]:
        src = ROOT / cfg["checkpoint_root"] / arm / model["name"] / model["checkpoint_name"]
        dst_dir = ROOT / "checkpoints" / model["name"]
        dst = dst_dir / model["checkpoint_name"]
        if not src.exists():
            print(f"[WARN] Missing checkpoint: {src}")
            continue
        if dry_run:
            print(f"[DRY-RUN] Would copy {src} -> {dst}")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst}")


def save_decision(strategy: str, arm: str, dry_run: bool) -> None:
    winner_path = EXP_ROOT / "results" / "winner.json"
    winner = {}
    if winner_path.exists():
        winner = json.loads(winner_path.read_text(encoding="utf-8"))
    record = {
        "promoted_strategy": strategy,
        "promoted_arm": arm,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "winner_recommendation": winner,
    }
    out = ROOT / "results" / "imbalance_strategy_decision.json"
    if dry_run:
        print(f"[DRY-RUN] Would save decision to {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved {out}")


def cleanup_ablation(cfg: dict, winner_arm: str, archive: bool, dry_run: bool) -> None:
    if archive and not dry_run:
        archive_ablation_experiment(EXP_ROOT, ROOT, label=f"pre_cleanup_{winner_arm}")

    ckpt_root = ROOT / cfg["checkpoint_root"]
    for arm in cfg["arms"]:
        if arm == winner_arm:
            continue
        arm_dir = ckpt_root / arm
        if arm_dir.exists():
            if dry_run:
                print(f"[DRY-RUN] Would remove {arm_dir}")
            else:
                shutil.rmtree(arm_dir)
                print(f"Removed {arm_dir}")

    scripts_to_remove = [
        ROOT / "scripts" / "run_imbalance_ablation.py",
        ROOT / "scripts" / "evaluate_ablation.py",
        ROOT / "scripts" / "promote_imbalance_winner.py",
    ]
    for script in scripts_to_remove:
        if script.exists():
            if dry_run:
                print(f"[DRY-RUN] Would remove {script}")
            elif not archive:
                script.unlink()
                print(f"Removed {script}")

    arms_dir = EXP_ROOT / "arms"
    if arms_dir.exists() and not dry_run and not archive:
        shutil.rmtree(arms_dir)
        print(f"Removed {arms_dir}")

    if archive and not dry_run:
        archive_path = ROOT / "experiments" / "imbalance_ablation_archive.zip"
        shutil.make_archive(str(archive_path.with_suffix("")), "zip", EXP_ROOT)
        print(f"Archived experiment to {archive_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote imbalance strategy winner")
    parser.add_argument("--strategy", type=str, required=True, help="balanced_sampler | natural | cost_sensitive")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="Remove non-winner arms and ablation scripts")
    parser.add_argument("--archive", action="store_true", help="Zip experiment before cleanup")
    parser.add_argument("--skip-copy-checkpoints", action="store_true")
    parser.add_argument(
        "--no-archive-production",
        action="store_true",
        help="Do not archive current production checkpoints before promote",
    )
    args = parser.parse_args()

    strategy = normalize_strategy(args.strategy)
    cfg = load_exp_config()
    arm = find_arm_for_strategy(cfg, strategy)

    print(f"Promoting strategy={strategy} (arm={arm})")
    update_data_config(strategy, args.dry_run)
    if not args.skip_copy_checkpoints:
        copy_checkpoints(cfg, arm, args.dry_run, archive_first=not args.no_archive_production)
    save_decision(strategy, arm, args.dry_run)

    if args.cleanup or args.archive:
        cleanup_ablation(cfg, arm, args.archive, args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
