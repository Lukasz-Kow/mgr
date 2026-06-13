#!/usr/bin/env python3
"""Orchestrate imbalance A/B/C ablation training (12 runs)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.archive_utils import archive_production  # noqa: E402

EXP_ROOT = ROOT / "experiments" / "imbalance_ablation"
EXP_CONFIG = EXP_ROOT / "config.yaml"


def load_experiment_config() -> dict:
    with open(EXP_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def checkpoint_path(cfg: dict, arm: str, model: dict) -> Path:
    return (
        ROOT
        / cfg["checkpoint_root"]
        / arm
        / model["name"]
        / model["checkpoint_name"]
    )


def manifest_path(cfg: dict, arm: str, model_name: str) -> Path:
    root = ROOT / cfg["manifests_root"]
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{arm}_{model_name}.json"


def run_is_complete(cfg: dict, arm: str, model: dict) -> bool:
    ckpt = checkpoint_path(cfg, arm, model)
    manifest = manifest_path(cfg, arm, model["name"])
    return ckpt.exists() and manifest.exists()


def write_manifest(cfg: dict, arm: str, model: dict, strategy: str) -> None:
    manifest = {
        "arm": arm,
        "model": model["name"],
        "strategy": strategy,
        "seed": cfg.get("seed", 42),
        "checkpoint": str(checkpoint_path(cfg, arm, model)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path(cfg, arm, model["name"]).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def train_run(cfg: dict, arm: str, arm_cfg: dict, model: dict, force: bool) -> int:
    if not force and run_is_complete(cfg, arm, model):
        print(f"[SKIP] {arm}/{model['name']} — checkpoint exists")
        return 0

    strategy = arm_cfg["strategy"]
    ckpt_dir = ROOT / cfg["checkpoint_root"] / arm / model["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / model["script"]),
        "--config",
        str(ROOT / model["config"]),
        "--imbalance-strategy",
        strategy,
        "--checkpoint-dir",
        str(ckpt_dir),
    ]
    print(f"\n[TRAIN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode == 0:
        write_manifest(cfg, arm, model, strategy)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run imbalance ablation training")
    parser.add_argument("--arms", type=str, default=None, help="Comma-separated arms, e.g. arm_a,arm_b")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated models, e.g. hybrid")
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, run evaluation only")
    parser.add_argument(
        "--archive-production",
        action="store_true",
        help="Archive current production checkpoints/results before starting training",
    )
    parser.add_argument("--dry-run", action="store_true", help="With --archive-production: show what would be archived")
    args = parser.parse_args()

    if args.archive_production:
        print("[ARCHIVE] Saving production snapshot before ablation training...")
        archive_production(ROOT, dry_run=args.dry_run)

    cfg = load_experiment_config()
    arms = list(cfg["arms"].keys())
    models = cfg["models"]
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",")]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in models if m["name"] in wanted]

    if not args.eval_only:
        print("=" * 70)
        print("  IMBALANCE A/B/C ABLATION — TRAINING")
        print("=" * 70)
        for arm in arms:
            arm_cfg = cfg["arms"][arm]
            for model in models:
                rc = train_run(cfg, arm, arm_cfg, model, args.force)
                if rc != 0:
                    print(f"[ERROR] Training failed: {arm}/{model['name']}")
                    return rc

    eval_script = ROOT / "scripts" / "evaluate_ablation.py"
    if eval_script.exists():
        print("\n[EVAL] Running evaluate_ablation.py ...")
        return subprocess.run([sys.executable, str(eval_script)], cwd=str(ROOT)).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
