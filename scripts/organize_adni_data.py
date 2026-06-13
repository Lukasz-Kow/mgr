#!/usr/bin/env python3
"""
Inventory and optional organization of ADNI data under Data baseline/.

Usage:
    python scripts/organize_adni_data.py              # report only
    python scripts/organize_adni_data.py --execute    # copy/move (idempotent)
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "Data baseline"
ADNI_ROOT = DATASET_ROOT / "ADNI"
BASELINE_DIR = ADNI_ROOT / "baseline"
ADNI2_DIR = ADNI_ROOT / "ADNI2"
METADATA_DIR = DATASET_ROOT / "metadata"

BASELINE_CSV = METADATA_DIR / "baseline_2026-02-23.csv"
MCI_CN_CSV = METADATA_DIR / "mci_cn_scaled2_2026-06-09.csv"
REPORT_PATH = METADATA_DIR / "inventory_report.json"

ONEDRIVE_ROOT = Path(r"C:\Users\Lukas\OneDrive\Dokumenty\mgr\Data baseline")
ONEDRIVE_ADNI2 = ONEDRIVE_ROOT / "ADNI" / "ADNI2"
ONEDRIVE_MCI_CN_CSV = Path(r"C:\Users\Lukas\OneDrive\Dokumenty\mgr\MCI_CN_Scaled2_6_09_2026.csv")

ADNI_MAPPING = {"CN": 0, "MCI": 1, "LMCI": 1, "EMCI": 1}


def _subject_dirs(folder: Path) -> set[str]:
    if not folder.exists():
        return set()
    return {p.name for p in folder.iterdir() if p.is_dir()}


def _count_nii(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*.nii") if "Zone.Identifier" not in p.name)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _labeled_subjects(df: pd.DataFrame) -> set[str]:
    if df.empty or "Subject" not in df.columns:
        return set()
    labeled = df[df["Group"].isin(ADNI_MAPPING.keys())]
    return set(labeled["Subject"].unique())


def build_inventory() -> dict:
    baseline_subjects = _subject_dirs(BASELINE_DIR)
    adni2_subjects = _subject_dirs(ADNI2_DIR)
    legacy_subjects = _subject_dirs(ADNI_ROOT) - {"baseline", "ADNI2"}

    all_disk_subjects = baseline_subjects | adni2_subjects | legacy_subjects
    overlap = baseline_subjects & adni2_subjects

    baseline_df = _load_csv(BASELINE_CSV)
    mci_cn_df = _load_csv(MCI_CN_CSV)

    baseline_csv_subjects = _labeled_subjects(baseline_df)
    mci_cn_csv_subjects = _labeled_subjects(mci_cn_df)

    only_baseline_csv = sorted(baseline_csv_subjects - mci_cn_csv_subjects)
    only_mci_cn_csv = sorted(mci_cn_csv_subjects - baseline_csv_subjects)
    both_csv = sorted(baseline_csv_subjects & mci_cn_csv_subjects)

    unlabeled_on_disk = sorted(
        s for s in all_disk_subjects
        if s not in baseline_csv_subjects and s not in mci_cn_csv_subjects
    )

    return {
        "paths": {
            "dataset_root": str(DATASET_ROOT),
            "baseline_dir": str(BASELINE_DIR),
            "adni2_dir": str(ADNI2_DIR),
            "metadata_dir": str(METADATA_DIR),
        },
        "counts": {
            "baseline_subjects": len(baseline_subjects),
            "adni2_subjects": len(adni2_subjects),
            "legacy_root_subjects": len(legacy_subjects),
            "unique_disk_subjects": len(all_disk_subjects),
            "overlap_baseline_adni2": len(overlap),
            "baseline_nii_files": _count_nii(BASELINE_DIR),
            "adni2_nii_files": _count_nii(ADNI2_DIR),
            "legacy_root_nii_files": _count_nii(ADNI_ROOT) - _count_nii(BASELINE_DIR) - _count_nii(ADNI2_DIR),
            "baseline_csv_rows": len(baseline_df),
            "baseline_csv_subjects": len(baseline_csv_subjects),
            "mci_cn_csv_rows": len(mci_cn_df),
            "mci_cn_csv_subjects": len(mci_cn_csv_subjects),
            "subjects_in_both_csv": len(both_csv),
            "subjects_only_baseline_csv": len(only_baseline_csv),
            "subjects_only_mci_cn_csv": len(only_mci_cn_csv),
            "unlabeled_on_disk": len(unlabeled_on_disk),
        },
        "overlap_subjects": sorted(overlap),
        "only_baseline_csv_subjects": only_baseline_csv,
        "unlabeled_on_disk": unlabeled_on_disk,
        "files_present": {
            "baseline_csv": BASELINE_CSV.exists(),
            "mci_cn_csv": MCI_CN_CSV.exists(),
            "compat_root_csv": (PROJECT_ROOT / "Data_baseline_2_23_2026.csv").exists(),
        },
    }


def print_summary(report: dict) -> None:
    c = report["counts"]
    print("=" * 60)
    print("ADNI DATA INVENTORY")
    print("=" * 60)
    print(f"Baseline subjects:     {c['baseline_subjects']}")
    print(f"ADNI2 subjects:          {c['adni2_subjects']}")
    print(f"Unique disk subjects:    {c['unique_disk_subjects']}")
    print(f"Overlap (both cohorts):  {c['overlap_baseline_adni2']}")
    print(f"Baseline .nii files:     {c['baseline_nii_files']}")
    print(f"ADNI2 .nii files:        {c['adni2_nii_files']}")
    print(f"Baseline CSV subjects:   {c['baseline_csv_subjects']}")
    print(f"MCI_CN CSV subjects:     {c['mci_cn_csv_subjects']}")
    print(f"Only in baseline CSV:    {c['subjects_only_baseline_csv']}")
    print(f"Unlabeled on disk:       {c['unlabeled_on_disk']}")
    if report["only_baseline_csv_subjects"]:
        print(f"  -> {report['only_baseline_csv_subjects'][:10]}"
              f"{'...' if len(report['only_baseline_csv_subjects']) > 10 else ''}")
    if report["unlabeled_on_disk"]:
        print(f"  -> {report['unlabeled_on_disk']}")
    print("=" * 60)


def execute_organization() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    legacy = [p for p in ADNI_ROOT.iterdir() if p.is_dir() and p.name not in ("baseline", "ADNI2")]
    for subject_dir in legacy:
        dest = BASELINE_DIR / subject_dir.name
        if dest.exists():
            print(f"[skip] baseline/{subject_dir.name} already exists")
            continue
        print(f"[move] {subject_dir.name} -> baseline/")
        shutil.move(str(subject_dir), str(dest))

    if ONEDRIVE_ADNI2.exists() and not ADNI2_DIR.exists():
        print(f"[copy] ADNI2 from OneDrive -> {ADNI2_DIR}")
        shutil.copytree(ONEDRIVE_ADNI2, ADNI2_DIR)
    elif ONEDRIVE_ADNI2.exists():
        print("[info] ADNI2 dir already present in repo")

    root_baseline = PROJECT_ROOT / "Data_baseline_2_23_2026.csv"
    if BASELINE_CSV.exists() and not root_baseline.exists():
        shutil.copy2(BASELINE_CSV, root_baseline)
        print("[copy] compatibility CSV -> repo root")

    if ONEDRIVE_MCI_CN_CSV.exists() and not MCI_CN_CSV.exists():
        shutil.copy2(ONEDRIVE_MCI_CN_CSV, MCI_CN_CSV)
        print("[copy] MCI_CN CSV -> metadata/")


def main() -> int:
    parser = argparse.ArgumentParser(description="ADNI data inventory and organization")
    parser.add_argument("--execute", action="store_true", help="Run copy/move steps (idempotent)")
    args = parser.parse_args()

    if args.execute:
        execute_organization()

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    report = build_inventory()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"\nReport saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
