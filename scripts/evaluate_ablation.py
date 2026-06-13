#!/usr/bin/env python3
"""Evaluate imbalance ablation runs and build comparison matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_all import evaluate_model, load_model  # noqa: E402
from src.data import MCIDataModule
from src.training.eval_utils import load_evaluation_config
from src.training.optimizations import get_optimized_device

EXP_ROOT = ROOT / "experiments" / "imbalance_ablation"
EXP_CONFIG = EXP_ROOT / "config.yaml"

MODEL_TYPE_MAP = {
    "baseline": "baseline",
    "selective_net": "selectivenet",
    "evidential": "evidential",
    "hybrid": "hybrid",
}


def load_exp_config() -> dict:
    with open(EXP_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _model_m_cfg(model: dict, arm: str) -> dict:
    return {
        "name": f"{model['name']} ({arm})",
        "config": str(ROOT / model["config"]),
        "type": MODEL_TYPE_MAP.get(model["name"], model["name"]),
    }


def collect_metrics(
    cfg: dict,
    eval_cfg: dict,
    dm,
    device,
    arms: list[str],
    model_names: list[str],
) -> pd.DataFrame:
    rows = []
    report_specs = eval_cfg.get("report_specificities", [0.70, 0.80, 0.90, 0.95, 1.0])
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    for arm in arms:
        arm_cfg = cfg["arms"][arm]
        strategy = arm_cfg["strategy"]
        for model in cfg["models"]:
            if model["name"] not in model_names:
                continue

            ckpt_dir = ROOT / cfg["checkpoint_root"] / arm / model["name"]
            ckpt_file = ckpt_dir / model["checkpoint_name"]
            if not ckpt_file.exists():
                print(f"[SKIP] No checkpoint: {arm}/{model['name']} ({ckpt_file})")
                continue

            model_cfg_path = ROOT / model["config"]
            with open(model_cfg_path, "r", encoding="utf-8") as f:
                model_yaml = yaml.safe_load(f)
            model_yaml["checkpoint"]["dir"] = str(ckpt_dir)

            m_cfg = _model_m_cfg(model, arm)
            loaded = load_model(m_cfg, model_yaml, device)
            if loaded[0] is None:
                print(f"[SKIP] Could not load: {arm}/{model['name']}")
                continue
            net, ckpt_path, _ = loaded
            print(f"[EVAL] {arm}/{model['name']} — {ckpt_path}")

            result = evaluate_model(
                net, m_cfg, val_loader, test_loader, device, eval_cfg, model_cfg=model_yaml
            )
            metrics = result["metrics"]
            fp_red = metrics.get("fp_reduction", {}).get("abstention_20pct", {})

            per_run = {
                "arm": arm,
                "strategy": strategy,
                "model": model["name"],
                "auc": float(metrics.get("auc", 0)),
                "augrc": float(metrics.get("augrc", 0)),
                "abstention_pct": float(metrics.get("abstention_rate", 0)),
                "fp_reduction_20pct": float(fp_red.get("fp_reduction_rate", 0)),
            }
            vt_specs = metrics.get("val_to_test_at_specs", {})
            for spec in report_specs:
                spec_key = f"spec_{int(round(spec * 100))}"
                vt = vt_specs.get(spec_key, {})
                pct = int(round(spec * 100))
                per_run[f"sens_vt_{pct}"] = float(vt.get("sensitivity", 0))
                per_run[f"fp_vt_{pct}"] = int(vt.get("fp", 0))
                per_run[f"actual_spec_vt_{pct}"] = float(vt.get("actual_specificity", 0))

            rows.append(per_run)

            out_dir = ROOT / cfg["results_root"] / "per_run"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{arm}_{model['name']}.json", "w", encoding="utf-8") as f:
                json.dump({"summary": per_run, "metrics": _json_safe(metrics)}, f, indent=2)

    return pd.DataFrame(rows)


def build_comparison_matrix(df: pd.DataFrame, report_specs: list[float], arms: list[str]) -> pd.DataFrame:
    """Pivot table: rows = (model, metric, spec_pct), columns = arm_*."""
    if df.empty:
        return df

    matrix_rows = []
    metric_defs = [
        ("Sens", "sens_vt_{pct}", "max"),
        ("FP", "fp_vt_{pct}", "min"),
        ("ActualSpec", "actual_spec_vt_{pct}", "max"),
    ]
    extra_metrics = [
        ("AUGRC", "augrc", "min"),
        ("FP_reduction@20%", "fp_reduction_20pct", "max"),
    ]

    for model_name in sorted(df["model"].unique()):
        sub = df[df["model"] == model_name]
        for spec in report_specs:
            pct = int(round(spec * 100))
            for metric_label, col_tpl, best_dir in metric_defs:
                col = col_tpl.format(pct=pct)
                arm_vals = {row["arm"]: row.get(col, 0) for _, row in sub.iterrows()}
                row = {"model": model_name, "metric": metric_label, "spec_pct": pct}
                for arm in arms:
                    row[arm] = arm_vals.get(arm, None)
                if arm_vals:
                    row["best_arm"] = (
                        min(arm_vals, key=arm_vals.get)
                        if best_dir == "min"
                        else max(arm_vals, key=arm_vals.get)
                    )
                matrix_rows.append(row)

        for metric_label, col, best_dir in extra_metrics:
            arm_vals = {row["arm"]: row.get(col, 0) for _, row in sub.iterrows()}
            row = {"model": model_name, "metric": metric_label, "spec_pct": None}
            for arm in arms:
                row[arm] = arm_vals.get(arm, None)
            if arm_vals:
                row["best_arm"] = (
                    min(arm_vals, key=arm_vals.get)
                    if best_dir == "min"
                    else max(arm_vals, key=arm_vals.get)
                )
            matrix_rows.append(row)

    # Mean across models per arm (FP/Sens @ each spec)
    for spec in report_specs:
        pct = int(round(spec * 100))
        for metric_label, col_tpl, best_dir in metric_defs[:2]:
            col = col_tpl.format(pct=pct)
            arm_means = {}
            for arm in arms:
                vals = df.loc[df["arm"] == arm, col]
                if len(vals):
                    arm_means[arm] = float(vals.mean())
            row = {"model": "MEAN", "metric": metric_label, "spec_pct": pct}
            for arm in arms:
                row[arm] = arm_means.get(arm)
            if arm_means:
                row["best_arm"] = (
                    min(arm_means, key=arm_means.get)
                    if best_dir == "min"
                    else max(arm_means, key=arm_means.get)
                )
            matrix_rows.append(row)

    return pd.DataFrame(matrix_rows)


def pick_winner(df: pd.DataFrame, arms: list[str]) -> dict:
    """Select recommended arm using plan criteria (Hybrid priority, FP@80%, Sens>=0.20)."""
    if df.empty:
        return {"winner_arm": None, "winner_strategy": None, "reason": "no results"}

    hybrid = df[df["model"] == "hybrid"].copy()
    target = hybrid if not hybrid.empty else df.copy()

    eligible = target[target["sens_vt_80"] >= 0.20] if "sens_vt_80" in target.columns else target
    if eligible.empty:
        eligible = target

    ranked = eligible.sort_values(["fp_vt_80", "sens_vt_80"], ascending=[True, False])
    winner_row = ranked.iloc[0]

    rankings = []
    for _, row in ranked.iterrows():
        rankings.append({
            "arm": row["arm"],
            "strategy": row["strategy"],
            "model": row["model"],
            "fp_at_80_spec": int(row.get("fp_vt_80", 0)),
            "sens_at_80_spec": float(row.get("sens_vt_80", 0)),
            "fp_at_90_spec": int(row.get("fp_vt_90", 0)),
            "sens_at_90_spec": float(row.get("sens_vt_90", 0)),
            "fp_at_95_spec": int(row.get("fp_vt_95", 0)),
            "sens_at_95_spec": float(row.get("sens_vt_95", 0)),
            "augrc": float(row.get("augrc", 0)),
        })

    return {
        "winner_arm": winner_row["arm"],
        "winner_strategy": winner_row["strategy"],
        "winner_model_basis": winner_row["model"],
        "fp_at_80_spec": int(winner_row.get("fp_vt_80", 0)),
        "sens_at_80_spec": float(winner_row.get("sens_vt_80", 0)),
        "fp_at_90_spec": int(winner_row.get("fp_vt_90", 0)),
        "sens_at_90_spec": float(winner_row.get("sens_vt_90", 0)),
        "fp_at_95_spec": int(winner_row.get("fp_vt_95", 0)),
        "sens_at_95_spec": float(winner_row.get("sens_vt_95", 0)),
        "augrc": float(winner_row.get("augrc", 0)),
        "rankings": rankings,
        "reason": (
            "Lowest FP@80%Spec (val→test) on Hybrid with Sens@80>=0.20; "
            "tie-break: higher Sens@80, then Sens@90/95"
        ),
    }


def write_comparison_md(matrix: pd.DataFrame, winner: dict, path: Path) -> None:
    lines = [
        "# Imbalance ablation — comparison matrix\n",
        "## Recommended winner\n",
        f"- **Arm:** `{winner.get('winner_arm')}`",
        f"- **Strategy:** `{winner.get('winner_strategy')}`",
        f"- **FP@80%Spec:** {winner.get('fp_at_80_spec')} | **Sens@80%Spec:** {winner.get('sens_at_80_spec', 0):.4f}",
        f"- **Reason:** {winner.get('reason')}\n",
        "## Sens / FP @ 70/80/90/95% Spec (val→test)\n",
    ]
    if not matrix.empty and hasattr(matrix, "to_markdown"):
        lines.append(matrix.to_markdown(index=False))
    else:
        lines.append(str(matrix))
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_fp_by_arm_and_spec(df: pd.DataFrame, report_specs: list[float], arms: list[str], path: Path) -> None:
    if df.empty:
        return
    models = sorted(df["model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)
    x_labels = [f"{int(round(s * 100))}%" for s in report_specs]

    for ax, model_name in zip(axes[0], models):
        sub = df[df["model"] == model_name]
        width = 0.25
        x = range(len(report_specs))
        for i, arm in enumerate(arms):
            arm_sub = sub[sub["arm"] == arm]
            if arm_sub.empty:
                continue
            fps = [int(arm_sub.iloc[0].get(f"fp_vt_{int(round(s * 100))}", 0)) for s in report_specs]
            offset = (i - len(arms) / 2 + 0.5) * width
            ax.bar([xi + offset for xi in x], fps, width=width, label=arm)
        ax.set_title(model_name)
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Target Spec")
        ax.set_ylabel("FP count (val→test)")
        ax.legend(fontsize=8)

    fig.suptitle("False Positives by arm and specificity level")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hybrid_sens_spec_tradeoff(hybrid_df: pd.DataFrame, report_specs: list[float], path: Path) -> None:
    if hybrid_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in hybrid_df.iterrows():
        specs = []
        sens = []
        for s in report_specs:
            pct = int(round(s * 100))
            specs.append(row.get(f"actual_spec_vt_{pct}", s))
            sens.append(row.get(f"sens_vt_{pct}", 0))
        ax.plot(specs, sens, marker="o", label=f"{row['arm']} ({row['strategy']})")
    ax.set_xlabel("Actual Specificity (val→test)")
    ax.set_ylabel("Sensitivity (val→test)")
    ax.set_title("Hybrid: Sens vs Spec trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate imbalance ablation")
    parser.add_argument("--arms", type=str, default=None, help="Comma-separated arms")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names")
    args = parser.parse_args()

    cfg = load_exp_config()
    eval_cfg = load_evaluation_config()
    report_specs = eval_cfg.get("report_specificities", [0.70, 0.80, 0.90, 0.95, 1.0])

    arms = list(cfg["arms"].keys())
    model_names = [m["name"] for m in cfg["models"]]
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",")]
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]

    with open(ROOT / "configs" / "data_config.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    device = get_optimized_device("cuda")
    dm = MCIDataModule(
        metadata_csv=data_cfg["paths"]["metadata_csv"],
        preprocessor_config=data_cfg["preprocessing"],
        batch_size=data_cfg["dataloader"]["batch_size"],
        num_workers=data_cfg["dataloader"]["num_workers"],
    )

    df = collect_metrics(cfg, eval_cfg, dm, device, arms, model_names)
    results_dir = ROOT / cfg["results_root"]
    results_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(results_dir / "all_runs_summary.csv", index=False)
    matrix = build_comparison_matrix(df, report_specs, arms)
    matrix.to_csv(results_dir / "comparison_matrix.csv", index=False)

    hybrid_focus = df[df["model"] == "hybrid"] if not df.empty else df
    hybrid_focus.to_csv(results_dir / "hybrid_focus.csv", index=False)

    winner = pick_winner(df, arms)
    (results_dir / "winner.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")

    write_comparison_md(matrix, winner, results_dir / "comparison_matrix.md")
    plot_fp_by_arm_and_spec(df, report_specs, arms, results_dir / "fp_by_arm_and_spec.png")
    plot_hybrid_sens_spec_tradeoff(hybrid_focus, report_specs, results_dir / "sens_spec_tradeoff_hybrid.png")

    print(f"\nSaved results to {results_dir}")
    print(f"  comparison_matrix.csv ({len(matrix)} rows)")
    print(f"  winner.json -> {winner.get('winner_arm')} / {winner.get('winner_strategy')}")
    if df.empty:
        print("  [WARN] No checkpoints evaluated - run training first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
