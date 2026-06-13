"""Shared evaluation config loading and validation metric helpers."""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.evaluation.metrics import MetricsTracker


def load_evaluation_config(path: str = "configs/evaluation_config.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    defaults = {
        "target_specificity": 0.80,
        "report_specificities": [0.70, 0.80, 0.90, 0.95, 1.0],
        "threshold_protocol": "val_to_test",
        "positive_class": 1,
        "abstention_levels": [0.1, 0.2, 0.3, 0.5],
        "calibration": {"enabled": False, "method": "none"},
        "abstention": {
            "enabled": False,
            "protocol": "val_to_test",
            "target_coverage": 0.80,
            "uncertainty_type": "epistemic",
        },
        "threshold_strategies": ["fixed_specificity"],
        "bootstrap": {"enabled": False, "n_iterations": 1000, "metrics": ["sensitivity", "specificity", "auc"]},
    }
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)
        if "calibration" in loaded:
            defaults["calibration"] = {**defaults.get("calibration", {}), **loaded["calibration"]}
        if "abstention" in loaded:
            defaults["abstention"] = {**defaults.get("abstention", {}), **loaded["abstention"]}
        if "bootstrap" in loaded:
            defaults["bootstrap"] = {**defaults.get("bootstrap", {}), **loaded["bootstrap"]}
    return defaults


def create_metrics_tracker(
    eval_cfg: Dict[str, Any], num_classes: int = 2
) -> MetricsTracker:
    report_specs: List[float] = eval_cfg.get(
        "report_specificities", [0.80, 0.90, 1.0]
    )
    return MetricsTracker(
        num_classes=num_classes,
        target_specificity=eval_cfg.get("target_specificity", 0.80),
        positive_class=eval_cfg.get("positive_class", 1),
        abstention_levels=eval_cfg.get("abstention_levels", [0.1, 0.2, 0.3]),
        report_specificities=report_specs,
    )


def get_monitor_value(metrics: Dict[str, Any], monitor_key: str) -> float:
    """Resolve checkpoint monitor value from flat or nested metric dict."""
    key = monitor_key.replace("val_", "")

    if key in ("composite", "val_composite"):
        auc = float(metrics.get("auc", 0.0))
        sens = float(metrics.get("metrics_at_target_spec", {}).get("sensitivity", 0.0))
        return 0.5 * auc + 0.5 * sens

    if key in ("augrc", "val_augrc"):
        return float(metrics.get("augrc", 0.0))

    if key in ("auc", "val_auc"):
        return float(metrics.get("auc", 0.0))

    if key in metrics:
        return float(metrics[key])

    nested_map = {
        "sensitivity_at_target_spec": ("metrics_at_target_spec", "sensitivity"),
        "specificity_at_target_spec": ("metrics_at_target_spec", "actual_specificity"),
    }
    if key in nested_map:
        parent, child = nested_map[key]
        return float(metrics.get(parent, {}).get(child, 0.0))

    return float(metrics.get(monitor_key, 0.0))


def format_validation_log(val_metrics: Dict[str, Any], target_spec: float) -> str:
    """Standard validation log line aligned with thesis metrics."""
    ms = val_metrics.get("metrics_at_target_spec", {})
    sens = ms.get("sensitivity", 0.0)
    actual_spec = ms.get("actual_specificity", 0.0)
    bal_acc = val_metrics.get("balanced_accuracy", 0.0)
    augrc = val_metrics.get("augrc", 0.0)
    cov = 1.0 - val_metrics.get("abstention_rate", 0.0)
    spec_argmax = val_metrics.get("specificity", 0.0)

    return (
        f"Sens@{target_spec * 100:.0f}%Spec={sens:.4f} (actual Spec={actual_spec:.4f}) | "
        f"BalancedAcc={bal_acc:.4f} | AUGRC={augrc:.4f} | Cov={cov:.4f} | "
        f"Spec@argmax={spec_argmax:.4f}"
    )
