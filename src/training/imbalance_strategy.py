"""Imbalance handling strategies for MCI vs CN training."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

STRATEGIES: Dict[str, Dict[str, bool]] = {
    "balanced_sampler": {"balance_classes": True, "use_class_weights": False},
    "natural": {"balance_classes": False, "use_class_weights": False},
    "cost_sensitive": {"balance_classes": False, "use_class_weights": True},
}

ARM_ALIASES: Dict[str, str] = {
    "arm_c": "balanced_sampler",
    "arm_a": "natural",
    "arm_b": "cost_sensitive",
    "arm_c_balanced_sampler": "balanced_sampler",
    "arm_a_natural": "natural",
    "arm_b_cost_sensitive": "cost_sensitive",
}

DEFAULT_STRATEGY = "balanced_sampler"


def normalize_strategy(name: Optional[str]) -> str:
    """Map arm alias or strategy name to a canonical strategy key."""
    if not name:
        return DEFAULT_STRATEGY
    key = name.strip().lower()
    if key in ARM_ALIASES:
        return ARM_ALIASES[key]
    if key in STRATEGIES:
        return key
    raise ValueError(
        f"Unknown imbalance strategy '{name}'. "
        f"Valid: {list(STRATEGIES.keys())} or arms {list(ARM_ALIASES.keys())}"
    )


def resolve_imbalance_settings(
    strategy: Optional[str] = None,
    data_cfg: Optional[Dict[str, Any]] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve training imbalance settings from strategy name and configs.

    Priority: explicit strategy > data_cfg.imbalance_strategy > legacy balance_classes.
    """
    if strategy is None and data_cfg:
        strategy = data_cfg.get("imbalance_strategy")
        if strategy is None:
            legacy_balance = data_cfg.get("dataloader", {}).get("balance_classes", False)
            strategy = "balanced_sampler" if legacy_balance else "natural"

    canonical = normalize_strategy(strategy)
    settings = dict(STRATEGIES[canonical])
    settings["imbalance_strategy"] = canonical

    if model_cfg and model_cfg.get("training", {}).get("use_class_weights") is True:
        if canonical != "cost_sensitive":
            settings["use_class_weights"] = True

    return settings


def add_imbalance_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--imbalance-strategy",
        type=str,
        default=None,
        help="Override imbalance strategy: balanced_sampler | natural | cost_sensitive",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint output directory",
    )


def apply_train_overrides(
    args: argparse.Namespace,
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Return resolved imbalance settings and apply checkpoint-dir override."""
    settings = resolve_imbalance_settings(
        strategy=getattr(args, "imbalance_strategy", None),
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    )
    if getattr(args, "checkpoint_dir", None):
        model_cfg.setdefault("checkpoint", {})["dir"] = args.checkpoint_dir
    return settings
