"""Abstention threshold fitting, alternative classification thresholds, bootstrap CI."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, roc_curve, roc_auc_score


def fit_coverage_threshold(
    scores: np.ndarray,
    target_coverage: float,
    higher_is_better: bool = True,
) -> float:
    """
    Fit threshold so that (1 - target_coverage) fraction is abstained.

    higher_is_better=True: abstain when score < threshold (confidence).
    higher_is_better=False: abstain when score > threshold (uncertainty).
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return float("nan")

    n_abstain = max(0, int(np.floor(n * (1.0 - target_coverage))))
    if n_abstain <= 0:
        return float("-inf") if higher_is_better else float("inf")
    if n_abstain >= n:
        return float("inf") if higher_is_better else float("-inf")

    sorted_scores = np.sort(scores)
    if higher_is_better:
        return float(sorted_scores[n_abstain])
    return float(sorted_scores[-(n_abstain + 1)])


def apply_abstention_mask(
    preds: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    higher_is_better: bool = True,
) -> np.ndarray:
    """Set predictions to -1 where abstention rule triggers."""
    preds = np.asarray(preds, dtype=int).copy()
    scores = np.asarray(scores, dtype=float)

    if higher_is_better:
        if np.isfinite(threshold):
            preds[scores < threshold] = -1
    else:
        if np.isfinite(threshold):
            preds[scores > threshold] = -1
    return preds


def fit_threshold_youden(labels: np.ndarray, probs_pos: np.ndarray) -> float:
    """Probability threshold maximizing Youden's J (TPR - FPR)."""
    labels = np.asarray(labels)
    probs_pos = np.asarray(probs_pos, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, probs_pos)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5


def fit_threshold_max_f1(labels: np.ndarray, probs_pos: np.ndarray) -> float:
    """Probability threshold maximizing F1 on given split."""
    labels = np.asarray(labels)
    probs_pos = np.asarray(probs_pos, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5
    candidates = np.unique(np.round(probs_pos, 4))
    if len(candidates) < 2:
        candidates = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        preds = (probs_pos >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def fit_selection_threshold_coverage(
    selection_probs: np.ndarray,
    target_coverage: float,
) -> float:
    """SelectiveNet: abstain when g(x) < threshold; fit for target coverage."""
    return fit_coverage_threshold(selection_probs, target_coverage, higher_is_better=True)


def get_abstention_scores(
    raw: Dict,
    model_type: str,
    uncertainty_type: str = "epistemic",
) -> Tuple[np.ndarray, bool]:
    """
    Return (scores, higher_is_better) for abstention fitting.

    Baseline / SelectiveNet: confidence or selection prob (higher = keep).
    EDL / Hybrid: epistemic uncertainty (higher = abstain).
    """
    if model_type == "selectivenet":
        scores = raw.get("selection_probs", raw["confidences"])
        return scores, True
    if model_type in ("evidential", "hybrid"):
        if uncertainty_type == "aleatoric" and "aleatoric" in raw:
            return raw["aleatoric"], False
        if uncertainty_type == "total" and "epistemic" in raw and "aleatoric" in raw:
            return raw["epistemic"] + raw["aleatoric"], False
        return raw.get("epistemic", 1.0 - raw["confidences"]), False
    return raw["confidences"], True


def bootstrap_metric_ci(
    labels: np.ndarray,
    probs_pos: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    labels = np.asarray(labels)
    probs_pos = np.asarray(probs_pos, dtype=float)
    n = len(labels)
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            stats.append(metric_fn(labels[idx], probs_pos[idx]))
        except Exception:
            continue
    if not stats:
        point = metric_fn(labels, probs_pos)
        return point, point, point
    arr = np.array(stats)
    point = float(metric_fn(labels, probs_pos))
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def bootstrap_sensitivity_at_threshold(
    labels: np.ndarray,
    probs_pos: np.ndarray,
    threshold: float,
    positive_class: int = 1,
) -> float:
    preds = (probs_pos >= threshold).astype(int)
    pos = positive_class
    tp = np.sum((preds == pos) & (labels == pos))
    fn = np.sum((preds != pos) & (labels == pos))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def bootstrap_specificity_at_threshold(
    labels: np.ndarray,
    probs_pos: np.ndarray,
    threshold: float,
    positive_class: int = 1,
) -> float:
    preds = (probs_pos >= threshold).astype(int)
    neg = 1 - positive_class
    tn = np.sum((preds == neg) & (labels == neg))
    fp = np.sum((preds == positive_class) & (labels == neg))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def bootstrap_auc(labels: np.ndarray, probs_pos: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, probs_pos))
