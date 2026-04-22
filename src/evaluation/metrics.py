"""
Evaluation metrics for MCI classification with rejection/abstention.

Key metrics according to project requirements:
1. Risk-Coverage Curve
2. AUGRC (Area Under Generalized Risk-Coverage curve)
3. Sensitivity @ Fixed Specificity  
4. Standard metrics: Accuracy, Precision, Recall, F1, AUC-ROC
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from typing import Tuple, Dict, List, Optional
import torch


def compute_risk_coverage(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    num_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Risk-Coverage curve.
    
    Risk = Error rate on non-abstained samples
    Coverage = Fraction of non-abstained samples
    
    Args:
        predictions: Predicted classes (can include -1 for abstention)
        labels: Ground truth labels
        confidences: Confidence scores (for threshold sweep)
        num_thresholds: Number of thresholds to evaluate
        
    Returns:
        coverages: Coverage values (1.0 to ~0.0)
        risks: Risk (error rate) at each coverage
        thresholds: Threshold values used
    """
    n = len(predictions)
    
    if confidences is None:
        # If no conf scores, use single point (current predictions)
        non_abstained = predictions != -1
        coverage = non_abstained.sum() / n if n > 0 else 0
        
        if non_abstained.sum() > 0:
            risk = 1 - accuracy_score(
                labels[non_abstained],
                predictions[non_abstained]
            )
        else:
            risk = 0.0  # No predictions = no risk
        
        return np.array([coverage]), np.array([risk]), np.array([0.0])
    
    # Threshold sweep on confidences
    thresholds = np.linspace(0, 1, num_thresholds)
    coverages = []
    risks = []
    
    for thresh in thresholds:
        # Accept if confidence >= threshold
        accepted = confidences >= thresh
        coverage = accepted.sum() / n if n > 0 else 0
        
        if accepted.sum() > 0:
            risk = 1 - accuracy_score(
                labels[accepted],
                predictions[accepted]
            )
        else:
            risk = 0.0
        
        coverages.append(coverage)
        risks.append(risk)
    
    return np.array(coverages), np.array(risks), thresholds


def compute_augrc(
    coverages: np.ndarray,
    risks: np.ndarray
) -> float:
    """
    Compute Area Under Generalized Risk-Coverage curve.
    
    AUGRC = ∫ Risk(c) dc from c=0 to c=1
    
    Lower AUGRC = Better (less risk at all coverage levels)
    
    Args:
        coverages: Coverage values (should be sorted descending)
        risks: Risk values
        
    Returns:
        augrc: Area under the curve
    """
    # Sort by coverage (descending)
    sorted_indices = np.argsort(coverages)[::-1]
    coverages_sorted = coverages[sorted_indices]
    risks_sorted = risks[sorted_indices]
    
    # Integrate using trapezoidal rule
    # np.trapz may return negative if coverages are descending, take abs
    augrc = abs(np.trapz(risks_sorted, coverages_sorted))
    
    return augrc


def compute_sensitivity_at_specificity(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_specificity: float = 0.95,
    positive_class: int = 1
) -> Tuple[float, float]:
    """
    Compute sensitivity (TPR) at a fixed specificity (TNR).
    
    Important for medical diagnosis: minimize false positives (FP)
    by ensuring high specificity, while maximizing sensitivity.
    
    Args:
        labels: Ground truth labels
        probabilities: Predicted probabilities for positive class
        target_specificity: Desired specificity (TPR when negative)
        positive_class: Which class is positive
        
    Returns:
        sensitivity: Sensitivity at target specificity
        threshold: Threshold that achieves this operating point
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probabilities, pos_label=positive_class)
    
    # Specificity = 1 - FPR → TNR
    specificity = 1 - fpr
    
    # Find threshold closest to target specificity
    idx = np.argmin(np.abs(specificity - target_specificity))
    
    sensitivity = tpr[idx]
    threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    actual_specificity = specificity[idx]
    
    return sensitivity, threshold, actual_specificity


def compute_sensitivity_at_multiple_specificities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_specificities: List[float] = [0.80, 0.90, 0.95],
    positive_class: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    Compute sensitivity at multiple specificity thresholds.
    
    Critical for medical diagnosis: shows trade-off between detecting
    MCI patients (sensitivity) and avoiding false alarms (specificity).
    
    Args:
        labels: Ground truth labels
        probabilities: Predicted probabilities for positive class
        target_specificities: List of desired specificity levels
        positive_class: Which class is positive (MCI=1)
        
    Returns:
        Dict mapping specificity level to {sensitivity, threshold, actual_specificity}
    """
    results = {}
    for spec in target_specificities:
        sens, thresh, actual_spec = compute_sensitivity_at_specificity(
            labels, probabilities, target_specificity=spec,
            positive_class=positive_class
        )
        key = f"sens_at_{int(spec*100)}spec"
        results[key] = {
            'sensitivity': sens,
            'threshold': thresh,
            'actual_specificity': actual_spec
        }
    return results


def compute_fp_reduction_at_abstention(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    abstention_levels: List[float] = [0.10, 0.20, 0.30]
) -> Dict[str, Dict[str, float]]:
    """
    Compute False Positive reduction rate when rejecting uncertain samples.
    
    Key question: "If we let the model abstain on the 20% most uncertain
    samples, how many False Positives do we eliminate?"
    
    This directly measures the clinical value of uncertainty estimation:
    fewer FP = fewer healthy patients subjected to unnecessary procedures.
    
    Args:
        predictions: Predicted classes
        labels: Ground truth labels
        confidences: Confidence scores (higher = more confident)
        abstention_levels: Fraction of samples to reject (sorted by uncertainty)
        
    Returns:
        Dict mapping abstention level to {fp_before, fp_after, fp_reduction_rate,
        accuracy_after, coverage}
    """
    n = len(predictions)
    
    # Baseline FP count (no abstention)
    fp_baseline = np.sum((predictions == 1) & (labels == 0))
    
    results = {}
    for level in abstention_levels:
        # Sort by confidence (ascending = least confident first)
        sorted_idx = np.argsort(confidences)
        n_reject = int(n * level)
        
        # Keep only the most confident samples
        keep_idx = sorted_idx[n_reject:]
        
        if len(keep_idx) == 0:
            continue
        
        preds_kept = predictions[keep_idx]
        labels_kept = labels[keep_idx]
        
        fp_after = np.sum((preds_kept == 1) & (labels_kept == 0))
        accuracy_after = np.mean(preds_kept == labels_kept)
        
        fp_reduction = (fp_baseline - fp_after) / max(fp_baseline, 1)
        
        key = f"abstention_{int(level*100)}pct"
        results[key] = {
            'fp_before': int(fp_baseline),
            'fp_after': int(fp_after),
            'fp_reduction_rate': float(fp_reduction),
            'accuracy_after': float(accuracy_after),
            'coverage': float(1.0 - level)
        }
    
    return results


def compute_standard_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        predictions: Predicted classes (excluding -1 abstentions)
        labels: Ground truth labels
        probabilities: Predicted probabilities (for AUC-ROC)
        
    Returns:
        metrics: Dict with accuracy, precision, recall, f1, auc
    """
    metrics = {}
    
    # Filter out abstentions
    if -1 in predictions:
        mask = predictions != -1
        predictions = predictions[mask]
        labels = labels[mask]
        if probabilities is not None:
            probabilities = probabilities[mask]
    
    if len(predictions) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0
        }
    
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, average='binary', zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, average='binary', zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, average='binary', zero_division=0)
    
    # AUC-ROC (if probabilities provided)
    if probabilities is not None:
        if len(np.unique(labels)) > 1:  # Need both classes for AUC
            # For binary, use prob of positive class
            if probabilities.ndim == 2:
                probs_positive = probabilities[:, 1]
            else:
                probs_positive = probabilities
            
            metrics['auc'] = roc_auc_score(labels, probs_positive)
        else:
            metrics['auc'] = 0.5  # Default for single class
    
    return metrics


def compute_confusion_matrix_with_abstention(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2
) -> Tuple[np.ndarray, int]:
    """
    Compute confusion matrix, tracking abstentions separately.
    
    Args:
        predictions: Predicted classes (can include -1 for abstention)
        labels: Ground truth labels
        num_classes: Number of actual classes (excluding abstention)
        
    Returns:
        conf_matrix: Confusion matrix (num_classes x num_classes)
        num_abstained: Number of abstained samples
    """
    # Count abstentions
    abstained_mask = predictions == -1
    num_abstained = abstained_mask.sum()
    
    # Compute confusion matrix on non-abstained
    if abstained_mask.sum() < len(predictions):
        conf_matrix = confusion_matrix(
            labels[~abstained_mask],
            predictions[~abstained_mask],
            labels=list(range(num_classes))
        )
    else:
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    return conf_matrix, num_abstained


class MetricsTracker:
    """Helper class to track metrics during evaluation."""
    
    def __init__(self, num_classes: int = 2):
        """
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all tracked values."""
        self.all_predictions = []
        self.all_labels = []
        self.all_confidences = []
        self.all_probabilities = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Update with batch results.
        
        Args:
            predictions: Batch predictions
            labels: Batch labels
            confidences: Batch confidence scores
            probabilities: Batch probability distributions
        """
        self.all_predictions.append(predictions.cpu().numpy())
        self.all_labels.append(labels.cpu().numpy())
        
        if confidences is not None:
            self.all_confidences.append(confidences.cpu().numpy())
        
        if probabilities is not None:
            self.all_probabilities.append(probabilities.cpu().numpy())
    
    def compute_all_metrics(self) -> Dict:
        """
        Compute all metrics from tracked values.
        
        Returns:
            Dictionary with all computed metrics
        """
        predictions = np.concatenate(self.all_predictions)
        labels = np.concatenate(self.all_labels)
        
        confidences = (
            np.concatenate(self.all_confidences)
            if self.all_confidences else None
        )
        probabilities = (
            np.concatenate(self.all_probabilities)
            if self.all_probabilities else None
        )
        
        metrics = {}
        
        # Standard metrics (on non-abstained)
        metrics.update(compute_standard_metrics(predictions, labels, probabilities))
        
        # Confusion matrix
        conf_matrix, num_abstained = compute_confusion_matrix_with_abstention(
            predictions, labels, self.num_classes
        )
        metrics['confusion_matrix'] = conf_matrix
        metrics['num_abstained'] = num_abstained
        metrics['abstention_rate'] = num_abstained / len(predictions)
        
        # Risk-Coverage
        if confidences is not None:
            # Filter for non-abstained for confidence-based sweep
            mask = predictions != -1
            coverages, risks, thresholds = compute_risk_coverage(
                predictions[mask],
                labels[mask],
                confidences[mask] if mask.sum() > 0 else None
            )
            metrics['risk_coverage'] = {
                'coverages': coverages,
                'risks': risks,
                'thresholds': thresholds
            }
            metrics['augrc'] = compute_augrc(coverages, risks)
        
        # Sensitivity @ multiple Specificity levels
        if probabilities is not None and probabilities.ndim == 2:
            probs_pos = probabilities[:, 1]
            
            # Multi-threshold sensitivity
            multi_sens = compute_sensitivity_at_multiple_specificities(
                labels, probs_pos,
                target_specificities=[0.80, 0.90, 0.95]
            )
            for key, vals in multi_sens.items():
                metrics[key] = vals['sensitivity']
            
            # Legacy key for backward compatibility
            metrics['sensitivity_at_95spec'] = multi_sens.get(
                'sens_at_95spec', {}).get('sensitivity', 0.0)
            metrics['threshold_at_95spec'] = multi_sens.get(
                'sens_at_95spec', {}).get('threshold', 0.0)
            metrics['actual_specificity'] = multi_sens.get(
                'sens_at_95spec', {}).get('actual_specificity', 0.0)
        
        # FP Reduction at various abstention levels
        if confidences is not None:
            fp_reduction = compute_fp_reduction_at_abstention(
                predictions, labels, confidences,
                abstention_levels=[0.10, 0.20, 0.30]
            )
            metrics['fp_reduction'] = fp_reduction
        
        return metrics


if __name__ == '__main__':
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Dummy data
    np.random.seed(42)
    labels = np.random.randint(0, 2, 100)
    predictions = labels.copy()
    # Add some errors
    error_idx = np.random.choice(100, 20, replace=False)
    predictions[error_idx] = 1 - predictions[error_idx]
    # Add some abstentions
    abstain_idx = np.random.choice(100, 10, replace=False)
    predictions[abstain_idx] = -1
    
    confidences = np.random.rand(100)
    probabilities = np.random.rand(100, 2)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Test standard metrics
    metrics = compute_standard_metrics(predictions, labels, probabilities)
    print("Standard metrics:", metrics)
    
    # Test confusion matrix
    conf_mat, n_abstained = compute_confusion_matrix_with_abstention(predictions, labels)
    print(f"\nConfusion matrix:\n{conf_mat}")
    print(f"Abstained: {n_abstained}")
    
    # Test risk-coverage
    coverages, risks, thresholds = compute_risk_coverage(
        predictions[predictions != -1],
        labels[predictions != -1],
        confidences[predictions != -1]
    )
    print(f"\nRisk-Coverage points: {len(coverages)}")
    augrc = compute_augrc(coverages, risks)
    print(f"AUGRC: {augrc:.4f}")
    
    # Test sensitivity @ specificity
    sens, thresh, spec = compute_sensitivity_at_specificity(labels, probabilities[:, 1])
    print(f"\nSensitivity @ 95% Spec: {sens:.4f} (actual spec: {spec:.4f})")
    
    print("\n✓ Metrics test passed!")
