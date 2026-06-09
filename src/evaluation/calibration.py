"""Post-hoc probability calibration (fit on val, apply on test)."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


class TemperatureScaler:
    """Temperature scaling on logits (Guo et al.)."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / temperature.clamp(min=0.01)
            loss = F.cross_entropy(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(temperature.detach().clamp(min=0.01).item())
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        scaled = torch.tensor(logits, dtype=torch.float32) / self.temperature
        return F.softmax(scaled, dim=1).detach().numpy()

    def transform_probs(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-10
        logits = np.log(np.clip(probs, eps, 1.0))
        return self.transform_logits(logits)


class PlattScaler:
    """Platt scaling (sigmoid) on P(positive class)."""

    def __init__(self):
        self.model: Optional[LogisticRegression] = None

    def fit(self, probs_pos: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        x = probs_pos.reshape(-1, 1)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(x, labels)
        return self

    def transform_probs_pos(self, probs_pos: np.ndarray) -> np.ndarray:
        if self.model is None:
            return probs_pos
        x = probs_pos.reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


def create_calibrator(method: str):
    method = (method or "none").lower()
    if method == "temperature":
        return TemperatureScaler()
    if method == "platt":
        return PlattScaler()
    return None


def calibrate_probabilities(
    calibrator,
    val_logits: Optional[np.ndarray],
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_logits: Optional[np.ndarray],
    test_probs: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit on val, return (val_probs_cal, test_probs_cal) as full (N,2) arrays."""
    method = (method or "none").lower()
    if calibrator is None or method == "none":
        return val_probs, test_probs

    if method == "temperature":
        if val_logits is None or test_logits is None:
            eps = 1e-10
            val_logits = np.log(np.clip(val_probs, eps, 1.0))
            test_logits = np.log(np.clip(test_probs, eps, 1.0))
        calibrator.fit(val_logits, val_labels)
        return calibrator.transform_logits(val_logits), calibrator.transform_logits(test_logits)

    if method == "platt":
        calibrator.fit(val_probs[:, 1], val_labels)
        val_pos = calibrator.transform_probs_pos(val_probs[:, 1])
        test_pos = calibrator.transform_probs_pos(test_probs[:, 1])
        val_cal = np.stack([1 - val_pos, val_pos], axis=1)
        test_cal = np.stack([1 - test_pos, test_pos], axis=1)
        return val_cal, test_cal

    return val_probs, test_probs
