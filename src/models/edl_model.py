"""Evidential Deep Learning model (backbone + Dirichlet head)."""

import torch
import torch.nn as nn
from typing import Tuple

from .evidential_layer import EvidentialLayer, compute_uncertainty


class EDLModel(nn.Module):
    """Lightweight EDL container without dropout (distinct from Hybrid)."""

    def __init__(self, backbone: nn.Module, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.evidential_head = EvidentialLayer(backbone.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.evidential_head(features)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor):
        self.eval()
        alpha = self.forward(x)
        strength = alpha.sum(dim=1, keepdim=True)
        probabilities = alpha / strength
        predictions = torch.argmax(probabilities, dim=1)
        epistemic_unc, aleatoric_unc, total_unc = compute_uncertainty(alpha)
        unc_dict = {
            "epistemic": epistemic_unc.cpu(),
            "aleatoric": aleatoric_unc.cpu(),
            "total": total_unc.cpu(),
            "strength": strength.squeeze(1).cpu(),
        }
        return predictions, probabilities, epistemic_unc, unc_dict

    @torch.no_grad()
    def predict_with_rejection(
        self,
        x: torch.Tensor,
        uncertainty_threshold: float = 0.5,
        uncertainty_type: str = "epistemic",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions, probabilities, _, unc_dict = self.predict_with_uncertainty(x)
        if uncertainty_type == "aleatoric":
            uncertainties = unc_dict["aleatoric"]
        elif uncertainty_type == "total":
            uncertainties = unc_dict["total"]
        else:
            uncertainties = unc_dict["epistemic"]
        uncertainties = uncertainties.to(x.device)
        is_abstained = uncertainties > uncertainty_threshold
        predictions_out = predictions.clone()
        predictions_out[is_abstained] = -1
        return predictions_out, probabilities, uncertainties, is_abstained
