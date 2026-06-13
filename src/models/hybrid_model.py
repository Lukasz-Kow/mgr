"""
Hybrid Model: ResNet (2D/3D) backbone + Evidential head.

This combines:
- Deep feature extraction from ResNet
- Uncertainty quantification from Evidential Learning
- Rejection mechanism based on epistemic uncertainty
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

from .evidential_layer import EvidentialLayer, compute_uncertainty


class HybridEvidentialModel(nn.Module):
    """
    Hybrid model combining backbone + evidential head.
    
    Current: 2D ResNet + EDL
    Future: 3D ResNet + EDL (same architecture, different backbone)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            backbone: Feature extractor (ResNet2D or ResNet3D)
            num_classes: Number of classes
            dropout: Dropout before evidential layer
        """
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Evidential head
        self.dropout = nn.Dropout(dropout)
        self.evidential_head = EvidentialLayer(
            in_features=backbone.feature_dim,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
            
        Returns:
            alpha: Dirichlet parameters (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Evidential output
        features = self.dropout(features)
        alpha = self.evidential_head(features)
        
        return alpha
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Prediction with uncertainty quantification.
        
        Args:
            x: Input tensor
            
        Returns:
            predictions: Predicted class indices (B,)
            probabilities: Expected probabilities (B, num_classes)
            uncertainties: Epistemic uncertainties (B,)
            uncertainty_dict: Dict with all uncertainty components
        """
        self.eval()
        
        # Forward
        alpha = self.forward(x)
        
        # Expected probability: p_i = α_i / S
        strength = alpha.sum(dim=1, keepdim=True)
        probabilities = alpha / strength
        
        # Predictions
        predictions = torch.argmax(probabilities, dim=1)
        
        # Uncertainties
        epistemic_unc, aleatoric_unc, total_unc = compute_uncertainty(alpha)
        
        uncertainty_dict = {
            'epistemic': epistemic_unc.cpu(),
            'aleatoric': aleatoric_unc.cpu(),
            'total': total_unc.cpu(),
            'strength': strength.squeeze(1).cpu()
        }
        
        return predictions, probabilities, epistemic_unc, uncertainty_dict
    
    @torch.no_grad()
    def predict_with_rejection(
        self,
        x: torch.Tensor,
        uncertainty_threshold: float = 0.5,
        uncertainty_type: str = 'epistemic'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction with rejection based on uncertainty.
        
        Rejection rule: If uncertainty > threshold → abstain (return -1)
        
        Args:
            x: Input tensor
            uncertainty_threshold: Threshold for abstention
            uncertainty_type: 'epistemic', 'aleatoric', or 'total'
            
        Returns:
            predictions: Predicted classes or -1 for abstention (B,)
            probabilities: Expected probabilities (B, num_classes)
            uncertainties: Selected uncertainty values (B,)
            is_abstained: Boolean mask for abstained samples (B,)
        """
        predictions, probabilities, _, unc_dict = self.predict_with_uncertainty(x)
        
        # Select uncertainty type
        if uncertainty_type == 'epistemic':
            uncertainties = unc_dict['epistemic']
        elif uncertainty_type == 'aleatoric':
            uncertainties = unc_dict['aleatoric']
        elif uncertainty_type == 'total':
            uncertainties = unc_dict['total']
        else:
            raise ValueError(f"Unknown uncertainty_type: {uncertainty_type}")
        
        uncertainties = uncertainties.to(x.device)
        
        # Abstain if uncertainty > threshold
        is_abstained = uncertainties > uncertainty_threshold
        predictions_with_rejection = predictions.clone()
        predictions_with_rejection[is_abstained] = -1
        
        return predictions_with_rejection, probabilities, uncertainties, is_abstained

    @staticmethod
    def composite_abstention_score(
        epistemic_unc: torch.Tensor,
        strength: torch.Tensor,
        strength_min: float = 2.0,
    ) -> torch.Tensor:
        """Higher score = more confident; combines uncertainty and evidence strength."""
        strength_factor = torch.clamp(strength / strength_min, max=1.0)
        return (1.0 - epistemic_unc) * strength_factor

    @torch.no_grad()
    def predict_with_dual_gate(
        self,
        x: torch.Tensor,
        prob_threshold: float,
        uncertainty_threshold: float = 0.5,
        strength_min: float = 2.0,
        uncertainty_type: str = 'epistemic',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dual-gate prediction: Spec threshold (Gate 1) + uncertainty abstention (Gate 2).

        Gate 1: prob_MCI < prob_threshold → CN (controls FP)
        Gate 2: high uncertainty or low evidence strength → abstain
        """
        self.eval()
        alpha = self.forward(x)
        strength = alpha.sum(dim=1)
        probabilities = alpha / strength.unsqueeze(1)
        prob_mci = probabilities[:, 1]

        if uncertainty_type == 'aleatoric':
            _, aleatoric_unc, _ = compute_uncertainty(alpha)
            uncertainties = aleatoric_unc
        elif uncertainty_type == 'total':
            _, _, total_unc = compute_uncertainty(alpha)
            uncertainties = total_unc
        else:
            epistemic_unc, _, _ = compute_uncertainty(alpha)
            uncertainties = epistemic_unc

        predictions = torch.zeros(len(prob_mci), dtype=torch.long, device=x.device)
        is_abstained = torch.zeros(len(prob_mci), dtype=torch.bool, device=x.device)

        mci_candidate = prob_mci >= prob_threshold
        low_strength = strength < strength_min
        high_unc = uncertainties > uncertainty_threshold

        predictions[mci_candidate] = 1
        abstain_mask = mci_candidate & (high_unc | low_strength)
        is_abstained[abstain_mask] = True
        predictions[is_abstained] = -1

        return predictions, probabilities, uncertainties, is_abstained


if __name__ == '__main__':
    # Test hybrid model
    print("Testing Hybrid Evidential Model...")
    
    from backbone import get_backbone

    backbone = get_backbone({
        'type': 'monai', 'use_3d': True, 'arch_3d': 'resnet10', 'pretrained': False,
    })
    model = HybridEvidentialModel(backbone, num_classes=2)

    x = torch.randn(2, 1, 64, 64, 64)
    
    # Forward pass
    alpha = model(x)
    print(f"Alpha shape: {alpha.shape}")
    print(f"Alpha (first sample): {alpha[0]}")
    
    # Prediction with uncertainty
    preds, probs, unc, unc_dict = model.predict_with_uncertainty(x)
    print(f"\nPredictions: {preds}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Epistemic uncertainty: {unc}")
    print(f"Uncertainty dict keys: {unc_dict.keys()}")
    
    # Prediction with rejection
    preds_rej, probs, unc_values, abstained = model.predict_with_rejection(
        x,
        uncertainty_threshold=0.5,
        uncertainty_type='epistemic'
    )
    print(f"\nPredictions with rejection: {preds_rej}")
    print(f"Abstained mask: {abstained}")
    print(f"Num abstained: {abstained.sum().item()}/{len(abstained)}")
    
    print("\n✓ Hybrid model test passed!")
