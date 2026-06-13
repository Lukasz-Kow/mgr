"""
SelectiveNet implementation based on Geifman & El-Yaniv (2019).

Paper: "SelectiveNet: A Deep Neural Network with an Integrated Reject Option"
URL: https://arxiv.org/abs/1901.09192

Architecture:
- Prediction head: Standard classification
- Selection head: Single neuron predicting selection probability
- Auxiliary head: For regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SelectiveNet(nn.Module):
    """
    SelectiveNet with integrated reject option.
    
    Three heads:
    1. Prediction head (f): Classifies input
    2. Selection head (g): Decides whether to predict or abstain
    3. Auxiliary head (h): Helps training the body
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        dropout: float = 0.5,
        selection_dropout: float = 0.3
    ):
        """
        Args:
            backbone: Feature extractor
            num_classes: Number of classes
            dropout: Dropout for prediction/auxiliary heads
            selection_dropout: Dropout for selection head
        """
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature_dim = backbone.feature_dim
        
        # Prediction head (f)
        self.pred_dropout = nn.Dropout(dropout)
        self.pred_head = nn.Linear(self.feature_dim, num_classes)
        
        # Selection head (g) - single neuron with sigmoid (no dropout: train/eval consistency)
        self.select_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary head (h) - for regularization during training
        self.aux_dropout = nn.Dropout(dropout)
        self.aux_head = nn.Linear(self.feature_dim, num_classes)
        
    def forward(
        self,
        x: torch.Tensor,
        return_selection: bool = True,
        return_auxiliary: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_selection: Whether to return selection probabilities
            return_auxiliary: Whether to return auxiliary logits (for training)
            
        Returns:
            pred_logits: Prediction logits (B, num_classes)
            selection_probs: Selection probabilities (B,) if return_selection=True
            aux_logits: Auxiliary logits (B, num_classes) if return_auxiliary=True
        """
        # Extract features
        features = self.backbone(x)  # (B, feature_dim)
        
        # Prediction head
        pred_logits = self.pred_head(self.pred_dropout(features))
        
        outputs = [pred_logits]
        
        # Selection head
        if return_selection:
            selection_probs = self.select_head(features).squeeze(-1)  # (B,)
            outputs.append(selection_probs)
        
        # Auxiliary head
        if return_auxiliary:
            aux_logits = self.aux_head(self.aux_dropout(features))
            outputs.append(aux_logits)
        
        return tuple(outputs) if len(outputs) > 1 else outputs[0]
    
    @torch.no_grad()
    def predict_with_selection(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction with selection mechanism.
        
        Args:
            x: Input tensor
            threshold: Selection threshold (g(x) > threshold → predict, else abstain)
            
        Returns:
            predictions: Predicted classes or -1 for abstention (B,)
            confidences: Softmax confidences (B,)
            selection_probs: Selection probabilities from g(x) (B,)
            is_abstained: Boolean mask for abstained samples (B,)
        """
        self.eval()
        
        pred_logits, selection_probs = self.forward(
            x, 
            return_selection=True,
            return_auxiliary=False
        )
        
        # Get predictions and confidences
        probs = F.softmax(pred_logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        
        # Abstain if selection probability < threshold
        is_abstained = selection_probs < threshold
        predictions_with_selection = predictions.clone()
        predictions_with_selection[is_abstained] = -1
        
        return predictions_with_selection, confidences, selection_probs, is_abstained


class SelectiveNetLoss(nn.Module):
    """
    Loss function for SelectiveNet.
    
    L = L_selective + λ * L_auxiliary
    
    L_selective = (1/c) * Σ g(x) * CE(f(x), y)
    where c = (1/n) * Σ g(x) is the coverage (penalized symmetrically around target_coverage)
    
    L_auxiliary = CE(h(x), y)
    """
    
    def __init__(
        self,
        target_coverage: float = 0.8,
        alpha: float = 0.5,
        aux_weight: float = 0.3,
        coverage_penalty: float = 10.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            target_coverage: Desired coverage (% of samples not abstained)
            alpha: Weight for classification loss (higher = prioritize accuracy)
            aux_weight: Weight for auxiliary loss
            coverage_penalty: Penalty for not meeting target coverage
            class_weights: Optional CE class weights (cost-sensitive)
        """
        super().__init__()
        
        self.target_coverage = target_coverage
        self.alpha = alpha
        self.aux_weight = aux_weight
        self.coverage_penalty = coverage_penalty
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
        self.aux_ce_loss = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        selection_probs: torch.Tensor,
        aux_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute SelectiveNet loss.
        
        Args:
            pred_logits: Prediction logits (B, num_classes)
            selection_probs: Selection probabilities (B,)
            aux_logits: Auxiliary logits (B, num_classes)
            labels: Ground truth labels (B,)
            
        Returns:
            total_loss: Combined loss
            metrics: Dict with loss components and coverage
        """
        batch_size = pred_logits.size(0)
        
        # Prediction loss (weighted by selection)
        pred_ce = self.ce_loss(pred_logits, labels)  # (B,)
        
        # Coverage constraint
        coverage = selection_probs.mean()  # Empirical coverage
        
        # Selective loss: normalize by effective coverage (prevents g→1 gaming)
        effective_cov = torch.clamp(coverage, min=self.target_coverage)
        if effective_cov > 0:
            selective_loss = (selection_probs * pred_ce).sum() / (
                effective_cov.detach() * batch_size
            )
        else:
            selective_loss = pred_ce.mean()
        
        # Symmetric coverage penalty: target coverage, not just lower bound
        cov_dev = coverage - self.target_coverage
        coverage_loss = self.coverage_penalty * cov_dev ** 2
        
        # Auxiliary loss (standard CE on auxiliary head)
        aux_loss = self.aux_ce_loss(aux_logits, labels)
        
        # Total loss (alpha weights classification vs selection)
        total_loss = self.alpha * selective_loss + coverage_loss + self.aux_weight * aux_loss
        
        # Metrics for monitoring
        metrics = {
            'selective_loss': selective_loss.item(),
            'coverage_loss': coverage_loss.item(),
            'aux_loss': aux_loss.item(),
            'coverage': coverage.item(),
            'coverage_deviation': cov_dev.item(),
            'target_coverage': self.target_coverage
        }
        
        return total_loss, metrics


if __name__ == '__main__':
    # Test SelectiveNet
    print("Testing SelectiveNet...")
    
    from backbone import get_backbone

    backbone = get_backbone({
        'type': 'monai', 'use_3d': True, 'arch_3d': 'resnet10', 'pretrained': False,
    })
    model = SelectiveNet(backbone, num_classes=2)
    
    x = torch.randn(2, 1, 64, 64, 64)
    labels = torch.randint(0, 2, (2,))
    
    # Forward pass
    pred_logits, selection_probs, aux_logits = model(
        x,
        return_selection=True,
        return_auxiliary=True
    )
    
    print(f"Pred logits shape: {pred_logits.shape}")
    print(f"Selection probs shape: {selection_probs.shape}")
    print(f"Selection probs: {selection_probs}")
    print(f"Aux logits shape: {aux_logits.shape}")
    
    # Test loss
    criterion = SelectiveNetLoss(target_coverage=0.8)
    loss, metrics = criterion(pred_logits, selection_probs, aux_logits, labels)
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test prediction with selection
    preds, confs, sel_probs, abstained = model.predict_with_selection(x, threshold=0.5)
    print(f"\nPredictions: {preds}")
    print(f"Abstained: {abstained}")
    print(f"Num abstained: {abstained.sum().item()}/{len(abstained)}")
    
    print("\n✓ SelectiveNet test passed!")
