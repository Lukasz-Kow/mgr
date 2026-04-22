"""
Evidential Deep Learning layer and loss functions.

Based on:
- Sensoy et al. (2018) "Evidential Deep Learning to Quantify Classification Uncertainty"
- Uses Dirichlet distribution to model uncertainty

Key concept:
Instead of outputting point estimates P(y|x), output parameters of a Dirichlet
distribution Dir(α), where α represents evidence for each class.

Uncertainty decomposition:
- Aleatoric uncertainty: Data noise (irreducible)
- Epistemic uncertainty: Model uncertainty (reducible with more data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class EvidentialLayer(nn.Module):
    """
    Evidential output layer that predicts Dirichlet parameters.
    
    Output: α (alpha) parameters of Dirichlet distribution
    Evidence: e = α - 1  (e >= 0)
    Uncertainty: u = K / S, where S = Σ α_i, K = num_classes
    """
    
    def __init__(self, in_features: int, num_classes: int = 2):
        """
        Args:
            in_features: Input feature dimension
            num_classes: Number of classes
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Output layer predicts evidence (non-negative)
        self.evidence_layer = nn.Linear(in_features, num_classes)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Feature tensor (B, in_features)
            
        Returns:
            alpha: Dirichlet parameters (B, num_classes), all > 1
        """
        # Predict evidence (use softplus to ensure non-negative)
        # e = softplus(logits) ensures e >= 0
        evidence = F.softplus(self.evidence_layer(features))
        
        # α = e + 1 (alpha must be > 1 for Dirichlet)
        alpha = evidence + 1.0
        
        return alpha


def compute_uncertainty(alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute uncertainties from Dirichlet parameters.
    
    Args:
        alpha: Dirichlet parameters (B, K)
        
    Returns:
        epistemic_unc: Epistemic uncertainty (B,) - lack of evidence
        aleatoric_unc: Aleatoric uncertainty (B,) - data noise
        total_unc: Total uncertainty (B,)
    """
    num_classes = alpha.size(1)
    strength = alpha.sum(dim=1)  # S = Σ α_i
    
    # Expected probability: p_i = α_i / S
    prob = alpha / strength.unsqueeze(1)
    
    # Epistemic uncertainty (vacuity): u = K / S
    # High when little evidence (all α_i ≈ 1)
    epistemic_unc = num_classes / strength
    
    # Aleatoric uncertainty (dissonance): expected entropy
    # High when evidence is conflicting
    aleatoric_unc = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
    
    # Total uncertainty
    total_unc = epistemic_unc + aleatoric_unc
    
    return epistemic_unc, aleatoric_unc, total_unc


class EvidentialLoss(nn.Module):
    """
    Loss function for Evidential Deep Learning.
    
    L = L_bayes + λ * L_KL
    
    L_bayes: Bayes risk (expected loss under Dirichlet)
    L_KL: KL divergence from uniform (encourages high evidence when wrong)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        kl_weight: float = 1.0,
        kl_anneal_start: int = 0,
        kl_anneal_end: int = 20
    ):
        """
        Args:
            num_classes: Number of classes
            kl_weight: Max weight for KL divergence term (reached after annealing)
            kl_anneal_start: Epoch to start annealing KL weight
            kl_anneal_end: Epoch to reach full KL weight
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.kl_anneal_start = kl_anneal_start
        self.kl_anneal_end = kl_anneal_end
        
        self.current_epoch = 0
    
    def forward(
        self,
        alpha: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute evidential loss (Sensoy et al. 2018).
        
        L = MSE + Variance + λ_t · KL
        
        MSE: (y_j - α_j/S)^2
        Variance: α_j(S - α_j) / (S²(S+1))  — penalizes spread evidence
        KL: KL[Dir(α̃) || Dir(1)] — regularization on incorrect classes
        
        Args:
            alpha: Dirichlet parameters (B, K)
            labels: Ground truth labels (B,)
            
        Returns:
            loss: Total loss
            metrics: Dict with loss components
        """
        batch_size = alpha.size(0)
        
        # One-hot encode labels
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Strength (sum of alphas)
        S = alpha.sum(dim=1, keepdim=True)  # (B, 1)
        
        # === BAYES RISK ===
        # Expected probability: p_j = α_j / S
        expected_prob = alpha / S
        
        # MSE term: Σ_j (y_j - α_j/S)^2
        mse_loss = torch.sum((labels_one_hot - expected_prob) ** 2, dim=1)
        
        # Variance term: Σ_j α_j(S - α_j) / (S²(S+1))
        # This term penalizes the model for NOT concentrating evidence
        # on a single class — critical for decisive predictions
        variance = torch.sum(
            alpha * (S - alpha) / (S ** 2 * (S + 1)),
            dim=1
        )
        
        bayes_loss = (mse_loss + variance).mean()
        
        # === KL DIVERGENCE REGULARIZATION ===
        # KL(Dir(α̃) || Dir(1)) where α̃ removes evidence for correct class
        # This penalizes confident WRONG predictions
        alpha_0 = torch.ones_like(alpha)  # Uniform prior
        
        # Only compute KL for incorrect predictions (where y_i = 0 but α_i > 1)
        kl_alpha = (1 - labels_one_hot) * alpha + labels_one_hot * alpha_0
        kl_div = torch.sum(
            (alpha - kl_alpha) * (torch.digamma(alpha) - torch.digamma(S)),
            dim=1
        )
        kl_loss = kl_div.mean()
        
        # Anneal KL weight (λ_t: 0 → kl_weight over epochs)
        kl_coeff = self._get_kl_coefficient()
        
        # Total loss
        total_loss = bayes_loss + kl_coeff * kl_loss
        
        # Metrics
        epistemic_unc, aleatoric_unc, total_unc = compute_uncertainty(alpha)
        
        metrics = {
            'bayes_loss': bayes_loss.item(),
            'mse_loss': mse_loss.mean().item(),
            'variance_loss': variance.mean().item(),
            'kl_loss': kl_loss.item(),
            'kl_coeff': kl_coeff,
            'mean_epistemic_unc': epistemic_unc.mean().item(),
            'mean_aleatoric_unc': aleatoric_unc.mean().item(),
            'mean_total_unc': total_unc.mean().item(),
            'mean_strength': S.mean().item()
        }
        
        return total_loss, metrics

    
    def _get_kl_coefficient(self) -> float:
        """Get KL coefficient with annealing."""
        if self.current_epoch < self.kl_anneal_start:
            return 0.0
        elif self.current_epoch >= self.kl_anneal_end:
            return self.kl_weight
        else:
            # Linear annealing
            progress = (self.current_epoch - self.kl_anneal_start) / \
                      (self.kl_anneal_end - self.kl_anneal_start)
            return progress * self.kl_weight
    
    def set_epoch(self, epoch: int):
        """Set current epoch for KL annealing."""
        self.current_epoch = epoch


if __name__ == '__main__':
    # Test evidential layer and loss
    print("Testing Evidential Layer...")
    
    layer = EvidentialLayer(in_features=512, num_classes=2)
    
    # Dummy features
    features = torch.randn(8, 512)
    labels = torch.randint(0, 2, (8,))
    
    # Forward
    alpha = layer(features)
    print(f"Alpha shape: {alpha.shape}")
    print(f"Alpha values (first sample): {alpha[0]}")
    print(f"Alpha >= 1: {(alpha >= 1).all()}")
    
    # Uncertainty
    epi_unc, ale_unc, total_unc = compute_uncertainty(alpha)
    print(f"\nEpistemic uncertainty: {epi_unc}")
    print(f"Aleatoric uncertainty: {ale_unc}")
    print(f"Total uncertainty: {total_unc}")
    
    # Loss
    criterion = EvidentialLoss(num_classes=2, kl_weight=0.1)
    criterion.set_epoch(5)
    
    loss, metrics = criterion(alpha, labels)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    print("\n✓ Evidential layer test passed!")
