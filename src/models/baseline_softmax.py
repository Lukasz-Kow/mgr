"""
Baseline model: Standard classifier with Softmax Response (SR) rejection.

This serves as the baseline approach for comparison with SelectiveNet and EDL.
Rejection mechanism: Simple threshold on max softmax probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BaselineSoftmaxModel(nn.Module):
    """
    Baseline classification model with standard softmax output.
    
    Rejection mechanism: If max(softmax_probs) < threshold → ABSTAIN
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            backbone: Feature extractor (e.g., ResNet)
            num_classes: Number of output classes (2 for MCI/CN)
            dropout: Dropout probability before final FC layer
        """
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(backbone.feature_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
            
        Returns:
            logits: Shape (B, num_classes) - raw logits before softmax
        """
        # Extract features
        features = self.backbone(x)  # (B, feature_dim)
        
        # Classifier
        features = self.dropout(features)
        logits = self.fc(features)  # (B, num_classes)
        
        return logits
    
    def predict_with_confidence(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction with confidence scores.
        
        Args:
            x: Input tensor
            
        Returns:
            predictions: Predicted class indices (B,)
            confidences: Max softmax probability (B,)
            probabilities: Full softmax probabilities (B, num_classes)
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        
        confidences, predictions = torch.max(probabilities, dim=1)
        
        return predictions, confidences, probabilities
    
    def predict_with_rejection(
        self,
        x: torch.Tensor,
        threshold: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction with rejection mechanism (Softmax Response).
        
        Rejection rule: If max(P(y|x)) < threshold → abstain (return -1)
        
        Args:
            x: Input tensor
            threshold: Confidence threshold for acceptance
            
        Returns:
            predictions: Predicted class or -1 for abstention (B,)
            confidences: Max softmax probability (B,)
            is_abstained: Boolean mask for abstained samples (B,)
        """
        predictions, confidences, _ = self.predict_with_confidence(x)
        
        # Abstain if confidence < threshold
        is_abstained = confidences < threshold
        predictions_with_rejection = predictions.clone()
        predictions_with_rejection[is_abstained] = -1  # -1 indicates abstention
        
        return predictions_with_rejection, confidences, is_abstained


class BaselineTrainer:
    """Helper class for training baseline model."""
    
    def __init__(
        self,
        model: BaselineSoftmaxModel,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda'
    ):
        """
        Args:
            model: BaselineSoftmaxModel instance
            criterion: Loss function (e.g., CrossEntropyLoss)
            optimizer: Optimizer
            device: Device to train on
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
    
    def train_step(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
    ) -> float:
        """
        Single training step.
        
        Args:
            images: Batch of images
            labels: Batch of labels
            
        Returns:
            loss: Scalar loss value
        """
        self.model.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def eval_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Single evaluation step.
        
        Args:
            images: Batch of images
            labels: Batch of labels
            
        Returns:
            loss: Scalar loss value
            predictions: Predicted classes
            probabilities: Class probabilities
        """
        self.model.eval()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        
        # Predictions
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        return loss.item(), predictions.cpu(), probabilities.cpu()


if __name__ == '__main__':
    # Test baseline model
    print("Testing Baseline Softmax Model...")
    
    from backbone import ResNetBackbone2D
    
    backbone = ResNetBackbone2D(arch='resnet18', pretrained=False)
    model = BaselineSoftmaxModel(backbone, num_classes=2)
    
    # Dummy input
    x = torch.randn(4, 1, 224, 224)
    
    # Forward pass
    logits = model(x)
    print(f"Logits shape: {logits.shape}")
    
    # Prediction with confidence
    predictions, confidences, probs = model.predict_with_confidence(x)
    print(f"Predictions: {predictions}")
    print(f"Confidences: {confidences}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Prediction with rejection
    preds_rej, confs, abstained = model.predict_with_rejection(x, threshold=0.8)
    print(f"Predictions with rejection: {preds_rej}")
    print(f"Abstained mask: {abstained}")
    print(f"Num abstained: {abstained.sum().item()}/{len(abstained)}")
    
    print("✓ Baseline model test passed!")
