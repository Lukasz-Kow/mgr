"""
3D Grad-CAM for MRI volume interpretability.

Generates gradient-weighted class activation maps to visualize
which brain regions drive model predictions. Critical for verifying
that models focus on clinically relevant structures (hippocampus,
lateral ventricles) rather than artifacts.

Based on: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations
from Deep Networks via Gradient-based Localization"
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple


class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D volumes.
    
    Registers forward/backward hooks on a target convolutional layer
    to capture activations and gradients, then computes a weighted
    combination that highlights discriminative regions.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer_name: str = 'backbone.layer4'):
        """
        Args:
            model: The trained model (Baseline, EDL, Hybrid, etc.)
            target_layer_name: Dot-separated path to the target layer
                               (e.g., 'backbone.layer4' for ResNet)
        """
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # Navigate to target layer
        target_layer = self._get_layer(target_layer_name)
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _get_layer(self, layer_name: str) -> torch.nn.Module:
        """Navigate to a nested layer by dot-separated name."""
        module = self.model
        for part in layer_name.split('.'):
            module = getattr(module, part)
        return module
    
    def _forward_hook(self, module, input, output):
        """Capture activations during forward pass."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass."""
        self.gradients = grad_output[0].detach()
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single input volume.
        
        Args:
            input_tensor: Input tensor (1, C, D, H, W) or (C, D, H, W)
            target_class: Class to generate CAM for. If None, uses predicted class.
            
        Returns:
            heatmap: 3D numpy array (D, H, W) normalized to [0, 1]
        """
        # Ensure batch dimension
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor.requires_grad_(True)
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle different model outputs
        if output.shape[1] > 2:
            # EDL/Hybrid: output is alpha (Dirichlet params)
            # Use expected probability for target class
            strength = output.sum(dim=1, keepdim=True)
            probs = output / strength
        else:
            # Baseline: output is logits
            probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = probs.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        target_score = probs[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Compute Grad-CAM
        # Global average pooling of gradients → weights
        if self.gradients.dim() == 5:
            # 3D: (B, C, D, H, W)
            weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        else:
            # 2D: (B, C, H, W)
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, D, H, W) or (B, 1, H, W)
        
        # ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Resize to input spatial dimensions
        if cam.dim() == 5:
            # 3D
            cam = F.interpolate(
                cam,
                size=input_tensor.shape[2:],  # (D, H, W)
                mode='trilinear',
                align_corners=False
            )
        else:
            # 2D
            cam = F.interpolate(
                cam,
                size=input_tensor.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


def plot_gradcam_slices(
    volume: np.ndarray,
    heatmap: np.ndarray,
    true_label: int,
    pred_label: int,
    model_name: str,
    confidence: float = None,
    output_path: Optional[str] = None,
    slice_indices: Optional[Tuple[int, int, int]] = None
):
    """
    Plot Grad-CAM heatmap overlaid on MRI slices (axial, coronal, sagittal).
    
    Args:
        volume: 3D numpy array (D, H, W) — the original MRI
        heatmap: 3D numpy array (D, H, W) — Grad-CAM output [0, 1]
        true_label: Ground truth label
        pred_label: Predicted label
        model_name: Name of the model
        confidence: Model confidence (optional)
        output_path: Path to save the figure
        slice_indices: (axial_idx, coronal_idx, sagittal_idx). If None, use center.
    """
    class_names = {0: 'CN (Healthy)', 1: 'MCI (Impaired)'}
    
    D, H, W = volume.shape
    
    if slice_indices is None:
        slice_indices = (D // 2, H // 2, W // 2)
    
    ax_idx, cor_idx, sag_idx = slice_indices
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f'{model_name} — Grad-CAM\n'
        f'True: {class_names.get(true_label, true_label)} | '
        f'Pred: {class_names.get(pred_label, pred_label)}'
        + (f' | Conf: {confidence:.2f}' if confidence else ''),
        fontsize=14, fontweight='bold'
    )
    
    # Row 1: Original slices
    slices_orig = [
        volume[ax_idx, :, :],      # Axial
        volume[:, cor_idx, :],      # Coronal
        volume[:, :, sag_idx],      # Sagittal
    ]
    titles = [f'Axial (z={ax_idx})', f'Coronal (y={cor_idx})', f'Sagittal (x={sag_idx})']
    
    # Row 2: Grad-CAM overlay
    slices_cam = [
        heatmap[ax_idx, :, :],
        heatmap[:, cor_idx, :],
        heatmap[:, :, sag_idx],
    ]
    
    for i in range(3):
        # Original
        axes[0, i].imshow(slices_orig[i].T, cmap='gray', origin='lower')
        axes[0, i].set_title(titles[i])
        axes[0, i].axis('off')
        
        # Overlay
        axes[1, i].imshow(slices_orig[i].T, cmap='gray', origin='lower')
        im = axes[1, i].imshow(
            slices_cam[i].T, cmap='jet', alpha=0.4, origin='lower',
            vmin=0, vmax=1
        )
        axes[1, i].set_title(f'{titles[i]} + Grad-CAM')
        axes[1, i].axis('off')
    
    # Colorbar
    fig.colorbar(im, ax=axes[1, :], shrink=0.6, label='Activation intensity')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        # Also save PDF for paper
        fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
        print(f"✅ Grad-CAM saved: {output_path}")
    
    plt.close(fig)
    return fig


def generate_gradcam_for_samples(
    model,
    dataloader,
    model_name: str,
    output_dir: str,
    target_layer: str = 'backbone.layer4',
    max_samples: int = 5,
    device: str = 'cpu'
):
    """
    Generate Grad-CAM visualizations for selected test samples.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        model_name: Name for file naming
        output_dir: Directory to save outputs
        target_layer: Layer name for Grad-CAM
        max_samples: Maximum number of samples to visualize
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gradcam = GradCAM3D(model, target_layer_name=target_layer)
    
    count = 0
    for images, labels, metadata in dataloader:
        for i in range(images.shape[0]):
            if count >= max_samples:
                return
            
            single_image = images[i:i+1].to(device)
            label = labels[i].item()
            
            # Generate heatmap
            heatmap = gradcam.generate(single_image, target_class=None)
            
            # Get prediction
            with torch.no_grad():
                output = model(single_image)
                if hasattr(output, 'sum'):
                    strength = output.sum(dim=1, keepdim=True)
                    probs = output / strength
                    pred = probs.argmax(dim=1).item()
                    conf = probs.max(dim=1).values.item()
                else:
                    pred = output.argmax(dim=1).item()
                    conf = None
            
            # Plot
            volume = images[i].squeeze(0).numpy()  # (D, H, W)
            
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            plot_gradcam_slices(
                volume=volume,
                heatmap=heatmap,
                true_label=label,
                pred_label=pred,
                model_name=model_name,
                confidence=conf,
                output_path=output_dir / f'gradcam_{safe_name}_sample{count+1}.png'
            )
            
            count += 1
