"""
Backbone models for MCI classification.

Current: 2D ResNet implementation
Future: Easy migration to 3D ResNet when ADNI data available

For 3D migration, replace with:
- torchvision.models.video.r3d_18 or
- med3D ResNet3D or
- Custom 3D-ResNet implementation
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNetBackbone2D(nn.Module):
    """
    2D ResNet backbone for feature extraction.
    
    NOTE: This is the 2D mock version. 
    When migrating to 3D ADNI data, replace with ResNet3D.
    """
    
    def __init__(
        self,
        arch: str = 'resnet18',
        pretrained: bool = True,
        in_channels: int = 1,  # Grayscale MRI
        feature_dim: int = 512
    ):
        """
        Args:
            arch: ResNet architecture ('resnet18', 'resnet34', 'resnet50')
            pretrained: Use ImageNet pretrained weights
            in_channels: Number of input channels (1 for grayscale MRI)
            feature_dim: Output feature dimension
        """
        super().__init__()
        
        self.arch = arch
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        
        # Load pretrained ResNet
        if arch == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif arch == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif arch == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Modify first conv layer for grayscale input (1 channel instead of 3)
        if in_channels != 3:
            original_conv = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Initialize new conv layer
            if pretrained and in_channels == 1:
                # Average RGB weights for grayscale
                with torch.no_grad():
                    base_model.conv1.weight[:, 0:1, :, :] = \
                        original_conv.weight.mean(dim=1, keepdim=True)
        
        # Remove final FC layer (we'll add custom heads)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        
        # Global average pooling is already part of ResNet
        # Output shape: (batch, feature_dim, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor shape (B, 1, H, W) for 2D
               TODO (3D): shape (B, 1, D, H, W) when using 3D model
               
        Returns:
            features: Shape (B, feature_dim)
        """
        features = self.encoder(x)  # (B, feature_dim, 1, 1)
        features = features.flatten(1)  # (B, feature_dim)
        return features


class ResNet3DBackbone(nn.Module):
    """
    3D ResNet backbone (Simple 3D-CNN implementation).
    """
    
    def __init__(
        self,
        arch: str = 'resnet3d_18',
        pretrained: bool = False,
        in_channels: int = 1,
        feature_dim: int = 512
    ):
        super().__init__()
        
        # Simple 3D feature extractor (placeholder for more complex architecture)
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.feature_dim = 512
        
    def _conv_block(self, in_f, out_f):
        return nn.Sequential(
            nn.Conv3d(in_f, out_f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_f),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input shape (B, 1, D, H, W)
        Returns:
            features: Shape (B, feature_dim)
        """
        features = self.encoder(x)
        features = features.flatten(1)
        return features


def get_backbone(config: dict, force_3d: bool = False) -> nn.Module:
    """
    Factory function to create backbone model.
    
    Args:
        config: Configuration dict with backbone settings
        force_3d: If True, use 3D model (raises error if not implemented)
        
    Returns:
        Backbone model instance
    """
    use_3d = config.get('use_3d', False) or force_3d
    
    if use_3d:
        # For future ADNI 3D data
        return ResNet3DBackbone(
            arch=config.get('arch_3d', 'resnet3d_18'),
            pretrained=config.get('pretrained', False),
            in_channels=config.get('in_channels', 1)
        )
    else:
        # Current 2D mock implementation
        return ResNetBackbone2D(
            arch=config.get('arch_2d', 'resnet18'),
            pretrained=config.get('pretrained', True),
            in_channels=config.get('in_channels', 1)
        )


if __name__ == '__main__':
    # Test 2D backbone
    print("Testing 2D ResNet Backbone...")
    model = ResNetBackbone2D(arch='resnet18', pretrained=False)
    
    # Dummy input: batch=2, channels=1, height=224, width=224
    x = torch.randn(2, 1, 224, 224)
    features = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dim: {model.feature_dim}")
    print("✓ 2D Backbone test passed!")
