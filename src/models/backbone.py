"""
Backbone models for MCI classification.

Supports:
- ResNetBackbone2D (ImageNet pretrained, legacy)
- ResNet3DBackbone (simple 3D CNN placeholder, Phase 1)
- MONAIResNet3DBackbone (MONAI ResNet3D + optional MedicalNet weights, Phase 2)
"""

import logging
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_RESNET_DEPTH_MAP = {
    'resnet10': 10,
    'resnet18': 18,
    'resnet34': 34,
    'resnet50': 50,
}

_SHORTCUT_BY_DEPTH = {
    10: 'B',
    18: 'A',
    34: 'A',
    50: 'B',
}


class ResNetBackbone2D(nn.Module):
    """2D ResNet backbone for feature extraction."""

    def __init__(
        self,
        arch: str = 'resnet18',
        pretrained: bool = True,
        in_channels: int = 1,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.arch = arch
        self.in_channels = in_channels
        self.backbone_type = '2d'

        if arch == 'resnet18':
            base_model = models.resnet18(weights='DEFAULT' if pretrained else None)
            self.feature_dim = 512
        elif arch == 'resnet34':
            base_model = models.resnet34(weights='DEFAULT' if pretrained else None)
            self.feature_dim = 512
        elif arch == 'resnet50':
            base_model = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        if in_channels != 3:
            original_conv = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    base_model.conv1.weight[:, 0:1, :, :] = (
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )

        self.encoder = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return features.flatten(1)


class ResNet3DBackbone(nn.Module):
    """Simple 3D CNN backbone (Phase 1 placeholder)."""

    def __init__(
        self,
        arch: str = 'resnet3d_18',
        pretrained: bool = False,
        in_channels: int = 1,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.backbone_type = 'simple'
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.feature_dim = 512

    def _conv_block(self, in_f, out_f):
        return nn.Sequential(
            nn.Conv3d(in_f, out_f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_f),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)


class MONAIResNet3DBackbone(nn.Module):
    """MONAI 3D ResNet backbone with optional MedicalNet pretrained weights."""

    def __init__(
        self,
        arch: str = 'resnet10',
        pretrained: str = 'medicalnet',
        in_channels: int = 1,
        shortcut_type: Optional[str] = None,
        bias_downsample: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        try:
            from monai.networks.nets import (
                resnet10, resnet18, resnet34, resnet50,
            )
        except ImportError as e:
            raise ImportError(
                "MONAI is required for MONAIResNet3DBackbone. "
                "Install with: pip install monai"
            ) from e

        self.arch = arch
        self.backbone_type = 'monai'
        depth = _RESNET_DEPTH_MAP.get(arch.replace('resnet3d_', 'resnet'), 10)
        if shortcut_type is None:
            shortcut_type = _SHORTCUT_BY_DEPTH.get(depth, 'B')

        builders = {
            10: resnet10,
            18: resnet18,
            34: resnet34,
            50: resnet50,
        }
        if depth not in builders:
            raise ValueError(f"Unsupported MONAI ResNet depth: {depth}")

        self.model = builders[depth](
            pretrained=False,
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=num_classes,
            feed_forward=False,
            shortcut_type=shortcut_type,
            bias_downsample=bias_downsample,
        )

        # Infer feature dim from MONAI fc layer
        if hasattr(self.model, 'fc') and isinstance(self.model.fc, nn.Linear):
            self.feature_dim = self.model.fc.in_features
        else:
            self.feature_dim = 512

        if pretrained and str(pretrained).lower() in ('medicalnet', 'true', 'yes'):
            self.load_medicalnet_weights(depth)

    def load_medicalnet_weights(self, depth: int = 10) -> None:
        """Load MedicalNet pretrained weights from HuggingFace (MONAI helper)."""
        state_dict = None
        try:
            from monai.networks.nets.resnet import get_pretrained_resnet_medicalnet
            state_dict = get_pretrained_resnet_medicalnet(
                depth, device='cpu', datasets23=True
            )
            logger.info("Loaded MedicalNet weights via MONAI HuggingFace helper")
        except Exception as e:
            logger.warning(f"MONAI HF loader failed: {e}. Trying pretrained=True...")
            try:
                from monai.networks.nets import resnet10, resnet18
                builders = {10: resnet10, 18: resnet18}
                tmp = builders[depth](
                    pretrained=True,
                    spatial_dims=3,
                    n_input_channels=1,
                    feed_forward=False,
                    bias_downsample=False,
                )
                state_dict = tmp.state_dict()
                del tmp
            except Exception as e2:
                logger.warning(f"Could not load MedicalNet weights: {e2}")
                return

        if state_dict is None:
            return

        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace('module.', '')] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.info(f"MedicalNet load missing keys (expected fc): {len(missing)}")
        if unexpected:
            logger.info(f"MedicalNet load unexpected keys: {len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if out.dim() > 2:
            return out.flatten(1)
        return out

    def encoder_parameters(self):
        """Parameters of MONAI ResNet excluding the final classifier."""
        for name, param in self.model.named_parameters():
            if not name.startswith('fc'):
                yield param

    def set_encoder_requires_grad(self, requires_grad: bool) -> None:
        for name, param in self.model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = requires_grad


def is_monai_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Heuristic: MONAI ResNet keys use backbone.model.layer1, not encoder.0."""
    keys = list(state_dict.keys())
    has_monai = any('model.layer1' in k for k in keys)
    has_simple = any('encoder.0' in k for k in keys)
    return has_monai and not has_simple


def get_backbone(config: dict, force_3d: bool = False) -> nn.Module:
    """
    Factory function to create backbone model.

    config keys:
      type: 'simple' | 'monai' (default simple for 3D)
      use_3d, arch_3d, pretrained, in_channels, shortcut_type, bias_downsample
    """
    use_3d = config.get('use_3d', False) or force_3d
    backbone_type = config.get('type', 'simple').lower()

    if use_3d and backbone_type == 'monai':
        arch = config.get('arch_3d', 'resnet10').replace('resnet3d_', 'resnet')
        return MONAIResNet3DBackbone(
            arch=arch,
            pretrained=config.get('pretrained', 'medicalnet'),
            in_channels=config.get('in_channels', 1),
            shortcut_type=config.get('shortcut_type'),
            bias_downsample=config.get('bias_downsample', False),
        )

    if use_3d:
        return ResNet3DBackbone(
            arch=config.get('arch_3d', 'resnet3d_18'),
            pretrained=config.get('pretrained', False),
            in_channels=config.get('in_channels', 1),
        )

    return ResNetBackbone2D(
        arch=config.get('arch_2d', 'resnet18'),
        pretrained=config.get('pretrained', True),
        in_channels=config.get('in_channels', 1),
    )


if __name__ == '__main__':
    print("Testing backbones...")
    simple = ResNet3DBackbone(pretrained=False)
    x = torch.randn(1, 1, 64, 64, 64)
    print(f"Simple 3D: {simple(x).shape}")
