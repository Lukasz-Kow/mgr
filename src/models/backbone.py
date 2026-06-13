"""
Backbone models for MCI classification.

MONAI ResNet3D with optional MedicalNet pretrained weights.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict

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
    """Heuristic: MONAI ResNet keys use backbone.model.layer1."""
    keys = list(state_dict.keys())
    return any('model.layer1' in k for k in keys)


def get_backbone(config: dict, force_3d: bool = False) -> nn.Module:
    """
    Factory function to create MONAI 3D ResNet backbone.

    config keys:
      type: 'monai' (required)
      use_3d, arch_3d, pretrained, in_channels, shortcut_type, bias_downsample
    """
    backbone_type = config.get('type', 'monai').lower()
    if backbone_type != 'monai':
        raise ValueError(
            f"Only MONAI backbone is supported (got type={backbone_type!r}). "
            "Set model.backbone.type: monai in config."
        )

    arch = config.get('arch_3d', 'resnet10').replace('resnet3d_', 'resnet')
    return MONAIResNet3DBackbone(
        arch=arch,
        pretrained=config.get('pretrained', 'medicalnet'),
        in_channels=config.get('in_channels', 1),
        shortcut_type=config.get('shortcut_type'),
        bias_downsample=config.get('bias_downsample', False),
    )


if __name__ == '__main__':
    print("Testing MONAI backbone...")
    bb = get_backbone({
        'type': 'monai',
        'use_3d': True,
        'arch_3d': 'resnet10',
        'pretrained': False,
    })
    x = torch.randn(1, 1, 64, 64, 64)
    print(f"MONAI 3D: {bb(x).shape}")
