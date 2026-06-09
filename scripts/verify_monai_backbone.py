#!/usr/bin/env python3
"""Verify MONAI ResNet3D backbone + MedicalNet weights on available GPU."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.backbone import MONAIResNet3DBackbone, get_backbone


def main() -> int:
    print("=" * 60)
    print("MONAI Backbone Verification")
    print("=" * 60)

    try:
        import monai
        print(f"  [OK] monai {monai.__version__}")
    except ImportError:
        print("  [FAIL] monai not installed. Run: pip install monai huggingface_hub")
        return 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    cfg = {
        'type': 'monai',
        'use_3d': True,
        'arch_3d': 'resnet10',
        'pretrained': 'medicalnet',
        'in_channels': 1,
        'bias_downsample': False,
    }

    print("\n  Building MONAI ResNet3D-10 + MedicalNet...")
    backbone = get_backbone(cfg, force_3d=True).to(device)
    print(f"  feature_dim: {backbone.feature_dim}")
    print(f"  backbone_type: {backbone.backbone_type}")

    x = torch.randn(1, 1, 128, 128, 128, device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        out = backbone(x)
    print(f"  Forward output shape: {tuple(out.shape)}")

    if device.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"  Peak VRAM: {peak_mb:.0f} MB")
        if peak_mb > 3800:
            print("  [WARN] VRAM > 3.8GB — may OOM on GTX 1050 during training")
        else:
            print("  [OK] VRAM within GTX 1050 budget")

    print("\nAll checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
