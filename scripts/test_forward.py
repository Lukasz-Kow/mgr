#!/usr/bin/env python3
"""Quick forward pass test for baseline 3D model."""
import sys, yaml, torch
sys.path.insert(0, '.')

from src.models.backbone import get_backbone
from src.models import BaselineSoftmaxModel

with open('configs/baseline_config.yaml') as f:
    config = yaml.safe_load(f)

backbone = get_backbone(config['model']['backbone'], force_3d=True)
model = BaselineSoftmaxModel(backbone, num_classes=2, dropout=0.5)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

x = torch.randn(2, 1, 128, 128, 128)
print(f'Input shape: {x.shape}')
out = model(x)
print(f'Output shape: {out.shape}')
print('Forward pass OK!')
