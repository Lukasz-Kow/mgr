"""Fine-tuning helpers for MONAI MedicalNet backbones."""

from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn


def get_backbone_module(model: nn.Module) -> Optional[nn.Module]:
    """Return backbone submodule if present."""
    if hasattr(model, 'backbone'):
        return model.backbone
    return None


def apply_encoder_freeze(model: nn.Module, freeze: bool) -> None:
    """Freeze or unfreeze MONAI encoder weights."""
    backbone = get_backbone_module(model)
    if backbone is None:
        return
    if hasattr(backbone, 'set_encoder_requires_grad'):
        backbone.set_encoder_requires_grad(not freeze)
    elif hasattr(backbone, 'encoder'):
        for p in backbone.encoder.parameters():
            p.requires_grad = not freeze


def build_optimizer_param_groups(
    model: nn.Module,
    config: Dict[str, Any],
    default_lr: float,
    default_wd: float,
) -> List[Dict]:
    """
    Build optimizer param groups for fine-tuning.
    Head params use head_lr; encoder uses encoder_lr when fine_tune config present.
    """
    fine_tune = config.get('model', {}).get('fine_tune', {})
    if not fine_tune:
        return [{'params': model.parameters(), 'lr': default_lr}]

    backbone = get_backbone_module(model)
    head_lr = fine_tune.get('head_lr', default_lr)
    encoder_lr = fine_tune.get('encoder_lr', default_lr * 0.1)

    encoder_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.') and backbone is not None:
            if hasattr(backbone, 'backbone_type') and backbone.backbone_type == 'monai':
                if '.model.' in name and not name.endswith('fc.weight') and not name.endswith('fc.bias'):
                    encoder_params.append(param)
                    continue
            elif name.startswith('backbone.encoder.'):
                encoder_params.append(param)
                continue
        head_params.append(param)

    groups = []
    if encoder_params:
        groups.append({'params': encoder_params, 'lr': encoder_lr})
    if head_params:
        groups.append({'params': head_params, 'lr': head_lr})
    if not groups:
        groups = [{'params': model.parameters(), 'lr': default_lr}]
    return groups


def setup_fine_tune_for_epoch(model: nn.Module, config: Dict[str, Any], epoch: int) -> None:
    """Apply encoder freeze during warmup epochs of fine-tuning."""
    fine_tune = config.get('model', {}).get('fine_tune', {})
    freeze_epochs = fine_tune.get('freeze_encoder_epochs', 0)
    if freeze_epochs <= 0:
        return
    apply_encoder_freeze(model, freeze=epoch <= freeze_epochs)


def should_count_early_stopping(config: Dict[str, Any], epoch: int) -> bool:
    """Return False during warmup_epochs (early stopping disabled)."""
    warmup = config.get('training', {}).get('warmup_epochs', 0)
    return epoch > warmup
