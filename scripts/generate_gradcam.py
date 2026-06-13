#!/usr/bin/env python3
"""Generate Grad-CAM visualizations for thesis case studies."""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MCIDataModule
from src.models.backbone import get_backbone
from src.models.hybrid_model import HybridEvidentialModel
from src.visualization.explainability import generate_gradcam_for_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['hybrid', 'baseline', 'evidential'], default='hybrid')
    parser.add_argument('--max-samples', type=int, default=3)
    parser.add_argument('--output-dir', type=str, default='results/gradcam')
    args = parser.parse_args()

    config_map = {
        'hybrid': ('configs/hybrid_config.yaml', 'checkpoints/hybrid'),
        'baseline': ('configs/baseline_config.yaml', 'checkpoints/baseline'),
        'evidential': ('configs/evidential_config.yaml', 'checkpoints/evidential'),
    }
    cfg_path, ckpt_dir = config_map[args.model]

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    with open('configs/data_config.yaml') as f:
        data_cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = get_backbone(cfg['model']['backbone'], force_3d=True)

    if args.model == 'hybrid':
        model = HybridEvidentialModel(
            backbone=backbone,
            num_classes=cfg['model']['classifier']['num_classes'],
            dropout=cfg['model']['classifier']['dropout'],
        )
    else:
        from src.models.baseline_softmax import BaselineSoftmaxModel
        from src.models.edl_model import EDLModel
        if args.model == 'baseline':
            model = BaselineSoftmaxModel(backbone, cfg['model']['classifier']['num_classes'],
                                         dropout=cfg['model']['classifier']['dropout'])
        else:
            model = EDLModel(backbone, num_classes=cfg['model']['classifier']['num_classes'])

    ckpt_path = None
    for ext in ['best_model.pt', 'best_model.pth']:
        p = Path(ckpt_dir) / ext
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        print(f'No checkpoint in {ckpt_dir}')
        return 1

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)

    dm = MCIDataModule(
        metadata_csv='data_metadata_adni.csv',
        preprocessor_config=data_cfg['preprocessing'],
        batch_size=1,
        num_workers=0,
        augmentation_config=data_cfg,
    )
    test_loader = dm.test_dataloader()

    model_label = f'{args.model} (MONAI)'
    generate_gradcam_for_samples(
        model=model,
        dataloader=test_loader,
        model_name=model_label,
        output_dir=args.output_dir,
        target_layer='backbone.layer4',
        max_samples=args.max_samples,
        device=str(device),
    )
    print(f'Grad-CAM saved to {args.output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
