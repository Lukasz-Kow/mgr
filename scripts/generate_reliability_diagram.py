#!/usr/bin/env python3
"""Generate reliability (calibration) diagrams for thesis models."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MCIDataModule
from src.evaluation.calibration import create_calibrator, calibrate_probabilities
from src.models.backbone import get_backbone, is_monai_state_dict
from src.models.baseline_softmax import BaselineSoftmaxModel
from src.models.selective_net import SelectiveNet
from src.models.edl_model import EDLModel
from src.models.hybrid_model import HybridEvidentialModel
from src.training.eval_utils import load_evaluation_config


MODELS_PHASE2 = [
    ('Baseline (SR)', 'configs/baseline_config.yaml', 'checkpoints/baseline', 'baseline'),
    ('SelectiveNet', 'configs/selectivenet_config.yaml', 'checkpoints/selective_net', 'selectivenet'),
    ('Evidential (EDL)', 'configs/evidential_config.yaml', 'checkpoints/evidential', 'evidential'),
    ('Hybrid (3D-ResNet-EDL)', 'configs/hybrid_config.yaml', 'checkpoints/hybrid', 'hybrid'),
]


def reliability_diagram(probs_pos, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_acc = []
    bin_conf = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs_pos >= bins[i]) & (probs_pos < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs_pos >= bins[i]) & (probs_pos <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_acc.append(labels[mask].mean())
        bin_conf.append(probs_pos[mask].mean())
        bin_counts.append(mask.sum())

    return np.array(bin_centers), np.array(bin_acc), np.array(bin_conf), np.array(bin_counts)


def load_model(name, config_path, ckpt_dir, model_type, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = None
    for ext in ['best_model.pt', 'best_model.pth']:
        p = ckpt_dir / ext
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    if 'config' in ckpt and ckpt['config']:
        cfg = ckpt['config']

    backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
    if model_type == 'baseline':
        model = BaselineSoftmaxModel(backbone, cfg['model']['classifier']['num_classes'],
                                     dropout=cfg['model']['classifier']['dropout'])
    elif model_type == 'selectivenet':
        model = SelectiveNet(backbone, cfg['model']['classifier']['num_classes'],
                             dropout=cfg['model']['classifier']['dropout'],
                             selection_dropout=cfg['model']['classifier'].get('selection_dropout', 0))
    elif model_type == 'evidential':
        model = EDLModel(backbone, num_classes=cfg['model']['classifier']['num_classes'])
    else:
        model = HybridEvidentialModel(backbone, cfg['model']['classifier']['num_classes'],
                                      dropout=cfg['model']['classifier']['dropout'])

    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, cfg


def collect_probs(model, model_type, loader, device):
    all_probs, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc='Collecting', leave=False):
            images = images.to(device)
            if model_type == 'selectivenet':
                logits, _, _ = model(images, return_selection=True, return_auxiliary=True)
            elif model_type in ('evidential', 'hybrid'):
                evidence = model(images)
                alpha = evidence + 1
                probs = alpha / alpha.sum(dim=1, keepdim=True)
                all_probs.append(probs.cpu().numpy())
                all_logits.append(np.log(np.clip(probs.cpu().numpy(), 1e-10, 1)))
                all_labels.append(labels.numpy())
                continue
            else:
                logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    return (np.concatenate(all_probs), np.concatenate(all_logits),
            np.concatenate(all_labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='results/calibration')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = load_evaluation_config()
    cal_method = eval_cfg.get('calibration', {}).get('method', 'temperature')

    with open('configs/data_config.yaml') as f:
        data_cfg = yaml.safe_load(f)

    dm = MCIDataModule(
        metadata_csv='data_metadata_adni.csv',
        preprocessor_config=data_cfg['preprocessing'],
        batch_size=1,
        num_workers=0,
        augmentation_config=data_cfg,
    )
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for idx, (name, cfg_path, ckpt_dir, mtype) in enumerate(MODELS_PHASE2):
        model, _ = load_model(name, cfg_path, ckpt_dir, mtype, device)
        if model is None:
            print(f'Skipping {name}: no checkpoint')
            continue

        val_probs, val_logits, val_labels = collect_probs(model, mtype, val_loader, device)
        test_probs, test_logits, test_labels = collect_probs(model, mtype, test_loader, device)

        calibrator = create_calibrator(cal_method)
        _, test_cal = calibrate_probabilities(
            calibrator, val_logits, val_probs, val_labels,
            test_logits, test_probs, cal_method,
        )

        centers, acc, conf, counts = reliability_diagram(test_cal[:, 1], test_labels)
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax.bar(centers, acc, width=0.08, alpha=0.6, label='Accuracy')
        ax.plot(centers, conf, 'ro-', label='Confidence')
        ax.set_title(name)
        ax.set_xlabel('Confidence (P(MCI))')
        ax.set_ylabel('Accuracy')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Reliability diagrams (calibrated, test set)', fontsize=12)
    plt.tight_layout()
    out = out_dir / 'reliability_diagrams.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
