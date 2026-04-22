#!/usr/bin/env python3
"""
Error Consistency Analysis — identifies "hard cases" where multiple models fail.

For each hard case, extracts epistemic uncertainty from EDL/Hybrid models
to verify whether the uncertainty estimation correctly flags difficult samples.

This script answers the key research question:
"Does the EDL model know when it doesn't know?"
"""

import sys
import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MCIDataModule
from src.models.backbone import get_backbone
from src.models.baseline_softmax import BaselineSoftmaxModel
from src.models.selective_net import SelectiveNet
from src.models.hybrid_model import HybridEvidentialModel
from src.models.evidential_layer import EvidentialLayer, compute_uncertainty


# Lightweight EDL container (same as in train_evidential.py)
class EDLModel(torch.nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.evidential_head = EvidentialLayer(backbone.feature_dim, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.evidential_head(features)


def load_all_models(device='cpu'):
    """Load all 4 trained models."""
    models = {}
    
    # --- Baseline ---
    with open('configs/baseline_config.yaml') as f:
        cfg = yaml.safe_load(f)
    backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
    model = BaselineSoftmaxModel(
        backbone=backbone,
        num_classes=cfg['model']['classifier']['num_classes'],
        dropout=cfg['model']['classifier']['dropout']
    )
    ckpt = torch.load('checkpoints/baseline/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    models['Baseline'] = {'model': model, 'type': 'softmax', 'config': cfg}
    
    # --- SelectiveNet ---
    with open('configs/selectivenet_config.yaml') as f:
        cfg = yaml.safe_load(f)
    backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
    model = SelectiveNet(
        backbone=backbone,
        num_classes=cfg['model']['classifier']['num_classes'],
        dropout=cfg['model']['classifier']['dropout'],
        selection_dropout=cfg['model']['classifier']['selection_dropout']
    )
    ckpt = torch.load('checkpoints/selective_net/best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    models['SelectiveNet'] = {'model': model, 'type': 'selective', 'config': cfg}
    
    # --- EDL ---
    with open('configs/evidential_config.yaml') as f:
        cfg = yaml.safe_load(f)
    backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
    model = EDLModel(backbone, num_classes=cfg['model']['classifier']['num_classes'])
    ckpt = torch.load('checkpoints/evidential/best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    models['EDL'] = {'model': model, 'type': 'evidential', 'config': cfg}
    
    # --- Hybrid ---
    with open('configs/hybrid_config.yaml') as f:
        cfg = yaml.safe_load(f)
    backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
    model = HybridEvidentialModel(
        backbone=backbone,
        num_classes=cfg['model']['classifier']['num_classes'],
        dropout=cfg['model']['classifier']['dropout']
    )
    ckpt = torch.load('checkpoints/hybrid/best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    models['Hybrid'] = {'model': model, 'type': 'evidential', 'config': cfg}
    
    return models


def get_predictions(model, model_type, images, device):
    """Get predictions, confidence, and uncertainty from a model."""
    images = images.to(device)
    
    with torch.no_grad():
        if model_type == 'softmax':
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confidence = probs.max(dim=1).values
            uncertainty = 1.0 - confidence
            return preds.cpu(), confidence.cpu(), uncertainty.cpu()
            
        elif model_type == 'selective':
            logits, selection, _ = model(images, return_selection=True, return_auxiliary=True)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confidence = selection
            uncertainty = 1.0 - selection
            return preds.cpu(), confidence.cpu(), uncertainty.cpu()
            
        elif model_type == 'evidential':
            alpha = model(images)
            S = alpha.sum(dim=1, keepdim=True)
            probs = alpha / S
            preds = probs.argmax(dim=1)
            epi_unc, _, _ = compute_uncertainty(alpha)
            confidence = 1.0 - epi_unc
            return preds.cpu(), confidence.cpu(), epi_unc.cpu()


def analyze():
    parser = argparse.ArgumentParser(description='Analyze Error Consistency')
    parser.add_argument('--output-dir', type=str, default='results/error_analysis')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    with open('configs/data_config.yaml') as f:
        data_cfg = yaml.safe_load(f)
    
    dm = MCIDataModule(
        metadata_csv='data_metadata_adni.csv',
        preprocessor_config=data_cfg['preprocessing'],
        batch_size=4,
        num_workers=0,
        augmentation_config=data_cfg
    )
    test_loader = dm.test_dataloader()
    
    # Load models
    print("Loading models...")
    models = load_all_models(device)
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    
    # Collect predictions for all samples
    all_results = []
    sample_idx = 0
    
    for images, labels, metadata in tqdm(test_loader, desc="Analyzing"):
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            single_image = images[i:i+1]
            label = labels[i].item()
            
            result = {
                'sample_idx': sample_idx,
                'true_label': label,
                'true_class': 'CN' if label == 0 else 'MCI',
            }
            
            n_wrong = 0
            for model_name, model_data in models.items():
                preds, conf, unc = get_predictions(
                    model_data['model'], model_data['type'],
                    single_image, device
                )
                
                pred = preds[0].item()
                is_correct = (pred == label)
                
                result[f'{model_name}_pred'] = pred
                result[f'{model_name}_correct'] = is_correct
                result[f'{model_name}_confidence'] = conf[0].item()
                result[f'{model_name}_uncertainty'] = unc[0].item()
                
                if not is_correct:
                    n_wrong += 1
            
            result['n_models_wrong'] = n_wrong
            result['is_hard_case'] = (n_wrong >= 3)  # 3+ models fail
            
            all_results.append(result)
            sample_idx += 1
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'error_analysis.csv', index=False)
    print(f"\n✅ Error analysis saved: {output_dir / 'error_analysis.csv'}")
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("  ERROR CONSISTENCY ANALYSIS")
    print("="*60)
    
    total = len(df)
    hard_cases = df[df['is_hard_case']]
    
    print(f"\nTotal test samples: {total}")
    print(f"Hard cases (≥3 models wrong): {len(hard_cases)} ({len(hard_cases)/total:.1%})")
    
    # Per-model accuracy
    print("\n--- Per-model accuracy ---")
    for model_name in models.keys():
        acc = df[f'{model_name}_correct'].mean()
        print(f"  {model_name}: {acc:.4f}")
    
    # Hard cases analysis
    if len(hard_cases) > 0:
        print(f"\n--- Hard cases (n={len(hard_cases)}) ---")
        print(f"  CN (healthy): {(hard_cases['true_label'] == 0).sum()}")
        print(f"  MCI (impaired): {(hard_cases['true_label'] == 1).sum()}")
        
        # Key question: Does EDL flag high uncertainty for hard cases?
        print("\n--- Uncertainty on hard vs easy cases ---")
        for model_name in ['EDL', 'Hybrid']:
            if f'{model_name}_uncertainty' in df.columns:
                hard_unc = hard_cases[f'{model_name}_uncertainty'].mean()
                easy_unc = df[~df['is_hard_case']][f'{model_name}_uncertainty'].mean()
                ratio = hard_unc / max(easy_unc, 1e-6)
                print(f"  {model_name}: Hard={hard_unc:.4f}, Easy={easy_unc:.4f}, Ratio={ratio:.2f}x")
                
                if ratio > 1.5:
                    print(f"    ✅ {model_name} correctly signals HIGHER uncertainty for hard cases")
                else:
                    print(f"    ⚠️  {model_name} does NOT differentiate hard cases well")
    
    # === VISUALIZATIONS ===
    
    # 1. Scatter: Uncertainty vs Correctness
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Uncertainty vs Prediction Correctness', fontsize=14, fontweight='bold')
    
    for idx, model_name in enumerate(['EDL', 'Hybrid']):
        if f'{model_name}_uncertainty' not in df.columns:
            continue
        
        ax = axes[idx]
        
        correct_mask = df[f'{model_name}_correct']
        
        ax.scatter(
            df[correct_mask][f'{model_name}_uncertainty'],
            df[correct_mask][f'{model_name}_confidence'],
            c='green', alpha=0.6, label='Correct', s=60
        )
        ax.scatter(
            df[~correct_mask][f'{model_name}_uncertainty'],
            df[~correct_mask][f'{model_name}_confidence'],
            c='red', alpha=0.6, label='Incorrect', s=60, marker='x'
        )
        
        ax.set_xlabel('Epistemic Uncertainty')
        ax.set_ylabel('Confidence')
        ax.set_title(f'{model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'uncertainty_vs_correctness.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'uncertainty_vs_correctness.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Plot saved: {output_dir / 'uncertainty_vs_correctness.png'}")
    
    # 2. Model agreement heatmap
    agreement_matrix = np.zeros((len(models), len(models)))
    model_names = list(models.keys())
    
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            agreement = (df[f'{m1}_pred'] == df[f'{m2}_pred']).mean()
            agreement_matrix[i, j] = agreement
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        agreement_matrix, annot=True, fmt='.2f',
        xticklabels=model_names, yticklabels=model_names,
        cmap='YlOrRd', vmin=0.5, vmax=1.0, ax=ax
    )
    ax.set_title('Model Agreement Matrix\n(Fraction of samples with same prediction)')
    plt.tight_layout()
    fig.savefig(output_dir / 'model_agreement_matrix.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'model_agreement_matrix.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Plot saved: {output_dir / 'model_agreement_matrix.png'}")
    
    # 3. Hard cases detail table
    if len(hard_cases) > 0:
        detail_cols = ['sample_idx', 'true_class', 'n_models_wrong']
        for m in models.keys():
            detail_cols.extend([f'{m}_pred', f'{m}_uncertainty'])
        
        hard_detail = hard_cases[detail_cols].round(4)
        hard_detail.to_csv(output_dir / 'hard_cases_detail.csv', index=False)
        print(f"✅ Hard cases detail: {output_dir / 'hard_cases_detail.csv'}")
    
    print(f"\n{'='*60}")
    print("  ✅ ERROR ANALYSIS COMPLETE")
    print(f"  Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    analyze()
