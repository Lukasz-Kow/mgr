#!/usr/bin/env python3
"""FP vs Coverage report, high-confidence subset, and case agreement table."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MCIDataModule
from src.training.eval_utils import load_evaluation_config
from src.training.optimizations import get_optimized_device
from src.evaluation.abstention import (
    fit_coverage_threshold,
    apply_abstention_mask,
    get_abstention_scores,
)
from scripts.evaluate_all import (
    load_model,
    evaluate_model,
    _tracker_metrics,
)


COVERAGE_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
HIGH_CONF_COVERAGE = 0.70


def _metrics_at_coverage(val_raw, test_raw, model_type, eval_cfg, coverage):
    """Compute test metrics when abstaining to achieve target coverage."""
    unc_type = eval_cfg.get('abstention', {}).get('uncertainty_type', 'epistemic')
    val_scores, higher = get_abstention_scores(val_raw, model_type, unc_type)
    test_scores, _ = get_abstention_scores(test_raw, model_type, unc_type)
    tau = fit_coverage_threshold(val_scores, coverage, higher_is_better=higher)
    preds = apply_abstention_mask(
        test_raw['predictions'].copy(), test_scores, tau, higher_is_better=higher,
    )
    probs = test_raw['probabilities']
    m = _tracker_metrics(preds, test_raw['labels'], test_raw['confidences'], probs, eval_cfg)

    labels = test_raw['labels']
    fp_before = int(np.sum((test_raw['predictions'] == 1) & (labels == 0)))
    mask = preds != -1
    fp_after = int(np.sum((preds[mask] == 1) & (labels[mask] == 0))) if mask.any() else 0
    fp_red = (fp_before - fp_after) / max(fp_before, 1)

    tn, fp, fn, tp = 0, 0, 0, 0
    if mask.any():
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels[mask], preds[mask], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'Coverage': coverage,
        'FP_before': fp_before,
        'FP_after': fp_after,
        'FP_reduction%': fp_red,
        'Sensitivity': sens,
        'Specificity': spec,
        'Abstention%': m.get('abstention_rate', 0),
        'AUGRC': m.get('augrc', 0),
    }


def generate_abstention_report(
    all_results=None,
    eval_cfg=None,
    results_dir=None,
    target_spec=0.80,
):
    """Build FP@coverage tables from precomputed all_results or by loading models."""
    results_dir = Path(results_dir or 'results')
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = eval_cfg or load_evaluation_config()

    if all_results is None:
        raise ValueError("all_results required when called from evaluate_all")

    fp_rows = []
    high_conf_rows = []
    agreement_rows = []

    for model_name, data in all_results.items():
        val_raw = data.get('val_raw')
        test_raw = data.get('test_raw')
        model_type = data.get('model_type', 'baseline')
        if val_raw is None or test_raw is None:
            continue

        for cov in COVERAGE_LEVELS:
            row = _metrics_at_coverage(val_raw, test_raw, model_type, eval_cfg, cov)
            row['Model'] = model_name
            fp_rows.append(row)

        hc = _metrics_at_coverage(val_raw, test_raw, model_type, eval_cfg, HIGH_CONF_COVERAGE)
        unc_type = eval_cfg.get('abstention', {}).get('uncertainty_type', 'epistemic')
        val_scores, higher = get_abstention_scores(val_raw, model_type, unc_type)
        test_scores, _ = get_abstention_scores(test_raw, model_type, unc_type)
        tau = fit_coverage_threshold(val_scores, HIGH_CONF_COVERAGE, higher_is_better=higher)
        hc_preds = apply_abstention_mask(
            test_raw['predictions'].copy(), test_scores, tau, higher_is_better=higher,
        )
        mask = hc_preds != -1
        hc_acc = float(np.mean(hc_preds[mask] == test_raw['labels'][mask])) if mask.any() else 0.0
        high_conf_rows.append({
            'Model': model_name,
            'Coverage': HIGH_CONF_COVERAGE,
            'Accuracy': f"{hc_acc:.4f}",
            'Sensitivity': f"{hc['Sensitivity']:.4f}",
            'Specificity': f"{hc['Specificity']:.4f}",
            'FP_after': hc['FP_after'],
            'Abstention%': f"{hc['Abstention%']:.2%}",
        })

    if fp_rows:
        fp_df = pd.DataFrame(fp_rows)
        cols = ['Model', 'Coverage', 'FP_before', 'FP_after', 'FP_reduction%',
                'Sensitivity', 'Specificity', 'Abstention%', 'AUGRC']
        fp_df = fp_df[cols]
        fp_csv = results_dir / 'fp_coverage.csv'
        fp_df.to_csv(fp_csv, index=False)
        print(f"  Saved: {fp_csv}")

        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name in fp_df['Model'].unique():
            sub = fp_df[fp_df['Model'] == model_name]
            ax.plot(sub['Coverage'], sub['FP_after'], marker='o', label=model_name)
        ax.set_xlabel('Coverage')
        ax.set_ylabel('False Positives (test)')
        ax.set_title('FP vs Coverage (val→test abstention threshold)')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fp_png = results_dir / 'fp_coverage.png'
        fig.savefig(fp_png, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fp_png}")

    if high_conf_rows:
        hc_df = pd.DataFrame(high_conf_rows)
        hc_path = results_dir / 'high_confidence_subset.csv'
        hc_df.to_csv(hc_path, index=False)
        print(f"  Saved: {hc_path}")

    if all_results:
        labels = list(all_results.values())[0]['labels']
        for idx in range(len(labels)):
            row = {'Index': idx, 'TrueLabel': int(labels[idx])}
            for model_name, data in all_results.items():
                pred = int(data['predictions'][idx])
                row[f'{model_name}_pred'] = pred
                row[f'{model_name}_abstain'] = pred == -1
            agreement_rows.append(row)
        agree_df = pd.DataFrame(agreement_rows)
        agree_path = results_dir / 'case_agreement.csv'
        agree_df.to_csv(agree_path, index=False)
        print(f"  Saved: {agree_path} ({len(agree_df)} test samples)")


def main():
    eval_cfg = load_evaluation_config()
    device = get_optimized_device('cuda')
    with open('configs/data_config.yaml') as f:
        data_cfg = yaml.safe_load(f)

    dm = MCIDataModule(
        metadata_csv=data_cfg['paths']['metadata_csv'],
        preprocessor_config=data_cfg['preprocessing'],
        batch_size=data_cfg['dataloader']['batch_size'],
        num_workers=data_cfg['dataloader']['num_workers'],
    )
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    model_configs = [
        {'name': 'Baseline (SR)', 'config': 'configs/baseline_config.yaml', 'type': 'baseline'},
        {'name': 'SelectiveNet', 'config': 'configs/selectivenet_config.yaml', 'type': 'selectivenet'},
        {'name': 'Evidential (EDL)', 'config': 'configs/evidential_config.yaml', 'type': 'evidential'},
        {'name': 'Hybrid (3D-ResNet-EDL)', 'config': 'configs/hybrid_config.yaml', 'type': 'hybrid'},
    ]

    all_results = {}
    for m_cfg in model_configs:
        cfg_path = Path(m_cfg['config'])
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        loaded = load_model(m_cfg, cfg, device)
        if loaded[0] is None:
            continue
        model, _, bb_label = loaded
        result = evaluate_model(model, m_cfg, val_loader, test_loader, device, eval_cfg, cfg)
        result['backbone'] = bb_label
        result['model_type'] = m_cfg['type']
        all_results[m_cfg['name']] = result
        del model

    generate_abstention_report(all_results=all_results, eval_cfg=eval_cfg)


if __name__ == '__main__':
    main()
