#!/usr/bin/env python3
"""
Kompletny skrypt ewaluacyjny: porównanie wszystkich modeli na zbiorze testowym.

Generuje:
1. Tabelę wyników (Accuracy, F1, AUC, AUGRC, Sensitivity@95%Spec)
2. Krzywe Risk-Coverage
3. Krzywe ROC z punktem 95% specificity
4. Macierze konfuzji
5. Histogramy niepewności (dla modeli ewidencyjnych)
6. Case studies (analiza trudnych przypadków)
"""

import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MCIDataModule
from src.models.backbone import get_backbone, is_monai_state_dict
from src.models.baseline_softmax import BaselineSoftmaxModel
from src.models.selective_net import SelectiveNet
from src.models.hybrid_model import HybridEvidentialModel
from src.models.edl_model import EDLModel
from src.models.evidential_layer import compute_uncertainty
from src.training.eval_utils import load_evaluation_config, create_metrics_tracker
from src.training.optimizations import get_optimized_device
from src.evaluation.metrics import (
    fit_threshold_at_specificity,
    metrics_with_fixed_threshold,
    compute_metrics_at_specificity,
    compute_val_to_test_metrics_at_specs,
)
from src.evaluation.calibration import create_calibrator, calibrate_probabilities
from src.evaluation.abstention import (
    fit_coverage_threshold,
    apply_abstention_mask,
    get_abstention_scores,
    fit_threshold_youden,
    fit_threshold_max_f1,
    bootstrap_metric_ci,
    bootstrap_sensitivity_at_threshold,
    bootstrap_specificity_at_threshold,
    bootstrap_auc,
)

from src.visualization.plot_curves import (
    plot_risk_coverage_comparison,
    plot_roc_curves_comparison,
    plot_confusion_matrices,
)
from src.visualization.uncertainty_plots import (
    plot_uncertainty_histograms,
    plot_uncertainty_scatter,
    plot_uncertainty_vs_evidence,
)
from src.visualization.case_studies import (
    generate_case_studies,
    find_interesting_cases,
)


def _backbone_label(cfg: dict, state_dict: dict) -> str:
    bb = cfg.get('model', {}).get('backbone', {})
    pre = bb.get('pretrained', 'medicalnet')
    return f"MONAI+{pre}" if pre else "MONAI"


def load_model(m_cfg, cfg, device):
    """Załaduj model z checkpointu."""
    # Szukaj checkpointu (obsługa .pt i .pth)
    ckpt_dir = Path(cfg['checkpoint']['dir'])
    ckpt_path = None
    for ext in ['best_model.pt', 'best_model.pth']:
        candidate = ckpt_dir / ext
        if candidate.exists():
            ckpt_path = candidate
            break

    if ckpt_path is None:
        print(f"  ⚠️ Checkpoint for {m_cfg['name']} not found in {ckpt_dir}")
        return None, None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    state_keys = list(state_dict.keys())

    if 'config' in checkpoint and checkpoint['config']:
        saved_bb = checkpoint['config'].get('model', {}).get('backbone', {})
        if saved_bb:
            cfg = dict(cfg)
            cfg['model'] = dict(cfg.get('model', {}))
            cfg['model']['backbone'] = {**cfg['model'].get('backbone', {}), **saved_bb}
    elif is_monai_state_dict(state_dict):
        cfg = dict(cfg)
        cfg['model'] = dict(cfg.get('model', {}))
        cfg['model']['backbone'] = {
            **cfg['model'].get('backbone', {}),
            'type': 'monai',
            'use_3d': True,
            'arch_3d': 'resnet10',
        }

    has_dropout = any('dropout' in k for k in state_keys)
    has_evidential_head = any('evidential_head' in k for k in state_keys)

    backbone = get_backbone(cfg['model']['backbone'], force_3d=True)

    if m_cfg['type'] == 'baseline':
        model = BaselineSoftmaxModel(backbone, num_classes=2).to(device)
    elif m_cfg['type'] == 'selectivenet':
        model = SelectiveNet(backbone, num_classes=2).to(device)
    elif m_cfg['type'] == 'evidential':
        if has_dropout and has_evidential_head:
            model = HybridEvidentialModel(backbone, num_classes=2).to(device)
        else:
            model = EDLModel(backbone, num_classes=2).to(device)
    elif m_cfg['type'] == 'hybrid':
        model = HybridEvidentialModel(backbone, num_classes=2).to(device)
    else:
        raise ValueError(f"Nieznany typ modelu: {m_cfg['type']}")

    model.load_state_dict(state_dict)
    model.eval()
    bb_label = _backbone_label(cfg, state_dict)
    print(f"  ✅ Załadowano: {ckpt_path} (3D, {bb_label})")

    return model, ckpt_path, bb_label


def _run_inference(model, m_cfg, loader, device, model_cfg=None, split_name='test', eval_cfg=None):
    """Collect raw argmax predictions; abstention applied later in evaluate_model."""
    eval_cfg = eval_cfg or {}
    hybrid_cfg = eval_cfg.get('hybrid', {}).get('dual_gate', {})
    hybrid_dual_gate = m_cfg['type'] == 'hybrid' and hybrid_cfg.get('enabled', False)
    strength_min = float(hybrid_cfg.get('strength_min', 2.0))
    preds_list, labels_list, confs_list, probs_list, logits_list = [], [], [], [], []
    epistemic_list, aleatoric_list, strength_list, selection_list = [], [], [], []
    metadata = []

    with torch.no_grad():
        for images, labels, batch_meta in tqdm(
            loader, desc=f"  {m_cfg['name']} [{split_name}]"
        ):
            images, labels = images.to(device), labels.to(device)

            if m_cfg['type'] == 'baseline':
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                confidences, preds = torch.max(probs, dim=1)
                logits_list.append(logits.cpu().numpy())

            elif m_cfg['type'] == 'selectivenet':
                pred_logits, selection_probs = model(images, return_selection=True)
                probs = torch.softmax(pred_logits, dim=1)
                confidences, preds = torch.max(probs, dim=1)
                selection_list.append(selection_probs.cpu().numpy())
                logits_list.append(pred_logits.cpu().numpy())

            elif m_cfg['type'] in ['evidential', 'hybrid']:
                alpha = model(images)
                strength = alpha.sum(dim=1, keepdim=True)
                probs = alpha / strength
                preds = torch.argmax(probs, dim=1)
                epistemic_unc, aleatoric_unc, _ = compute_uncertainty(alpha)
                if hybrid_dual_gate:
                    confidences = HybridEvidentialModel.composite_abstention_score(
                        epistemic_unc,
                        strength.squeeze(1),
                        strength_min=strength_min,
                    )
                else:
                    confidences = 1.0 - epistemic_unc
                logits_list.append(torch.log(probs.clamp(min=1e-8)).cpu().numpy())
                epistemic_list.append(epistemic_unc.cpu().numpy())
                aleatoric_list.append(aleatoric_unc.cpu().numpy())
                strength_list.append(strength.squeeze(1).cpu().numpy())
            else:
                raise ValueError(f"Unknown type: {m_cfg['type']}")

            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            confs_list.append(confidences.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            metadata.extend(batch_meta)

    out = {
        'predictions': np.concatenate(preds_list),
        'labels': np.concatenate(labels_list),
        'confidences': np.concatenate(confs_list),
        'probabilities': np.concatenate(probs_list),
        'logits': np.concatenate(logits_list),
        'metadata': metadata,
    }
    if epistemic_list:
        out['epistemic'] = np.concatenate(epistemic_list)
        out['aleatoric'] = np.concatenate(aleatoric_list)
        out['strength'] = np.concatenate(strength_list)
    if selection_list:
        out['selection_probs'] = np.concatenate(selection_list)
    return out


def _tracker_metrics(preds, labels, confidences, probs, eval_cfg):
    tracker = create_metrics_tracker(eval_cfg, num_classes=2)
    tracker.update(
        torch.tensor(preds),
        torch.tensor(labels),
        confidences=torch.tensor(confidences),
        probabilities=torch.tensor(probs),
    )
    return tracker.compute_all_metrics()


def evaluate_model(
    model, m_cfg, val_loader, test_loader, device, eval_cfg, model_cfg=None,
    abstention_coverage_override=None,
):
    """Evaluate on val+test with calibration, abstention val→test, and multi-threshold."""
    target_spec = eval_cfg.get('target_specificity', 0.80)
    positive_class = eval_cfg.get('positive_class', 1)
    protocol = eval_cfg.get('threshold_protocol', 'val_to_test')
    cal_cfg = eval_cfg.get('calibration', {})
    cal_enabled = cal_cfg.get('enabled', False)
    cal_method = cal_cfg.get('method', 'temperature')
    abst_cfg = eval_cfg.get('abstention', {})
    boot_cfg = eval_cfg.get('bootstrap', {})

    val_raw = _run_inference(model, m_cfg, val_loader, device, model_cfg, 'val', eval_cfg)
    test_raw = _run_inference(model, m_cfg, test_loader, device, model_cfg, 'test', eval_cfg)

    val_probs, test_probs = val_raw['probabilities'], test_raw['probabilities']
    if cal_enabled and cal_method not in (None, 'none'):
        calibrator = create_calibrator(cal_method)
        val_probs, test_probs = calibrate_probabilities(
            calibrator,
            val_raw.get('logits'),
            val_raw['probabilities'],
            val_raw['labels'],
            test_raw.get('logits'),
            test_raw['probabilities'],
            cal_method,
        )

    test_preds_no_abst = test_raw['predictions'].copy()
    metrics_no_abstention = _tracker_metrics(
        test_preds_no_abst, test_raw['labels'],
        test_raw['confidences'], test_probs, eval_cfg,
    )

    test_preds = test_preds_no_abst.copy()
    abst_threshold = None
    target_coverage = abstention_coverage_override or abst_cfg.get('target_coverage', 0.80)
    if abst_cfg.get('enabled', False):
        unc_type = abst_cfg.get('uncertainty_type', 'epistemic')
        val_scores, higher = get_abstention_scores(val_raw, m_cfg['type'], unc_type)
        test_scores, _ = get_abstention_scores(test_raw, m_cfg['type'], unc_type)
        abst_threshold = fit_coverage_threshold(val_scores, target_coverage, higher)
        test_preds = apply_abstention_mask(
            test_preds, test_scores, abst_threshold, higher_is_better=higher,
        )

    metrics_with_abstention = _tracker_metrics(
        test_preds, test_raw['labels'],
        test_raw['confidences'], test_probs, eval_cfg,
    )
    metrics = metrics_with_abstention
    metrics['metrics_no_abstention'] = metrics_no_abstention
    metrics['metrics_with_abstention'] = metrics_with_abstention
    metrics['abstention_threshold'] = abst_threshold
    metrics['target_coverage'] = target_coverage

    val_thresh_metrics = {}
    test_valthresh_metrics = {}
    threshold = None
    if protocol == 'val_to_test':
        threshold = fit_threshold_at_specificity(
            val_raw['labels'], val_probs[:, 1], target_spec, positive_class
        )
        if np.isfinite(threshold):
            val_thresh_metrics = metrics_with_fixed_threshold(
                val_raw['labels'], val_probs[:, 1], threshold, positive_class
            )
            test_valthresh_metrics = metrics_with_fixed_threshold(
                test_raw['labels'], test_probs[:, 1], threshold, positive_class
            )
        metrics['val_threshold'] = threshold
        metrics['metrics_val_threshold_on_val'] = val_thresh_metrics
        metrics['metrics_val_threshold_on_test'] = test_valthresh_metrics

    report_specs = eval_cfg.get('report_specificities', [0.70, 0.80, 0.90, 0.95, 1.0])
    if protocol == 'val_to_test':
        metrics['val_to_test_at_specs'] = compute_val_to_test_metrics_at_specs(
            val_raw['labels'],
            val_probs[:, 1],
            test_raw['labels'],
            test_probs[:, 1],
            report_specs,
            positive_class,
        )

    metrics['threshold_strategies'] = {}
    for strategy in eval_cfg.get('threshold_strategies', ['fixed_specificity']):
        if strategy == 'fixed_specificity':
            t = fit_threshold_at_specificity(
                val_raw['labels'], val_probs[:, 1], target_spec, positive_class
            )
        elif strategy == 'youden':
            t = fit_threshold_youden(val_raw['labels'], val_probs[:, 1])
        elif strategy == 'max_f1':
            t = fit_threshold_max_f1(val_raw['labels'], val_probs[:, 1])
        else:
            continue
        if np.isfinite(t):
            metrics['threshold_strategies'][strategy] = {
                'threshold': float(t),
                'val': metrics_with_fixed_threshold(
                    val_raw['labels'], val_probs[:, 1], t, positive_class
                ),
                'test': metrics_with_fixed_threshold(
                    test_raw['labels'], test_probs[:, 1], t, positive_class
                ),
            }

    if cal_enabled:
        metrics['metrics_at_target_spec_calibrated'] = compute_metrics_at_specificity(
            test_raw['labels'], test_probs[:, 1], target_spec, positive_class
        )

    metrics['bootstrap_ci'] = {}
    if boot_cfg.get('enabled', False) and threshold is not None and np.isfinite(threshold):
        n_boot = boot_cfg.get('n_iterations', 1000)
        labels_t, probs_t = test_raw['labels'], test_probs[:, 1]
        for mname in boot_cfg.get('metrics', ['sensitivity', 'specificity', 'auc']):
            if mname == 'sensitivity':
                fn = lambda y, p, th=threshold, pc=positive_class: bootstrap_sensitivity_at_threshold(y, p, th, pc)
            elif mname == 'specificity':
                fn = lambda y, p, th=threshold, pc=positive_class: bootstrap_specificity_at_threshold(y, p, th, pc)
            elif mname == 'auc':
                fn = bootstrap_auc
            else:
                continue
            mean, lo, hi = bootstrap_metric_ci(labels_t, probs_t, fn, n_boot=n_boot)
            metrics['bootstrap_ci'][mname] = {'mean': mean, 'ci_low': lo, 'ci_high': hi}

    result = {
        'predictions': test_preds,
        'predictions_no_abstention': test_preds_no_abst,
        'labels': test_raw['labels'],
        'confidences': test_raw['confidences'],
        'probabilities': test_probs,
        'probabilities_raw': test_raw['probabilities'],
        'metrics': metrics,
        'metadata': test_raw['metadata'],
        'val_raw': val_raw,
        'test_raw': test_raw,
    }
    if 'epistemic' in test_raw:
        result['epistemic'] = test_raw['epistemic']
        result['aleatoric'] = test_raw['aleatoric']
        result['strength'] = test_raw['strength']
    if 'selection_probs' in test_raw:
        result['selection_probs'] = test_raw['selection_probs']
    return result


def generate_results_table(
    all_results: dict,
    target_spec: float = 0.80,
    report_specs: list = None,
) -> pd.DataFrame:
    """Generuje tabelę podsumowującą wyniki wszystkich modeli."""
    report_specs = report_specs or [0.80, 0.90, 1.0]
    rows = []
    for model_name, data in all_results.items():
        m = data['metrics']
        ms = m.get('metrics_at_target_spec', {})
        ms_cal = m.get('metrics_at_target_spec_calibrated', {})
        ms_vt = m.get('metrics_val_threshold_on_test', {})

        fp_red = m.get('fp_reduction', {}).get('abstention_20pct', {})
        boot = m.get('bootstrap_ci', {}).get('sensitivity', {})
        sens_ci = ''
        if boot:
            sens_ci = f"{boot.get('ci_low', 0):.3f}-{boot.get('ci_high', 0):.3f}"

        row = {
            'Model': model_name,
            'Backbone': data.get('backbone', ''),
            'Accuracy': f"{m['accuracy']:.4f}",
            'AUC-ROC': f"{m.get('auc', 0):.4f}",
            'AUGRC': f"{m.get('augrc', 0):.4f}",
            'Coverage': f"{1.0 - m.get('abstention_rate', 0):.2%}",
            f'Sens@{target_spec:.0%}Spec (raw)': f"{ms.get('sensitivity', 0):.4f}",
            f'Sens@{target_spec:.0%}Spec (cal)': f"{ms_cal.get('sensitivity', ms.get('sensitivity', 0)):.4f}",
            f'Sens@{target_spec:.0%}Spec (val→test)': f"{ms_vt.get('sensitivity', 0):.4f}",
            f'Actual Spec (val→test)': f"{ms_vt.get('actual_specificity', 0):.4f}",
            f'Sens@{target_spec:.0%}Spec (val→test) CI': sens_ci,
            'Abstention%': f"{m.get('abstention_rate', 0):.2%}",
            'FP reduction @20%': f"{fp_red.get('fp_reduction_rate', 0):.2%}",
        }
        for spec in report_specs:
            key = f"sens_at_{int(spec * 100)}spec"
            if key in m:
                row[f'Sens@{spec:.0%}Spec (test ROC)'] = f"{m[key]:.4f}"
            spec_key = f"spec_{int(round(spec * 100))}"
            vt_specs = m.get('val_to_test_at_specs', {})
            if spec_key in vt_specs:
                vt = vt_specs[spec_key]
                row[f'Sens@{spec:.0%}Spec (val→test)'] = f"{vt.get('sensitivity', 0):.4f}"
                row[f'FP@{spec:.0%}Spec (val→test)'] = str(vt.get('fp', 0))
                row[f'ActualSpec@{spec:.0%} (val→test)'] = f"{vt.get('actual_specificity', 0):.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate():
    """Główna funkcja ewaluacyjna."""
    print("=" * 70)
    print("  FINALNA EWALUACJA MODELI – ZBIÓR TESTOWY ADNI  ")
    print("=" * 70)

    device = get_optimized_device('cuda')
    print(f"  Device: {device}")

    # ── Konfiguracja danych i ewaluacji ──────────────────────────────────
    with open('configs/data_config.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)

    eval_cfg = load_evaluation_config()
    target_spec = eval_cfg.get('target_specificity', 0.80)
    report_specs = eval_cfg.get('report_specificities', [0.80, 0.90, 1.0])
    print(f"  Target Specificity: {target_spec:.2%}")
    print(f"  Threshold protocol: {eval_cfg.get('threshold_protocol', 'val_to_test')}")
    print(f"  Calibration: {eval_cfg.get('calibration', {})}")

    dm = MCIDataModule(
        metadata_csv=data_cfg['paths']['metadata_csv'],
        preprocessor_config=data_cfg['preprocessing'],
        batch_size=data_cfg['dataloader']['batch_size'],
        num_workers=data_cfg['dataloader']['num_workers']
    )
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    model_configs = [
        {'name': 'Baseline (SR)', 'config': 'configs/baseline_config.yaml', 'type': 'baseline'},
        {'name': 'SelectiveNet', 'config': 'configs/selectivenet_config.yaml', 'type': 'selectivenet'},
        {'name': 'Evidential (EDL)', 'config': 'configs/evidential_config.yaml', 'type': 'evidential'},
        {'name': 'Hybrid (3D-ResNet-EDL)', 'config': 'configs/hybrid_config.yaml', 'type': 'hybrid'},
    ]

    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── ETAP 1: Ewaluacja wszystkich modeli ──────────────────────────────
    print("\n" + "─" * 70)
    print("  ETAP 1: Ewaluacja modeli")
    print("─" * 70)

    for m_cfg in model_configs:
        cfg_path = Path(m_cfg['config'])
        if not cfg_path.exists():
            print(f"  ⏩ Pomijam {m_cfg['name']} (brak configu: {cfg_path})")
            continue

        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        print(f"\n📊 {m_cfg['name']}:")
        loaded = load_model(m_cfg, cfg, device)
        if loaded[0] is None:
            continue
        model, ckpt_path, bb_label = loaded

        result = evaluate_model(
            model, m_cfg, val_loader, test_loader, device, eval_cfg, model_cfg=cfg
        )
        result['backbone'] = bb_label
        result['model_type'] = m_cfg['type']
        all_results[m_cfg['name']] = result

        m = result['metrics']
        ms = m.get('metrics_at_target_spec', {})
        ms_vt = m.get('metrics_val_threshold_on_test', {})
        print(f"     Accuracy:  {m['accuracy']:.4f}")
        print(f"     F1:        {m['f1']:.4f}")
        print(f"     AUC:       {m.get('auc', 0):.4f}")
        print(f"     AUGRC:     {m.get('augrc', 0):.4f}")
        print(f"     Abstention: {m.get('abstention_rate', 0):.2%}  Coverage: {1-m.get('abstention_rate',0):.2%}")
        print(f"     --- @ {target_spec*100:.0f}% Spec (raw test ROC) ---")
        print(f"     Sensitivity: {ms.get('sensitivity', 0):.4f}")
        print(f"     --- val→test threshold ---")
        print(f"     Sensitivity: {ms_vt.get('sensitivity', 0):.4f} (Spec={ms_vt.get('actual_specificity', 0):.4f})")

        # Zwolnij pamięć GPU
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not all_results:
        print("\n❌ Brak modeli do ewaluacji. Sprawdź checkpointy.")
        return

    # ── ETAP 2: Tabela wyników ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  ETAP 2: Tabela wyników")
    print("─" * 70)

    results_df = generate_results_table(
        all_results, target_spec=target_spec, report_specs=report_specs
    )
    csv_path = results_dir / 'final_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n{results_df.to_string(index=False)}")
    print(f"\n✅ Tabela zapisana: {csv_path}")

    abst_rows = []
    for model_name, data in all_results.items():
        m_abst = data['metrics']
        m_no = m_abst.get('metrics_no_abstention', m_abst)
        for mode, m in [('no_abstention', m_no), ('with_abstention', m_abst)]:
            ms_vt = m.get('metrics_val_threshold_on_test', {})
            abst_rows.append({
                'Model': model_name,
                'Mode': mode,
                'AUC': f"{m.get('auc', 0):.4f}",
                'AUGRC': f"{m.get('augrc', 0):.4f}",
                'Abstention%': f"{m.get('abstention_rate', 0):.2%}",
                f'Sens@{target_spec:.0%}Spec (val→test)': f"{ms_vt.get('sensitivity', 0):.4f}",
            })
    abst_df = pd.DataFrame(abst_rows)
    abst_csv = results_dir / 'comparison_with_abstention.csv'
    abst_df.to_csv(abst_csv, index=False)
    print(f"\n✅ Tabela abstencji: {abst_csv}")

    # ── ETAP 3: Wizualizacja krzywych ────────────────────────────────────
    print("\n" + "─" * 70)
    print("  ETAP 3: Wizualizacja krzywych")
    print("─" * 70)

    # 3a. Risk-Coverage
    rc_data = {}
    for name, data in all_results.items():
        m = data['metrics']
        if 'risk_coverage' in m:
            rc_data[name] = {
                'coverages': m['risk_coverage']['coverages'],
                'risks': m['risk_coverage']['risks'],
                'augrc': m.get('augrc', 0),
            }
    if rc_data:
        plot_risk_coverage_comparison(rc_data, results_dir / 'risk_coverage_comparison.png')

    # 3b. ROC
    roc_data = {}
    for name, data in all_results.items():
        probs = data['probabilities']
        probs_pos = probs[:, 1] if probs.ndim == 2 else probs
        roc_data[name] = {
            'labels': data['labels'],
            'probabilities': probs_pos,
        }
    if roc_data:
        plot_roc_curves_comparison(
            roc_data, 
            results_dir / 'roc_curves_comparison.png',
            target_specificity=target_spec
        )

    # 3c. Macierze konfuzji (używamy punktu pracy z target specificity)
    cm_data = {}
    for name, data in all_results.items():
        m = data['metrics']
        ms = m.get('metrics_at_target_spec', {})
        threshold = ms.get('threshold', 0.5)
        
        probs = data['probabilities']
        probs_pos = probs[:, 1] if probs.ndim == 2 else probs
        
        # Wyznacz predykcje dla ustalonego punktu pracy
        preds_at_spec = (probs_pos >= threshold).astype(int)
        
        # Jeśli model miał abstencje w oryginalnych predykcjach, zachowaj je?
        # Tutaj lepiej pokazać pełną macierz dla punktu pracy bez abstencji 
        # LUB zintegrować to. Przyjmijmy predykcje dla ustalonego progu.
        
        cm_data[name] = {
            'predictions': preds_at_spec,
            'labels': data['labels'],
        }
    if cm_data:
        plot_confusion_matrices(cm_data, results_dir / 'confusion_matrices_at_spec.png')

    # ── ETAP 4: Histogramy niepewności ────────────────────────────────────
    print("\n" + "─" * 70)
    print("  ETAP 4: Histogramy niepewności")
    print("─" * 70)

    for name, data in all_results.items():
        if 'epistemic' in data:
            print(f"\n📊 Generowanie histogramów dla: {name}")

            plot_uncertainty_histograms(
                data['predictions'], data['labels'],
                data['epistemic'], data['aleatoric'],
                model_name=name, output_dir=results_dir,
            )
            plot_uncertainty_scatter(
                data['predictions'], data['labels'],
                data['epistemic'], data['aleatoric'],
                model_name=name,
                output_path=results_dir / f'uncertainty_scatter_{name.replace(" ", "_").lower()}.png',
            )
            if 'strength' in data:
                plot_uncertainty_vs_evidence(
                    data['predictions'], data['labels'],
                    data['strength'],
                    model_name=name,
                    output_path=results_dir / f'evidence_strength_{name.replace(" ", "_").lower()}.png',
                )

    # ── ETAP 5: Case Studies ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  ETAP 5: Case Studies")
    print("─" * 70)

    # Przygotuj dane do identyfikacji interesujących przypadków
    pred_dict = {name: data['predictions'] for name, data in all_results.items()}
    conf_dict = {name: data['confidences'] for name, data in all_results.items()}
    unc_dict = {name: data.get('epistemic', None) for name, data in all_results.items()}
    unc_dict = {k: v for k, v in unc_dict.items() if v is not None}

    # Użyj etykiet z pierwszego modelu (identyczne)
    first_labels = list(all_results.values())[0]['labels']

    # Znajdź interesujące przypadki
    interesting = find_interesting_cases(
        pred_dict, first_labels, conf_dict, unc_dict, max_cases=5
    )

    if interesting:
        print(f"\n  Znaleziono {len(interesting)} interesujących przypadków:")
        for ic in interesting:
            print(f"    Index {ic['index']}: {ic['case_type']}")

        # Załaduj woluminy dla case studies
        test_dataset = dm.test_dataset()
        case_data_list = []

        for ic in interesting:
            idx = ic['index']
            try:
                image, label, meta = test_dataset[idx]
                volume = image.numpy()  # (1, D, H, W)

                case_models = {}
                for model_name, data in all_results.items():
                    pred = int(data['predictions'][idx])
                    conf = float(data['confidences'][idx])
                    is_abs = pred == -1
                    unc_val = float(data['epistemic'][idx]) if 'epistemic' in data else None

                    case_models[model_name] = {
                        'prediction': pred,
                        'confidence': conf,
                        'is_abstained': is_abs,
                        'uncertainty': unc_val,
                    }

                patient_id = meta.get('path', f'Sample #{idx}')
                # Skróć ścieżkę do nazwy pliku
                patient_id = Path(patient_id).stem if '/' in str(patient_id) else patient_id

                case_data_list.append({
                    'patient_id': patient_id,
                    'true_label': int(label),
                    'volume': volume,
                    'models': case_models,
                    'case_type': ic['case_type'],
                })
            except Exception as e:
                print(f"    ⚠️ Nie udało się załadować próbki {idx}: {e}")

        if case_data_list:
            generate_case_studies(case_data_list, results_dir / 'case_studies')
    else:
        print("  ℹ️ Nie znaleziono interesujących przypadków do case studies.")

    # ── ETAP 6: Raport FP vs Coverage ────────────────────────────────────
    print("\n" + "─" * 70)
    print("  ETAP 6: Raport FP vs Coverage")
    print("─" * 70)
    from scripts.generate_abstention_report import generate_abstention_report
    generate_abstention_report(
        all_results=all_results,
        eval_cfg=eval_cfg,
        results_dir=results_dir,
        target_spec=target_spec,
    )

    # ── Podsumowanie ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅ EWALUACJA ZAKOŃCZONA")
    print("=" * 70)
    print(f"\n  Wyniki zapisane w: {results_dir.resolve()}")

    # Lista wygenerowanych plików
    generated = list(results_dir.glob('**/*'))
    generated = [f for f in generated if f.is_file()]
    print(f"  Wygenerowane pliki ({len(generated)}):")
    for f in sorted(generated):
        size_kb = f.stat().st_size / 1024
        print(f"    📄 {f.relative_to(results_dir)} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    evaluate()
