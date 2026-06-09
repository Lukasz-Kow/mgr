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
from src.models.backbone import get_backbone, ResNetBackbone2D, is_monai_state_dict
from src.models.baseline_softmax import BaselineSoftmaxModel
from src.models.selective_net import SelectiveNet
from src.models.hybrid_model import HybridEvidentialModel
from src.models.evidential_layer import EvidentialLayer, compute_uncertainty
from src.training.eval_utils import load_evaluation_config, create_metrics_tracker
from src.training.optimizations import get_optimized_device
from src.evaluation.metrics import (
    fit_threshold_at_specificity,
    metrics_with_fixed_threshold,
    compute_metrics_at_specificity,
)
from src.evaluation.calibration import create_calibrator, calibrate_probabilities

import torch.nn as nn


# ── EDLModel (identyczny jak w train_evidential.py) ──────────────────────────
# Lekka klasa bez Dropout, użyta do treningu modelu Evidential (nie Hybrid).
class EDLModel(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.evidential_head = EvidentialLayer(backbone.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.evidential_head(features)

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
    bb_type = bb.get('type', 'simple')
    if bb_type == 'monai' or is_monai_state_dict(state_dict):
        pre = bb.get('pretrained', 'medicalnet')
        return f"MONAI+{pre}" if pre else "MONAI"
    return "simple CNN"


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

    # Auto-detekcja: 2D (Conv2d) vs 3D (Conv3d) backbone
    # Sprawdzamy kształt pierwszej warstwy konwolucyjnej
    is_2d_checkpoint = False
    for key in state_keys:
        if 'conv' in key and 'weight' in key:
            weight_shape = state_dict[key].shape
            if len(weight_shape) == 4:  # Conv2d: (out, in, kH, kW)
                is_2d_checkpoint = True
            elif len(weight_shape) == 5:  # Conv3d: (out, in, kD, kH, kW)
                is_2d_checkpoint = False
            break

    # Auto-detekcja: EDLModel vs HybridEvidentialModel
    # EDLModel nie ma klucza 'dropout.*' ani nie używa prefiksu 'evidential_head'
    # Natomiast HybridEvidentialModel ma 'dropout' + 'evidential_head'
    has_dropout = any('dropout' in k for k in state_keys)
    has_evidential_head = any('evidential_head' in k for k in state_keys)

    # Buduj model – architektura dopasowana do checkpointu
    if m_cfg['type'] == 'baseline':
        if is_2d_checkpoint:
            # Wykryj architekturę z kształtu wag
            arch = cfg['model']['backbone'].get('arch_2d', 'resnet18')
            # Sprawdź czy to resnet50 (feature_dim=2048) czy resnet18/34 (512)
            fc_key = [k for k in state_keys if k.endswith('fc.weight')]
            if fc_key:
                fc_shape = state_dict[fc_key[0]].shape
                if fc_shape[1] == 2048:
                    arch = 'resnet50'
                elif fc_shape[1] == 512:
                    arch = 'resnet18'  # lub resnet34
            backbone = ResNetBackbone2D(
                arch=arch,
                pretrained=False,
                in_channels=cfg['model']['backbone'].get('in_channels', 1)
            )
        else:
            backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
        model = BaselineSoftmaxModel(backbone, num_classes=2).to(device)

    elif m_cfg['type'] == 'selectivenet':
        if is_2d_checkpoint:
            arch = cfg['model']['backbone'].get('arch_2d', 'resnet18')
            backbone = ResNetBackbone2D(arch=arch, pretrained=False,
                                        in_channels=cfg['model']['backbone'].get('in_channels', 1))
        else:
            backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
        model = SelectiveNet(backbone, num_classes=2).to(device)

    elif m_cfg['type'] == 'evidential':
        if is_2d_checkpoint:
            arch = cfg['model']['backbone'].get('arch_2d', 'resnet18')
            backbone = ResNetBackbone2D(arch=arch, pretrained=False,
                                        in_channels=cfg['model']['backbone'].get('in_channels', 1))
        else:
            backbone = get_backbone(cfg['model']['backbone'], force_3d=True)

        if has_dropout and has_evidential_head:
            # Trenowany jako HybridEvidentialModel
            model = HybridEvidentialModel(backbone, num_classes=2).to(device)
        else:
            # Trenowany jako EDLModel (bez dropout)
            model = EDLModel(backbone, num_classes=2).to(device)

    elif m_cfg['type'] == 'hybrid':
        if is_2d_checkpoint:
            arch = cfg['model']['backbone'].get('arch_2d', 'resnet18')
            backbone = ResNetBackbone2D(arch=arch, pretrained=False,
                                        in_channels=cfg['model']['backbone'].get('in_channels', 1))
        else:
            backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
        model = HybridEvidentialModel(backbone, num_classes=2).to(device)

    else:
        raise ValueError(f"Nieznany typ modelu: {m_cfg['type']}")

    model.load_state_dict(state_dict)
    model.eval()
    arch_label = "2D" if is_2d_checkpoint else "3D"
    bb_label = _backbone_label(cfg, state_dict)
    print(f"  ✅ Załadowano: {ckpt_path} ({arch_label}, {bb_label})")

    return model, ckpt_path, bb_label


def _run_inference(model, m_cfg, loader, device, model_cfg=None, split_name='test'):
    """Collect predictions, probabilities and logits for a dataloader."""
    selection_threshold = 0.5
    if model_cfg and 'selective_net' in model_cfg:
        selection_threshold = model_cfg['selective_net'].get('selection_threshold', 0.5)

    preds_list, labels_list, confs_list, probs_list, logits_list = [], [], [], [], []
    epistemic_list, aleatoric_list, strength_list = [], [], []
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
                preds, _, sel_probs, _ = model.predict_with_selection(
                    images, threshold=selection_threshold
                )
                confidences = sel_probs
                logits_list.append(pred_logits.cpu().numpy())

            elif m_cfg['type'] in ['evidential', 'hybrid']:
                alpha = model(images)
                strength = alpha.sum(dim=1, keepdim=True)
                probs = alpha / strength
                preds = torch.argmax(probs, dim=1)
                epistemic_unc, aleatoric_unc, _ = compute_uncertainty(alpha)
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
    return out


def evaluate_model(model, m_cfg, val_loader, test_loader, device, eval_cfg, model_cfg=None):
    """Evaluate on val+test with optional calibration and val→test threshold."""
    target_spec = eval_cfg.get('target_specificity', 0.80)
    positive_class = eval_cfg.get('positive_class', 1)
    protocol = eval_cfg.get('threshold_protocol', 'val_to_test')
    cal_cfg = eval_cfg.get('calibration', {})
    cal_enabled = cal_cfg.get('enabled', False)
    cal_method = cal_cfg.get('method', 'temperature')

    val_raw = _run_inference(model, m_cfg, val_loader, device, model_cfg, 'val')
    test_raw = _run_inference(model, m_cfg, test_loader, device, model_cfg, 'test')

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

    tracker = create_metrics_tracker(eval_cfg, num_classes=2)
    tracker.update(
        torch.tensor(test_raw['predictions']),
        torch.tensor(test_raw['labels']),
        confidences=torch.tensor(test_raw['confidences']),
        probabilities=torch.tensor(test_probs),
    )
    metrics = tracker.compute_all_metrics()

    val_thresh_metrics = {}
    test_valthresh_metrics = {}
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

    if cal_enabled:
        metrics['metrics_at_target_spec_calibrated'] = compute_metrics_at_specificity(
            test_raw['labels'], test_probs[:, 1], target_spec, positive_class
        )

    result = {
        'predictions': test_raw['predictions'],
        'labels': test_raw['labels'],
        'confidences': test_raw['confidences'],
        'probabilities': test_probs,
        'probabilities_raw': test_raw['probabilities'],
        'metrics': metrics,
        'metadata': test_raw['metadata'],
    }
    if 'epistemic' in test_raw:
        result['epistemic'] = test_raw['epistemic']
        result['aleatoric'] = test_raw['aleatoric']
        result['strength'] = test_raw['strength']
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

        row = {
            'Model': model_name,
            'Backbone': data.get('backbone', ''),
            'Accuracy': f"{m['accuracy']:.4f}",
            'AUC-ROC': f"{m.get('auc', 0):.4f}",
            'AUGRC': f"{m.get('augrc', 0):.4f}",
            f'Sens@{target_spec:.0%}Spec (raw)': f"{ms.get('sensitivity', 0):.4f}",
            f'Sens@{target_spec:.0%}Spec (cal)': f"{ms_cal.get('sensitivity', ms.get('sensitivity', 0)):.4f}",
            f'Sens@{target_spec:.0%}Spec (val→test)': f"{ms_vt.get('sensitivity', 0):.4f}",
            'Abstention%': f"{m.get('abstention_rate', 0):.2%}",
        }
        for spec in report_specs:
            key = f"sens_at_{int(spec * 100)}spec"
            if key in m:
                row[f'Sens@{spec:.0%}Spec'] = f"{m[key]:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate():
    """Główna funkcja ewaluacyjna."""
    parser = argparse.ArgumentParser(description='Evaluate all models on test set')
    parser.add_argument(
        '--include-phase2',
        action='store_true',
        help='Include Phase 2 MONAI models (all 4 *_monai checkpoints)',
    )
    args = parser.parse_args()

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
        {'name': 'Baseline (SR)', 'config': 'configs/baseline_config_phase1.yaml', 'type': 'baseline'},
        {'name': 'SelectiveNet', 'config': 'configs/selectivenet_config_phase1.yaml', 'type': 'selectivenet'},
        {'name': 'Evidential (EDL)', 'config': 'configs/evidential_config_phase1.yaml', 'type': 'evidential'},
        {'name': 'Hybrid (3D-ResNet-EDL)', 'config': 'configs/hybrid_config_phase1.yaml', 'type': 'hybrid'},
    ]
    if args.include_phase2:
        model_configs.extend([
            {'name': 'Baseline (MONAI)', 'config': 'configs/baseline_config.yaml', 'type': 'baseline'},
            {'name': 'SelectiveNet (MONAI)', 'config': 'configs/selectivenet_config.yaml', 'type': 'selectivenet'},
            {'name': 'Evidential (MONAI)', 'config': 'configs/evidential_config.yaml', 'type': 'evidential'},
            {'name': 'Hybrid (MONAI)', 'config': 'configs/hybrid_config.yaml', 'type': 'hybrid'},
        ])

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
        all_results[m_cfg['name']] = result

        m = result['metrics']
        ms = m.get('metrics_at_target_spec', {})
        ms_vt = m.get('metrics_val_threshold_on_test', {})
        print(f"     Accuracy:  {m['accuracy']:.4f}")
        print(f"     F1:        {m['f1']:.4f}")
        print(f"     AUC:       {m.get('auc', 0):.4f}")
        print(f"     AUGRC:     {m.get('augrc', 0):.4f}")
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
