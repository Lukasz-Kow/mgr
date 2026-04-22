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
from src.models.backbone import get_backbone, ResNetBackbone2D
from src.models.baseline_softmax import BaselineSoftmaxModel
from src.models.selective_net import SelectiveNet
from src.models.hybrid_model import HybridEvidentialModel
from src.models.evidential_layer import EvidentialLayer, compute_uncertainty
from src.evaluation.metrics import MetricsTracker

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

    # Załaduj checkpoint i sprawdź klucze state_dict
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    state_keys = list(state_dict.keys())

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
    print(f"  ✅ Załadowano: {ckpt_path} (backbone: {arch_label})")

    return model, ckpt_path


def evaluate_model(model, m_cfg, test_loader, device):
    """
    Ewaluuje jeden model na zbiorze testowym.

    Returns:
        Dict z wynikami: predictions, labels, confidences, probabilities,
        metrics, uncertainty_data (dla modeli ewidencyjnych), metadata
    """
    tracker = MetricsTracker(num_classes=2)

    all_preds = []
    all_labels = []
    all_confs = []
    all_probs = []
    all_metadata = []

    # Dane specyficzne dla EDL
    all_epistemic = []
    all_aleatoric = []
    all_strength = []

    with torch.no_grad():
        for images, labels, metadata in tqdm(test_loader, desc=f"  Evaluating {m_cfg['name']}"):
            images, labels = images.to(device), labels.to(device)

            if m_cfg['type'] == 'baseline':
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                confidences, preds = torch.max(probs, dim=1)
                tracker.update(preds, labels, confidences=confidences, probabilities=probs)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_confs.append(confidences.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

            elif m_cfg['type'] == 'selectivenet':
                pred_logits, selection_probs = model(images, return_selection=True)
                probs = torch.softmax(pred_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                tracker.update(preds, labels, confidences=selection_probs, probabilities=probs)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_confs.append(selection_probs.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

            elif m_cfg['type'] in ['evidential', 'hybrid']:
                alpha = model(images)
                strength = alpha.sum(dim=1, keepdim=True)
                probs = alpha / strength
                preds = torch.argmax(probs, dim=1)

                # Niepewności
                epistemic_unc, aleatoric_unc, total_unc = compute_uncertainty(alpha)
                confidences = 1.0 - epistemic_unc

                tracker.update(preds, labels, confidences=confidences, probabilities=probs)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_confs.append(confidences.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_epistemic.append(epistemic_unc.cpu().numpy())
                all_aleatoric.append(aleatoric_unc.cpu().numpy())
                all_strength.append(strength.squeeze(1).cpu().numpy())

            all_metadata.extend(metadata)

    # Concatenate
    result = {
        'predictions': np.concatenate(all_preds),
        'labels': np.concatenate(all_labels),
        'confidences': np.concatenate(all_confs),
        'probabilities': np.concatenate(all_probs),
        'metrics': tracker.compute_all_metrics(),
        'metadata': all_metadata,
    }

    # EDL-specific
    if all_epistemic:
        result['epistemic'] = np.concatenate(all_epistemic)
        result['aleatoric'] = np.concatenate(all_aleatoric)
        result['strength'] = np.concatenate(all_strength)

    return result


def generate_results_table(all_results: dict) -> pd.DataFrame:
    """Generuje tabelę podsumowującą wyniki wszystkich modeli."""
    rows = []
    for model_name, data in all_results.items():
        m = data['metrics']
        
        # FP Reduction at 20% abstention
        fp_red_20 = 0.0
        if 'fp_reduction' in m and 'abstention_20pct' in m['fp_reduction']:
            fp_red_20 = m['fp_reduction']['abstention_20pct']['fp_reduction_rate']
        
        rows.append({
            'Model': model_name,
            'Accuracy': f"{m['accuracy']:.4f}",
            'F1': f"{m['f1']:.4f}",
            'AUC-ROC': f"{m.get('auc', 0):.4f}",
            'AUGRC': f"{m.get('augrc', 0):.4f}",
            'Sens@80%Spec': f"{m.get('sens_at_80spec', 0):.4f}",
            'Sens@90%Spec': f"{m.get('sens_at_90spec', 0):.4f}",
            'Sens@95%Spec': f"{m.get('sensitivity_at_95spec', 0):.4f}",
            'FP_Red@20%Abs': f"{fp_red_20:.2%}",
            'Abstention%': f"{m.get('abstention_rate', 0):.2%}",
        })
    return pd.DataFrame(rows)


def evaluate():
    """Główna funkcja ewaluacyjna."""
    print("=" * 70)
    print("  FINALNA EWALUACJA MODELI – ZBIÓR TESTOWY ADNI  ")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ── Konfiguracja danych ──────────────────────────────────────────────
    with open('configs/data_config.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)

    dm = MCIDataModule(
        metadata_csv=data_cfg['paths']['metadata_csv'],
        preprocessor_config=data_cfg['preprocessing'],
        batch_size=data_cfg['dataloader']['batch_size'],
        num_workers=data_cfg['dataloader']['num_workers']
    )
    test_loader = dm.test_dataloader()

    # ── Modele do ewaluacji ──────────────────────────────────────────────
    model_configs = [
        {'name': 'Baseline (SR)',          'config': 'configs/baseline_config.yaml',     'type': 'baseline'},
        {'name': 'SelectiveNet',           'config': 'configs/selectivenet_config.yaml',  'type': 'selectivenet'},
        {'name': 'Evidential (EDL)',       'config': 'configs/evidential_config.yaml',    'type': 'evidential'},
        {'name': 'Hybrid (3D-ResNet-EDL)', 'config': 'configs/hybrid_config.yaml',       'type': 'hybrid'},
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
        model, ckpt_path = load_model(m_cfg, cfg, device)
        if model is None:
            continue

        result = evaluate_model(model, m_cfg, test_loader, device)
        all_results[m_cfg['name']] = result

        # Drukuj metryki
        m = result['metrics']
        print(f"     Accuracy:  {m['accuracy']:.4f}")
        print(f"     F1:        {m['f1']:.4f}")
        print(f"     AUC:       {m.get('auc', 0):.4f}")
        print(f"     AUGRC:     {m.get('augrc', 0):.4f}")
        print(f"     Sens@95%:  {m.get('sensitivity_at_95spec', 0):.4f}")

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

    results_df = generate_results_table(all_results)
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
        if probs.ndim == 2:
            probs_pos = probs[:, 1]
        else:
            probs_pos = probs
        roc_data[name] = {
            'labels': data['labels'],
            'probabilities': probs_pos,
        }
    if roc_data:
        plot_roc_curves_comparison(roc_data, results_dir / 'roc_curves_comparison.png')

    # 3c. Macierze konfuzji
    cm_data = {}
    for name, data in all_results.items():
        cm_data[name] = {
            'predictions': data['predictions'],
            'labels': data['labels'],
        }
    if cm_data:
        plot_confusion_matrices(cm_data, results_dir / 'confusion_matrices.png')

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
