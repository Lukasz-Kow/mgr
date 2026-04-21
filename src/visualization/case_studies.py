#!/usr/bin/env python3
"""
Moduł Case Studies – analiza użyteczności klinicznej.

Generuje wizualizacje dla trudnych przypadków pacjentów,
pokazując jak różne modele radzą sobie z niejednoznacznymi diagnozami.
Kluczowe: demonstracja, że model Hybrydowy potrafi odmówić diagnozy
tam, gdzie Baseline daje False Positive.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch


def _extract_slices(
    volume: np.ndarray,
    slice_fractions: Tuple[float, ...] = (0.5,)
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Wyciąga przekroje (sagittalny, koronalny, osiowy) z woluminu 3D.

    Args:
        volume: Numpy array shape (D, H, W) lub (1, D, H, W)
        slice_fractions: Frakcje wymiarów do wycinania (0.5 = środek)

    Returns:
        Lista tuple (sagittal, coronal, axial) dla każdej frakcji
    """
    if volume.ndim == 4:
        volume = volume[0]  # Usuń wymiar kanału

    slices_list = []
    D, H, W = volume.shape

    for frac in slice_fractions:
        sagittal = volume[int(D * frac), :, :]      # Sagittalny (D)
        coronal = volume[:, int(H * frac), :]        # Koronalny (H)
        axial = volume[:, :, int(W * frac)]           # Osiowy (W)
        slices_list.append((sagittal, coronal, axial))

    return slices_list


def _format_decision(prediction, confidence, is_abstained, uncertainty=None):
    """
    Formatuje decyzję modelu do czytelnego tekstu.

    Returns:
        (text, color) – tekst decyzji i kolor ramki
    """
    if is_abstained:
        text = f"⚠️ ODMOWA DIAGNOZY\nNiepewność: {uncertainty:.1%}" if uncertainty else "⚠️ ODMOWA"
        return text, '#EA580C'  # orange

    class_name = 'MCI' if prediction == 1 else 'CN'
    text = f"Diagnoza: {class_name}\nPewność: {confidence:.1%}"
    if uncertainty is not None:
        text += f"\nNiepewność: {uncertainty:.1%}"
    return text, '#16A34A' if prediction == 1 else '#2563EB'


def generate_case_studies(
    case_data: List[Dict],
    output_dir: Path = Path('results/case_studies'),
    save_pdf: bool = True
) -> List[Path]:
    """
    Generuje wizualizacje case studies.

    Każdy case to dict zawierający:
    - 'patient_id': str – identyfikator pacjenta
    - 'true_label': int – rzeczywista etykieta (0=CN, 1=MCI)
    - 'volume': np.ndarray – wolumin 3D (D, H, W) lub (1, D, H, W)
    - 'models': Dict[str, dict] – per-model wyniki:
        - 'prediction': int
        - 'confidence': float
        - 'is_abstained': bool
        - 'uncertainty': float (opcja)
    - 'case_type': str – opis typu case study

    Args:
        case_data: Lista case dictionaries
        output_dir: Katalog wyjściowy
        save_pdf: Czy zapisywać PDF

    Returns:
        Lista ścieżek do zapisanych plików
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    class_names = {0: 'CN (zdrowy)', 1: 'MCI'}

    for case_idx, case in enumerate(case_data):
        patient_id = case.get('patient_id', f'Pacjent #{case_idx + 1}')
        true_label = case['true_label']
        volume = case['volume']
        models = case['models']
        case_type = case.get('case_type', '')

        # Wyciągnij środkowe przekroje
        slices = _extract_slices(volume, slice_fractions=(0.5,))[0]
        slice_names = ['Sagittalny', 'Koronalny', 'Osiowy']

        n_models = len(models)

        # Layout: wiersz 1 = MRI wycinki, wiersze 2+ = decyzje modeli
        fig = plt.figure(figsize=(14, 4 + 2.5 * n_models))

        # ── Wiersz 1: Wycinki MRI ────────────────────────────────────────
        gs = fig.add_gridspec(1 + n_models, 3, hspace=0.4, wspace=0.15,
                              height_ratios=[1.2] + [0.6] * n_models)

        for col, (sl, sl_name) in enumerate(zip(slices, slice_names)):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(sl.T if col == 0 else sl, cmap='gray', aspect='auto')
            ax.set_title(f'Przekrój {sl_name}', fontsize=11, fontweight='bold')
            ax.axis('off')

        # ── Wiersze 2+: Decyzje modeli ──────────────────────────────────
        model_colors_map = {
            'Baseline (SR)':          '#2563EB',
            'SelectiveNet':           '#16A34A',
            'Evidential (EDL)':       '#EA580C',
            'Hybrid (3D-ResNet-EDL)': '#DC2626',
        }

        for row_idx, (model_name, model_result) in enumerate(models.items()):
            pred = model_result['prediction']
            conf = model_result['confidence']
            is_abs = model_result['is_abstained']
            unc = model_result.get('uncertainty', None)

            decision_text, decision_color = _format_decision(pred, conf, is_abs, unc)

            # Jeden merged subplot na cały wiersz
            ax = fig.add_subplot(gs[1 + row_idx, :])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            # Tło kolorowe
            bg_alpha = 0.15 if not is_abs else 0.25
            bg_color = decision_color
            ax.axhspan(0, 1, facecolor=bg_color, alpha=bg_alpha)

            # Lewa strona: nazwa modelu
            ax.text(0.02, 0.5, model_name,
                    fontsize=12, fontweight='bold', va='center',
                    color=model_colors_map.get(model_name, '#333'))

            # Prawa strona: decyzja
            ax.text(0.98, 0.5, decision_text,
                    fontsize=11, va='center', ha='right',
                    color=decision_color, fontweight='bold')

            # Obramowanie
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(decision_color)
                spine.set_linewidth(2)

        # ── Tytuł ogólny ────────────────────────────────────────────────
        true_class = class_names.get(true_label, str(true_label))
        suptitle = f'{patient_id} – Rzeczywista klasa: {true_class}'
        if case_type:
            suptitle += f'\n({case_type})'
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

        # Zapisz
        path = output_dir / f'case_{case_idx + 1:02d}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        if save_pdf:
            fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)

        saved_paths.append(path)
        print(f"✅ Case study saved: {path}")

    return saved_paths


def find_interesting_cases(
    all_predictions: Dict[str, np.ndarray],
    all_labels: np.ndarray,
    all_confidences: Dict[str, np.ndarray],
    all_uncertainties: Optional[Dict[str, np.ndarray]] = None,
    max_cases: int = 5,
) -> List[Dict]:
    """
    Automatycznie identyfikuje najciekawsze przypadki do case studies.

    Kryteria:
    1. Baseline FP → Hybrid abstains (najważniejszy!)
    2. Baseline FP → Hybrid correct (CN)
    3. Trudne przypadki: hybryda i baseline mają niską pewność
    4. Łatwe przypadki: hybryda pewna, baseline pewny (kontrola)

    Args:
        all_predictions: Dict[model_name -> predictions array]
        all_labels: Ground truth labels
        all_confidences: Dict[model_name -> confidence array]
        all_uncertainties: Dict[model_name -> uncertainty array] (opcja)
        max_cases: Max liczba case studies

    Returns:
        Lista indices z metadanymi (do użycia z generate_case_studies po
        załadowaniu woluminów)
    """
    cases = []

    baseline_preds = all_predictions.get('Baseline (SR)', None)
    hybrid_preds = all_predictions.get('Hybrid (3D-ResNet-EDL)', None)

    if baseline_preds is None or hybrid_preds is None:
        print("⚠️ Brak predykcji Baseline lub Hybrid – pomijam case studies")
        return cases

    n_samples = len(all_labels)

    # ── Typ 1: Baseline FP, Hybrid ABSTAINS ──────────────────────────────
    baseline_fp = (baseline_preds == 1) & (all_labels == 0)
    hybrid_abstains = hybrid_preds == -1

    fp_abstain_indices = np.where(baseline_fp & hybrid_abstains)[0]
    if len(fp_abstain_indices) > 0:
        # Sortuj po niepewności hybrid (malejąco – najciekawsze)
        if all_uncertainties and 'Hybrid (3D-ResNet-EDL)' in all_uncertainties:
            hybrid_unc = all_uncertainties['Hybrid (3D-ResNet-EDL)']
            sorted_idx = fp_abstain_indices[np.argsort(hybrid_unc[fp_abstain_indices])[::-1]]
        else:
            sorted_idx = fp_abstain_indices
        for idx in sorted_idx[:2]:
            cases.append({
                'index': int(idx),
                'case_type': 'Baseline FP → Hybrid odmawia diagnozy (kluczowy!)',
            })

    # ── Typ 2: Baseline FP, Hybrid correct ──────────────────────────────
    hybrid_correct = hybrid_preds == all_labels
    fp_correct_indices = np.where(baseline_fp & hybrid_correct)[0]
    if len(fp_correct_indices) > 0:
        for idx in fp_correct_indices[:1]:
            cases.append({
                'index': int(idx),
                'case_type': 'Baseline FP → Hybrid poprawna diagnoza CN',
            })

    # ── Typ 3: Trudne – obie niepewne ────────────────────────────────────
    if all_uncertainties and 'Hybrid (3D-ResNet-EDL)' in all_uncertainties:
        hybrid_unc = all_uncertainties['Hybrid (3D-ResNet-EDL)']
        baseline_conf = all_confidences.get('Baseline (SR)', np.ones(n_samples))

        # Oba niepewne: hybrid unc > median, baseline conf < median
        unc_median = np.median(hybrid_unc)
        conf_median = np.median(baseline_conf)
        hard_mask = (hybrid_unc > unc_median) & (baseline_conf < conf_median)
        hard_indices = np.where(hard_mask)[0]
        if len(hard_indices) > 0:
            idx = hard_indices[np.argmax(hybrid_unc[hard_indices])]
            cases.append({
                'index': int(idx),
                'case_type': 'Trudny przypadek – oba modele niepewne',
            })

    # ── Typ 4: Kontrola – pewna, poprawna diagnoza ──────────────────────
    both_correct = (baseline_preds == all_labels) & (hybrid_preds == all_labels)
    if all_uncertainties and 'Hybrid (3D-ResNet-EDL)' in all_uncertainties:
        hybrid_unc = all_uncertainties['Hybrid (3D-ResNet-EDL)']
        easy_correct = both_correct & (hybrid_unc < np.percentile(hybrid_unc, 25))
    else:
        easy_correct = both_correct

    easy_indices = np.where(easy_correct & (all_labels == 1))[0]  # MCI poprawne
    if len(easy_indices) > 0:
        cases.append({
            'index': int(easy_indices[0]),
            'case_type': 'Kontrola – pewna poprawna diagnoza MCI',
        })

    return cases[:max_cases]
