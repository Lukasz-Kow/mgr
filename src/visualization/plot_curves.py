#!/usr/bin/env python3
"""
Wizualizacja krzywych porównawczych modeli.

Generuje:
1. Risk-Coverage Curve – porównanie wszystkich modeli
2. ROC Curve – z zaznaczonym punktem 95% Specificity
3. Macierze konfuzji – heatmapy per model
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, auc, confusion_matrix


# ── Global style ─────────────────────────────────────────────────────────────
# Jasny styl – czytelny w druku (praca magisterska)
STYLE_DEFAULTS = dict(
    figure_dpi=300,
    font_size=12,
    title_size=14,
    label_size=12,
    legend_size=10,
    grid_alpha=0.3,
    line_width=2.0,
)

# Profesjonalna paleta kolorów (dobra w druku i na ekranie)
MODEL_COLORS = {
    'Baseline (SR)':             '#2563EB',  # blue
    'SelectiveNet':              '#16A34A',  # green
    'Evidential (EDL)':          '#EA580C',  # orange
    'Hybrid (3D-ResNet-EDL)':    '#DC2626',  # red
}

MODEL_MARKERS = {
    'Baseline (SR)':             'o',
    'SelectiveNet':              's',
    'Evidential (EDL)':          '^',
    'Hybrid (3D-ResNet-EDL)':    'D',
}


def _apply_style():
    """Stosuje globalny styl matplotlib."""
    plt.rcParams.update({
        'font.size': STYLE_DEFAULTS['font_size'],
        'axes.titlesize': STYLE_DEFAULTS['title_size'],
        'axes.labelsize': STYLE_DEFAULTS['label_size'],
        'legend.fontsize': STYLE_DEFAULTS['legend_size'],
        'figure.dpi': STYLE_DEFAULTS['figure_dpi'],
        'axes.grid': True,
        'grid.alpha': STYLE_DEFAULTS['grid_alpha'],
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    sns.set_palette("deep")


def _save_figure(fig, output_path: Path, save_pdf: bool = True):
    """Zapisuje wykres w PNG i opcjonalnie PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=STYLE_DEFAULTS['figure_dpi'],
                bbox_inches='tight', facecolor='white')
    if save_pdf:
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── 1. Risk-Coverage Curve ──────────────────────────────────────────────────

def plot_risk_coverage_comparison(
    model_results: Dict[str, dict],
    output_path: Path = Path('results/risk_coverage_comparison.png'),
    save_pdf: bool = True
) -> Path:
    """
    Rysuje krzywe Risk-Coverage dla wielu modeli na jednym wykresie.

    Args:
        model_results: Dict z kluczami = nazwa modelu, wartości = dict z:
            - 'coverages': np.ndarray
            - 'risks': np.ndarray
            - 'augrc': float
        output_path: Ścieżka zapisu wykresu
        save_pdf: Czy zapisywać również PDF

    Returns:
        Ścieżka do zapisanego pliku PNG
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name, data in model_results.items():
        coverages = data['coverages']
        risks = data['risks']
        augrc = data.get('augrc', 0.0)
        color = MODEL_COLORS.get(model_name, '#888888')

        # Sortuj po coverage malejąco dla czytelności
        sort_idx = np.argsort(coverages)[::-1]
        coverages_sorted = coverages[sort_idx]
        risks_sorted = risks[sort_idx]

        ax.plot(
            coverages_sorted, risks_sorted,
            label=f'{model_name} (AUGRC={augrc:.4f})',
            color=color,
            linewidth=STYLE_DEFAULTS['line_width'],
            alpha=0.9,
        )

        # Zaznacz punkt pełnego pokrycia (coverage=1.0)
        if len(coverages_sorted) > 0:
            ax.scatter(
                coverages_sorted[0], risks_sorted[0],
                color=color, marker=MODEL_MARKERS.get(model_name, 'o'),
                s=60, zorder=5, edgecolors='white', linewidth=0.5,
            )

    ax.set_xlabel('Coverage (udział zaakceptowanych próbek)', fontsize=13)
    ax.set_ylabel('Risk (współczynnik błędu)', fontsize=13)
    ax.set_title('Krzywe Risk-Coverage – porównanie modeli', fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=-0.01)
    ax.invert_xaxis()  # Coverage od 1.0 do 0.0 (konwencja)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Anotacja: idealna krzywa
    ax.annotate(
        '← Niższy AUGRC = lepszy model',
        xy=(0.5, 0.02), fontsize=9, color='gray', fontstyle='italic',
        ha='center',
    )

    _save_figure(fig, output_path, save_pdf)
    print(f"✅ Risk-Coverage saved: {output_path}")
    return output_path


# ── 2. ROC Curve ────────────────────────────────────────────────────────────

def plot_roc_curves_comparison(
    model_results: Dict[str, dict],
    output_path: Path = Path('results/roc_curves_comparison.png'),
    target_specificity: float = 0.95,
    save_pdf: bool = True
) -> Path:
    """
    Rysuje krzywe ROC z zaznaczonym punktem 95% Specificity.

    Args:
        model_results: Dict z kluczami = nazwa modelu, wartości = dict z:
            - 'labels': np.ndarray (ground truth)
            - 'probabilities': np.ndarray (prob of positive class)
            - 'sensitivity_at_95spec': float (opcjonalnie)
        output_path: Ścieżka zapisu
        target_specificity: Docelowa specyficzność
        save_pdf: Czy zapisywać PDF

    Returns:
        Ścieżka do pliku
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Linia diagonalna (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random (AUC=0.50)')

    for model_name, data in model_results.items():
        labels = data['labels']
        probs = data['probabilities']
        color = MODEL_COLORS.get(model_name, '#888888')

        # Compute ROC
        fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Linia ROC
        ax.plot(
            fpr, tpr,
            label=f'{model_name} (AUC={roc_auc:.3f})',
            color=color,
            linewidth=STYLE_DEFAULTS['line_width'],
        )

        # Zaznacz punkt 95% Specificity
        specificity = 1 - fpr
        idx = np.argmin(np.abs(specificity - target_specificity))
        sens = tpr[idx]
        spec_actual = specificity[idx]

        ax.scatter(
            fpr[idx], tpr[idx],
            color=color, s=120, zorder=5,
            marker=MODEL_MARKERS.get(model_name, 'o'),
            edgecolors='black', linewidth=1.5,
        )
        # Etykieta obok punktu
        offset_x = 0.03
        offset_y = -0.04 if sens > 0.5 else 0.04
        ax.annotate(
            f'Sens={sens:.2f}\n@Spec={spec_actual:.0%}',
            xy=(fpr[idx], tpr[idx]),
            xytext=(fpr[idx] + offset_x, tpr[idx] + offset_y),
            fontsize=8,
            color=color,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
        )

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax.set_title('Krzywe ROC – porównanie modeli\n(punkt ● = operating point @ 95% Specificity)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Zacieniowanie strefy "wysoka specyficzność"
    ax.axvspan(0, 1 - target_specificity, alpha=0.06, color='green',
               label=f'_Strefa ≥{target_specificity:.0%} Spec')

    _save_figure(fig, output_path, save_pdf)
    print(f"✅ ROC Curves saved: {output_path}")
    return output_path


# ── 3. Macierze konfuzji ───────────────────────────────────────────────────

def plot_confusion_matrices(
    model_results: Dict[str, dict],
    output_path: Path = Path('results/confusion_matrices.png'),
    class_names: List[str] = None,
    save_pdf: bool = True
) -> Path:
    """
    Rysuje macierze konfuzji jako heatmapy dla wielu modeli.

    Args:
        model_results: Dict z kluczami = nazwa modelu, wartości = dict z:
            - 'predictions': np.ndarray
            - 'labels': np.ndarray
            - 'num_abstained': int (opcjonalnie)
        output_path: Ścieżka zapisu
        class_names: Nazwy klas (domyślnie ['CN', 'MCI'])
        save_pdf: Czy zapisywać PDF

    Returns:
        Ścieżka do pliku
    """
    _apply_style()

    if class_names is None:
        class_names = ['CN', 'MCI']

    n_models = len(model_results)
    if n_models == 0:
        return output_path

    # Grid layout: max 2 kolumny
    n_cols = min(n_models, 2)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (model_name, data) in enumerate(model_results.items()):
        ax = axes[idx]

        preds = data['predictions']
        labels = data['labels']

        # Filtruj abstencje
        mask = preds != -1
        preds_filtered = preds[mask]
        labels_filtered = labels[mask]
        n_abstained = (~mask).sum()

        cm = confusion_matrix(labels_filtered, preds_filtered, labels=list(range(len(class_names))))

        # Heatmapa
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=False,
            annot_kws={'size': 16, 'fontweight': 'bold'},
            linewidths=1, linecolor='white',
        )
        ax.set_xlabel('Predykcja', fontsize=12)
        ax.set_ylabel('Rzeczywista klasa', fontsize=12)

        title = model_name
        if n_abstained > 0:
            total = len(preds)
            title += f'\n(abstencje: {n_abstained}/{total} = {n_abstained/total:.1%})'
        ax.set_title(title, fontsize=12, fontweight='bold')

    # Ukryj puste subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Macierze konfuzji – porównanie modeli', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()

    _save_figure(fig, output_path, save_pdf)
    print(f"✅ Confusion Matrices saved: {output_path}")
    return output_path
