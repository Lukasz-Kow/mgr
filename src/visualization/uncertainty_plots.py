#!/usr/bin/env python3
"""
Wizualizacja rozkładów niepewności modeli ewidencyjnych.

Generuje:
1. Histogramy KDE – niepewność epistemiczna / aleatoryczna wg poprawności
2. Scatter plot – epistemiczna vs aleatoryczna
3. Box plot – siła dowodów (S) vs poprawność predykcji
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional


# ── Style ────────────────────────────────────────────────────────────────────

GROUP_COLORS = {
    'Poprawne CN':     '#2563EB',  # blue
    'Poprawne MCI':    '#16A34A',  # green
    'Fałszywie Pozy. (FP)': '#DC2626',  # red
    'Fałszywie Neg. (FN)': '#EA580C',  # orange
}


def _classify_samples(predictions, labels, num_classes=2):
    """
    Dzieli próbki na 4 grupy: poprawne CN, poprawne MCI, FP, FN.

    Konwencja: 0 = CN (negatywna), 1 = MCI (pozytywna).

    Returns:
        Dict[str, np.ndarray] – maski boolean per grupa
    """
    correct = predictions == labels
    groups = {
        'Poprawne CN':     correct & (labels == 0),
        'Poprawne MCI':    correct & (labels == 1),
        'Fałszywie Pozy. (FP)': ~correct & (labels == 0),  # zdrowy → MCI
        'Fałszywie Neg. (FN)': ~correct & (labels == 1),   # MCI → zdrowy
    }
    return groups


# ── 1. Histogramy niepewności ────────────────────────────────────────────────

def plot_uncertainty_histograms(
    predictions: np.ndarray,
    labels: np.ndarray,
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    model_name: str = 'Hybrid (3D-ResNet-EDL)',
    output_dir: Path = Path('results'),
    save_pdf: bool = True
) -> list:
    """
    Generuje nakładane histogramy KDE dla niepewności epistemicznej i aleatorycznej,
    podzielone wg poprawności predykcji.

    Args:
        predictions: Predykcje modelu (N,)
        labels: Etykiety rzeczywiste (N,)
        epistemic: Niepewność epistemiczna (N,)
        aleatoric: Niepewność aleatoryczna (N,)
        model_name: Nazwa modelu do tytułu
        output_dir: Katalog wyjściowy
        save_pdf: Czy zapisywać PDF

    Returns:
        Lista ścieżek do zapisanych plików
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = _classify_samples(predictions, labels)
    saved_paths = []

    for unc_name, unc_values, suffix in [
        ('Niepewność epistemiczna', epistemic, 'epistemic'),
        ('Niepewność aleatoryczna', aleatoric, 'aleatoric'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for group_name, mask in groups.items():
            if mask.sum() == 0:
                continue
            color = GROUP_COLORS.get(group_name, '#888888')
            values = unc_values[mask]

            # KDE histogram
            sns.kdeplot(
                values, ax=ax,
                label=f'{group_name} (n={mask.sum()})',
                color=color,
                fill=True, alpha=0.25,
                linewidth=2,
                warn_singular=False,
            )

        ax.set_xlabel(unc_name, fontsize=13)
        ax.set_ylabel('Gęstość', fontsize=13)
        ax.set_title(
            f'{unc_name} – rozkład wg poprawności predykcji\n({model_name})',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
        ax.set_xlim(left=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--')

        path = output_dir / f'uncertainty_histogram_{suffix}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        if save_pdf:
            fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved_paths.append(path)
        print(f"✅ Histogram saved: {path}")

    return saved_paths


# ── 2. Scatter plot: epistemiczna vs aleatoryczna ────────────────────────────

def plot_uncertainty_scatter(
    predictions: np.ndarray,
    labels: np.ndarray,
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    model_name: str = 'Hybrid (3D-ResNet-EDL)',
    output_path: Path = Path('results/uncertainty_scatter.png'),
    save_pdf: bool = True
) -> Path:
    """
    Scatter plot: niepewność epistemiczna vs aleatoryczna,
    kolorowana wg poprawności predykcji.

    Args:
        predictions, labels: Predykcje i etykiety
        epistemic, aleatoric: Wartości niepewności
        model_name: Nazwa modelu
        output_path: Ścieżka zapisu
        save_pdf: Czy zapisywać PDF

    Returns:
        Ścieżka do pliku
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    groups = _classify_samples(predictions, labels)

    fig, ax = plt.subplots(figsize=(9, 7))

    for group_name, mask in groups.items():
        if mask.sum() == 0:
            continue
        color = GROUP_COLORS.get(group_name, '#888888')
        ax.scatter(
            epistemic[mask], aleatoric[mask],
            label=f'{group_name} (n={mask.sum()})',
            color=color,
            alpha=0.5, s=30, edgecolors='white', linewidth=0.3,
        )

    ax.set_xlabel('Niepewność epistemiczna (vacuity)', fontsize=13)
    ax.set_ylabel('Niepewność aleatoryczna (dissonance)', fontsize=13)
    ax.set_title(
        f'Epistemiczna vs Aleatoryczna niepewność\n({model_name})',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    if save_pdf:
        fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"✅ Scatter plot saved: {output_path}")
    return output_path


# ── 3. Box plot: siła dowodów vs poprawność ─────────────────────────────────

def plot_uncertainty_vs_evidence(
    predictions: np.ndarray,
    labels: np.ndarray,
    strength: np.ndarray,
    model_name: str = 'Hybrid (3D-ResNet-EDL)',
    output_path: Path = Path('results/evidence_strength_boxplot.png'),
    save_pdf: bool = True
) -> Path:
    """
    Box plot: siła dowodów (S = Σα) dla predykcji poprawnych vs błędnych.

    Args:
        predictions, labels: Predykcje i etykiety
        strength: Dirichlet strength S (N,)
        model_name: Nazwa modelu
        output_path: Ścieżka zapisu
        save_pdf: Czy zapisywać PDF

    Returns:
        Ścieżka do pliku
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = predictions == labels
    data_correct = strength[correct]
    data_wrong = strength[~correct]

    fig, ax = plt.subplots(figsize=(8, 6))

    box_data = []
    box_labels = []
    box_colors = []

    if len(data_correct) > 0:
        box_data.append(data_correct)
        box_labels.append(f'Poprawne\n(n={len(data_correct)})')
        box_colors.append('#16A34A')

    if len(data_wrong) > 0:
        box_data.append(data_wrong)
        box_labels.append(f'Błędne\n(n={len(data_wrong)})')
        box_colors.append('#DC2626')

    if len(box_data) == 0:
        plt.close(fig)
        return output_path

    bp = ax.boxplot(
        box_data, labels=box_labels,
        patch_artist=True, widths=0.5,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel('Siła dowodów S = Σα', fontsize=13)
    ax.set_title(
        f'Siła dowodów Dirichleta – poprawne vs błędne predykcje\n({model_name})',
        fontsize=14, fontweight='bold'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Dodaj medianę jako tekst
    for i, d in enumerate(box_data):
        median = np.median(d)
        ax.annotate(
            f'mediana: {median:.1f}',
            xy=(i + 1, median),
            xytext=(i + 1.35, median),
            fontsize=10, fontweight='bold', color=box_colors[i],
            arrowprops=dict(arrowstyle='->', color=box_colors[i], lw=1.0),
        )

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    if save_pdf:
        fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"✅ Evidence strength box plot saved: {output_path}")
    return output_path
