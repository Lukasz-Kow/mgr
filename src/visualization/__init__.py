"""
Moduł wizualizacji dla projektu klasyfikacji MCI.

Zawiera narzędzia do generowania:
- Krzywych Risk-Coverage i ROC
- Macierzy konfuzji
- Histogramów niepewności (epistemiczna / aleatoryczna)
- Case studies z wycinkami skanów MRI
"""

from .plot_curves import (
    plot_risk_coverage_comparison,
    plot_roc_curves_comparison,
    plot_confusion_matrices,
)
from .uncertainty_plots import (
    plot_uncertainty_histograms,
    plot_uncertainty_scatter,
    plot_uncertainty_vs_evidence,
)
from .case_studies import generate_case_studies

__all__ = [
    # Krzywe porównawcze
    'plot_risk_coverage_comparison',
    'plot_roc_curves_comparison',
    'plot_confusion_matrices',
    # Niepewność
    'plot_uncertainty_histograms',
    'plot_uncertainty_scatter',
    'plot_uncertainty_vs_evidence',
    # Case studies
    'generate_case_studies',
]
