#!/usr/bin/env python3
"""
Skrypt wnioskowania (inference) dla pojedynczego skanu MRI.

Użycie:
    python scripts/predict.py path/to/scan.nii.gz
    python scripts/predict.py path/to/scan.nii.gz --model hybrid --threshold 0.5
    python scripts/predict.py path/to/scan.nii.gz --visualize

Zwraca:
    - Diagnozę (MCI / CN / ODMOWA DIAGNOZY)
    - Pewność modelu
    - Niepewność epistemiczną i aleatoryczną (dla modeli ewidencyjnych)
    - Siłę dowodów Dirichleta (S)
"""

import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import MRIPreprocessor
from src.models.backbone import get_backbone, ResNetBackbone2D
from src.models.baseline_softmax import BaselineSoftmaxModel
from src.models.selective_net import SelectiveNet
from src.models.hybrid_model import HybridEvidentialModel
from src.models.evidential_layer import EvidentialLayer, compute_uncertainty
import torch.nn as nn


# EDLModel – identyczny jak w train_evidential.py
class EDLModel(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.evidential_head = EvidentialLayer(backbone.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.evidential_head(features)


# ── Stałe ────────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: 'CN (Cognitively Normal)', 1: 'MCI (Mild Cognitive Impairment)'}
CLASS_NAMES_SHORT = {0: 'CN', 1: 'MCI'}

MODEL_TYPE_MAP = {
    'baseline':    {'config': 'configs/baseline_config.yaml',    'type': 'baseline'},
    'selectivenet':{'config': 'configs/selectivenet_config.yaml','type': 'selectivenet'},
    'evidential':  {'config': 'configs/evidential_config.yaml',  'type': 'evidential'},
    'hybrid':      {'config': 'configs/hybrid_config.yaml',      'type': 'hybrid'},
}


def load_model_for_inference(model_key: str, device: torch.device):
    """
    Ładuje model na podstawie klucza.

    Args:
        model_key: 'baseline', 'selectivenet', 'evidential', 'hybrid'
        device: torch device

    Returns:
        (model, model_type, config)
    """
    if model_key not in MODEL_TYPE_MAP:
        raise ValueError(
            f"Nieznany model: {model_key}. "
            f"Dostępne: {list(MODEL_TYPE_MAP.keys())}"
        )

    m_info = MODEL_TYPE_MAP[model_key]
    cfg_path = Path(m_info['config'])

    if not cfg_path.exists():
        raise FileNotFoundError(f"Brak pliku konfiguracyjnego: {cfg_path}")

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Szukaj checkpointu
    ckpt_dir = Path(cfg['checkpoint']['dir'])
    ckpt_path = None
    for fname in ['best_model.pt', 'best_model.pth']:
        candidate = ckpt_dir / fname
        if candidate.exists():
            ckpt_path = candidate
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            f"Brak checkpointu w {ckpt_dir}. "
            f"Najpierw wytrenuj model."
        )

    # Buduj model – architektura musi odpowiadać treningowi!
    if m_info['type'] == 'baseline':
        backbone = ResNetBackbone2D(
            arch=cfg['model']['backbone'].get('arch_2d', 'resnet18'),
            pretrained=False,
            in_channels=cfg['model']['backbone'].get('in_channels', 1)
        )
        model = BaselineSoftmaxModel(backbone, num_classes=2).to(device)
    elif m_info['type'] == 'selectivenet':
        backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
        model = SelectiveNet(backbone, num_classes=2).to(device)
    elif m_info['type'] == 'evidential':
        backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
        model = EDLModel(backbone, num_classes=2).to(device)
    elif m_info['type'] == 'hybrid':
        backbone = get_backbone(cfg['model']['backbone'], force_3d=True)
        model = HybridEvidentialModel(backbone, num_classes=2).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, m_info['type'], cfg


def preprocess_scan(scan_path: str, config: dict) -> torch.Tensor:
    """
    Wczytuje i przetwarza skan NIfTI.

    Args:
        scan_path: Ścieżka do pliku .nii.gz
        config: Config z sekcją preprocessing

    Returns:
        Tensor shape (1, 1, D, H, W) – gotowy do inference
    """
    # Ładuj data config dla preprocessora
    with open('configs/data_config.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)

    preprocessor = MRIPreprocessor(
        target_size=tuple(data_cfg['preprocessing']['target_size']),
        normalize_method=data_cfg['preprocessing']['normalize_method']
    )

    # Preprocess (zwraca tensor (1, D, H, W))
    tensor = preprocessor.preprocess(scan_path)

    # Dodaj wymiar batch: (1, 1, D, H, W)
    tensor = tensor.unsqueeze(0)

    return tensor


def predict_single(model, model_type, tensor, device, threshold=0.5):
    """
    Wykonuje predykcję na pojedynczym skanie.

    Returns:
        Dict z wynikami
    """
    tensor = tensor.to(device)

    with torch.no_grad():
        if model_type == 'baseline':
            preds, confs, probs = model.predict_with_confidence(tensor)
            pred = preds[0].item()
            conf = confs[0].item()
            is_abstained = conf < threshold

            return {
                'prediction': pred,
                'confidence': conf,
                'is_abstained': is_abstained,
                'probabilities': probs[0].cpu().numpy(),
                'rejection_method': 'Softmax Response',
                'threshold': threshold,
            }

        elif model_type == 'selectivenet':
            preds_rej, confs, sel_probs, abstained = model.predict_with_selection(
                tensor, threshold=threshold
            )
            pred = preds_rej[0].item()
            is_abs = abstained[0].item()

            return {
                'prediction': pred if not is_abs else -1,
                'confidence': confs[0].item(),
                'selection_probability': sel_probs[0].item(),
                'is_abstained': is_abs,
                'rejection_method': 'SelectiveNet Selection Head',
                'threshold': threshold,
            }

        elif model_type == 'hybrid':
            # HybridEvidentialModel ma wbudowaną metodę predict_with_uncertainty
            preds, probs, epi_unc, unc_dict = model.predict_with_uncertainty(tensor)

            pred = preds[0].item()
            epi = unc_dict['epistemic'][0].item()
            ale = unc_dict['aleatoric'][0].item()
            total = unc_dict['total'][0].item()
            strength_val = unc_dict['strength'][0].item()

            is_abstained = epi > threshold

            return {
                'prediction': pred,
                'confidence': 1.0 - epi,
                'is_abstained': is_abstained,
                'probabilities': probs[0].cpu().numpy(),
                'epistemic_uncertainty': epi,
                'aleatoric_uncertainty': ale,
                'total_uncertainty': total,
                'dirichlet_strength': strength_val,
                'rejection_method': 'Evidential Epistemic Uncertainty',
                'threshold': threshold,
            }

        elif model_type == 'evidential':
            # EDLModel nie ma predict_with_uncertainty – liczymy ręcznie
            alpha = model(tensor)
            strength = alpha.sum(dim=1, keepdim=True)
            probs = alpha / strength
            pred = torch.argmax(probs, dim=1)[0].item()

            epi_unc, ale_unc, total_unc = compute_uncertainty(alpha)
            epi = epi_unc[0].item()
            ale = ale_unc[0].item()
            total = total_unc[0].item()
            strength_val = strength[0, 0].item()

            is_abstained = epi > threshold

            return {
                'prediction': pred,
                'confidence': 1.0 - epi,
                'is_abstained': is_abstained,
                'probabilities': probs[0].cpu().numpy(),
                'epistemic_uncertainty': epi,
                'aleatoric_uncertainty': ale,
                'total_uncertainty': total,
                'dirichlet_strength': strength_val,
                'rejection_method': 'Evidential Epistemic Uncertainty',
                'threshold': threshold,
            }


def visualize_prediction(scan_path, result, output_path=None):
    """
    Generuje wizualizację skanu z predykcją.

    Args:
        scan_path: Ścieżka do skanu
        result: Dict z wynikami predykcji
        output_path: Ścieżka wyjściowa (None = wyświetl)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nibabel as nib

    # Załaduj oryginalny skan
    img = nib.load(scan_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]

    D, H, W = data.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Wycinki
    slices = [
        (data[D // 2, :, :], 'Sagittalny'),
        (data[:, H // 2, :], 'Koronalny'),
        (data[:, :, W // 2], 'Osiowy'),
    ]

    for ax, (sl, name) in zip(axes, slices):
        ax.imshow(sl.T, cmap='gray', origin='lower')
        ax.set_title(name, fontsize=12)
        ax.axis('off')

    # Tytuł z predykcją
    pred = result['prediction']
    is_abs = result['is_abstained']

    if is_abs:
        title = '⚠️ ODMOWA DIAGNOZY – zbyt duża niepewność'
        title_color = '#EA580C'
    else:
        class_name = CLASS_NAMES.get(pred, str(pred))
        title = f'Diagnoza: {class_name}'
        title_color = '#DC2626' if pred == 1 else '#2563EB'

    conf = result.get('confidence', 0)
    subtitle = f'Pewność: {conf:.1%}'
    if 'epistemic_uncertainty' in result:
        subtitle += f' | Unc.epi: {result["epistemic_uncertainty"]:.1%}'
        subtitle += f' | S: {result["dirichlet_strength"]:.1f}'

    fig.suptitle(title, fontsize=15, fontweight='bold', color=title_color, y=1.05)
    fig.text(0.5, 0.98, subtitle, ha='center', fontsize=11, color='gray')

    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✅ Wizualizacja zapisana: {output_path}")
    else:
        plt.show()


def print_result(result: dict, scan_path: str):
    """Wyświetla wynik predykcji w czytelnej formie."""
    print()
    print("═" * 60)
    print(f"  🧠 PREDYKCJA MCI – Skan: {Path(scan_path).name}")
    print("═" * 60)

    pred = result['prediction']
    is_abs = result['is_abstained']

    if is_abs:
        print()
        print("  ⚠️  ODMOWA DIAGNOZY")
        print("  Powód: Zbyt duża niepewność modelu.")
        print("  Decyzja: Niejednoznaczny przypadek –")
        print("           wymagana dodatkowa analiza kliniczna.")
    else:
        class_name = CLASS_NAMES.get(pred, str(pred))
        print(f"\n  📋 Diagnoza:  {class_name}")

    print()
    print("  ─── Szczegóły ─────────────────────────────────")
    print(f"  Pewność modelu:      {result.get('confidence', 0):.1%}")

    if 'epistemic_uncertainty' in result:
        print(f"  Niepewność epist.:   {result['epistemic_uncertainty']:.1%}")
        print(f"  Niepewność aleat.:   {result['aleatoric_uncertainty']:.1%}")
        print(f"  Niepewność łączna:   {result['total_uncertainty']:.1%}")
        print(f"  Siła dowodów (S):    {result['dirichlet_strength']:.2f}")

    if 'selection_probability' in result:
        print(f"  Prawdop. selekcji:   {result['selection_probability']:.1%}")

    print(f"\n  Mechanizm odrzuceń:  {result.get('rejection_method', 'N/A')}")
    print(f"  Próg odrzucenia:     {result.get('threshold', 'N/A')}")

    if 'probabilities' in result:
        probs = result['probabilities']
        print(f"\n  P(CN):   {probs[0]:.4f}")
        print(f"  P(MCI):  {probs[1]:.4f}")

    print("═" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Predykcja MCI/CN dla pojedynczego skanu MRI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python scripts/predict.py data/patient_001.nii.gz
  python scripts/predict.py data/patient_001.nii.gz --model hybrid --threshold 0.3
  python scripts/predict.py data/patient_001.nii.gz --visualize --output results/pred.png
        """
    )
    parser.add_argument(
        'scan_path',
        type=str,
        help='Ścieżka do skanu MRI (format NIfTI: .nii lub .nii.gz)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='hybrid',
        choices=['baseline', 'selectivenet', 'evidential', 'hybrid'],
        help='Nazwa modelu (domyślnie: hybrid)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Próg dla mechanizmu odrzucenia (domyślnie: 0.5)'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generuj wizualizację skanu z predykcją'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Ścieżka wyjściowa dla wizualizacji (domyślnie: results/prediction.png)'
    )
    args = parser.parse_args()

    # Walidacja ścieżki
    scan_path = Path(args.scan_path)
    if not scan_path.exists():
        print(f"❌ Plik nie istnieje: {scan_path}")
        sys.exit(1)

    if not (str(scan_path).endswith('.nii') or str(scan_path).endswith('.nii.gz')):
        print(f"⚠️ Uwaga: Oczekiwany format NIfTI (.nii / .nii.gz), otrzymano: {scan_path.suffix}")

    # Załaduj model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    print(f"📂 Ładowanie modelu: {args.model}...")

    model, model_type, cfg = load_model_for_inference(args.model, device)
    print(f"✅ Model załadowany.")

    # Preprocess
    print(f"🔄 Przetwarzanie skanu: {scan_path.name}...")
    tensor = preprocess_scan(str(scan_path), cfg)
    print(f"   Wymiary tensora: {tensor.shape}")

    # Predykcja
    print(f"🧠 Predykcja (próg={args.threshold})...")
    result = predict_single(model, model_type, tensor, device, threshold=args.threshold)

    # Wyświetl wynik
    print_result(result, str(scan_path))

    # Wizualizacja
    if args.visualize:
        output_path = args.output or 'results/prediction.png'
        visualize_prediction(str(scan_path), result, output_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
