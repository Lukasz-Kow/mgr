#!/usr/bin/env python3
"""
Master script: sprawdza checkpointy, trenuje brakujące modele, uruchamia ewaluację.
"""
import sys
import os
import torch
import yaml
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHECKPOINTS = {
    'baseline':      'checkpoints/baseline/best_model.pth',
    'selective_net':  'checkpoints/selective_net/best_model.pt',
    'evidential':     'checkpoints/evidential/best_model.pt',
    'hybrid':         'checkpoints/hybrid/best_model.pt',
}

TRAIN_SCRIPTS = {
    'baseline':      'scripts/train_baseline.py',
    'selective_net':  'scripts/train_selectivenet.py',
    'evidential':     'scripts/train_evidential.py',
    'hybrid':         'scripts/train_hybrid.py',
}

def check_checkpoint(name, path):
    """Sprawdza checkpoint – czy istnieje i czy jest 3D."""
    if not os.path.exists(path):
        return 'NOT_FOUND', None
    
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    
    is_3d = False
    for k in list(sd.keys()):
        # Check any weight tensor
        if 'weight' in k and isinstance(sd[k], torch.Tensor):
            # 3D kernels are 5D (out_c, in_c, d, h, w)
            if len(sd[k].shape) == 5:
                is_3d = True
                break
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return '3D' if is_3d else '2D', size_mb


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    
    print("=" * 70)
    print("  MASTER PIPELINE: CHECK → TRAIN → EVALUATE")
    print("=" * 70)
    
    # ── ETAP 1: Sprawdź checkpointy ──────────────────────────────────────
    print("\n📋 ETAP 1: Sprawdzanie checkpointów...")
    needs_training = []
    
    for name, path in CHECKPOINTS.items():
        backbone_type, size = check_checkpoint(name, path)
        if backbone_type == 'NOT_FOUND':
            print(f"  ❌ {name}: BRAK CHECKPOINTU → wymaga treningu")
            needs_training.append(name)
        elif backbone_type == '2D':
            print(f"  ⚠️  {name}: Checkpoint 2D ({size:.1f}MB) → wymaga retreningu z 3D")
            needs_training.append(name)
        else:
            print(f"  ✅ {name}: Checkpoint 3D ({size:.1f}MB) → OK")
    
    # ── ETAP 2: Trenuj brakujące modele ──────────────────────────────────
    if needs_training:
        print(f"\n🔧 ETAP 2: Trening {len(needs_training)} modeli: {needs_training}")
        
        for name in needs_training:
            script = TRAIN_SCRIPTS[name]
            if not os.path.exists(script):
                print(f"  ⚠️  Pomijam {name} – brak skryptu {script}")
                continue
            
            print(f"\n{'─' * 60}")
            print(f"  🚀 Trening: {name}")
            print(f"{'─' * 60}")
            
            start = time.time()
            result = subprocess.run(
                [sys.executable, script],
                capture_output=False,
                text=True,
            )
            elapsed = time.time() - start
            
            if result.returncode != 0:
                print(f"  ❌ {name} – trening zakończony błędem (code={result.returncode})")
            else:
                print(f"  ✅ {name} – trening ukończony w {elapsed:.0f}s")
    else:
        print("\n✅ ETAP 2: Wszystkie checkpointy OK – pomijam trening")
    
    # ── ETAP 3: Weryfikacja po treningu ──────────────────────────────────
    print(f"\n📋 ETAP 3: Weryfikacja checkpointów po treningu...")
    all_ok = True
    for name, path in CHECKPOINTS.items():
        backbone_type, size = check_checkpoint(name, path)
        if backbone_type == '3D':
            print(f"  ✅ {name}: 3D ({size:.1f}MB)")
        else:
            print(f"  ❌ {name}: {backbone_type} – PROBLEM!")
            all_ok = False
    
    if not all_ok:
        print("\n❌ Nie wszystkie checkpointy są gotowe. Przerywam.")
        return 1
    
    # ── ETAP 4: Ewaluacja ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  🏆 ETAP 4: Uruchamiam pełną ewaluację")
    print(f"{'=' * 70}")
    
    result = subprocess.run(
        [sys.executable, 'scripts/evaluate_all.py'],
        capture_output=False,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"\n❌ Ewaluacja zakończona błędem (code={result.returncode})")
        return 1
    
    print("\n✅ Pipeline zakończony pomyślnie!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
