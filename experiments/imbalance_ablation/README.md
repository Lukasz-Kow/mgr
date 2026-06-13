# Eksperyment A/B/C — nierównowaga klas MCI vs CN

Porównanie trzech strategii treningu na 4 modelach (12 runów).

| Arm | Strategia | Opis |
|-----|-----------|------|
| arm_c | `balanced_sampler` | WeightedRandomSampler 50/50 (obecny setup) |
| arm_a | `natural` | Naturalny rozkład 34/66 + progi val→test |
| arm_b | `cost_sensitive` | Inverse-frequency w lossie |

## Uruchomienie

```powershell
conda activate mgr
cd C:\Users\Lukas\mgr

# Wszystkie 12 treningów (resume automatyczny)
python scripts/run_imbalance_ablation.py

# Tylko wybrane
python scripts/run_imbalance_ablation.py --arms arm_a --models hybrid

# Ewaluacja po treningu
python scripts/run_imbalance_ablation.py --eval-only
python scripts/evaluate_ablation.py

# Wdrożenie zwycięzcy
python scripts/promote_imbalance_winner.py --strategy natural --dry-run
python scripts/promote_imbalance_winner.py --strategy natural
# promote automatycznie archiwizuje produkcję przed nadpisaniem checkpointów

## Archiwizacja (nie tracić starych danych)

Przed nowym treningiem ablacji — snapshot obecnych checkpointów produkcyjnych:

```powershell
python scripts/archive_snapshot.py --target production
python scripts/run_imbalance_ablation.py --archive-production
```

Lista archiwów:

```powershell
python scripts/archive_snapshot.py --list
```

Struktura:

```
archives/
  production/2026-06-13_120000/
    manifest.json
    checkpoints/baseline/best_model.pth
    results/final_comparison.csv
  ablation/2026-06-13_pre_cleanup_arm_a/
    imbalance_ablation.zip
    manifest.json
``` --cleanup
```

## Wyniki

- `results/comparison_matrix.csv` — macierz pivot (model × metryka × arm)
- `results/hybrid_focus.csv` — wycinek dla modelu hybrydowego
- `results/winner.json` — auto-rekomendacja

## Metryki kluczowe

- FP @70/80/90/95% Spec (val→test)
- Sens @70/80/90/95% Spec (val→test)
- AUGRC, FP reduction @20%
