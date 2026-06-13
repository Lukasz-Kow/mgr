# Data baseline — ADNI

## Struktura

| Ścieżka | Zawartość |
|---|---|
| `ADNI/baseline/` | Stary zbiór: 147 pacjentów, wizyta `bl`, 871 plików `.nii` |
| `ADNI/ADNI2/` | Nowe pobranie: 959 folderów pacjentów, 1742 pliki `.nii` |
| `metadata/baseline_2026-02-23.csv` | Etykiety CN/MCI dla 147 pacjentów (baseline) |
| `metadata/mci_cn_scaled2_2026-06-09.csv` | Etykiety CN/MCI/EMCI/LMCI dla 1328 pacjentów |
| `metadata/inventory_report.json` | Wygenerowany raport (liczby, nakładanie, braki) |

## Statystyki (po uporządkowaniu)

- **978** unikalnych pacjentów na dysku (128 w obu kohortach)
- **978** pacjentów w `data_metadata_adni.csv` (zgodność 1:1 z danymi na dysku)
- Usunięto 20 pustych folderów ADNI2 bez plików `.nii` (niedokończone pobranie)
- Podział klas: ~34% CN, ~66% MCI

## Wybór skanu (1 na pacjenta)

Pipeline w `src/data/dataset_mapper.py` wybiera najlepszy skan według:

1. Wizyta: `bl` → `sc` → `init` → `scmri` → `v02`
2. Preprocessing: `Scaled_2` → `Scaled` → `N3` → pozostałe
3. Kohorta: `baseline/` ma pierwszeństwo przed `ADNI2/` (przy duplikatach)

## Komendy

```powershell
conda activate mgr

# Raport inwentaryzacji
python scripts/organize_adni_data.py

# Regeneracja pliku treningowego
python scripts/prepare_dataset.py --dataset_root "Data baseline" --output data_metadata_adni.csv
```

## Uwagi

- Pliki `.nii` są w `.gitignore` — foldery pacjentów są widoczne w IDE, skany mogą być ukryte (ustawienie „Hide ignored files”).
- Kopia zapasowa danych ADNI2 pozostaje w OneDrive: `Dokumenty\mgr\Data baseline\ADNI\ADNI2\`.
- 18 pacjentów ze starego CSV nie występuje w `mci_cn_scaled2` — etykiety pochodzą wyłącznie z `baseline_2026-02-23.csv`.
