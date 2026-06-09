# Redukcja Fałszywych Alarmów w Diagnostyce MCI

Hybrydowy model 3D-CNN wykorzystujący uczenie ewidencyjne (Evidential Learning) i mechanizm selektywnej predykcji do klasyfikacji MCI (Mild Cognitive Impairment) vs CN (Cognitive Normal).

## Opis Projektu

Projekt realizuje porównanie różnych metod selektywnej predykcji w diagnostyce medycznej:
1. **Baseline**: Softmax Response z progowaniem
2. **SelectiveNet**: Dedykowana architektura z głowicą selekcyjną
3. **Evidential Deep Learning**: Kwantyfikacja niepewności przez rozkład Dirichleta
4. **Hybrid**: 3D-ResNet-EDL łączący zalety deep features i evidential heads

## Struktura Projektu

```
mgr/
├── src/
│   ├── data/              # Data loading i preprocessing
│   ├── models/            # Architektury modeli
│   ├── training/          # Training loop, losses, optimizers
│   ├── evaluation/        # Metryki i ewaluacja
│   └── visualization/     # Plotting i wykresy
├── configs/               # Pliki konfiguracyjne YAML
├── scripts/               # Skrypty treningowe i ewaluacyjne
├── docs/                  # Dokumentacja pracy
├── environment.yml        # Conda environment (nazwa: mgr)
└── run_windows.ps1        # Uruchamianie na Windows bez WSL
```

## Instalacja — Windows (bez WSL, zalecane)

### Wymagania
- Windows 10/11
- Miniconda/Anaconda z środowiskiem `mgr`
- NVIDIA GPU z obsługą CUDA (np. GTX 1050)
- Dane ADNI w `Data baseline/ADNI/`

### 1. Środowisko Conda

```powershell
conda env create -f environment.yml
conda activate mgr
```

Jeśli środowisko `mgr` już istnieje, zaktualizuj pakiety:
```powershell
conda activate mgr
pip install -r requirements.txt
```

### 2. Jednorazowa konfiguracja PowerShell

Wyłącz aliasy Pythona ze Sklepu Microsoft (Ustawienia → Aliasy wykonywania aplikacji → wyłącz `python.exe`).

Zainicjalizuj conda w PowerShell:
```powershell
& "C:\Users\Lukas\miniconda3\Scripts\conda.exe" init powershell
```
Zrestartuj terminal, potem: `conda activate mgr`

### 3. Weryfikacja

```powershell
cd C:\Users\Lukas\mgr
.\run_windows.ps1 -Verify
```

### 4. Pełny potok

```powershell
.\run_windows.ps1 -PrepareData
.\run_windows.ps1 -TrainAll
.\run_windows.ps1 -Evaluate
```

Opcja czystego retreningu (usuwa checkpointy):
```powershell
.\run_windows.ps1 -Clean -PrepareData -TrainAll
```

### Faza 2 — MONAI ResNet3D-10 + MedicalNet (wszystkie 4 modele)

Faza 2 podmienia backbone na MONAI ResNet3D-10 z wagami MedicalNet (transfer learning z MRI). Checkpointy Fazy 1 pozostają w `checkpoints/baseline`, `selective_net`, `evidential`, `hybrid`; Faza 2 zapisuje do `checkpoints/*_monai`.

```powershell
pip install monai huggingface_hub
python scripts/verify_monai_backbone.py

# Retrening wszystkich 4 modeli MONAI (~10-14 h na GTX 1050)
.\run_windows.ps1 -Phase2 -TrainAll -Clean

# Ewaluacja porównawcza Faza 1 vs Faza 2 (8 wierszy w tabeli)
.\run_windows.ps1 -Phase2 -Evaluate
# lub: python scripts/evaluate_all.py --include-phase2
```

Wagi MedicalNet są pobierane z HuggingFace (`TencentMedicalNet/MedicalNet-Resnet10`) i cache'owane w `~/.cache/huggingface`. Przy braku internetu można ręcznie umieścić `resnet_10_23dataset.pth` w cache HF.

## Dataset

Projekt używa podzbioru ADNI (MCI vs CN):
- Dane obrazowe: `Data baseline/ADNI/`
- Etykiety: `Data_baseline_2_23_2026.csv`
- Metadane (generowane): `data_metadata_adni.csv`

```powershell
python scripts/prepare_adni_dataset.py
```

## Trening Modeli

Checkpoint wybierany po **Sens@80%Spec** (czułość przy ustalonej specyficzności).

```powershell
conda activate mgr
python scripts/train_baseline.py
python scripts/train_selectivenet.py
python scripts/train_evidential.py
python scripts/train_hybrid.py
```

Lub automatycznie (tylko brakujące modele):
```powershell
python scripts/check_and_train_all.py
```

**Uwaga:** `batch_size: 1` w configach — dostosowane do GTX 1050 (4 GB VRAM).

## Ewaluacja

```powershell
python scripts/evaluate_all.py
```

Wyniki trafiają do `results/` (tabela, krzywe ROC, Risk-Coverage).

## Metryki (SMART / Measurable)

- **Sensitivity @ 80% Specificity** — główna metryka kliniczna (protokół val→test)
- **AUGRC** — Area Under Generalized Risk-Coverage curve
- **Coverage** — pokrycie (udział nie-abstencji)
- Risk-Coverage, FP reduction, AUC-ROC

## Literatura

- Wen, J., et al. (2020). "Convolutional neural networks for classification of Alzheimer's disease"
- Geifman, Y., & El-Yaniv, R. (2019). "SelectiveNet"
- Sensoy, M., et al. (2018). "Evidential Deep Learning"

## Testy

```powershell
pytest tests/ -v
```
