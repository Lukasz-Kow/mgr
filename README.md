# Redukcja Fałszywych Alarmów w Diagnostyce MCI

Hybrydowy model 3D-CNN wykorzystujący uczenie ewidencyjne (Evidential Learning) i mechanizm selektywnej predykcji do klasyfikacji MCI (Mild Cognitive Impairment) vs CN (Cognitive Normal).

## 📋 Opis Projektu

Projekt realizuje porównanie różnych metod selektywnej predykcji w diagnostyce medycznej:
1. **Baseline**: Softmax Response z progowaniem
2. **SelectiveNet**: Dedykowana architektura z głowicą selekcyjną
3. **Evidential Deep Learning**: Kwantyfikacja niepewności przez rozkład Dirichleta
4. **Hybrid**: 3D-ResNet-EDL łączący zalety deep features i evidential heads

## 🗂️ Struktura Projektu

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
├── tests/                 # Unit testy
├── docs/                  # Dokumentacja
├── Alzheimer_MRI_4_classes_dataset/  # Dataset (nie w repo)
├── environment.yml        # Conda environment
└── requirements.txt       # Pip requirements
```

## 🚀 Instalacja

### Opcja 1: Conda (Zalecane)
```bash
conda env create -f environment.yml
conda activate mci_classification
```

### Opcja 2: Pip
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## 📊 Dataset

Projekt używa zbioru danych Alzheimer MRI z 4 klasami:
- **NonDemented** → CN (Cognitive Normal)
- **VeryMildDemented** + **MildDemented** → MCI
- **ModerateDemented** → Wykluczony z klasyfikacji binarnej

Dataset powinien być umieszczony w `Alzheimer_MRI_4_classes_dataset/`.

## 🔧 Użycie

### Preprocessing i Mapowanie Klas
```bash
python scripts/prepare_dataset.py --config configs/data_config.yaml
```

### Trening Modeli
```bash
# Baseline (Softmax Response)
python scripts/train_baseline.py --config configs/baseline_config.yaml

# SelectiveNet
python scripts/train_selectivenet.py --config configs/selectivenet_config.yaml

# Evidential Deep Learning
python scripts/train_evidential.py --config configs/evidential_config.yaml

# Hybrid Model (3D-ResNet-EDL)
python scripts/train_hybrid.py --config configs/hybrid_config.yaml
```

### Ewaluacja i Porównanie
```bash
# Wygenerowanie wspólnego wykresu Risk-Coverage dla wszystkich modeli
python src/visualization/plot_curves.py

# Finalne zestawienie metryk (Accuracy, AUGRC, Sens@95%Spec)
python scripts/evaluate_all.py
```

## 📈 Metryki

Projekt implementuje następujące metryki zgodnie z wymaganiami:
- **Risk-Coverage Curve**: Ryzyko vs pokrycie
- **AUGRC**: Area Under Generalized Risk-Coverage curve
- **Sensitivity @ Fixed Specificity**: Czułość przy TNR=95%
- Standard: Accuracy, Precision, Recall, F1, AUC-ROC

## 📚 Literatura

- Wen, J., et al. (2020). "Convolutional neural networks for classification of Alzheimer's disease"
- Geifman, Y., & El-Yaniv, R. (2019). "SelectiveNet: A Deep Neural Network with an Integrated Reject Option"
- Sensoy, M., et al. (2018). "Evidential Deep Learning to Quantify Classification Uncertainty"

## 🧪 Testy

```bash
pytest tests/ -v
```

## 📝 Autor

Adam Stefański - Praca Magisterska

## 📄 Licencja

Projekt edukacyjny - Praca Magisterska
