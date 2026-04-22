"""
Dataset Mapper - Mapowanie klas 4-klasowego datasetu na binarne MCI vs CN.

Class Mapping:
- NonDemented → 0 (CN - Cognitive Normal)
- VeryMildDemented → 1 (MCI - Mild Cognitive Impairment)
- MildDemented → 1 (MCI)
- ModerateDemented → EXCLUDED (zbyt zaawansowana demencja)
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import random


class DatasetMapper:
    """Klasa do mapowania klas Alzheimer dataset na binarne MCI vs CN."""
    
    # Mapowanie klas
    CLASS_MAPPING = {
        'NonDemented': 0,           # CN
        'VeryMildDemented': 1,      # MCI
        'MildDemented': 1,          # MCI
        'ModerateDemented': None    # Excluded
    }
    
    CLASS_NAMES = {
        0: 'CN',
        1: 'MCI'
    }
    
    def __init__(self, dataset_root: str):
        """
        Args:
            dataset_root: Ścieżka do głównego folderu datasetu
        """
        self.dataset_root = Path(dataset_root)
        self.metadata = []
        
    def scan_dataset(self) -> pd.DataFrame:
        """
        Skanuje dataset i tworzy metadane dla wszystkich obrazów.
        Obsługuje zarówno strukturę folderów klas (mock), jak i strukturę ADNI.
        
        Returns:
            DataFrame z kolumnami: path, original_class, label, class_name
        """
        # Sprawdź czy w folderze jest podfolder ADNI (nowa struktura)
        adni_path = self.dataset_root / 'ADNI'
        if adni_path.exists() and adni_path.is_dir():
            return self._scan_adni_dataset(adni_path)
            
        print("Skanowanie datasetu (struktura klasowa)...")
        for class_folder in self.dataset_root.iterdir():
            if not class_folder.is_dir():
                continue
                
            original_class = class_folder.name
            label = self.CLASS_MAPPING.get(original_class)
            
            # Pomijamy ModerateDemented
            if label is None:
                print(f"Pomijam klasę: {original_class}")
                continue
            
            class_name = self.CLASS_NAMES[label]
            
            # Przeszukuj wszystkie pliki (w tym w podfolderach)
            image_files = self._find_images(class_folder)
            
            for img_path in image_files:
                self.metadata.append({
                    'path': str(img_path),
                    'original_class': original_class,
                    'label': label,
                    'class_name': class_name
                })
        
        df = pd.DataFrame(self.metadata)
        self._print_stats(df)
        return df

    def _scan_adni_dataset(self, adni_root: Path) -> pd.DataFrame:
        """
        Skanuje strukturę ADNI i mapuje na klasy z CSV (jeśli dostępne).
        Zapewnia filtrowanie: 1 obraz na pacjenta, według priorytetu:
        Scaled_2 > Scaled > N3.
        """
        print(f"Skanowanie datasetu ADNI w: {adni_root}")
        
        # Znajdź wszystkie pliki .nii
        nii_files = list(adni_root.rglob('*.nii'))
        print(f"Znaleziono {len(nii_files)} plików .nii")
        
        # Jeśli mamy plik metadata CSV
        metadata_csv = Path('Data_baseline_2_23_2026.csv')
        if not metadata_csv.exists():
             metadata_csv = self.dataset_root.parent / 'Data_baseline_2_23_2026.csv'

        if not metadata_csv.exists():
            print("❌ BŁĄD: BRAK pliku metadanych Data_baseline_2_23_2026.csv!")
            return pd.DataFrame()

        print(f"Ładowanie metadanych z {metadata_csv}...")
        csv_df = pd.read_csv(metadata_csv)
        
        # Przygotuj słownik do szybkiego wyszukiwania informacji o obrazie
        # Image Data ID -> (Group, Description, Subject)
        image_info = {}
        for _, row in csv_df.iterrows():
            image_info[str(row['Image Data ID'])] = {
                'group': row['Group'],
                'description': row['Description'],
                'subject': row['Subject']
            }

        # ADNI mapping grupy na label
        ADNI_MAPPING = {
            'CN': 0,
            'MCI': 1,
            'LMCI': 1,
            'EMCI': 1
        }

        # Zbieraj wszystkich kandydatów pogrupowanych po pacjencie (Subject)
        subject_candidates = {}

        for img_path in nii_files:
            if 'Zone.Identifier' in img_path.name:
                continue
                
            # Wyciągnij ID obrazu (zwykle ID to nazwa folderu nadrzędnego)
            parent_id = img_path.parent.name
            
            info = image_info.get(parent_id)
            if info is None:
                # Spróbuj wyciągnąć z nazwy pliku
                name_parts = img_path.stem.split('_')
                if name_parts[-1].startswith('I'):
                    info = image_info.get(name_parts[-1])

            if info is None:
                continue

            group = info['group']
            label = ADNI_MAPPING.get(group)
            if label is None:
                continue # Pomiń AD lub inne niepasujące grupy
            
            subject = info['subject']
            desc = info['description'].upper()
            
            # Punktacja priorytetu (im niższa tym lepsza)
            priority = 4 # Default
            if 'SCALED_2' in desc: priority = 1
            elif 'SCALED' in desc: priority = 2
            elif 'N3' in desc: priority = 3
            
            candidate = {
                'path': str(img_path),
                'original_class': group,
                'label': label,
                'class_name': self.CLASS_NAMES[label],
                'subject': subject,
                'priority': priority,
                'description': info['description']
            }
            
            if subject not in subject_candidates:
                subject_candidates[subject] = []
            subject_candidates[subject].append(candidate)

        # Wybierz najlepszego kandydata dla każdego pacjenta
        final_samples = []
        for subject, candidates in subject_candidates.items():
            # Sortuj po priorytecie i weź pierwszy
            best_candidate = sorted(candidates, key=lambda x: x['priority'])[0]
            final_samples.append(best_candidate)

        df = pd.DataFrame(final_samples)
        
        # Usuwamy pomocnicze kolumny jeśli nie są potrzebne
        if not df.empty and 'priority' in df.columns:
            df = df.drop(columns=['priority'])
            
        self._print_stats(df)
        return df

    def _print_stats(self, df: pd.DataFrame):
        """Pomocnicza funkcja do statystyk."""
        if len(df) == 0:
            print("Dataset jest pusty!")
            return
            
        print("\n" + "="*50)
        print("Dataset Statistics:")
        print("="*50)
        print(f"Total images: {len(df)}")
        print("\nClass distribution:")
        for class_name in df['class_name'].unique():
            count = len(df[df['class_name'] == class_name])
            percentage = (count / len(df)) * 100
            print(f"  {class_name}: {count:5d} ({percentage:5.1f}%)")
        print("="*50)

    def _find_images(self, folder: Path) -> List[Path]:
        """
        Znajduje wszystkie obrazy w folderze (rekurencyjnie).
        
        Args:
            folder: Folder do przeszukania
            
        Returns:
            Lista ścieżek do obrazów
        """
        # Obsługa zarówno 2D jak i 3D
        image_extensions = {'.jpg', '.jpeg', '.png', '.nii', '.nii.gz'}
        images = []
        
        for file_path in folder.rglob('*'):
            if file_path.is_file() and any(file_path.name.lower().endswith(ext) for ext in image_extensions):
                # Pomijaj pliki Zone.Identifier (Windows)
                if 'Zone.Identifier' not in file_path.name:
                    images.append(file_path)
        
        return images
    
    @staticmethod
    def _extract_patient_id(filepath: str) -> str:
        """
        Wyciąga ID pacjenta z nazwy pliku obrazu 2D.
        
        Konwencja nazewnictwa w Alzheimer_MRI_4_classes_dataset:
            '<patient_num> (<slice_num>).jpg' → patient_id = '<subfolder>_<patient_num>'
            '<patient_num>.jpg' → patient_id = '<subfolder>_<patient_num>'
        
        Przykłady:
            'NonDemented_7th_part/9 (11).jpg' → 'NonDemented_7th_part_9'
            'NonDemented_1st_part/12.jpg' → 'NonDemented_1st_part_12'
            
        Dla plików ADNI (.nii), zwraca Subject ID z kolumny 'subject' (obsługiwane osobno).
        
        Args:
            filepath: Ścieżka do pliku obrazu
            
        Returns:
            Unikalny identyfikator pacjenta
        """
        p = Path(filepath)
        filename = p.stem  # np. '9 (11)' lub '8'
        parent = p.parent.name  # np. 'NonDemented_7th_part'
        
        # Wyciągnij numer pacjenta (cyfry na początku nazwy pliku)
        match = re.match(r'^(\d+)', filename)
        if match:
            patient_num = match.group(1)
            return f"{parent}_{patient_num}"
        return f"{parent}_{filename}"

    def create_splits(
        self, 
        df: pd.DataFrame, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Tworzy podział train/val/test na poziomie pacjentów (subject-level split).
        
        Zgodnie z Wen et al. (2020), podział musi gwarantować, że WSZYSTKIE
        obrazy (plastry MRI) tego samego pacjenta trafiają do tego samego zbioru.
        Zapobiega to wyciekowi danych (data leakage) i zawyżaniu wyników.
        
        Args:
            df: DataFrame z metadanymi
            train_ratio: Proporcja zbioru treningowego
            val_ratio: Proporcja zbioru walidacyjnego
            test_ratio: Proporcja zbioru testowego
            stratify: Czy zachować proporcje klas w każdym zbiorze
            random_seed: Seed dla reproducibility
            
        Returns:
            DataFrame z dodatkową kolumną 'split'
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Sumy ratio muszą wynosić 1.0"
        
        random.seed(random_seed)
        
        df = df.copy()
        df['split'] = None
        
        # Wyciągnij patient_id jeśli brak kolumny 'subject'
        if 'subject' not in df.columns:
            df['subject'] = df['path'].apply(self._extract_patient_id)
            print(f"Wyodrębniono {df['subject'].nunique()} unikalnych pacjentów z nazw plików.")
        
        if stratify:
            # Stratified split na poziomie pacjentów - zachowuje proporcje klas
            for label in df['label'].unique():
                class_df = df[df['label'] == label]
                unique_subjects = class_df['subject'].unique().tolist()
                random.shuffle(unique_subjects)
                
                n = len(unique_subjects)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                train_subjects = set(unique_subjects[:n_train])
                val_subjects = set(unique_subjects[n_train:n_train+n_val])
                test_subjects = set(unique_subjects[n_train+n_val:])
                
                # Przypisz WSZYSTKIE obrazy pacjenta do tego samego splitu
                for idx in class_df.index:
                    subj = df.loc[idx, 'subject']
                    if subj in train_subjects:
                        df.loc[idx, 'split'] = 'train'
                    elif subj in val_subjects:
                        df.loc[idx, 'split'] = 'val'
                    else:
                        df.loc[idx, 'split'] = 'test'
        else:
            # Random split na poziomie pacjentów
            unique_subjects = df['subject'].unique().tolist()
            random.shuffle(unique_subjects)
            
            n = len(unique_subjects)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_subjects = set(unique_subjects[:n_train])
            val_subjects = set(unique_subjects[n_train:n_train+n_val])
            test_subjects = set(unique_subjects[n_train+n_val:])
            
            for idx in df.index:
                subj = df.loc[idx, 'subject']
                if subj in train_subjects:
                    df.loc[idx, 'split'] = 'train'
                elif subj in val_subjects:
                    df.loc[idx, 'split'] = 'val'
                else:
                    df.loc[idx, 'split'] = 'test'
        
        # Walidacja: sprawdź brak wycieku między zbiorami
        self._validate_no_leakage(df)
        
        # Statystyki splits
        print("\\n" + "="*50)
        print("Split Statistics (subject-level split, Wen et al. 2020):")
        print("="*50)
        for split in ['train', 'val', 'test']:
            split_df = df[df['split'] == split]
            n_subjects = split_df['subject'].nunique()
            print(f"\\n{split.upper()}:")
            print(f"  Total images: {len(split_df)}")
            print(f"  Total subjects: {n_subjects}")
            for class_name in split_df['class_name'].unique():
                class_split = split_df[split_df['class_name'] == class_name]
                count = len(class_split)
                n_subj = class_split['subject'].nunique()
                print(f"    {class_name}: {count} images, {n_subj} subjects")
        print("="*50)
        
        return df
    
    def _validate_no_leakage(self, df: pd.DataFrame):
        """
        Sprawdza, czy żaden pacjent nie występuje w więcej niż jednym zbiorze.
        Zgłasza błąd jeśli wykryto wyciek danych.
        """
        split_pairs = [('train', 'val'), ('train', 'test'), ('val', 'test')]
        for s1, s2 in split_pairs:
            subjects_s1 = set(df[df['split'] == s1]['subject'])
            subjects_s2 = set(df[df['split'] == s2]['subject'])
            overlap = subjects_s1 & subjects_s2
            if overlap:
                print(f"⚠️  WYCIEK DANYCH: {len(overlap)} pacjentów w {s1} i {s2}: {list(overlap)[:5]}...")
            else:
                print(f"✅ Brak wycieku między {s1} a {s2}")
        
        total_subjects = df['subject'].nunique()
        print(f"\\nŁącznie unikalnych pacjentów: {total_subjects}")
    
    def save_metadata(self, df: pd.DataFrame, output_path: str):
        """
        Zapisuje metadane do pliku CSV.
        
        Args:
            df: DataFrame z metadanymi
            output_path: Ścieżka do pliku wyjściowego CSV
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"\\nMetadane zapisane do: {output_path}")


def main():
    """Przykład użycia."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Map Alzheimer dataset classes to MCI vs CN')
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='Alzheimer_MRI_4_classes_dataset',
        help='Path to dataset root folder'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_metadata.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Train set ratio'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Tworzenie mappera
    mapper = DatasetMapper(args.dataset_root)
    
    # Skanowanie datasetu
    df = mapper.scan_dataset()
    
    # Tworzenie splits
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    df = mapper.create_splits(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
        stratify=True,
        random_seed=args.seed
    )
    
    # Zapisanie
    mapper.save_metadata(df, args.output)
    

if __name__ == '__main__':
    main()
