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
from typing import Dict, List, Optional, Tuple
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

    ADNI_GROUP_MAPPING = {
        'CN': 0,
        'MCI': 1,
        'LMCI': 1,
        'EMCI': 1,
    }

    VISIT_PRIORITY = {
        'bl': 1,
        'sc': 2,
        'init': 3,
        'scmri': 4,
        'v02': 5,
    }

    METADATA_BASELINE_CSV = 'baseline_2026-02-23.csv'
    METADATA_MCI_CN_CSV = 'mci_cn_scaled2_2026-06-09.csv'
    
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

    def _resolve_metadata_csv_paths(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Locate baseline and MCI_CN metadata CSV files."""
        candidates = [
            self.dataset_root / 'metadata',
            self.dataset_root.parent / 'metadata',
            Path.cwd() / 'Data baseline' / 'metadata',
            Path.cwd(),
        ]
        baseline_csv = None
        mci_cn_csv = None
        for base in candidates:
            b = base / self.METADATA_BASELINE_CSV
            m = base / self.METADATA_MCI_CN_CSV
            if baseline_csv is None and b.exists():
                baseline_csv = b
            if mci_cn_csv is None and m.exists():
                mci_cn_csv = m
        if baseline_csv is None:
            for name in ('Data_baseline_2_23_2026.csv', self.METADATA_BASELINE_CSV):
                for base in (Path.cwd(), self.dataset_root.parent, self.dataset_root):
                    p = base / name
                    if p.exists():
                        baseline_csv = p
                        break
                if baseline_csv is not None:
                    break
        return baseline_csv, mci_cn_csv

    def _load_adni_image_info(self) -> Dict[str, dict]:
        """Merge baseline and MCI_CN CSV metadata keyed by Image Data ID."""
        baseline_csv, mci_cn_csv = self._resolve_metadata_csv_paths()
        image_info: Dict[str, dict] = {}

        def _ingest(df: pd.DataFrame, source: str, overwrite: bool = False) -> None:
            for _, row in df.iterrows():
                image_id = str(row['Image Data ID'])
                if image_id in image_info and not overwrite:
                    continue
                image_info[image_id] = {
                    'group': row['Group'],
                    'description': row['Description'],
                    'subject': row['Subject'],
                    'visit': str(row.get('Visit', '')),
                    'source': source,
                }

        if mci_cn_csv is not None:
            print(f"Loading metadata from {mci_cn_csv}...")
            _ingest(pd.read_csv(mci_cn_csv), 'mci_cn_scaled2')
        if baseline_csv is not None:
            print(f"Loading metadata from {baseline_csv}...")
            _ingest(pd.read_csv(baseline_csv), 'baseline', overwrite=True)

        if not image_info:
            print("[ERROR] No ADNI metadata CSV files found!")
        return image_info

    def _adni_scan_roots(self, adni_root: Path) -> List[Tuple[str, Path]]:
        """Return (cohort_name, path) pairs to scan for NIfTI files."""
        roots: List[Tuple[str, Path]] = []
        baseline_dir = adni_root / 'baseline'
        adni2_dir = adni_root / 'ADNI2'
        if baseline_dir.is_dir():
            roots.append(('baseline', baseline_dir))
        if adni2_dir.is_dir():
            roots.append(('ADNI2', adni2_dir))
        if not roots:
            roots.append(('baseline', adni_root))
        return roots

    @staticmethod
    def _extract_image_id(img_path: Path) -> Optional[str]:
        parent_id = img_path.parent.name
        if parent_id.startswith('I'):
            return parent_id
        match = re.search(r'_(I\d+)$', img_path.stem)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _visit_priority(visit: str) -> int:
        return DatasetMapper.VISIT_PRIORITY.get(str(visit).lower(), 99)

    @staticmethod
    def _preprocess_priority(description: str) -> int:
        desc = description.upper()
        if 'SCALED_2' in desc:
            return 1
        if 'SCALED' in desc:
            return 2
        if 'N3' in desc:
            return 3
        return 4

    @staticmethod
    def _cohort_priority(cohort: str) -> int:
        return 1 if cohort == 'baseline' else 2

    def _relative_data_path(self, img_path: Path) -> str:
        for base in (Path.cwd(), self.dataset_root.parent, self.dataset_root):
            try:
                return str(img_path.resolve().relative_to(base.resolve()))
            except ValueError:
                continue
        return str(img_path)

    def _scan_adni_dataset(self, adni_root: Path) -> pd.DataFrame:
        """
        Skanuje strukturę ADNI (baseline + ADNI2) i mapuje na klasy z CSV.
        Wybiera 1 obraz na pacjenta według priorytetu:
        visit (bl > sc > init > scmri > v02), preprocessing (Scaled_2 > Scaled > N3),
        cohort (baseline > ADNI2).
        """
        print(f"Skanowanie datasetu ADNI w: {adni_root}")
        image_info = self._load_adni_image_info()
        if not image_info:
            return pd.DataFrame()

        nii_files: List[Tuple[str, Path]] = []
        for cohort, scan_root in self._adni_scan_roots(adni_root):
            for img_path in scan_root.rglob('*.nii'):
                if 'Zone.Identifier' not in img_path.name:
                    nii_files.append((cohort, img_path))
        print(f"Znaleziono {len(nii_files)} plików .nii")

        subject_candidates: Dict[str, list] = {}

        for cohort, img_path in nii_files:
            image_id = self._extract_image_id(img_path)
            if image_id is None:
                continue
            info = image_info.get(image_id)
            if info is None:
                continue

            group = info['group']
            label = self.ADNI_GROUP_MAPPING.get(group)
            if label is None:
                continue

            subject = info['subject']
            visit = info.get('visit', '')
            description = info['description']
            sort_key = (
                self._visit_priority(visit),
                self._preprocess_priority(description),
                self._cohort_priority(cohort),
            )

            candidate = {
                'path': self._relative_data_path(img_path),
                'original_class': group,
                'label': label,
                'class_name': self.CLASS_NAMES[label],
                'subject': subject,
                'description': description,
                'visit': visit,
                'cohort': cohort,
                'image_id': image_id,
                '_sort_key': sort_key,
            }
            subject_candidates.setdefault(subject, []).append(candidate)

        final_samples = []
        for candidates in subject_candidates.values():
            best = sorted(candidates, key=lambda x: x['_sort_key'])[0]
            del best['_sort_key']
            final_samples.append(best)

        df = pd.DataFrame(final_samples)
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
                print(f"[WARNING] DATA LEAKAGE: {len(overlap)} patients in {s1} and {s2}: {list(overlap)[:5]}...")
            else:
                print(f"[OK] No leakage between {s1} and {s2}")
        
        total_subjects = df['subject'].nunique()
        print(f"\\nTotal unique subjects: {total_subjects}")
    
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
        print(f"\\nMetadata saved to: {output_path}")


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
