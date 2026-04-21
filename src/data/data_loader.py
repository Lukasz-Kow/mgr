"""
PyTorch Dataset and DataLoader for MCI classification.

Current: 2D implementation for mock dataset
Future: Easy migration to 3D ADNI data
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
from pathlib import Path

from .preprocessing import MRIPreprocessor, AugmentationPipeline


class MCIDataset(Dataset):
    """
    Dataset dla klasyfikacji MCI vs CN.
    
    Supports both 2D (current) and 3D (future) data.
    """
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        preprocessor: MRIPreprocessor,
        augmentation: Optional[AugmentationPipeline] = None,
        split: str = 'train'
    ):
        """
        Args:
            metadata_df: DataFrame z kolumnami: path, label, split
            preprocessor: MRIPreprocessor instance
            augmentation: Opcjonalny augmentation pipeline
            split: 'train', 'val', lub 'test'
        """
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.split = split
        
        # Filtruj dane według split
        self.data = metadata_df[metadata_df['split'] == split].reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError(f"No data found for split: {split}")
        
        print(f"[{split.upper()}] Loaded {len(self.data)} samples")
        
        # Statystyki klas
        class_counts = self.data['label'].value_counts().sort_index()
        for label, count in class_counts.items():
            class_name = self.data[self.data['label'] == label]['class_name'].iloc[0]
            print(f"  {class_name} (label={label}): {count} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Zwraca pojedynczy sample.
        
        Returns:
            image: Tensor shape (1, H, W) dla 2D lub (1, D, H, W) dla 3D
            label: Integer label (0=CN, 1=MCI)
            metadata: Dict z dodatkowymi informacjami
        """
        row = self.data.iloc[idx]
        
        # Load and preprocess
        image = self.preprocessor.preprocess(row['path'])
        
        # Apply augmentation (only for training)
        if self.augmentation is not None and self.split == 'train':
            image = self.augmentation(image)
        
        label = int(row['label'])
        
        # Metadata
        metadata = {
            'path': row['path'],
            'class_name': row['class_name'],
            'original_class': row.get('original_class', 'unknown')
        }
        
        return image, label, metadata


class MCIDataModule:
    """
    DataModule zarządzający DataLoaders dla train/val/test.
    Inspirowane PyTorch Lightning DataModule.
    """
    
    def __init__(
        self,
        metadata_csv: str,
        preprocessor_config: dict,
        batch_size: int = 16,
        num_workers: int = 4,
        augmentation_config: Optional[dict] = None
    ):
        """
        Args:
            metadata_csv: Ścieżka do pliku CSV z metadanymi
            preprocessor_config: Config dla preprocessora
            batch_size: Batch size
            num_workers: Liczba workerów dla DataLoader
            augmentation_config: Config dla augmentacji (tylko train)
        """
        self.metadata_csv = metadata_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load metadata
        self.metadata_df = pd.read_csv(metadata_csv)
        print(f"Loaded metadata from {metadata_csv}")
        print(f"Total samples: {len(self.metadata_df)}")
        
        # Create preprocessors
        from .preprocessing import get_preprocessor, get_augmentation
        self.preprocessor = get_preprocessor(preprocessor_config)
        
        # Augmentation tylko dla train
        self.train_augmentation = get_augmentation(
            augmentation_config or {}, 
            is_train=True
        )
        
        # Datasets (lazy initialization)
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
    
    def train_dataset(self) -> MCIDataset:
        """Get or create train dataset."""
        if self._train_dataset is None:
            self._train_dataset = MCIDataset(
                self.metadata_df,
                self.preprocessor,
                self.train_augmentation,
                split='train'
            )
        return self._train_dataset
    
    def val_dataset(self) -> MCIDataset:
        """Get or create validation dataset."""
        if self._val_dataset is None:
            self._val_dataset = MCIDataset(
                self.metadata_df,
                self.preprocessor,
                augmentation=None,  # No augmentation for val
                split='val'
            )
        return self._val_dataset
    
    def test_dataset(self) -> MCIDataset:
        """Get or create test dataset."""
        if self._test_dataset is None:
            self._test_dataset = MCIDataset(
                self.metadata_df,
                self.preprocessor,
                augmentation=None,  # No augmentation for test
                split='test'
            )
        return self._test_dataset
    
    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Create train DataLoader."""
        return DataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch for stable training
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Oblicza wagi klas dla imbalanced dataset (inverse frequency).
        
        Returns:
            Tensor shape (num_classes,) z wagami
        """
        train_df = self.metadata_df[self.metadata_df['split'] == 'train']
        class_counts = train_df['label'].value_counts().sort_index()
        
        total = len(train_df)
        weights = torch.tensor([
            total / count for count in class_counts
        ], dtype=torch.float32)
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        print(f"Class weights: {weights.tolist()}")
        return weights


# Helper function dla custom collate
def mci_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of (image, label, metadata) tuples
        
    Returns:
        images: Batched tensor
        labels: Batched labels
        metadata_list: List of metadata dicts
    """
    images, labels, metadata_list = zip(*batch)
    
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels, list(metadata_list)
