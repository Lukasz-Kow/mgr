"""
Preprocessing utilities for MRI images.

NOTE: Current implementation for 2D images (JPG).
When migrating to 3D ADNI data (NIfTI), update:
- Load functions to use nibabel
- Normalization to handle 3D volumes
- Add skull stripping if needed
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import torch


class MRIPreprocessor:
    """Preprocessor dla obrazów MRI - wersja 2D z możliwością rozszerzenia do 3D."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_method: str = 'zscore'  # 'zscore', 'minmax', or 'none'
    ):
        """
        Args:
            target_size: Docelowy rozmiar obrazu (height, width)
            normalize_method: Metoda normalizacji intensywności
        """
        self.target_size = target_size
        self.normalize_method = normalize_method
        
    def load_image(self, path: str) -> np.ndarray:
        """
        Ładuje obraz (2D JPG/PNG lub 3D NIfTI).
        
        Args:
            path: Ścieżka do pliku obrazu (relatywna lub absolutna)
            
        Returns:
            Numpy array z wartościami float
        """
        import os
        from pathlib import Path as _Path
        
        # Rozwiąż ścieżkę relatywną względem katalogu głównego projektu
        # (aby działało zarówno z WSL jak i z Windows przez \\wsl.localhost\...)
        if not os.path.isabs(path):
            # Szukamy katalogu projektu: src/data/preprocessing.py → ../../
            project_root = _Path(__file__).resolve().parent.parent.parent
            resolved = project_root / path
            if resolved.exists():
                path = str(resolved)
            # Jeśli nie istnieje, spróbuj oryginalnej ścieżki (może CWD jest poprawny)
        
        path_lower = path.lower()
        if path_lower.endswith('.nii') or path_lower.endswith('.nii.gz'):
            import nibabel as nib
            img = nib.load(path)
            # Pobierz dane i zorientuj do standardu RAS+ jeśli trzeba
            # (Dla uproszczenia bierzemy surowe dane)
            img_array = img.get_fdata(dtype=np.float32)
            
            # ADNI NIfTI często mają 4 wymiary (ostatni to czas/modality), 
            # bierzemy pierwszy wolumin jeśli tak jest.
            if len(img_array.shape) == 4:
                img_array = img_array[..., 0]
            
            return img_array
        else:
            # Fallback do 2D
            img = Image.open(path).convert('L')  # Grayscale
            img_array = np.array(img, dtype=np.float32)
            return img_array
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize obrazu (2D lub 3D).
        
        Args:
            image: Obraz shape (H, W) lub (D, H, W)
            
        Returns:
            Resized image
        """
        if len(image.shape) == 3:
            # 3D Resize using scipy.ndimage.zoom
            from scipy.ndimage import zoom
            
            target_d, target_h, target_w = self.target_size
            curr_d, curr_h, curr_w = image.shape
            
            zoom_factors = (target_d/curr_d, target_h/curr_h, target_w/curr_w)
            # order=1 to interpolacja liniowa (szybsza i zwykle wystarczająca)
            return zoom(image, zoom_factors, order=1)
        else:
            # 2D Resize
            img_pil = Image.fromarray(image)
            # target_size to zwykle (H, W) dla 2D
            size2d = (self.target_size[-2], self.target_size[-1])
            img_resized = img_pil.resize(size2d, Image.BILINEAR)
            return np.array(img_resized, dtype=np.float32)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizuje intensywności obrazu.
        
        Args:
            image: Obraz (2D lub 3D)
            
        Returns:
            Znormalizowany obraz
        """
        if self.normalize_method == 'zscore':
            # Z-score normalization (mean=0, std=1)
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                image = (image - mean) / std
            else:
                image = image - mean
                
        elif self.normalize_method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
            else:
                image = np.zeros_like(image)
                
        elif self.normalize_method == 'none':
            pass
        else:
            raise ValueError(f"Unknown normalize_method: {self.normalize_method}")
        
        return image
    
    def preprocess(self, path: str) -> torch.Tensor:
        """
        Pełny pipeline preprocessing dla pojedynczego obrazu.
        
        Args:
            path: Ścieżka do pliku obrazu
            
        Returns:
            Tensor shape (1, H, W) dla 2D lub (1, D, H, W) dla 3D
        """
        # Load
        image = self.load_image(path)
        
        # Resize
        image = self.resize(image)
        
        # Normalize
        image = self.normalize(image)
        
        # Convert to tensor with channel dimension
        tensor = torch.from_numpy(image).unsqueeze(0)  # Add channel dim
        
        return tensor


class AugmentationPipeline:
    """Augmentacje dla obrazów medycznych - kompatybilne z 2D i 3D."""
    
    def __init__(
        self,
        horizontal_flip: bool = True,
        rotation_range: float = 15.0,  # degrees
        random_brightness: float = 0.1,
        random_contrast: float = 0.1
    ):
        """
        Args:
            horizontal_flip: Czy stosować horizontal flip
            rotation_range: Zakres rotacji w stopniach (+/-)
            random_brightness: Zakres zmiany jasności
            random_contrast: Zakres zmiany kontrastu
        """
        self.horizontal_flip = horizontal_flip
        self.rotation_range = rotation_range
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Aplikuje augmentacje do tensora (obsługuje 2D i 3D).
        
        Args:
            tensor: Input tensor (1, H, W) dla 2D lub (1, D, H, W) dla 3D
            
        Returns:
            Augmented tensor
        """
        # Random horizontal flip (along the last dimension)
        if self.horizontal_flip and torch.rand(1).item() > 0.5:
            # W 2D (C, H, W) flipujemy W (dim=2)
            # W 3D (C, D, H, W) flipujemy W (dim=3)
            tensor = torch.flip(tensor, dims=[-1])
        
        # Simple rotation (only for 2D, for 3D it's more complex)
        if self.rotation_range > 0 and len(tensor.shape) == 3:
            import torchvision.transforms.functional as TF
            angle = (torch.rand(1).item() - 0.5) * 2 * self.rotation_range
            tensor = TF.rotate(tensor, angle, fill=0.0)
        # TODO: Add 3D rotation if needed using scipy or torch-based affine
        
        # Random brightness and contrast (dimension-agnostic)
        if self.random_brightness > 0:
            brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.random_brightness
            tensor = tensor * brightness_factor
            
        if self.random_contrast > 0:
            contrast_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.random_contrast
            mean = tensor.mean()
            tensor = (tensor - mean) * contrast_factor + mean
        
        return tensor


def get_preprocessor(config: dict) -> MRIPreprocessor:
    """
    Factory function do tworzenia preprocessora z config dict.
    
    Args:
        config: Dictionary z konfiguracją
        
    Returns:
        MRIPreprocessor instance
    """
    return MRIPreprocessor(
        target_size=tuple(config.get('target_size', [224, 224])),
        normalize_method=config.get('normalize_method', 'zscore')
    )


def get_augmentation(config: dict, is_train: bool = True) -> Optional[AugmentationPipeline]:
    """
    Factory function do tworzenia augmentation pipeline.
    
    Args:
        config: Dictionary z konfiguracją
        is_train: Czy to trening (augmentacje tylko dla train)
        
    Returns:
        AugmentationPipeline lub None
    """
    if not is_train:
        return None
    
    aug_config = config.get('augmentation', {})
    if not aug_config.get('enabled', True):
        return None
    
    return AugmentationPipeline(
        horizontal_flip=aug_config.get('horizontal_flip', True),
        rotation_range=aug_config.get('rotation_range', 15.0),
        random_brightness=aug_config.get('random_brightness', 0.1),
        random_contrast=aug_config.get('random_contrast', 0.1)
    )
