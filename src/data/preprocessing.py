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
from scipy.ndimage import rotate, shift


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
        rotation_range: float = 10.0,  # degrees
        random_brightness: float = 0.1,
        random_contrast: float = 0.1,
        shift_3d_range: float = 5.0,   # pixels
        noise_std: float = 0.01        # std of gaussian noise
    ):
        """
        Args:
            horizontal_flip: Czy stosować horizontal flip
            rotation_range: Zakres rotacji w stopniach (+/-) na każdą z 3 osi
            random_brightness: Zakres zmiany jasności
            random_contrast: Zakres zmiany kontrastu
            shift_3d_range: Zakres przesunięcia w pikselach (+/-)
            noise_std: Odchylenie standardowe szumu Gaussa
        """
        self.horizontal_flip = horizontal_flip
        self.rotation_range = rotation_range
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        self.shift_3d_range = shift_3d_range
        self.noise_std = noise_std

    def _random_3d_rotation(self, tensor: torch.Tensor) -> torch.Tensor:
        """Wykonuje losowe obroty 3D na wszystkich osiach."""
        # Kąty dla osi X, Y, Z
        angles = [np.random.uniform(-self.rotation_range, self.rotation_range) for _ in range(3)]
        
        # Konwersja do numpy (C, D, H, W) -> (D, H, W) dla pojedynczego kanału
        arr = tensor.squeeze(0).numpy()
        
        # Obroty na 3 płaszczyznach
        # Axial (płaszczyzna H-W, oś D)
        arr = rotate(arr, angles[0], axes=(1, 2), reshape=False, order=1, mode='constant', cval=0.0)
        # Coronal (płaszczyzna D-W, oś H)
        arr = rotate(arr, angles[1], axes=(0, 2), reshape=False, order=1, mode='constant', cval=0.0)
        # Sagittal (płaszczyzna D-H, oś W)
        arr = rotate(arr, angles[2], axes=(0, 1), reshape=False, order=1, mode='constant', cval=0.0)
        
        return torch.from_numpy(arr).unsqueeze(0)

    def _random_3d_shift(self, tensor: torch.Tensor) -> torch.Tensor:
        """Wykonuje losowe przesunięcie 3D."""
        shifts = [np.random.uniform(-self.shift_3d_range, self.shift_3d_range) for _ in range(3)]
        
        arr = tensor.squeeze(0).numpy()
        arr = shift(arr, shift=shifts, order=1, mode='constant', cval=0.0)
        
        return torch.from_numpy(arr).unsqueeze(0)

    def _random_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Dodaje szum Gaussa."""
        if self.noise_std > 0:
            noise = torch.randn_like(tensor) * self.noise_std
            return tensor + noise
        return tensor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Aplikuje intensywne augmentacje 3D dla ADNI lub 2D dla Kaggle.
        
        Args:
            tensor: Input tensor (1, D, H, W) lub (1, H, W)
            
        Returns:
            Augmented tensor
        """
        is_3d = (len(tensor.shape) == 4)
        
        if is_3d:
            # --- AUGMENTACJE 3D (ADNI) ---
            
            # 1. Losowa rotacja we wszystkich trzech osiach
            if self.rotation_range > 0:
                tensor = self._random_3d_rotation(tensor)
                
            # 2. Losowe przesunięcie (Shift)
            if self.shift_3d_range > 0:
                tensor = self._random_3d_shift(tensor)
                
            # 3. Horizontal Flip (anatomical left-right)
            # W 3D (C, D, H, W), oś W to dim=3
            if self.horizontal_flip and torch.rand(1).item() > 0.5:
                tensor = torch.flip(tensor, dims=[-1])
                
        else:
            # --- AUGMENTACJE 2D (Kaggle) ---
            if self.horizontal_flip and torch.rand(1).item() > 0.5:
                tensor = torch.flip(tensor, dims=[-1])
            
            if self.rotation_range > 0:
                import torchvision.transforms.functional as TF
                angle = (torch.rand(1).item() - 0.5) * 2 * self.rotation_range
                tensor = TF.rotate(tensor, angle, fill=0.0)
        
        # --- AUGMENTACJE WSPÓLNE (Intensywność / Szum) ---
        
        # 4. Szum Gaussa (kluczowy dla robustness EDL)
        if self.noise_std > 0:
            tensor = self._random_gaussian_noise(tensor)
        
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
        rotation_range=aug_config.get('rotation_range', 10.0),
        random_brightness=aug_config.get('random_brightness', 0.1),
        random_contrast=aug_config.get('random_contrast', 0.1),
        shift_3d_range=aug_config.get('shift_3d_range', 5.0),
        noise_std=aug_config.get('noise_std', 0.01)
    )
